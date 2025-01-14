import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from rle_assignment.ppo_eval import evaluate

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


gym.register_envs(ale_py)  # unnecessary but helpful for IDEs

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    eval_checkpoint: str = None
    """"if set, no training, only evaluation of the checkpoint is done"""

    # Algorithm specific arguments
    env_id: str = "ALE/SpaceInvaders-v5"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    # IEM specific arguments
    uncertainty_coef: float = 0.1
    """coefficient for the uncertainty reward"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayscaleObservation(env)
        env = gym.wrappers.FrameStackObservation(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        
        # Uncertainty estimation network - now predicts number of steps
        self.uncertainty_net = nn.Sequential(
            layer_init(nn.Linear(512 * 2, 256)),  # Takes current and next state features
            nn.ReLU(),
            layer_init(nn.Linear(256, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, 1)),  # Outputs predicted number of steps
            nn.ReLU()  # Ensure positive step prediction
        )

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), hidden

    def get_uncertainty(self, current_features, next_features):
        """Estimate the uncertainty between current and next state"""
        combined_features = torch.cat([current_features, next_features], dim=1)
        steps_pred = self.uncertainty_net(combined_features)
        return steps_pred


class StateBuffer:
    def __init__(self, num_steps, feature_dim):
        self.num_steps = num_steps
        self.feature_dim = feature_dim
        self.buffer = []  # Will store tuples of (state_features, timestamp)
        self.current_timestamp = 0
        
    def add(self, state_features):
        """Add state features to buffer with timestamp, maintaining max size"""
        self.buffer.append((state_features.detach(), self.current_timestamp))
        self.current_timestamp += 1
        if len(self.buffer) > self.num_steps:
            self.buffer.pop(0)
            
    def get_all_pairs(self):
        """Get all unique ordered pairs of states from buffer
        Returns:
            state1: Earlier states
            state2: Later states
            steps: Number of steps between states
        """
        if len(self.buffer) < 2:
            return None, None, None
            
        # Generate all unique ordered pairs
        pairs = []
        steps = []
        n = len(self.buffer)
        
        # Only take pairs where first state is equal or earlier than second state
        for i in range(n):
            for j in range(i, n):
                state1, t1 = self.buffer[i]
                state2, t2 = self.buffer[j]
                # States are already in chronological order due to buffer structure
                pairs.append((state1, state2))
                steps.append(t2 - t1)
        
        # Convert to tensors
        state1 = torch.stack([p[0] for p in pairs])
        state2 = torch.stack([p[1] for p in pairs])
        steps = torch.tensor(steps, device=state1.device).view(-1, 1)
        
        return state1, state2, steps


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{time.strftime('%Y%m%d_%H%M%S')}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if not args.eval_checkpoint:
        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        uncertainties = torch.zeros((args.num_steps, args.num_envs)).to(device)
        features = torch.zeros((args.num_steps, args.num_envs, 512)).to(device)  # Store features for uncertainty estimation
        
        # Initialize state buffer for each environment
        state_buffers = [StateBuffer(args.num_steps, 512) for _ in range(args.num_envs)]

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)
        next_features = agent.network(next_obs / 255.0)  # Get initial features

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                features[step] = next_features

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, current_features = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_obs = torch.Tensor(next_obs).to(device)
                next_done = torch.Tensor(np.logical_or(terminations, truncations)).to(device)
                
                # Calculate uncertainty-based intrinsic reward
                with torch.no_grad():
                    next_features = agent.network(next_obs / 255.0)
                    steps_pred = agent.get_uncertainty(current_features, next_features)
                    # Higher step prediction means more uncertainty/novelty
                    uncertainty = torch.tanh(steps_pred)
                    uncertainties[step] = uncertainty.flatten()
                    
                    # Add current state features to buffer
                    for env_idx in range(args.num_envs):
                        state_buffers[env_idx].add(current_features[env_idx])
                    
                    # Combine extrinsic and intrinsic rewards
                    combined_reward = reward + args.uncertainty_coef * uncertainty.cpu().numpy().flatten()
                    rewards[step] = torch.tensor(combined_reward).to(device).view(-1)

                if (terminations.any() or truncations.any()) and "episode" in infos:
                    for i in np.argwhere(infos["_episode"]):
                        print(f"global_step={global_step}, episodic_return={infos['episode']['r'][i]}")
                        writer.add_scalar("train/episodic_return", infos["episode"]["r"][i], global_step)
                        writer.add_scalar("train/episodic_length", infos["episode"]["l"][i], global_step)
                        writer.add_scalar("train/episodic_time", infos["episode"]["t"][i], global_step)
                        writer.add_scalar("train/uncertainty_reward", uncertainty.mean().item(), global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)
            b_uncertainties = uncertainties.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, current_features = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions.long()[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    # Uncertainty network loss
                    uncertainty_loss = 0
                    
                    # Get all unique pairs from each environment's buffer
                    for env_idx in range(args.num_envs):
                        state1, state2, steps = state_buffers[env_idx].get_all_pairs()
                        # Predict number of steps between states
                        steps_pred = agent.get_uncertainty(state1, state2)
                        # MSE loss between predicted and actual steps
                        uncertainty_loss += F.mse_loss(steps_pred, steps.float(), reduction='mean')
                    
                    uncertainty_loss = uncertainty_loss / args.num_envs
                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + uncertainty_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/uncertainty_loss", uncertainty_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/StepPerSecond", int(global_step / (time.time() - start_time)), global_step)

        if args.save_model:
            model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)

            print(f"model saved to {model_path}")
        
    model_path = args.eval_checkpoint if args.eval_checkpoint else model_path
    episodic_events = evaluate(
        model_path,
        make_env,
        args.env_id,
        eval_episodes=100,
        run_name=f"{run_name}-eval",
        Model=Agent,
        device=device,
    )

    for idx, event in enumerate(episodic_events):
        writer.add_scalar("eval/episodic_return", event['return'], idx)
        writer.add_scalar("eval/episodic_length", event['length'], idx)
        writer.add_scalar("eval/episodic_time", event['time'], idx)
    
    try:
        writer.close()
        envs.close()
    except:
        # somehow this code throws an "AttributeError: 'RecordVideo' object has no attribute 'enabled'" when running the evaluation code
        pass