import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import tyro
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (
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
    env_id: str = "ALE/SpaceInvaders-v5"
    """the id of the environment"""
    eval_episodes: int = 100
    """number of episodes to evaluate"""

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

class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self):
        return self.action_space.sample()

def evaluate_random(
    env_id: str,
    eval_episodes: int,
    run_name: str,
    capture_video: bool = True,
):
    """
    Evaluate a random policy on the environment.
    Returns a list of dictionaries, each containing the return, length, and time of the episode.
    """
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name)])
    agent = RandomAgent(envs.single_action_space)

    obs, _ = envs.reset()
    episodic_events = []
    
    while len(episodic_events) < eval_episodes:
        action = agent.get_action()
        next_obs, rewards, terminated, truncated, infos = envs.step([action])
        
        if (terminated or truncated) and "episode" in infos:
            print(f"eval_episode={len(episodic_events)}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}, episodic_time={infos['episode']['t']}")
            episodic_events += [{'return': infos['episode']['r'], 'length': infos['episode']['l'], 'time': infos['episode']['t']}]
        obs = next_obs

    return episodic_events

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    
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

    # Evaluate random policy
    episodic_events = evaluate_random(
        args.env_id,
        args.eval_episodes,
        run_name,
        args.capture_video
    )

    # Log results
    for idx, event in enumerate(episodic_events):
        writer.add_scalar("eval/episodic_return", event['return'], idx)
        writer.add_scalar("eval/episodic_length", event['length'], idx)
        writer.add_scalar("eval/episodic_time", event['time'], idx)
    
    writer.close()