from typing import Callable

import gymnasium as gym
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = True,
):
    """
    Evaluate the model on the environment.
    Returns a list of dictionaries, each containing the return, length, and time of the episode.
    """
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, capture_video, run_name)])
    agent = Model(envs).to(device)
    agent.load_state_dict(torch.load(model_path, map_location=device))
    agent.eval()

    obs, _ = envs.reset()
    episodic_events = []
    while len(episodic_events) < eval_episodes:
        actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
        next_obs, rewards, terminated , truncated, infos = envs.step(actions.cpu().numpy())
        if (terminated or truncated) and "episode" in infos:
            print(f"eval_episode={len(episodic_events)}, episodic_return={infos['episode']['r']}, episodic_length={infos['episode']['l']}, episodic_time={infos['episode']['t']}")
            episodic_events += [{'return': infos['episode']['r'], 'length': infos['episode']['l'], 'time': infos['episode']['t']}]
        obs = next_obs

    return episodic_events

