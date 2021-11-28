from pathlib import Path
from typing import Tuple, Optional, List, Callable

import hydra
import numpy as np

from pse.data.collect_utils import load_policy, compute_metric, ScriptedPolicy
from pse.envs.distracting_dmc import make
from pse.envs.extended_time_step import ExtendedTimeStep
from pse.envs.wrappers import ExtendedTimeStepWrapper

INIT_DATA_SEED = 42  # Answer to life, the universe and everything


def collect_and_save_data(task_name: str, snapshot_dir: str, max_episode_len: int, total_episodes: int,
                          episodes_per_seed: int, frame_stack: int, action_repeat: int, discount: float):
    print("Starting data collection.")
    snapshot_dir = Path(snapshot_dir)

    env = make(name=task_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=1, num_videos=1,
               background_videos=['bear'])
    action_shape = env.action_spec().shape
    del env
    policy = load_policy(snapshot_dir=snapshot_dir, action_shape=action_shape)
    num_seeds = total_episodes // episodes_per_seed
    max_steps = max_episode_len * episodes_per_seed
    episodes_dir = snapshot_dir.parent / 'metric_data'
    episodes_dir.mkdir(exist_ok=True)

    for seed in range(INIT_DATA_SEED, INIT_DATA_SEED + num_seeds):
        print(f"Seed {seed} starting to collect metric data.")
        episodes, paired_episodes = collect_pair_episodes(policy=policy, env_name=task_name, random_seed=seed,
                                                          max_steps=max_steps, max_episodes=episodes_per_seed,
                                                          frame_stack=frame_stack, action_repeat=action_repeat)
        print(f"Seed {seed} done collecting metric data. Saving...")
        for episode, paired_episode in zip(episodes, paired_episodes):
            # Write (obs1, obs2, metric) tuples
            processed_episode = process_episode(episode, paired_episode, gamma=discount)
            save_processed_episode(processed_episode, episodes_dir)


def collect_pair_episodes(policy: Callable[[np.ndarray], np.ndarray], env_name: str, frame_stack: int,
                          action_repeat: int, max_steps: Optional[int] = None, random_seed: Optional[int] = None,
                          max_episodes: int = 10) -> Tuple[List[List[ExtendedTimeStep]], List[List[ExtendedTimeStep]]]:
    env = make(name=env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=random_seed, num_videos=1,
               background_videos=['bear'])
    env_copy = make(name=env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=random_seed,
                    num_videos=1, background_videos=['bmx-bumps'])

    buffer = run_env(env=env, policy=policy, max_steps=max_steps, max_episodes=max_episodes)

    actions = [x.action for x in buffer]
    scripted_optimal_policy = ScriptedPolicy(actions=actions).act
    paired_buffer = run_env(env=env_copy, policy=scripted_optimal_policy, max_steps=max_steps,
                            max_episodes=max_episodes)

    episodes = get_complete_episodes(replay_buffer=buffer, num_episodes=max_episodes)
    paired_episodes = get_complete_episodes(replay_buffer=paired_buffer, num_episodes=max_episodes)
    return episodes, paired_episodes


def get_complete_episodes(replay_buffer: List[ExtendedTimeStep], num_episodes: int = 2) -> List[List[ExtendedTimeStep]]:
    terminal_steps = [int(x.step_type) for x in replay_buffer]
    episode_boundaries = np.where(np.array(terminal_steps) == 2)[0]
    episode_boundaries = np.append(episode_boundaries[::-1], [-1])[::-1]  # prepend -1
    return [replay_buffer[episode_boundaries[i] + 1: episode_boundaries[i + 1] + 1] for i in range(num_episodes)]


def _stack_observations(episode: List[ExtendedTimeStep]) -> np.ndarray:
    return np.stack([x.observation for x in episode], axis=0)


def process_episode(episode1: List[ExtendedTimeStep], episode2: List[ExtendedTimeStep],
                    gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs1, obs2 = _stack_observations(episode1), _stack_observations(episode2)
    metric = compute_metric(episode1, episode2, gamma)
    return obs1, obs2, metric


def run_env(env: ExtendedTimeStepWrapper, policy: Callable[[np.ndarray], np.ndarray], max_steps: int,
            max_episodes: int) -> List[ExtendedTimeStep]:
    replay_buffer: List[ExtendedTimeStep] = list()
    num_steps = 0
    num_episodes = 0
    extended_time_step = env.reset()

    while num_steps < max_steps and num_episodes < max_episodes:
        replay_buffer.append(extended_time_step)
        action = policy(extended_time_step.observation)
        extended_time_step = env.step(action=action)
        num_steps += 1

        if extended_time_step.last():
            replay_buffer.append(extended_time_step)
            extended_time_step = env.reset()
            num_episodes += 1

    return replay_buffer


def save_processed_episode(processed_episode: Tuple[np.ndarray, np.ndarray, np.ndarray], episodes_dir: Path):
    episode_number = len(list(episodes_dir.glob('*.npz')))
    pth = episodes_dir / f"episode{episode_number: 05d}.npz"
    np.savez(pth, obs1=processed_episode[0], obs2=processed_episode[1], metric_vals=processed_episode[2])


@hydra.main(config_path="../../configs", config_name="data_collection_config.yaml")
def main(cfg):
    collect_and_save_data(**cfg)


if __name__ == "__main__":
    main()
