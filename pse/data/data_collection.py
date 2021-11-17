from pathlib import Path
from typing import Tuple, Optional, List, Callable

from absl import app, flags
import numpy as np

from pse.data.collect_utils import load_policy, compute_metric, ScriptedPolicy
from pse.envs.distracting_dmc import make
from pse.envs.extended_time_step import ExtendedTimeStep
from pse.envs.wrappers import ExtendedTimeStepWrapper

INIT_DATA_SEED = 42  # Answer to life, the universe and everything
FLAGS = flags.FLAGS


def collect_and_save_data(env_name: str, snapshot_dir: Path, max_episode_len: int, total_episodes: int,
                          episodes_per_seed: int, frame_stack: int, action_repeat: int, discount: float):
    policy = load_policy(snapshot_dir=snapshot_dir)
    num_seeds = total_episodes // episodes_per_seed
    max_steps = (max_episode_len + 1) * episodes_per_seed
    episodes_dir = snapshot_dir.parent / 'episodes'
    episodes_dir.mkdir(exist_ok=True)

    processed_episodes = list()
    for seed in range(INIT_DATA_SEED, INIT_DATA_SEED + num_seeds):
        episodes, paired_episodes = collect_pair_episodes(policy=policy, env_name=env_name, random_seed=seed,
                                                          max_steps=max_steps, max_episodes=episodes_per_seed,
                                                          frame_stack=frame_stack, action_repeat=action_repeat)
        for episode, paired_episode in zip(episodes, paired_episodes):
            # Write (obs1, obs2, metric) tuples
            processed_episodes.append(process_episode(episode, paired_episode, gamma=discount))

    save_processed_episodes(processed_episodes, episodes_dir)


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
    episode_boundaries = np.append(episode_boundaries[::-1], [-2])[::-1]  # prepend -2
    return [replay_buffer[episode_boundaries[i] + 2: episode_boundaries[i + 1] + 1] for i in range(num_episodes)]


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
    obs = env.reset().observation

    while num_steps < max_steps and num_episodes < max_episodes:
        action = policy(obs)
        extended_time_step = env.step(action=action)
        replay_buffer.append(extended_time_step)
        num_steps += 1

        if extended_time_step.last():
            obs = env.reset().observation
            num_episodes += 1

    return replay_buffer


def save_processed_episodes(processed_episodes: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], episodes_dir: Path):
    for i, episode in enumerate(processed_episodes):
        pth = episodes_dir / f"episode{i: 05d}.npz"
        np.savez(pth, obs1=episode[0], obs2=episode[1], metric=episode[2])


def main(_):
    flags.DEFINE_integer('max_episode_len', 1000, 'Number of steps in an episode.')
    flags.DEFINE_string('env_name', 'cartpole-swingup', 'Name of the environment.')
    flags.DEFINE_integer('total_episodes', 500, 'Number of steps in an episode.')
    flags.DEFINE_integer('episodes_per_seed', 10, 'Number of episode per random seed.')
    flags.DEFINE_string('snapshot_dir', None, 'Directory where model snapshot is stored.')
    flags.DEFINE_integer('action_repeat', 2, '')
    flags.DEFINE_integer('frame_stack', 3, '')
    flags.DEFINE_float('discount', 0.99, '')

    collect_and_save_data(
        env_name=FLAGS.env_name,
        snapshot_dir=FLAGS.snapshot_dir,
        max_episode_len=FLAGS.max_episode_len,
        total_episodes=FLAGS.total_episodes,
        episodes_per_seed=FLAGS.episodes_per_seed,
        action_repeat=FLAGS.action_repeat,
        frame_stack=FLAGS.frame_stack,
        discount=FLAGS.discount)


if __name__ == "__main__":
    app.run(main)
