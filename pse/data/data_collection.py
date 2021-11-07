from pathlib import Path
from typing import Tuple, Optional, List, Callable

import dm_env
from absl import app, flags
import numpy as np

from pse.data.collect_utils import load_policy, compute_metric, ScriptedPolicy
from pse.data.trajectory import Trajectory
from pse.envs.dmc import make

INIT_DATA_SEED = 42  # Answer to life, the universe and everything
GAMMA = 0.99
FLAGS = flags.FLAGS


def collect_and_save_data(env_name: str, model_dir: Path, trial_suffix: str, max_episode_len: int, root_dir: Path,
                          total_episodes: int, episodes_per_seed: int, frame_stack: int, action_repeat: int,
                          discount: float):
    saved_model_dir = model_dir / env_name / trial_suffix / 'policies/greedy_policy'
    max_train_step = 500 * max_episode_len
    policy = load_policy(saved_model_dir=saved_model_dir, max_train_step=max_train_step)
    root_dir = root_dir / env_name / trial_suffix
    num_seeds = total_episodes // episodes_per_seed
    max_steps = (max_episode_len + 1) * episodes_per_seed

    for seed in range(INIT_DATA_SEED, INIT_DATA_SEED + num_seeds):
        episodes, paired_episodes = collect_pair_episodes(policy=policy, env_name=env_name, random_seed=seed,
                                                          max_steps=max_steps, max_episodes=episodes_per_seed,
                                                          frame_stack=frame_stack, action_repeat=action_repeat)
        for episode, paired_episode in zip(episodes, paired_episodes):
            # Write (obs1, obs2, metric) tuples
            processed_episode_tuple = process_episode(episode, paired_episode, gamma=GAMMA)
            tf_episode2_observer.write(processed_episode_tuple)


def collect_pair_episodes(policy: Callable[[np.ndarray], np.ndarray], env_name: str, frame_stack: int,
                          action_repeat: int, max_steps: Optional[int] = None, random_seed: Optional[int] = None,
                          max_episodes: int = 10) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    env = make(name=env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=random_seed)
    env_copy = make(name=env_name, frame_stack=frame_stack, action_repeat=action_repeat, seed=random_seed)

    buffer = run_env(env=env, policy=policy, max_steps=max_steps, max_episodes=max_episodes)

    actions = [x.action for x in buffer]
    scripted_optimal_policy = ScriptedPolicy(actions=actions).act
    paired_buffer = run_env(env=env_copy, policy=scripted_optimal_policy, max_steps=max_steps,
                            max_episodes=max_episodes)

    episodes = get_complete_episodes(replay_buffer=buffer, num_episodes=max_episodes)
    paired_episodes = get_complete_episodes(replay_buffer=paired_buffer, num_episodes=max_episodes)
    return episodes, paired_episodes


def get_complete_episodes(replay_buffer: List[Trajectory], num_episodes: int = 2) -> List[np.ndarray]:
    terminal_steps = [int(x.next_step_type) for x in replay_buffer]
    episode_boundaries = np.where(np.array(terminal_steps) == 2)[0]
    episode_boundaries = np.append(episode_boundaries[::-1], [-2])[::-1]
    return [replay_buffer[episode_boundaries[i] + 2 : episode_boundaries[i + 1] + 1]
            for i in range(num_episodes)]


def _get_pixels(episode) -> np.ndarray:
    if isinstance(episode, list):
        return np.stack([x.observation['pixels'] for x in episode], axis=0)
    else:
        return episode.observation['pixels']


def process_episode(episode1, episode2, gamma) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs1, obs2 = _get_pixels(episode1), _get_pixels(episode2)
    metric = compute_metric(episode1, episode2, gamma)
    return obs1, obs2, metric


def run_env(env: dm_env.Environment, policy: Callable[[np.ndarray], np.ndarray], max_steps: int,
            max_episodes: int) -> List[Trajectory]:
    replay_buffer: List[Trajectory] = list()
    num_steps = 0
    num_episodes = 0
    obs = env.reset()

    while num_steps < max_steps and num_episodes < max_episodes:
        action = policy(obs)
        obs, reward, done, _ = env.step(action=action)
        replay_buffer.append(Trajectory(obs, action, reward))
        num_steps += 1

        if done:
            obs = env.reset()
            num_episodes += 1

    return replay_buffer


def main(_):
    flags.DEFINE_integer('trial_id', 0, 'The trial ID from 0 to num_trials-1.')
    flags.DEFINE_integer('max_episode_len', 1000, 'Number of steps in an episode.')
    flags.DEFINE_string('env_name', 'cartpole-swingup', 'Name of the environment.')
    flags.DEFINE_string('root_dir', None, 'Path to output trajectories from data collection')
    flags.DEFINE_integer('seed', None, 'Random Seed for model_dir/data_dir')
    flags.DEFINE_integer('total_episodes', 500, 'Number of steps in an episode.')
    flags.DEFINE_integer('episodes_per_seed', 10, 'Number of episode per random seed.')
    flags.DEFINE_string('model_dir', None, 'Model directory for reading logs/summaries/checkpoints.')
    flags.DEFINE_integer('action_repeat', 2, '')
    flags.DEFINE_integer('frame_stack', 3, '')
    flags.DEFINE_float('discount', 0.99, '')

    if FLAGS.seed is not None:
        trial_suffix = f'{FLAGS.trial_id}/seed_{FLAGS.seed}'
    else:
        trial_suffix = str(FLAGS.trial_id)

    collect_and_save_data(
        env_name=FLAGS.env_name,
        model_dir=FLAGS.model_dir,
        trial_suffix=trial_suffix,
        max_episode_len=FLAGS.max_episode_len,
        root_dir=FLAGS.root_dir,
        total_episodes=FLAGS.total_episodes,
        episodes_per_seed=FLAGS.episodes_per_seed,
        action_repeat=FLAGS.action_repeat,
        frame_stack=FLAGS.frame_stack,
        discount=FLAGS.discount)


if __name__ == "__main__":
    app.run(main)
