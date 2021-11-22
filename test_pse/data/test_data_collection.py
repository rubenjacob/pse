import shutil
from pathlib import Path
from typing import Callable
from unittest import mock

import dm_env
import numpy as np
import pytest
from dm_env import specs
from dm_env._environment import TimeStep, StepType

from pse.data.data_collection import collect_pair_episodes, process_episode, save_processed_episode, \
    collect_and_save_data
from pse.envs.extended_time_step import ExtendedTimeStep
from pse.envs.wrappers import ExtendedTimeStepWrapper


class TestEnv(dm_env.Environment):
    def reset(self) -> TimeStep:
        return TimeStep(step_type=StepType.FIRST, reward=1.0, discount=0.99,
                        observation=np.zeros((9, 84, 84), dtype=np.float32))

    def step(self, action) -> TimeStep:
        return TimeStep(step_type=StepType.LAST, reward=1.0, discount=0.99,
                        observation=np.zeros((9, 84, 84), dtype=np.float32))

    def observation_spec(self):
        return specs.Array(shape=(9, 84, 84), dtype=np.float32)

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, minimum=-1., maximum=1.)


@pytest.fixture()
def test_env() -> ExtendedTimeStepWrapper:
    test_env = TestEnv()
    return ExtendedTimeStepWrapper(env=test_env)


@pytest.fixture()
def fake_policy() -> Callable[[np.ndarray], np.ndarray]:
    def policy(obs: np.ndarray) -> np.ndarray:
        return np.ones((2,))

    return policy


@mock.patch('pse.data.data_collection.make')
def test_collect_pair_episodes(make_mock: mock.MagicMock, test_env: ExtendedTimeStepWrapper,
                               fake_policy: Callable[[np.ndarray], np.ndarray]):
    make_mock.return_value = test_env
    max_episodes = 5
    max_steps = 10
    episodes, paired_episodes = collect_pair_episodes(policy=fake_policy, env_name='ball_in_cup_catch', frame_stack=3,
                                                      action_repeat=4, max_episodes=max_episodes, max_steps=max_steps)

    for episode, paired_episode in zip(episodes, paired_episodes):
        assert len(episode) == max_steps / max_episodes
        assert len(episode) == len(paired_episode)


@mock.patch('pse.data.data_collection.collect_pair_episodes')
@mock.patch('pse.data.data_collection.load_policy')
def test_collect_and_save_data(load_policy_mock: mock.MagicMock, collect_mock: mock.MagicMock,
                               fake_policy: Callable[[np.ndarray], np.ndarray]):
    episodes_dir = Path.cwd() / 'metric_data'
    shutil.rmtree(episodes_dir)
    num_episodes = 5

    episodes = [[ExtendedTimeStep(step_type=StepType.FIRST, reward=1.0, discount=0.99, action=np.ones((2,)),
                                  observation=np.zeros((9, 84, 84), dtype=np.float32)),
                 ExtendedTimeStep(step_type=StepType.LAST, reward=1.0, discount=0.99, action=np.ones((2,)),
                                  observation=np.zeros((9, 84, 84), dtype=np.float32))]] * num_episodes
    paired_episodes = episodes
    collect_mock.return_value = episodes, paired_episodes
    load_policy_mock.return_value = fake_policy

    collect_and_save_data(env_name='test_env', snapshot_dir=Path.cwd() / 'foo', max_episode_len=2, total_episodes=5,
                          episodes_per_seed=5, frame_stack=3, action_repeat=4, discount=0.99)

    assert len(list(episodes_dir.glob('*.npz'))) == num_episodes
