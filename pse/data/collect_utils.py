import os
from pathlib import Path
from typing import Union, Dict, List, Callable

import numpy as np
import torch

from pse.agents.drq import DrQAgent
from pse.data.trajectory import Trajectory


class ScriptedPolicy:
    def __init__(self, actions: List[np.ndarray]):
        self.scripted_actions = actions
        self.action_index = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        del obs  # unused

        action = self.scripted_actions[self.action_index]
        self.action_index += 1
        return action


def load_policy(saved_model_dir: Path, max_train_step: int, seconds_between_checkpoint_polls: int = 5,
                num_retries: int = 10) -> Callable[[np.ndarray], np.ndarray]:
    split = os.path.split(saved_model_dir)
    if not split[-1]:
        saved_model_dir = split[0]

    # TODO: Load from latest checkpoint
    agent = torch.load(saved_model_dir)

    if not issubclass(agent.__class__, DrQAgent):
        raise ValueError('The loaded agent does not have the correct class!')

    return agent.act


def _load_from_checkpoint() -> :


def _get_action(replay: Union[Trajectory, List[Trajectory]]) -> np.ndarray:
    if isinstance(replay, list):
        return np.array([x.action for x in replay])
    else:
        return replay.action


def _calculate_action_cost_matrix(action1: np.ndarray, action2: np.ndarray) -> np.ndarray:
    diff = np.expand_dims(action1, axis=1) - np.expand_dims(action2, axis=0)
    return np.cast(np.mean(np.abs(diff), axis=-1), dtype=np.float32)


def metric_fixed_point_fast(cost_matrix: np.ndarray, gamma: float = 0.99, eps: float = 1e-7) -> np.ndarray:
    """Dynamic programming for calculating PSM."""
    d = np.zeros_like(cost_matrix)

    def operator(d_cur):
        d_new = 1 * cost_matrix
        discounted_d_cur = gamma * d_cur
        d_new[:-1, :-1] += discounted_d_cur[1:, 1:]
        d_new[:-1, -1] += discounted_d_cur[1:, -1]
        d_new[-1, :-1] += discounted_d_cur[-1, 1:]
        return d_new

    while True:
        d_new = operator(d)
        if np.sum(np.abs(d - d_new)) < eps:
            break
        else:
            d = d_new[:]
    return d


def compute_metric(replay1: Union[Trajectory, List[Trajectory]], replay2: Union[Trajectory, List[Trajectory]],
                   gamma: float) -> np.ndarray:
    actions1, actions2 = _get_action(replay=replay1), _get_action(replay=replay2)
    action_cost = _calculate_action_cost_matrix(action1=actions1, action2=actions2)
    return metric_fixed_point_fast(cost_matrix=action_cost, gamma=gamma)
