from functools import partial
from pathlib import Path
from typing import Union, List, Callable, Tuple

import numpy as np

from pse.agents.utils import load_snapshot_payload

from pse.envs.extended_time_step import ExtendedTimeStep


class ScriptedPolicy:
    def __init__(self, actions: List[np.ndarray]):
        self.scripted_actions = actions
        self.action_index = 0

    def act(self, obs: np.ndarray) -> np.ndarray:
        action = self.scripted_actions[self.action_index]
        self.action_index += 1
        return action


def load_policy(snapshot_dir: Path, device: str = 'cpu') -> Callable[[np.ndarray], np.ndarray]:
    payload = load_snapshot_payload(snapshot_dir=snapshot_dir, device=device)
    return partial(payload['agent'].act, step=100000, eval_mode=True, device=device)  # step doesn't matter in eval mode


def _get_action(replay: List[ExtendedTimeStep]) -> np.ndarray:
    return np.array([x.action for x in replay])


def _calculate_action_cost_matrix(action1: np.ndarray, action2: np.ndarray) -> np.ndarray:
    diff = np.expand_dims(action1, axis=1) - np.expand_dims(action2, axis=0)
    return np.mean(np.abs(diff), axis=-1)


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


def compute_metric(replay1: List[ExtendedTimeStep], replay2: List[ExtendedTimeStep], gamma: float) -> np.ndarray:
    actions1, actions2 = _get_action(replay=replay1), _get_action(replay=replay2)
    action_cost = _calculate_action_cost_matrix(action1=actions1, action2=actions2)
    return metric_fixed_point_fast(cost_matrix=action_cost, gamma=gamma)
