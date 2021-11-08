import numpy as np

from pse.envs.distracting_dmc import make


def test_env_builds_and_outputs_correct_shape():
    test_env = make('ball_in_cup_catch', 3, 2, 0)
    obs = test_env.reset().observation
    obs = np.moveaxis(obs, 0, -1)
    assert np.all(obs.shape == (84, 84, 9))
