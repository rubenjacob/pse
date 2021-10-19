from typing import NamedTuple

import numpy as np


class Trajectory(NamedTuple('Trajectory', [('observation', np.ndarray),
                                           ('action', np.ndarray),
                                           ('reward', np.ndarray)])):
    pass
