import random
from pathlib import Path
from typing import Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset


def _load_episode(episode_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(str(episode_path))
    return data['obs1'], data['obs2'], data['metric_vals']


class MetricDataset(IterableDataset):
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode = list(self._data_dir.glob('*.npz'))[index]
        return _load_episode(episode)

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def _sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        episode = random.choice(list(self._data_dir.glob('*.npz')))
        return _load_episode(episode)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        while True:
            yield self._sample()


def make_metric_data_loader(data_dir: Path, num_workers: int) -> torch.utils.data.DataLoader:
    dataset = MetricDataset(data_dir=data_dir)
    return torch.utils.data.DataLoader(dataset=dataset, num_workers=num_workers)
