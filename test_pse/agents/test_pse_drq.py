from pathlib import Path
from typing import Optional, Iterator

import torch
from torch.utils.data import DataLoader

from pse.data.metric_dataset import make_metric_data_loader


class FakeAgent:
    def __init__(self, metric_data_dir: str):
        self.metric_data_loader = make_metric_data_loader(data_dir=metric_data_dir, num_workers=4)
        self._metric_data_iter = None

    @property
    def metric_data_iter(self) -> Optional[Iterator[DataLoader]]:
        if self.metric_data_loader is None:
            return None
        if self._metric_data_iter is None:
            self._metric_data_iter = iter(self.metric_data_loader)
        return self._metric_data_iter

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_metric_data_iter'] = None
        return d


def test_snapshot_saving():
    snapshot_path = Path.cwd() / 'fake_agent.pt'
    snapshot_path.unlink(missing_ok=True)

    fake_agent = FakeAgent(metric_data_dir='metric_data')
    _ = fake_agent.metric_data_iter
    with snapshot_path.open('wb') as f:
        torch.save({'agent': fake_agent}, f)

    assert snapshot_path.exists()
