from pathlib import Path

from pse.data.metric_dataset import MetricDataset


def test_data_loading():
    num_samples = 10
    buffer = MetricDataset(data_dir=str(Path.cwd() / 'metric_data'))

    for i, loaded_episode in enumerate(iter(buffer)):
        if i >= num_samples:
            break

        assert len(loaded_episode) == 3
        assert loaded_episode[0].shape == (2, 9, 84, 84)
        assert loaded_episode[1].shape == (2, 9, 84, 84)
        assert loaded_episode[2].shape == (2, 2)
