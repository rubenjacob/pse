from pathlib import Path
from unittest import mock

import pytest

from pse.utils.logger import Logger


@pytest.fixture()
def logger() -> Logger:
    return Logger(log_dir=Path('/tmp/test_log_dir'), use_tb=False, use_wandb=True, cfg={'run_name': 'wandb_test'})


@mock.patch('pse.utils.logger.wandb')
def test_logger_logs_to_wandb(wandb_mock: mock.MagicMock, logger: Logger):
    for step in range(5):
        with logger.log_and_dump_ctx(step=step, train_or_eval='train') as log:
            log('step', step)
            log('loss', 1 / (step + 1))
            log('episode', 1)

    assert len(wandb_mock.log.call_args_list) == 5
