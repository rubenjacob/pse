import csv
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Union, Optional, Tuple, List

import torch
import wandb
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

Value = Union[int, float]

COMMON_TRAIN_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                       ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                       ('episode_reward', 'R', 'float'),
                       ('buffer_size', 'BS', 'int'), ('fps', 'FPS', 'float'),
                       ('total_time', 'T', 'time')]

COMMON_EVAL_FORMAT = [('frame', 'F', 'int'), ('step', 'S', 'int'),
                      ('episode', 'E', 'int'), ('episode_length', 'L', 'int'),
                      ('episode_reward', 'R', 'float'),
                      ('total_time', 'T', 'time')]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value: Value, n: int = 1):
        self._sum += value
        self._count += n

    def value(self) -> Value:
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name: Path, formatting: List[Tuple[str, str, str]], use_wandb: bool):
        self._csv_file_name = csv_file_name
        self._formatting = formatting
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None
        self.use_wandb = use_wandb

    def log(self, key: str, value: Value, n: int = 1):
        self._meters[key].update(value, n)

    def _prime_meters(self) -> Dict[str, Value]:
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise f'invalid format type: {ty}'

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = [f'| {prefix: <14}']
        for key, disp_key, ty in self._formatting:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)

        if self.use_wandb:
            prefixed_data = {f'{prefix}/{key}': value for key, value in data.items()}
            wandb.log(prefixed_data)

        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir: Path, use_tb: bool, use_wandb: bool, cfg: Dict[str, Any]):
        self._log_dir = log_dir
        self._train_meters_group = MetersGroup(log_dir / 'train.csv', formatting=COMMON_TRAIN_FORMAT,
                                               use_wandb=use_wandb)
        self._eval_meters_group = MetersGroup(log_dir / 'eval.csv', formatting=COMMON_EVAL_FORMAT, use_wandb=use_wandb)
        if use_tb:
            self._summary_writer = SummaryWriter(log_dir=str(log_dir / 'tb'))
        else:
            self._summary_writer = None

    def _try_sw_log(self, key: str, value: Value, step: int):
        if self._summary_writer is not None:
            self._summary_writer.add_scalar(key, value, step)

    def log(self, key: str, value: Union[Value, torch.Tensor], step: int):
        assert key.startswith('train') or key.startswith('eval')
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value, step)
        meters_group = self._train_meters_group if key.startswith('train') else self._eval_meters_group
        meters_group.log(key, value)

    def log_metrics(self, metrics: Dict[str, Union[Value, torch.Tensor]], step: int, train_or_eval: str):
        for key, value in metrics.items():
            self.log(key=f'{train_or_eval}/{key}', value=value, step=step)

    def dump(self, step: int, train_or_eval: Optional[str] = None):
        if train_or_eval is None or train_or_eval == 'eval':
            self._eval_meters_group.dump(step=step, prefix='eval')
        if train_or_eval is None or train_or_eval == 'train':
            self._train_meters_group.dump(step=step, prefix='train')

    def log_and_dump_ctx(self, step: int, train_or_eval: str):
        return LogAndDumpCtx(self, step=step, train_or_eval=train_or_eval)


class LogAndDumpCtx:
    def __init__(self, logger: Logger, step: int, train_or_eval: str):
        self._logger = logger
        self._step = step
        self._train_or_eval = train_or_eval

    def __enter__(self):
        return self

    def __call__(self, key: str, value: Union[Value, torch.Tensor]):
        self._logger.log(key=f'{self._train_or_eval}/{key}', value=value, step=self._step)

    def __exit__(self, *args):
        self._logger.dump(step=self._step, train_or_eval=self._train_or_eval)
