from pathlib import Path
from typing import Iterator

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
from torch.utils.data import DataLoader

import utils
from pse.agents.drq import DrQAgent
from pse.envs import dmc
from pse.utils.logger import Logger
from pse.data.replay_buffer import make_replay_loader, ReplayBufferStorage
from pse.utils.video_recorder import VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec: specs.Array, action_spec: specs.Array, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:

    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.snapshot_dir = self.work_dir / 'snapshots'
        self.snapshot_dir.mkdir(exist_ok=True)

        self.cfg = cfg

        if self.cfg.use_wandb:
            wandb.init(config=cfg)

        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb, use_wandb=self.cfg.use_wandb, cfg=self.cfg)

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack, self.cfg.action_repeat, self.cfg.seed)

        self.agent: DrQAgent = make_agent(obs_spec=self.train_env.observation_spec(),
                                          action_spec=self.train_env.action_spec(),
                                          cfg=self.cfg.agent)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs=data_specs, replay_dir=self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(replay_dir=self.work_dir / 'buffer',
                                                max_size=self.cfg.replay_buffer_size,
                                                batch_size=self.cfg.batch_size,
                                                num_workers=self.cfg.replay_buffer_num_workers,
                                                save_snapshot=self.cfg.save_snapshot,
                                                nstep=self.cfg.nstep,
                                                discount=self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(root_dir=self.work_dir if self.cfg.save_video else None, log_to_wandb=True)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def global_episode(self) -> int:
        return self._global_episode

    @property
    def global_frame(self) -> int:
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self) -> Iterator[DataLoader]:
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(until=self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(env=self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(obs=time_step.observation, eval_mode=True)
                time_step = self.eval_env.step(action=action)
                self.video_recorder.record(env=self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(file_name=f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, train_or_eval='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(until=self.cfg.num_train_frames, action_repeat=self.cfg.action_repeat)
        seed_until_step = utils.Until(until=self.cfg.num_seed_frames, action_repeat=self.cfg.action_repeat)
        eval_every_step = utils.Every(every=self.cfg.eval_every_frames, action_repeat=self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step=time_step)
        metrics = None
        while train_until_step(step=self.global_step):
            if time_step.last():
                self._global_episode += 1
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame, train_or_eval='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                self.replay_storage.add(time_step=time_step)
                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(step=self.global_step):
                self.logger.log(key='eval_total_time', value=self.timer.total_time(), step=self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs=time_step.observation, eval_mode=False)

            # try to update the agent
            if not seed_until_step(step=self.global_step):
                metrics = self.agent.update(replay_iter=self.replay_iter, step=self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, train_or_eval='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.snapshot_dir / 'snapshot.pt'
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.snapshot_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path='configs/config.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    snapshot = Path.cwd() / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
