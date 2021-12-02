from pathlib import Path
from typing import Iterator, Optional

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
from torch.utils.data import DataLoader

from pse.agents import utils
from pse.agents.drq import DrQV2Agent
from pse.data.metric_dataset import make_metric_data_loader
from pse.envs import distracting_dmc
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
        self.work_dir = Path.home() / Path(cfg.root_dir)
        print(f'workspace: {self.work_dir}')
        self.snapshot_dir = self.work_dir / 'snapshots'
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.cfg = cfg

        if self.cfg.use_wandb:
            if self.cfg.resume_from_snapshot and self.cfg.wandb_resume_run_id is not None:
                print(f"Resuming W&B run with id {self.cfg.wandb_resume_run_id}")
                resume_args = {'resume': 'must', 'id': self.cfg.wandb_resume_run_id}
            else:
                resume_args = {}
            run_name = f"{self.cfg.task_name}_{self.cfg.experiment}"
            wandb.init(config=dict(self.cfg), project="pse", name=run_name, **resume_args)

        self.logger = Logger(self.work_dir, use_wandb=self.cfg.use_wandb)

        utils.set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)

        self.train_env = distracting_dmc.make(name=self.cfg.task_name,
                                              frame_stack=self.cfg.frame_stack,
                                              action_repeat=self.cfg.action_repeat,
                                              seed=self.cfg.seed,
                                              num_videos=self.cfg.num_train_videos,
                                              background_videos=self.cfg.train_background_videos,
                                              dynamic_background=self.cfg.dynamic_background)
        self.eval_env = distracting_dmc.make(name=self.cfg.task_name,
                                             frame_stack=self.cfg.frame_stack,
                                             action_repeat=self.cfg.action_repeat,
                                             seed=self.cfg.seed,
                                             num_videos=self.cfg.num_eval_videos,
                                             background_videos='validation',
                                             dynamic_background=self.cfg.dynamic_background)

        self.agent: DrQV2Agent = make_agent(obs_spec=self.train_env.observation_spec(),
                                            action_spec=self.train_env.action_spec(),
                                            cfg=self.cfg.agent)

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        buffer_dir = self.work_dir / 'buffer'
        self.replay_storage = ReplayBufferStorage(data_specs=data_specs, replay_dir=buffer_dir)
        self.replay_loader = make_replay_loader(replay_dir=buffer_dir,
                                                max_size=self.cfg.replay_buffer_size,
                                                batch_size=self.cfg.batch_size,
                                                num_workers=self.cfg.replay_buffer_num_workers,
                                                save_snapshot=self.cfg.save_snapshot,
                                                nstep=self.cfg.nstep,
                                                discount=self.cfg.discount)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(root_dir=self.work_dir if self.cfg.save_video else None,
                                            log_to_wandb=self.cfg.use_wandb)

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
                    # step doesn't matter in eval mode, so just pass any value
                    action = self.agent.act(obs=time_step.observation, step=10000, eval_mode=True)
                time_step = self.eval_env.step(action=action)
                self.video_recorder.record(env=self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(file_name=Path(f'{self.global_frame}.mp4'))

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
        save_every_step = utils.Every(every=self.cfg.snapshot_interval, action_repeat=self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step=time_step)
        metrics = None
        print("Started training...")
        while train_until_step(step=self.global_step):
            if time_step.last():
                print(f"Finished episode {self.global_episode}")
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
                if self.cfg.save_snapshot and save_every_step(step=self.global_step):
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(step=self.global_step) and self.global_step != 0:
                print(f"Evaluating for {self.cfg.num_eval_episodes}...")
                self.logger.log(key='eval_total_time', value=self.timer.total_time(), step=self.global_frame)
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(obs=time_step.observation, step=self.global_step, eval_mode=False)

            # try to update the agent
            if not seed_until_step(step=self.global_step):
                metrics = self.agent.update(replay_iter=self.replay_iter, step=self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, train_or_eval='train')

            # take env step
            print(action)
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot = self.snapshot_dir / f'snapshot-{self.global_step:07d}.pt'
        print(f"Saving snapshot to {str(snapshot)}")
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self, step_to_load: Optional[int] = None):
        payload = utils.load_snapshot_payload(snapshot_dir=self.snapshot_dir, step_to_load=step_to_load,
                                              device=self.cfg.device)
        for k, v in payload.items():
            self.__dict__[k] = v
