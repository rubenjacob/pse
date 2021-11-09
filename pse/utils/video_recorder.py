from pathlib import Path

import imageio
import wandb
from dm_env import Environment


class VideoRecorder:
    def __init__(self, root_dir: Path, render_size: int = 256, fps: int = 20, log_to_wandb: bool = True):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []
        self.log_to_wandb = log_to_wandb

    def init(self, env: Environment, enabled: bool = True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env: Environment):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(height=self.render_size,
                                           width=self.render_size,
                                           camera_id=0)
            else:
                frame = env.render()
            self.frames.append(frame)

    def save(self, file_name: Path):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)

            if self.log_to_wandb:
                wandb.log({"eval/video": wandb.Video(str(path))})
