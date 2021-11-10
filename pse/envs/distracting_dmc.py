from typing import Tuple

import numpy as np
from distracting_control import suite
from dm_control.suite.wrappers import action_scale

from pse.envs.wrappers import ActionDTypeWrapper, ActionRepeatWrapper, FrameStackWrapper, ExtendedTimeStepWrapper


def make(name: str,
         frame_stack: int,
         action_repeat: int,
         seed: int,
         frame_shape: Tuple[int, int, int] = (84, 84, 3),
         num_videos: int = 2,
         dynamic_background: bool = True) -> ExtendedTimeStepWrapper:
    splits = name.split('_')
    task = splits[-1]
    domain = '_'.join(splits[:-1])
    render_kwargs = {
        'width': frame_shape[0],
        'height': frame_shape[1],
        'random': seed,
        'camera_id': 2 if domain == 'quadruped' else 0
    }
    background_kwargs = {
        'num_videos': num_videos,
        'dynamic': dynamic_background,
        'dataset_videos': 'train'
    }

    # make sure reward is not visualized
    env = suite.load(domain,
                     task,
                     render_kwargs=render_kwargs,
                     background_kwargs=background_kwargs)
    pixels_key = 'pixels'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
