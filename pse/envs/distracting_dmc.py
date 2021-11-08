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
         task_kwargs=None,
         render_kwargs=None,
         camera_kwargs=None,
         background_kwargs=None,
         color_kwargs=None) -> ExtendedTimeStepWrapper:
    splits = name.split('_')
    task = splits[-1]
    domain = '_'.join(splits[:-1])
    render_kwargs = render_kwargs or {}
    render_kwargs['width'] = frame_shape[0]
    render_kwargs['height'] = frame_shape[1]

    if 'camera_id' not in render_kwargs:
        render_kwargs['camera_id'] = 2 if domain == 'quadruped' else 0
    if camera_kwargs and 'camera_id' not in camera_kwargs:
        camera_kwargs['camera_id'] = 2 if domain == 'quadruped' else 0

    # make sure reward is not visualized
    env = suite.load(domain,
                     task,
                     task_kwargs=task_kwargs,
                     render_kwargs=render_kwargs or {'random': seed},
                     camera_kwargs=camera_kwargs,
                     background_kwargs=background_kwargs,
                     color_kwargs=color_kwargs)
    pixels_key = 'pixels'
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env
