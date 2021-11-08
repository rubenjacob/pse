from collections import deque
from typing import Optional, Any, NamedTuple

import numpy as np
from dm_env import specs, TimeStep, Environment, StepType


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ActionRepeatWrapper(Environment):
    def __init__(self, env: Environment, num_repeats: int):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action: np.ndarray) -> TimeStep:
        reward = 0.0
        discount = 1.0
        time_step = None
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        if time_step is not None:
            return time_step._replace(reward=reward, discount=discount)
        else:
            raise ValueError(f"Number of repeats can't be smaller than 1 but is: {self._num_repeats}")

    def observation_spec(self) -> specs.Array:
        return self._env.observation_spec()

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def reset(self) -> TimeStep:
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(Environment):
    def __init__(self, env: Environment, num_frames: int, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(shape=np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]],
                                                                 axis=0),
                                            dtype=np.uint8,
                                            minimum=0,
                                            maximum=255,
                                            name='observation')

    def _transform_observation(self, time_step: TimeStep) -> TimeStep:
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step: TimeStep) -> TimeStep:
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self) -> TimeStep:
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action: np.ndarray) -> TimeStep:
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self) -> specs.BoundedArray:
        return self._obs_spec

    def action_spec(self) -> specs.Array:
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(Environment):
    def __init__(self, env: Environment, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action: np.ndarray) -> TimeStep:
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self) -> specs.BoundedArray:
        return self._env.observation_spec()

    def action_spec(self) -> specs.BoundedArray:
        return self._action_spec

    def reset(self) -> TimeStep:
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(Environment):
    def __init__(self, env: Environment):
        self._env = env

    def reset(self) -> ExtendedTimeStep:
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action: np.ndarray) -> ExtendedTimeStep:
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step: TimeStep, action: Optional[np.ndarray] = None) -> ExtendedTimeStep:
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self) -> specs.BoundedArray:
        return self._env.observation_spec()

    def action_spec(self) -> specs.BoundedArray:
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)
