# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from gym import spaces
from ..utils.wrappers import NormalizedBoxEnv
from gym.envs.mujoco import mujoco_env


class CircleEnv:
    """
    Simple 1D state, 1D action environment used for IRL toy experiments.
    True reward (conceptual): 1 if s^2 + a^2 within an annulus band around 1,
    else 0. (IRL uses external reward; this helper is for analysis.)

    Notes:
    - This environment mirrors other toy shape envs in this repo. It samples
      states from a provided dataset of x-coordinates and ignores the true
      reward during rollout since IRL algorithms compute rewards externally.
    - We still expose a helper to compute the true circle reward for analysis.
    """

    def __init__(self, x_coordinates, band_width=0.2):
        self.x_coordinates = x_coordinates
        # Reward band: |s^2 + a^2 - 1| <= band_width
        self.band_width = float(band_width)

        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        self.max_step_num = 100
        self.step_count = 0

    def seed(self, seed=0):
        mujoco_env.MujocoEnv.seed(self, seed)

    @staticmethod
    def circle_reward(s, a, tol=1e-3, band_width=None):
        # Backwards-compatible point reward or band reward if band_width provided
        if band_width is None:
            return float(abs(s * s + a * a - 1.0) <= tol)
        return float(abs(s * s + a * a - 1.0) <= band_width)

    def reset(self):
        idx = np.random.choice(self.x_coordinates.shape[0], size=1)
        state = self.x_coordinates[idx]
        return state

    def step(self, action):
        # For IRL training, reward is provided externally. Return 0 here.
        idx = np.random.choice(self.x_coordinates.shape[0], size=1)
        next_state = self.x_coordinates[idx]

        self.step_count += 1
        done = self.step_count >= self.max_step_num
        if done:
            self.step_count = 0

        return next_state, 0.0, done, {}

    def render(self):
        pass


def get_circle_env(**kwargs):
    # States uniformly sampled in [-1, 1]
    x = np.linspace(-1.0, 1.0, num=2000)
    band_width = kwargs.get('band_width', 0.1)
    return NormalizedBoxEnv(CircleEnv(x, band_width=band_width))


