#!/usr/bin/env python
"""Inspect the smallest useful Gymnasium/MuJoCo interaction loop."""

from __future__ import annotations

import gymnasium as gym
import gym_hil  # noqa: F401  # Importing registers the gym_hil environments.
import mujoco
import numpy as np


ENV_ID = "gym_hil/PandaPickCubeBase-v0"


def main() -> None:
    env = gym.make(ENV_ID, image_obs=True)
    try:
        observation, info = env.reset(seed=42)
        print(f"MuJoCo version: {mujoco.__version__}")
        print(f"Environment: {ENV_ID}")
        print(f"Action space: {env.action_space}")
        print(f"Observation keys: {list(observation)}")
        print(f"Robot state shape: {observation['agent_pos'].shape}")
        print(
            "Camera shapes:",
            {name: image.shape for name, image in observation["pixels"].items()},
        )
        print(f"Reset info: {info}")

        # [dx, dy, dz, dRx, dRy, dRz, gripper_delta].  A zero action asks
        # the simulator to advance time while holding the current command.
        action = np.zeros(7, dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        print(
            "One step:",
            {
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "success": info["succeed"],
            },
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
