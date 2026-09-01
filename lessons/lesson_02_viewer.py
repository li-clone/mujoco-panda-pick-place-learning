#!/usr/bin/env python
"""Open a MuJoCo window and hold the Panda at its reset pose."""

from __future__ import annotations

import time

import gymnasium as gym
import gym_hil  # noqa: F401  # Registers the gym_hil environments.
import numpy as np
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def main() -> None:
    base_env = gym.make("gym_hil/PandaPickCubeBase-v0")
    env = PassiveViewerWrapper(base_env)
    env.reset(seed=42)
    action = np.zeros(7, dtype=np.float32)

    print("MuJoCo viewer 已打开。关闭窗口或在终端按 Ctrl+C 退出。")
    print("鼠标左键旋转视角，右键平移，滚轮缩放。")
    try:
        while env._viewer.is_running():
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset(seed=42)
            time.sleep(env.unwrapped.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
