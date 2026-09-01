#!/usr/bin/env python
"""Open and close only the Panda gripper while keeping the TCP target fixed."""

from __future__ import annotations

import time

import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def wait_and_step(env: PassiveViewerWrapper, action: np.ndarray, steps: int) -> bool:
    for index in range(steps):
        if not env._viewer.is_running():
            return False
        # gripper_delta changes the setpoint; send it only on the first step.
        env.step(action if index == 0 else np.zeros(7, dtype=np.float32))
        time.sleep(env.unwrapped.control_dt)
    return True


def gripper_command(env: PassiveViewerWrapper) -> float:
    return float(env.unwrapped.get_robot_state()[14])


def main() -> None:
    env = PassiveViewerWrapper(PandaPickCubeGymEnv(control_dt=0.1))
    env.reset(seed=42)
    zero = np.zeros(7, dtype=np.float32)

    print(f"初始夹爪控制值：{gripper_command(env):.1f}（物理张开端）")
    print("2 秒后闭合夹爪。TCP 目标位置保持不变。")

    try:
        if not wait_and_step(env, zero, 20):
            return

        close_gripper = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        if not wait_and_step(env, close_gripper, 20):
            return
        print(f"闭合后控制值：  {gripper_command(env):.1f}（物理闭合端）")

        print("保持 2 秒，然后张开夹爪。")
        if not wait_and_step(env, zero, 20):
            return

        open_gripper = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
        if not wait_and_step(env, open_gripper, 20):
            return
        print(f"张开后控制值：  {gripper_command(env):.1f}（物理张开端）")
        print("关闭窗口或按 Ctrl+C 退出。")

        while env._viewer.is_running():
            env.step(zero)
            time.sleep(env.unwrapped.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
