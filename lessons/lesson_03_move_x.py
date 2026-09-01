#!/usr/bin/env python
"""Move only the Panda TCP along world +X, then hold its pose."""

from __future__ import annotations

import time

import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def main() -> None:
    env = PassiveViewerWrapper(PandaPickCubeGymEnv(control_dt=0.1))
    env.reset(seed=42)

    tcp_sensor = env.unwrapped.data.sensor("2f85/pinch_pos")
    start = tcp_sensor.data.copy()
    print(f"初始 TCP [x, y, z]：{np.round(start, 3)} m")
    print("2 秒后开始沿世界坐标 +X 移动 10 cm。")

    try:
        # First let the user see the initial pose.
        zero_action = np.zeros(7, dtype=np.float32)
        for _ in range(20):
            if not env._viewer.is_running():
                return
            env.step(zero_action)
            time.sleep(env.unwrapped.control_dt)

        # The base gym_hil environment interprets xyz as Cartesian displacement
        # in metres per control step.  20 * 0.005 m = 0.10 m.
        move_x = np.array([0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            if not env._viewer.is_running():
                return
            env.step(move_x)
            time.sleep(env.unwrapped.control_dt)

        end = tcp_sensor.data.copy()
        print(f"移动后 TCP [x, y, z]：{np.round(end, 3)} m")
        print(f"实际位移：{np.round(end - start, 3)} m")
        print("现在发送零动作保持姿态；关闭窗口或按 Ctrl+C 退出。")

        while env._viewer.is_running():
            env.step(zero_action)
            time.sleep(env.unwrapped.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
