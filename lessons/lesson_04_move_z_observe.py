#!/usr/bin/env python
"""Lift the Panda TCP along +Z and inspect the 18-D robot observation."""

from __future__ import annotations

import time

import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def print_state(label: str, agent_pos: np.ndarray) -> None:
    joint_pos = agent_pos[0:7]
    joint_vel = agent_pos[7:14]
    gripper = agent_pos[14]
    tcp = agent_pos[15:18]

    print(f"\n{label}")
    print(f"  关节角（度）：{np.round(np.rad2deg(joint_pos), 1)}")
    print(f"  关节速度：    {np.round(joint_vel, 3)} rad/s")
    print(f"  夹爪控制值：  {gripper:.1f}")
    print(f"  TCP [x,y,z]： {np.round(tcp, 3)} m")


def main() -> None:
    env = PassiveViewerWrapper(PandaPickCubeGymEnv(control_dt=0.1))
    observation, _ = env.reset(seed=42)
    start_state = observation["agent_pos"].copy()
    print_state("移动前的 18 维 agent_pos", start_state)
    print("\n2 秒后沿世界坐标 +Z 抬高约 10 cm。")

    zero_action = np.zeros(7, dtype=np.float32)
    try:
        for _ in range(20):
            if not env._viewer.is_running():
                return
            env.step(zero_action)
            time.sleep(env.unwrapped.control_dt)

        move_z = np.array([0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(20):
            if not env._viewer.is_running():
                return
            observation, _, _, _, _ = env.step(move_z)
            time.sleep(env.unwrapped.control_dt)

        # Let the controller settle so velocities approach zero.
        for _ in range(10):
            observation, _, _, _, _ = env.step(zero_action)
            time.sleep(env.unwrapped.control_dt)

        end_state = observation["agent_pos"].copy()
        print_state("移动并稳定后的 18 维 agent_pos", end_state)
        print(
            f"\nTCP 实际位移：{np.round(end_state[15:18] - start_state[15:18], 3)} m"
        )
        print("现在保持姿态；关闭窗口或按 Ctrl+C 退出。")

        while env._viewer.is_running():
            env.step(zero_action)
            time.sleep(env.unwrapped.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
