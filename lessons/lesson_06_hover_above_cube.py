#!/usr/bin/env python
"""Use the state observation to move the TCP above a randomly placed cube."""

from __future__ import annotations

import time

import numpy as np
from gym_hil.envs.panda_pick_gym_env import PandaPickCubeGymEnv
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper


def main() -> None:
    # gym_hil 0.1.14 samples the cube with NumPy's global RNG.
    np.random.seed(7)
    env = PassiveViewerWrapper(
        PandaPickCubeGymEnv(control_dt=0.1, random_block_position=True)
    )
    observation, _ = env.reset(seed=7)

    cube = observation["environment_state"].astype(np.float64)
    tcp_start = observation["agent_pos"][15:18].astype(np.float64)
    target = cube + np.array([0.0, 0.0, 0.15])

    print(f"方块坐标：    {np.round(cube, 3)} m")
    print(f"初始 TCP：    {np.round(tcp_start, 3)} m")
    print(f"悬停目标：    {np.round(target, 3)} m")
    print("2 秒后开始移动；夹爪不会闭合。")

    zero = np.zeros(7, dtype=np.float32)
    try:
        for _ in range(20):
            if not env._viewer.is_running():
                return
            env.step(zero)
            time.sleep(env.unwrapped.control_dt)

        for step_index in range(80):
            tcp = observation["agent_pos"][15:18].astype(np.float64)
            error = target - tcp
            if np.linalg.norm(error) < 0.006:
                break

            action = np.zeros(7, dtype=np.float32)
            # Limit every Cartesian increment to 8 mm for a smooth trajectory.
            action[:3] = np.clip(error, -0.008, 0.008)
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                raise RuntimeError("环境在到达悬停点之前结束")
            if step_index % 5 == 0:
                print(
                    f"step {step_index:02d}: TCP={np.round(tcp, 3)}, "
                    f"error={np.linalg.norm(error):.3f} m"
                )
            time.sleep(env.unwrapped.control_dt)
        else:
            raise RuntimeError("TCP 未在 80 个控制步内到达目标")

        tcp_end = observation["agent_pos"][15:18]
        final_error = float(np.linalg.norm(target - tcp_end))
        print(f"最终 TCP：    {np.round(tcp_end, 3)} m")
        print(f"目标误差：    {final_error:.4f} m")
        print("已到达方块上方；关闭窗口或按 Ctrl+C 退出。")

        while env._viewer.is_running():
            env.step(zero)
            time.sleep(env.unwrapped.control_dt)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
