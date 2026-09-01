#!/usr/bin/env python
"""Script one MuJoCo pick-and-place and save the two camera rollouts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco.viewer
import numpy as np
from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
from gym_hil.mujoco_gym_env import GymRenderingSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco_pick_place"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open a live MuJoCo viewer while also saving the camera videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = PandaArrangeBoxesGymEnv(
        seed=args.seed,
        control_dt=0.05,
        physics_dt=0.002,
        render_spec=GymRenderingSpec(height=256, width=256),
    )
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.viewer else None
    frames: dict[str, list[np.ndarray]] = {"front": [], "wrist": []}
    step_count = 0

    def record() -> None:
        front, wrist = env.render()
        frames["front"].append(front.copy())
        frames["wrist"].append(wrist.copy())

    def step(action: np.ndarray) -> None:
        nonlocal step_count
        if viewer is not None and not viewer.is_running():
            raise KeyboardInterrupt
        _, _, terminated, truncated, _ = env.step(action.astype(np.float32))
        step_count += 1
        record()
        if viewer is not None:
            viewer.sync()
            # Run at approximately real time only when someone is watching.
            time.sleep(env.control_dt)
        if terminated or truncated:
            raise RuntimeError("The environment ended before the scripted motion finished")

    def hold(steps: int, gripper_delta: float = 0.0) -> None:
        for index in range(steps):
            action = np.zeros(7, dtype=np.float32)
            # One command is enough to move the gripper setpoint to an end stop.
            action[-1] = gripper_delta if index == 0 else 0.0
            step(action)

    def move_tcp(target: np.ndarray, *, label: str) -> None:
        for _ in range(80):
            tcp = env.data.sensor("2f85/pinch_pos").data.copy()
            error = target - tcp
            if np.linalg.norm(error) < 0.006:
                print(f"{label:>16}: tcp={np.round(tcp, 3)}")
                return
            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(error, -0.012, 0.012)
            step(action)
        raise RuntimeError(f"TCP failed to reach {label}: target={target}, tcp={tcp}")

    try:
        np.random.seed(args.seed)
        env.reset(seed=args.seed)

        # PandaArrangeBoxes provides useful block/target physics.  This demo is
        # intentionally a single-object task, so hide and disable blocks 2-5.
        for index in range(2, 6):
            block_geom = env.model.geom(f"block{index}")
            block_geom.rgba[3] = 0.0
            block_geom.contype = 0
            block_geom.conaffinity = 0
            env.model.geom(f"target{index}").rgba[3] = 0.0

        if viewer is not None:
            viewer.sync()
        record()

        block = env.data.sensor("block1_pos").data.copy()
        target = env.data.sensor("target1_pos").data.copy()
        print(f"block1 start: {np.round(block, 3)}")
        print(f"target1:      {np.round(target, 3)}")

        # In gym_hil 0.1.14 the physical 2F-85 model is open at actuator 0
        # and closes when a positive delta moves its command to 255.
        hold(8, gripper_delta=-1.0)
        move_tcp(block + np.array([0.0, 0.0, 0.13]), label="above block")
        # The TCP is the gripper pinch site, not the cube centre.  Its lowest
        # collision-free grasp height is about 3.5 cm above the cube centre.
        move_tcp(block + np.array([0.0, 0.0, 0.035]), label="at block")
        hold(12, gripper_delta=1.0)
        move_tcp(block + np.array([0.0, 0.0, 0.18]), label="lift")
        move_tcp(target + np.array([0.0, 0.0, 0.18]), label="above target")
        move_tcp(target + np.array([0.0, 0.0, 0.055]), label="place")
        hold(12, gripper_delta=-1.0)
        move_tcp(target + np.array([0.0, 0.0, 0.18]), label="retreat")
        hold(6)

        final_block = env.data.sensor("block1_pos").data.copy()
        distance = float(np.linalg.norm(final_block - target))
        success = distance < 0.04
        print(f"block1 final: {np.round(final_block, 3)}")
        print(f"target distance: {distance:.4f} m")
        print(f"pick-place success: {success}")
        print(f"control steps: {step_count}")

        fps = round(1.0 / env.control_dt)
        for camera_name, camera_frames in frames.items():
            output = args.output_dir / f"{camera_name}.mp4"
            imageio.mimsave(output, camera_frames, fps=fps, codec="libx264")
            print(f"saved: {output}")

        if not success:
            raise SystemExit(1)
    except KeyboardInterrupt:
        print("viewer closed; demo stopped")
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
