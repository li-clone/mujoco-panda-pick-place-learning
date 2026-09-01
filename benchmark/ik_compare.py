#!/usr/bin/env python
"""Compare position-only DLS IK with gym_hil's built-in 6D OSC."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from gym_hil.envs.panda_arrange_boxes_gym_env import PandaArrangeBoxesGymEnv
from gym_hil.mujoco_gym_env import GymRenderingSpec

from controllers.dls_ik import DLSIKConfig, PandaDLSIKController


ControllerName = Literal["dls", "builtin"]
WAYPOINT_LABELS = ("above_block", "at_block", "lift", "above_target", "place", "retreat")
WAYPOINT_TOLERANCE = 0.006
WAYPOINT_SETTLE_STEPS = 3
CSV_FIELDS = (
    "controller",
    "seed",
    "block_start_x",
    "block_start_y",
    "block_start_z",
    "success",
    "target_distance",
    "total_steps",
    *(f"{label}_error" for label in WAYPOINT_LABELS),
    "mean_waypoint_error",
    "max_waypoint_error",
    "joint_limit_clips",
    "torque_clipped_substeps",
    "failure_reason",
)


class WaypointTimeout(RuntimeError):
    def __init__(self, label: str, error: float) -> None:
        super().__init__(f"waypoint {label} timed out at {error:.4f} m")
        self.label = label
        self.error = error


class EnvironmentEnded(RuntimeError):
    pass


@dataclass
class EpisodeResult:
    controller: str
    seed: int
    block_start_x: float
    block_start_y: float
    block_start_z: float
    success: bool = False
    target_distance: float = float("nan")
    total_steps: int = 0
    waypoint_errors: dict[str, float] = field(default_factory=dict)
    joint_limit_clips: int = 0
    torque_clipped_substeps: int = 0
    failure_reason: str = ""

    def csv_row(self) -> dict[str, object]:
        errors = list(self.waypoint_errors.values())
        row: dict[str, object] = {
            "controller": self.controller,
            "seed": self.seed,
            "block_start_x": self.block_start_x,
            "block_start_y": self.block_start_y,
            "block_start_z": self.block_start_z,
            "success": self.success,
            "target_distance": self.target_distance,
            "total_steps": self.total_steps,
            "mean_waypoint_error": float(np.mean(errors)) if errors else float("nan"),
            "max_waypoint_error": float(np.max(errors)) if errors else float("nan"),
            "joint_limit_clips": self.joint_limit_clips,
            "torque_clipped_substeps": self.torque_clipped_substeps,
            "failure_reason": self.failure_reason,
        }
        row.update(
            {f"{label}_error": self.waypoint_errors.get(label, float("nan")) for label in WAYPOINT_LABELS}
        )
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=("dls", "builtin"),
        default=("dls", "builtin"),
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mujoco_ik_compare"),
    )
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--damping", type=float, default=0.05)
    parser.add_argument("--max-joint-step", type=float, default=0.05)
    parser.add_argument("--joint-limit-margin", type=float, default=0.02)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.viewer and (args.episodes != 1 or len(args.controllers) != 1):
        parser.error("--viewer requires one controller and --episodes 1")
    return args


def prepare_environment(seed: int) -> tuple[PandaArrangeBoxesGymEnv, np.ndarray, np.ndarray]:
    """Create one deterministic single-block task for a paired rollout."""

    env = PandaArrangeBoxesGymEnv(
        seed=seed,
        control_dt=0.05,
        physics_dt=0.002,
        render_spec=GymRenderingSpec(height=256, width=256),
    )
    np.random.seed(seed)
    env.reset(seed=seed)

    for index in range(2, 6):
        block_geom = env.model.geom(f"block{index}")
        block_geom.rgba[3] = 0.0
        block_geom.contype = 0
        block_geom.conaffinity = 0
        env.model.geom(f"target{index}").rgba[3] = 0.0

    rng = np.random.default_rng(seed)
    block_xy = np.asarray([rng.uniform(0.30, 0.42), rng.uniform(-0.15, 0.15)])
    block_joint = env.data.joint("block1")
    block_joint.qpos[:3] = (*block_xy, env._block_z)
    block_joint.qpos[3:] = np.asarray([1.0, 0.0, 0.0, 0.0])
    block_joint.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)

    block = env.data.sensor("block1_pos").data.copy()
    target = env.data.sensor("target1_pos").data.copy()
    return env, block, target


class PickPlaceRunner:
    """Shared state machine with swappable Cartesian control paths."""

    def __init__(
        self,
        env: PandaArrangeBoxesGymEnv,
        controller_name: ControllerName,
        *,
        ik_config: DLSIKConfig,
        viewer: mujoco.viewer.Handle | None = None,
        save_video: bool = False,
    ) -> None:
        self.env = env
        self.controller_name = controller_name
        self.viewer = viewer
        self.save_video = save_video
        self.frames: dict[str, list[np.ndarray]] = {"front": [], "wrist": []}
        self.total_steps = 0
        self.waypoint_errors: dict[str, float] = {}
        self.dls = (
            PandaDLSIKController(
                env.model,
                env.data,
                control_dt=env.control_dt,
                config=ik_config,
            )
            if controller_name == "dls"
            else None
        )

    @property
    def tcp_position(self) -> np.ndarray:
        return self.env.data.sensor("2f85/pinch_pos").data.copy()

    def record(self) -> None:
        if not self.save_video:
            return
        front, wrist = self.env.render()
        self.frames["front"].append(front.copy())
        self.frames["wrist"].append(wrist.copy())

    def after_step(self) -> None:
        self.total_steps += 1
        self.record()
        if self.viewer is not None:
            if not self.viewer.is_running():
                raise KeyboardInterrupt
            self.viewer.sync()
            time.sleep(self.env.control_dt)

    def builtin_step(self, action: np.ndarray) -> None:
        _, _, terminated, truncated, _ = self.env.step(action.astype(np.float32))
        self.after_step()
        if terminated or truncated:
            raise EnvironmentEnded("environment ended before the state machine finished")

    def dls_step(self, target: np.ndarray, *, gripper_delta: float = 0.0) -> None:
        assert self.dls is not None
        self.dls.step(target, gripper_delta=gripper_delta)
        self.after_step()

    def dls_hold_step(self, joint_target: np.ndarray, *, gripper_delta: float = 0.0) -> None:
        assert self.dls is not None
        self.dls.track_joint_target(joint_target, gripper_delta=gripper_delta)
        self.after_step()

    def hold(self, steps: int, *, gripper_delta: float = 0.0) -> None:
        joint_target = self.dls.joint_position if self.dls is not None else None
        for index in range(steps):
            command = gripper_delta if index == 0 else 0.0
            if self.controller_name == "builtin":
                action = np.zeros(7, dtype=np.float32)
                action[-1] = command
                self.builtin_step(action)
            else:
                assert joint_target is not None
                self.dls_hold_step(joint_target, gripper_delta=command)

    def move_tcp(self, target: np.ndarray, *, label: str) -> None:
        final_error = float("inf")
        for _ in range(80):
            error = np.asarray(target) - self.tcp_position
            final_error = float(np.linalg.norm(error))
            if final_error < WAYPOINT_TOLERANCE:
                # Continue a few closed-loop corrections after first entry
                # into the tolerance ball.  This prevents a systematic
                # edge-of-tolerance grasp offset while keeping the same 6 mm
                # convergence criterion for both controllers.
                for _ in range(WAYPOINT_SETTLE_STEPS):
                    settle_error = np.asarray(target) - self.tcp_position
                    if self.controller_name == "builtin":
                        action = np.zeros(7, dtype=np.float32)
                        action[:3] = np.clip(settle_error, -0.012, 0.012)
                        self.builtin_step(action)
                    else:
                        self.dls_step(np.asarray(target))
                self.waypoint_errors[label] = float(
                    np.linalg.norm(np.asarray(target) - self.tcp_position)
                )
                return

            if self.controller_name == "builtin":
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(error, -0.012, 0.012)
                self.builtin_step(action)
            else:
                self.dls_step(np.asarray(target))

        self.waypoint_errors[label] = final_error
        raise WaypointTimeout(label, final_error)

    def run(self, block: np.ndarray, target: np.ndarray) -> None:
        self.record()
        self.hold(8, gripper_delta=-1.0)
        self.move_tcp(block + np.asarray([0.0, 0.0, 0.13]), label="above_block")
        self.move_tcp(block + np.asarray([0.0, 0.0, 0.02]), label="at_block")
        self.hold(12, gripper_delta=1.0)
        self.move_tcp(block + np.asarray([0.0, 0.0, 0.18]), label="lift")
        self.move_tcp(target + np.asarray([0.0, 0.0, 0.18]), label="above_target")
        self.move_tcp(target + np.asarray([0.0, 0.0, 0.055]), label="place")
        self.hold(12, gripper_delta=-1.0)
        self.move_tcp(target + np.asarray([0.0, 0.0, 0.18]), label="retreat")
        self.hold(6)


def save_videos(
    frames: dict[str, list[np.ndarray]], output_dir: Path, *, fps: int
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for camera_name, camera_frames in frames.items():
        if camera_frames:
            imageio.mimsave(
                output_dir / f"{camera_name}.mp4",
                camera_frames,
                fps=fps,
                codec="libx264",
            )


def run_episode(
    controller_name: ControllerName,
    seed: int,
    *,
    ik_config: DLSIKConfig,
    viewer_enabled: bool = False,
    save_video: bool = False,
    video_dir: Path | None = None,
) -> EpisodeResult:
    env, block, target = prepare_environment(seed)
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if viewer_enabled else None
    runner = PickPlaceRunner(
        env,
        controller_name,
        ik_config=ik_config,
        viewer=viewer,
        save_video=save_video,
    )
    result = EpisodeResult(
        controller=controller_name,
        seed=seed,
        block_start_x=float(block[0]),
        block_start_y=float(block[1]),
        block_start_z=float(block[2]),
    )

    try:
        if viewer is not None:
            viewer.sync()
        runner.run(block, target)
        final_block = env.data.sensor("block1_pos").data.copy()
        result.target_distance = float(np.linalg.norm(final_block - target))
        result.success = result.target_distance < 0.04
        if not result.success:
            result.failure_reason = "task_miss"
    except WaypointTimeout as error:
        result.failure_reason = f"waypoint_timeout:{error.label}"
    except EnvironmentEnded:
        result.failure_reason = "environment_terminated"
    except KeyboardInterrupt:
        result.failure_reason = "viewer_closed"
    except Exception as error:  # Keep a batch benchmark running and report the failing type.
        result.failure_reason = f"exception:{type(error).__name__}"
    finally:
        if not np.isfinite(result.target_distance):
            final_block = env.data.sensor("block1_pos").data.copy()
            result.target_distance = float(np.linalg.norm(final_block - target))
        result.total_steps = runner.total_steps
        result.waypoint_errors = runner.waypoint_errors.copy()
        if runner.dls is not None:
            result.joint_limit_clips = runner.dls.limit_clip_count
            result.torque_clipped_substeps = runner.dls.torque_clip_count
        if save_video and video_dir is not None:
            save_videos(runner.frames, video_dir, fps=round(1.0 / env.control_dt))
        if viewer is not None:
            viewer.close()
        env.close()
    return result


def finite_values(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(results: list[EpisodeResult]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for controller in sorted({result.controller for result in results}):
        group = [result for result in results if result.controller == controller]
        distances = finite_values([result.target_distance for result in group])
        waypoint_errors = finite_values(
            [error for result in group for error in result.waypoint_errors.values()]
        )
        summary[controller] = {
            "episodes": len(group),
            "successes": sum(result.success for result in group),
            "success_rate": float(np.mean([result.success for result in group])),
            "target_distance_mean": float(np.mean(distances)),
            "target_distance_p95": float(np.percentile(distances, 95)),
            "waypoint_error_mean": float(np.mean(waypoint_errors)) if waypoint_errors.size else None,
            "waypoint_error_p95": (
                float(np.percentile(waypoint_errors, 95)) if waypoint_errors.size else None
            ),
            "control_steps_mean": float(np.mean([result.total_steps for result in group])),
            "joint_limit_clips_total": sum(result.joint_limit_clips for result in group),
            "torque_clipped_substeps_total": sum(
                result.torque_clipped_substeps for result in group
            ),
            "failure_counts": dict(
                Counter(result.failure_reason for result in group if result.failure_reason)
            ),
        }
    return summary


def write_outputs(
    results: list[EpisodeResult], summary: dict[str, dict[str, object]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(result.csv_row() for result in results)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")


def print_summary(summary: dict[str, dict[str, object]]) -> None:
    print("\ncontroller  success       target mean/p95    waypoint mean/p95   mean steps")
    print("----------  ------------  -----------------  ------------------  ----------")
    for controller, metrics in summary.items():
        waypoint_mean = metrics["waypoint_error_mean"]
        waypoint_p95 = metrics["waypoint_error_p95"]
        waypoint_text = (
            f"{waypoint_mean:.4f}/{waypoint_p95:.4f}"
            if waypoint_mean is not None and waypoint_p95 is not None
            else "n/a"
        )
        print(
            f"{controller:<10}  {metrics['successes']:>2}/{metrics['episodes']:<2} "
            f"({metrics['success_rate'] * 100:5.1f}%)  "
            f"{metrics['target_distance_mean']:.4f}/{metrics['target_distance_p95']:.4f}   "
            f"{waypoint_text:<18}  {metrics['control_steps_mean']:.1f}"
        )
        if metrics["failure_counts"]:
            print(f"  failures: {metrics['failure_counts']}")


def main() -> None:
    args = parse_args()
    ik_config = DLSIKConfig(
        damping=args.damping,
        max_joint_step=args.max_joint_step,
        joint_limit_margin=args.joint_limit_margin,
    )
    results: list[EpisodeResult] = []

    for seed in range(args.seed_start, args.seed_start + args.episodes):
        for controller in args.controllers:
            video_dir = args.output_dir / "videos" / f"{controller}_seed_{seed}"
            result = run_episode(
                controller,
                seed,
                ik_config=ik_config,
                viewer_enabled=args.viewer,
                save_video=args.save_video,
                video_dir=video_dir,
            )
            results.append(result)
            print(
                f"{controller:>7} seed={seed:03d} success={str(result.success):<5} "
                f"distance={result.target_distance:.4f} steps={result.total_steps} "
                f"failure={result.failure_reason or '-'}"
            )

    # Fail loudly if paired scenarios drift apart.
    by_seed: dict[int, list[EpisodeResult]] = {}
    for result in results:
        by_seed.setdefault(result.seed, []).append(result)
    for seed, group in by_seed.items():
        positions = np.asarray(
            [[result.block_start_x, result.block_start_y, result.block_start_z] for result in group]
        )
        if not np.allclose(positions, positions[0], atol=0.0, rtol=0.0):
            raise RuntimeError(f"paired initial block positions differ for seed {seed}")

    summary = summarize(results)
    write_outputs(results, summary, args.output_dir)
    print_summary(summary)
    print(f"\nwrote: {args.output_dir / 'episodes.csv'}")
    print(f"wrote: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
