#!/usr/bin/env python
"""Tests for the position-only Panda DLS IK comparison."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from controllers.dls_ik import DLSIKConfig, PandaDLSIKController
from benchmark.ik_compare import (
    prepare_environment,
    run_episode,
    summarize,
    write_outputs,
)


class DLSIKNumericalTest(unittest.TestCase):
    def setUp(self) -> None:
        self.env, _, _ = prepare_environment(0)

    def tearDown(self) -> None:
        self.env.close()

    def test_jacobian_and_formula(self) -> None:
        config = DLSIKConfig(posture_gain=0.0)
        controller = PandaDLSIKController(
            self.env.model,
            self.env.data,
            control_dt=self.env.control_dt,
            config=config,
        )
        jacobian = controller.position_jacobian()
        self.assertEqual(jacobian.shape, (3, 7))
        self.assertTrue(np.all(np.isfinite(jacobian)))

        target = controller.tcp_position + np.asarray([1e-3, -2e-3, 1e-3])
        error = target - controller.tcp_position
        expected = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + config.damping**2 * np.eye(3), error
        )
        actual = controller.compute_delta_q(target)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_near_singular_result_is_finite_and_step_limited(self) -> None:
        controller = PandaDLSIKController(
            self.env.model, self.env.data, control_dt=self.env.control_dt
        )
        self.env.data.qpos[controller.qpos_ids] = controller.joint_upper - 1e-4
        self.env.data.qvel[controller.dof_ids] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

        delta_q = controller.compute_delta_q(controller.tcp_position + np.ones(3))
        self.assertTrue(np.all(np.isfinite(delta_q)))
        self.assertLessEqual(np.linalg.norm(delta_q), controller.config.max_joint_step + 1e-12)

        joint_target, _ = controller.compute_joint_target(
            controller.tcp_position + np.asarray([0.5, -0.5, 0.5])
        )
        self.assertTrue(np.all(joint_target >= controller.joint_lower))
        self.assertTrue(np.all(joint_target <= controller.joint_upper))

    def test_pd_torque_is_actuator_limited(self) -> None:
        controller = PandaDLSIKController(
            self.env.model, self.env.data, control_dt=self.env.control_dt
        )
        torque, clipped = controller.compute_pd_torque(controller.joint_position + 10.0)
        self.assertTrue(clipped)
        self.assertTrue(np.all(torque >= controller.torque_lower))
        self.assertTrue(np.all(torque <= controller.torque_upper))

    def test_collision_free_position_convergence(self) -> None:
        controller = PandaDLSIKController(
            self.env.model, self.env.data, control_dt=self.env.control_dt
        )
        target = controller.tcp_position + np.asarray([0.04, 0.03, -0.05])
        initial_error = float(np.linalg.norm(target - controller.tcp_position))
        errors = []
        for _ in range(80):
            diagnostics = controller.step(target)
            errors.append(diagnostics.error_after)
            if diagnostics.error_after < 0.006:
                break
        self.assertLess(errors[-1], 0.006)
        self.assertLess(errors[-1], initial_error)


class PickPlaceIntegrationTest(unittest.TestCase):
    def test_dls_completes_fixed_seed_pick_place(self) -> None:
        result = run_episode("dls", 0, ik_config=DLSIKConfig())
        self.assertTrue(result.success, result.failure_reason)
        self.assertLess(result.target_distance, 0.04)
        self.assertEqual(set(result.waypoint_errors), {
            "above_block",
            "at_block",
            "lift",
            "above_target",
            "place",
            "retreat",
        })

    def test_paired_scenarios_and_serialized_summary(self) -> None:
        results = []
        for seed in (0, 1):
            results.append(run_episode("dls", seed, ik_config=DLSIKConfig()))
            results.append(run_episode("builtin", seed, ik_config=DLSIKConfig()))

        for seed in (0, 1):
            pair = [result for result in results if result.seed == seed]
            positions = np.asarray(
                [[item.block_start_x, item.block_start_y, item.block_start_z] for item in pair]
            )
            np.testing.assert_array_equal(positions[0], positions[1])

        summary = summarize(results)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_outputs(results, summary, output_dir)
            with (output_dir / "episodes.csv").open(encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            with (output_dir / "summary.json").open(encoding="utf-8") as file:
                serialized_summary = json.load(file)

        self.assertEqual(len(rows), 4)
        self.assertEqual(serialized_summary, summary)
        for controller in ("dls", "builtin"):
            controller_rows = [row for row in rows if row["controller"] == controller]
            csv_success_rate = np.mean([row["success"] == "True" for row in controller_rows])
            self.assertEqual(serialized_summary[controller]["success_rate"], csv_success_rate)


if __name__ == "__main__":
    unittest.main()
