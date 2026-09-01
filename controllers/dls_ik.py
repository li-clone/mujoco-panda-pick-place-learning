#!/usr/bin/env python
"""Position-only damped-least-squares IK for the gym_hil Panda model."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


PANDA_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
PANDA_ACTUATOR_NAMES = tuple(f"actuator{index}" for index in range(1, 8))


@dataclass(frozen=True)
class DLSIKConfig:
    """Numerical IK and joint-space control settings."""

    damping: float = 0.05
    max_joint_step: float = 0.05
    joint_limit_margin: float = 0.02
    solver_tolerance: float = 1e-5
    max_solver_iterations: int = 100
    posture_gain: float = 0.2
    kp: np.ndarray = field(
        # 1.5x the initial [120,120,100,80,40,30,20] proposal removes the
        # steady-state error caused by the grasped 0.1 kg payload.  Actuator
        # ctrlrange clipping remains the final safety bound.
        default_factory=lambda: np.asarray([180.0, 180.0, 150.0, 120.0, 60.0, 45.0, 30.0])
    )
    nominal_joint_position: np.ndarray = field(
        default_factory=lambda: np.asarray([0.06, 0.05, 0.24, -2.73, -0.01, 2.76, 1.09])
    )

    def __post_init__(self) -> None:
        if self.damping <= 0:
            raise ValueError("damping must be positive")
        if self.max_joint_step <= 0:
            raise ValueError("max_joint_step must be positive")
        if self.joint_limit_margin < 0:
            raise ValueError("joint_limit_margin must be non-negative")
        if self.solver_tolerance <= 0:
            raise ValueError("solver_tolerance must be positive")
        if self.max_solver_iterations < 1:
            raise ValueError("max_solver_iterations must be at least 1")
        if self.posture_gain < 0:
            raise ValueError("posture_gain must be non-negative")
        if np.asarray(self.kp).shape != (7,) or np.any(np.asarray(self.kp) <= 0):
            raise ValueError("kp must contain seven positive gains")
        if np.asarray(self.nominal_joint_position).shape != (7,):
            raise ValueError("nominal_joint_position must contain seven values")


@dataclass(frozen=True)
class DLSIKStep:
    """Diagnostics produced by one outer-loop IK/control step."""

    error_before: float
    error_after: float
    delta_q_norm: float
    limit_clipped: bool
    torque_clipped_substeps: int


class PandaDLSIKController:
    """3D DLS IK followed by joint PD torque control.

    The controller intentionally solves position only.  MuJoCo supplies the
    Panda kinematics through ``mj_jacSite``; no hand-written forward
    kinematics or orientation error is used.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        control_dt: float,
        config: DLSIKConfig | None = None,
        site_name: str = "pinch",
        gripper_actuator_name: str = "fingers_actuator",
    ) -> None:
        self.model = model
        self.data = data
        self.config = config or DLSIKConfig()

        self.site_id = model.site(site_name).id
        self.joint_ids = np.asarray([model.joint(name).id for name in PANDA_JOINT_NAMES])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids].copy()
        self.dof_ids = model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids = np.asarray(
            [model.actuator(name).id for name in PANDA_ACTUATOR_NAMES]
        )
        self.gripper_actuator_id = model.actuator(gripper_actuator_name).id

        self.joint_lower = model.jnt_range[self.joint_ids, 0] + self.config.joint_limit_margin
        self.joint_upper = model.jnt_range[self.joint_ids, 1] - self.config.joint_limit_margin
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("joint limit margin leaves an empty valid range")

        self.torque_lower = model.actuator_ctrlrange[self.actuator_ids, 0].copy()
        self.torque_upper = model.actuator_ctrlrange[self.actuator_ids, 1].copy()
        self.kp = np.asarray(self.config.kp, dtype=np.float64)
        self.kd = 2.0 * np.sqrt(self.kp)

        physics_dt = float(model.opt.timestep)
        self.n_substeps = int(round(control_dt / physics_dt))
        if self.n_substeps < 1 or not np.isclose(self.n_substeps * physics_dt, control_dt):
            raise ValueError("control_dt must be a positive integer multiple of physics_dt")

        self.limit_clip_count = 0
        self.torque_clip_count = 0
        self._ik_data = mujoco.MjData(model)
        self.nominal_joint_position = np.asarray(
            self.config.nominal_joint_position, dtype=np.float64
        ).copy()

    @property
    def tcp_position(self) -> np.ndarray:
        return self.data.site_xpos[self.site_id].copy()

    @property
    def joint_position(self) -> np.ndarray:
        return self.data.qpos[self.qpos_ids].copy()

    def position_jacobian(self, data: mujoco.MjData | None = None) -> np.ndarray:
        """Return the 3x7 world-frame translational TCP Jacobian."""

        source_data = self.data if data is None else data
        jacobian_position = np.zeros((3, self.model.nv), dtype=np.float64)
        jacobian_rotation = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(
            self.model,
            source_data,
            jacobian_position,
            jacobian_rotation,
            self.site_id,
        )
        return jacobian_position[:, self.dof_ids]

    def compute_delta_q(
        self, target_position: np.ndarray, data: mujoco.MjData | None = None
    ) -> np.ndarray:
        """Compute a norm-limited DLS joint increment for a 3D target."""

        target = np.asarray(target_position, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError("target_position must have shape (3,)")

        source_data = self.data if data is None else data
        jacobian = self.position_jacobian(source_data)
        position_error = target - source_data.site_xpos[self.site_id]
        regularized = (
            jacobian @ jacobian.T
            + self.config.damping**2 * np.eye(3, dtype=np.float64)
        )
        damped_inverse = jacobian.T @ np.linalg.solve(regularized, np.eye(3))
        delta_q = damped_inverse @ position_error

        # Position-only IK leaves four redundant directions on a 7-DoF arm.
        # A weak nominal-posture term selects a stable gripper-down solution
        # without adding orientation error to the primary task.
        q = source_data.qpos[self.qpos_ids]
        nullspace = np.eye(7) - damped_inverse @ jacobian
        delta_q += self.config.posture_gain * nullspace @ (self.nominal_joint_position - q)

        norm = float(np.linalg.norm(delta_q))
        if norm > self.config.max_joint_step:
            delta_q *= self.config.max_joint_step / norm
        return delta_q

    def solve_joint_target(
        self, target_position: np.ndarray
    ) -> tuple[np.ndarray, bool, float]:
        """Iterate DLS on scratch MuJoCo data without changing physical state."""

        target = np.asarray(target_position, dtype=np.float64)
        self._ik_data.qpos[:] = self.data.qpos
        self._ik_data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self._ik_data)

        clipped_any = False
        first_delta_norm = 0.0
        for iteration in range(self.config.max_solver_iterations):
            error = target - self._ik_data.site_xpos[self.site_id]
            if np.linalg.norm(error) < self.config.solver_tolerance:
                break
            delta_q = self.compute_delta_q(target, self._ik_data)
            if iteration == 0:
                first_delta_norm = float(np.linalg.norm(delta_q))
            raw_target = self._ik_data.qpos[self.qpos_ids] + delta_q
            joint_target = np.clip(raw_target, self.joint_lower, self.joint_upper)
            clipped_any = clipped_any or not np.array_equal(raw_target, joint_target)
            self._ik_data.qpos[self.qpos_ids] = joint_target
            mujoco.mj_forward(self.model, self._ik_data)

        return self._ik_data.qpos[self.qpos_ids].copy(), clipped_any, first_delta_norm

    def compute_joint_target(self, target_position: np.ndarray) -> tuple[np.ndarray, bool]:
        """Apply DLS and joint-limit clipping to form the next reference."""

        raw_target = self.joint_position + self.compute_delta_q(target_position)
        joint_target = np.clip(raw_target, self.joint_lower, self.joint_upper)
        clipped = not np.array_equal(raw_target, joint_target)
        if clipped:
            self.limit_clip_count += 1
        return joint_target, clipped

    def compute_pd_torque(self, joint_target: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return gravity-compensated, actuator-limited joint torque."""

        target = np.asarray(joint_target, dtype=np.float64)
        if target.shape != (7,):
            raise ValueError("joint_target must have shape (7,)")

        q = self.data.qpos[self.qpos_ids]
        dq = self.data.qvel[self.dof_ids]
        raw_torque = self.kp * (target - q) - self.kd * dq + self.data.qfrc_bias[self.dof_ids]
        torque = np.clip(raw_torque, self.torque_lower, self.torque_upper)
        clipped = not np.array_equal(raw_torque, torque)
        return torque, clipped

    def set_gripper_delta(self, gripper_delta: float) -> None:
        """Update the gym_hil gripper setpoint with its native delta semantics."""

        actuator = self.gripper_actuator_id
        lower, upper = self.model.actuator_ctrlrange[actuator]
        current_fraction = (self.data.ctrl[actuator] - lower) / (upper - lower)
        next_fraction = np.clip(current_fraction + gripper_delta, 0.0, 1.0)
        self.data.ctrl[actuator] = lower + next_fraction * (upper - lower)

    def track_joint_target(
        self, joint_target: np.ndarray, *, gripper_delta: float = 0.0
    ) -> int:
        """Hold one joint reference for a complete outer control interval."""

        target = np.clip(
            np.asarray(joint_target, dtype=np.float64), self.joint_lower, self.joint_upper
        )
        if target.shape != (7,):
            raise ValueError("joint_target must have shape (7,)")
        self.set_gripper_delta(gripper_delta)
        clipped_substeps = 0
        for _ in range(self.n_substeps):
            torque, torque_clipped = self.compute_pd_torque(target)
            self.data.ctrl[self.actuator_ids] = torque
            clipped_substeps += int(torque_clipped)
            mujoco.mj_step(self.model, self.data)
        self.torque_clip_count += clipped_substeps
        return clipped_substeps

    def step(self, target_position: np.ndarray, *, gripper_delta: float = 0.0) -> DLSIKStep:
        """Run one DLS outer step and one PD-controlled MuJoCo interval."""

        target = np.asarray(target_position, dtype=np.float64)
        error_before = float(np.linalg.norm(target - self.tcp_position))
        joint_target, limit_clipped, delta_q_norm = self.solve_joint_target(target)
        if limit_clipped:
            self.limit_clip_count += 1

        clipped_substeps = self.track_joint_target(
            joint_target, gripper_delta=gripper_delta
        )
        error_after = float(np.linalg.norm(target - self.tcp_position))
        return DLSIKStep(
            error_before=error_before,
            error_after=error_after,
            delta_q_norm=delta_q_norm,
            limit_clipped=limit_clipped,
            torque_clipped_substeps=clipped_substeps,
        )
