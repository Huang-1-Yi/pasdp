"""Quaternion attitude mapping with a rotation-vector controller boundary.

All attitude composition, limiting, and filtering uses ``Rotation`` objects
(unit quaternions internally).  The returned public pose remains xyz+rotvec so
the proven AUBO controller and shared-memory layout can be reused unchanged.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import bounded, rotation_matrix


def _position(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("position must be a finite 3-vector")
    return value.copy()


def _pose(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("pose must be finite xyz+rotation-vector")
    return value.copy()


def _signs(value) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,) or not np.isin(value, (-1.0, 1.0)).all():
        raise ValueError("axis signs must contain three +1/-1 values")
    return value.copy()


def operator_rotation_vector(pose, home, operator_basis_base) -> np.ndarray:
    """Relative attitude expressed in operator axes, without Euler angles."""
    pose = _pose(pose)
    home = _pose(home)
    basis = Rotation.from_matrix(rotation_matrix(operator_basis_base))
    relative = Rotation.from_rotvec(pose[3:]) * Rotation.from_rotvec(home[3:]).inv()
    return (basis.inv() * relative * basis).as_rotvec()


class QuaternionPoseMapper:
    """Map a T265 pose to a bounded robot target using quaternion operations."""

    def __init__(
        self,
        mapping,
        home,
        scale=0.8,
        rotation_scale=1.0,
        radius=0.15,
        angle=np.deg2rad(30.0),
        speed=0.05,
        angular_speed=np.deg2rad(30.0),
        tau=0.06,
        deadzone=0.0002,
        operator_basis_t265=None,
        rotation_tau=None,
        rotation_basis_t265=None,
        rotation_axis_signs=None,
        position_axis_signs=None,
    ):
        self.mapping = rotation_matrix(mapping)
        self.operator_basis_t265 = rotation_matrix(
            np.eye(3) if operator_basis_t265 is None else operator_basis_t265
        )
        self.rotation_basis_t265 = rotation_matrix(
            self.operator_basis_t265
            if rotation_basis_t265 is None
            else rotation_basis_t265
        )
        self.operator_basis_base = rotation_matrix(
            self.mapping @ self.operator_basis_t265
        )
        self.rotation_axis_signs = _signs(
            [1.0, 1.0, 1.0]
            if rotation_axis_signs is None
            else rotation_axis_signs
        )
        self.position_axis_signs = _signs(
            [1.0, 1.0, 1.0]
            if position_axis_signs is None
            else position_axis_signs
        )
        numeric = (scale, rotation_scale, speed, angular_speed, tau, deadzone)
        if not all(np.isfinite(v) and v >= 0 for v in numeric):
            raise ValueError("mapping parameters must be finite and nonnegative")
        if radius is not None and (not np.isfinite(radius) or radius <= 0):
            raise ValueError("translation radius must be positive or None")
        if angle is not None and (
            not np.isfinite(angle) or angle <= 0 or angle >= np.pi
        ):
            raise ValueError("rotation limit must be in (0, pi), or None")
        if angle is None and rotation_scale not in (0.0, 1.0):
            raise ValueError("unbounded rotation requires scale 0 or 1")
        self.rotation_tau = tau if rotation_tau is None else rotation_tau
        if not np.isfinite(self.rotation_tau) or self.rotation_tau < 0:
            raise ValueError("rotation tau must be finite and nonnegative")
        self.home = _pose(home)
        self.pose = self.home.copy()
        self.scale = float(scale)
        self.rotation_scale = float(rotation_scale)
        self.radius = radius
        self.angle = angle
        self.speed = float(speed)
        self.angular_speed = float(angular_speed)
        self.tau = float(tau)
        self.deadzone = float(deadzone)
        self.hand_rotvec = np.zeros(3)
        self.limited_rotvec = np.zeros(3)

    def anchor(self, position, orientation: Rotation, robot_pose):
        self.hand = _position(position)
        self.hand_rotation = orientation
        self.robot = _pose(robot_pose)
        self.pose = self.robot.copy()
        self.hand_rotvec = np.zeros(3)
        self.limited_rotvec = operator_rotation_vector(
            self.pose, self.home, self.operator_basis_base
        )

    def step(self, position, orientation: Rotation, dt, *, limit_speed=True):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        dt = min(float(dt), 0.05)

        hand_delta_world = self.operator_basis_t265.T @ (
            _position(position) - self.hand
        )
        delta = self.operator_basis_base @ (
            hand_delta_world * self.position_axis_signs
        ) * self.scale
        length = float(np.linalg.norm(delta))
        delta *= max(0.0, 1.0 - self.deadzone / max(length, 1e-12))
        raw_target_p = self.robot[:3] + delta
        target_p = (
            raw_target_p
            if self.radius is None
            else self.home[:3]
            + bounded(raw_target_p - self.home[:3], self.radius)
        )

        sensor_basis = Rotation.from_matrix(self.rotation_basis_t265)
        base_basis = Rotation.from_matrix(self.operator_basis_base)
        hand_delta = (
            sensor_basis.inv()
            * (orientation * self.hand_rotation.inv())
            * sensor_basis
        )
        # The rotation logarithm is continuous through Euler gimbal lock.  It
        # is used only for calibrated axis signs/gain, then reconstructed as a
        # unit quaternion rotation.
        self.hand_rotvec = hand_delta.as_rotvec() * self.rotation_axis_signs
        gesture_rotvec = self.hand_rotvec * self.rotation_scale
        if self.angle is not None:
            gesture_rotvec = bounded(gesture_rotvec, self.angle)
        elif self.rotation_scale == 0.0:
            gesture_rotvec[:] = 0.0
        gesture = Rotation.from_rotvec(gesture_rotvec)

        raw_target_r = (
            base_basis
            * gesture
            * base_basis.inv()
            * Rotation.from_rotvec(self.robot[3:])
        )
        home_r = Rotation.from_rotvec(self.home[3:])
        relative_home = raw_target_r * home_r.inv()
        limited_relative = relative_home.as_rotvec()
        if self.angle is not None:
            limited_relative = bounded(limited_relative, self.angle)
        target_r = Rotation.from_rotvec(limited_relative) * home_r
        self.limited_rotvec = (
            base_basis.inv() * (target_r * home_r.inv()) * base_basis
        ).as_rotvec()

        alpha = -np.expm1(-dt / self.tau) if self.tau else 1.0
        rotation_alpha = (
            -np.expm1(-dt / self.rotation_tau) if self.rotation_tau else 1.0
        )
        position_step = alpha * (target_p - self.pose[:3])
        self.pose[:3] += (bounded(position_step, self.speed * dt)
                         if limit_speed else position_step)
        current_r = Rotation.from_rotvec(self.pose[3:])
        step = rotation_alpha * (target_r * current_r.inv()).as_rotvec()
        if limit_speed:
            step = bounded(step, self.angular_speed * dt)
        self.pose[3:] = (Rotation.from_rotvec(step) * current_r).as_rotvec()
        return self.pose.copy()
