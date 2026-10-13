"""Offline quaternion mapping acceptance; no robot or camera access."""

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.envs.realword.aubo_t265.pose_mapper import (
    QuaternionPoseMapper,
    operator_rotation_vector,
)


HOME = np.array([0.5, 0.1, 0.4, 0.2, -0.1, 0.3])


class QuaternionMapperTests(unittest.TestCase):
    def mapper(self, **kwargs):
        defaults = dict(
            mapping=np.eye(3),
            home=HOME,
            operator_basis_t265=np.eye(3),
            rotation_basis_t265=np.eye(3),
            tau=0.0,
            rotation_tau=0.0,
            speed=10.0,
            angular_speed=10.0,
        )
        defaults.update(kwargs)
        return QuaternionPoseMapper(**defaults)

    def test_pitch_crosses_euler_gimbal_lock_continuously(self):
        mapper = self.mapper(
            angle=np.deg2rad(120.0), angular_speed=np.deg2rad(10000.0)
        )
        mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        poses = [
            mapper.step(
                np.zeros(3), Rotation.from_euler("y", pitch, degrees=True), 0.02
            )
            for pitch in (88.0, 89.0, 90.0, 91.0, 92.0)
        ]
        increments = [
            (
                Rotation.from_rotvec(current[3:])
                * Rotation.from_rotvec(previous[3:]).inv()
            ).magnitude()
            for previous, current in zip(poses, poses[1:])
        ]
        np.testing.assert_allclose(increments, np.deg2rad(1.0), atol=1e-10)

    def test_quaternion_antipodes_map_to_same_target(self):
        rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.2])
        q = rotation.as_quat()
        first, second = self.mapper(), self.mapper()
        for mapper in (first, second):
            mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        a = first.step(np.zeros(3), Rotation.from_quat(q), 0.02)
        b = second.step(np.zeros(3), Rotation.from_quat(-q), 0.02)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_total_rotation_limit_is_geodesic(self):
        mapper = self.mapper(angle=np.deg2rad(30.0))
        mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        pose = mapper.step(
            np.zeros(3),
            Rotation.from_euler("xyz", [70.0, 60.0, 50.0], degrees=True),
            0.05,
        )
        relative = (
            Rotation.from_rotvec(pose[3:])
            * Rotation.from_rotvec(HOME[3:]).inv()
        )
        self.assertLessEqual(relative.magnitude(), np.deg2rad(30.0) + 1e-12)

    def test_angular_speed_limit(self):
        mapper = self.mapper(
            angle=np.deg2rad(120.0), angular_speed=np.deg2rad(10.0)
        )
        mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        pose = mapper.step(
            np.zeros(3), Rotation.from_euler("x", 90.0, degrees=True), 0.02
        )
        relative = (
            Rotation.from_rotvec(pose[3:])
            * Rotation.from_rotvec(HOME[3:]).inv()
        )
        self.assertLessEqual(relative.magnitude(), np.deg2rad(0.2) + 1e-12)

    def test_reanchor_has_no_pose_jump(self):
        robot = np.r_[
            HOME[:3] + [0.02, -0.01, 0.03],
            Rotation.from_euler("xyz", [0.1, 0.2, -0.1]).as_rotvec(),
        ]
        mapper = self.mapper()
        hand_rotation = Rotation.from_euler("xyz", [0.2, -0.3, 0.4])
        mapper.anchor([1.0, 2.0, 3.0], hand_rotation, robot)
        result = mapper.step([1.0, 2.0, 3.0], hand_rotation, 0.02)
        np.testing.assert_allclose(result, robot, atol=1e-12)

    def test_left_translation_signs_do_not_change_rotation(self):
        normal = self.mapper()
        inverted = self.mapper(position_axis_signs=[-1, -1, 1])
        orientation = Rotation.from_euler("xyz", [0.1, 0.2, 0.3])
        for mapper in (normal, inverted):
            mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        a = normal.step([0.01, 0.02, 0.03], orientation, 0.02)
        b = inverted.step([0.01, 0.02, 0.03], orientation, 0.02)
        np.testing.assert_allclose(a[3:], b[3:], atol=1e-12)
        np.testing.assert_allclose(
            b[:3] - HOME[:3],
            (a[:3] - HOME[:3]) * [-1, -1, 1],
            atol=1e-12,
        )

    def test_operator_diagnostic_uses_rotation_vector(self):
        pose = HOME.copy()
        basis = Rotation.from_euler("z", 0.4).as_matrix()
        delta = Rotation.from_rotvec(basis @ np.array([0.1, 0.2, -0.1]))
        pose[3:] = (delta * Rotation.from_rotvec(HOME[3:])).as_rotvec()
        np.testing.assert_allclose(
            operator_rotation_vector(pose, HOME, basis),
            [0.1, 0.2, -0.1],
            atol=1e-12,
        )

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            self.mapper(rotation_axis_signs=[1, 0, 1])
        mapper = self.mapper()
        mapper.anchor(np.zeros(3), Rotation.identity(), HOME)
        with self.assertRaises(ValueError):
            mapper.step([0, np.nan, 0], Rotation.identity(), 0.02)


if __name__ == "__main__":
    unittest.main()
