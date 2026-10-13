"""左右独立高度下限和旋转矩阵等价性的离线验证。"""
import unittest
import numpy as np
from diffusion_policy.envs.realword.aubo_t265.config import DEFAULT_CONFIG, load_config
from diffusion_policy.envs.realword.aubo_t265.calibration import load_calibration
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_t265.tests import test_height_recovery as recovery_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_strict_orientation as strict_tests


class ArmHeightLimitTests(unittest.TestCase):
    def workspace(self):
        c = load_config()
        w = DualWorkspace(c['arms'], dict(c['workspace'], reference='gravity_base',
                          max_downward_distance={'left': .57, 'right': .55}))
        for side in ('left', 'right'):
            w.update_gravity(side, [0., 9.81, 0.])
        return w

    def rig(self, version, side, depth):
        c, r, loop = recovery_tests.HeightRecoveryTests().rig(version)
        c.workspace, c.workspace_side = self.workspace(), side
        r.get_state.return_value['eef_pose'][1] = depth
        c.workspace.set_locked_orientation(side, strict_tests.StrictOrientationTests.locked)
        return c, r, loop

    def test_both_configs_use_their_configured_height_limits(self):
        for path in (DEFAULT_CONFIG, DEFAULT_CONFIG.with_name('dual_no_smoothing.json')):
            config = load_config(path)
            w = DualWorkspace(config['arms'], config['workspace'])
            # 现场高度由所选 JSON 决定，不要求两份配置相同或固定数值。
            for side, distance in config['workspace']['max_downward_distance'].items():
                self.assertEqual(w.minimum_height(side), -distance)
            self.assertEqual(w.target_margin, 0.)

    def test_560_mm_allowed_only_on_left(self):
        w = self.workspace()
        pose = np.array([.1, .56, -.2, .1, .2, .3])
        w.validate_target('left', pose)
        self.assertIsNone(w.recovery_target('left', pose))
        np.testing.assert_array_equal(w.limit_target('left', pose), pose)
        with self.assertRaisesRegex(RuntimeError, 'limit -550.00 mm'):
            w.validate_target('right', pose)
        self.assertAlmostEqual(w.limit_target('right', pose)[1], .55)

    def test_projection_and_feedback_recovery_use_each_inclusive_floor(self):
        w = self.workspace()
        for side, distance in (('left', .57), ('right', .55)):
            pose = np.array([.1, distance, -.2, .1, .2, .3])
            w.validate_target(side, pose)
            self.assertIsNone(w.recovery_target(side, pose))
            pose[1] += .001
            original = pose.copy()
            goal = w.recovery_target(side, pose)
            self.assertAlmostEqual(goal[1], distance)
            np.testing.assert_array_equal(goal[[0, 2, 3, 4, 5]], pose[[0, 2, 3, 4, 5]])
            np.testing.assert_array_equal(pose, original)
            w.validate_feedback(side, pose)
            with self.assertRaisesRegex(RuntimeError, f'limit {-distance * 1000:.2f} mm'):
                w.validate_target(side, pose)

    def test_controller_recovery_both_versions_both_sides_preserves_lock(self):
        helper = strict_tests.StrictOrientationTests()
        for version in (2, 3):
            for side, floor in (('left', .57), ('right', .55)):
                c, r, loop = self.rig(version, side, floor + .001)
                original = r.get_state.return_value['eef_pose'].copy()
                space, target = helper.sample(version, c, r, loop)
                helper.send(version, c, r, loop, space, target)
                self.assertAlmostEqual(loop.height_recovery_pose[1], floor)
                np.testing.assert_array_equal(loop.target_pose[3:], helper.locked)
                np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], original)

    def test_left_between_old_and_new_floor_needs_no_recovery(self):
        helper = strict_tests.StrictOrientationTests()
        for version in (2, 3):
            c, r, loop = self.rig(version, 'left', .56)
            loop.target_pose = r.get_state.return_value['eef_pose'].copy()
            if version == 3:
                loop.last_space = 'joint'
            helper.sample(version, c, r, loop)
            self.assertIsNone(loop.height_recovery_pose)
            self.assertAlmostEqual(loop.target_pose[1], .56)

    def test_saved_matrix_z_equals_gravity_projection(self):
        config = load_config()
        for arm in config['arms']:
            calibration = load_calibration(arm['calibration'])
            rotation = np.asarray(calibration['operator_basis_base'])
            gravity = np.asarray(calibration['gravity'])
            up = -gravity / np.linalg.norm(gravity)
            np.testing.assert_allclose(rotation[:, 2], up, atol=1e-12)
            pose = np.array([.2, .4, -.3])
            self.assertAlmostEqual((rotation.T @ pose)[2], up @ pose)
            self.assertIsNone(arm['world_from_base'])

    def test_legacy_scalar_still_applies_to_both_sides(self):
        config = load_config()
        w = DualWorkspace(config['arms'], dict(config['workspace'], max_downward_distance=.55))
        for side in ('left', 'right'):
            self.assertEqual(w.minimum_height(side), -.55)

    def test_invalid_mappings_fail_without_robot_access(self):
        config = load_config()
        invalid = ({'left': .57}, {'left': .57, 'rigth': .55},
                   {'left': .57, 'right': .55, 'extra': .6},
                   {'left': 0., 'right': .55}, {'left': -.57, 'right': .55},
                   {'left': np.nan, 'right': .55}, {'left': np.inf, 'right': .55},
                   {'left': True, 'right': .55}, {'left': '.57', 'right': .55}, None)
        for limits in invalid:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                DualWorkspace(config['arms'], dict(config['workspace'], max_downward_distance=limits))

    def test_world_transform_keeps_translation_in_height(self):
        # 世界原点的高度要包含平移；当前重力模式不隐式假设平移为零。
        transform = np.eye(4)
        transform[2, 3] = .4
        w = DualWorkspace([{'side': side, 'world_from_base': transform}
                           for side in ('left', 'right')],
                          {'enabled': True, 'reference': 'world', 'minimum_world_z': -.2})
        pose = np.array([.1, .2, -.7, .1, .2, .3])
        self.assertAlmostEqual(w.height('left', pose), -.3)
        limited = w.limit_target('left', pose)
        self.assertAlmostEqual(limited[2], -.6)
        self.assertAlmostEqual(w.height('left', limited), -.2)


if __name__ == '__main__':
    unittest.main()
