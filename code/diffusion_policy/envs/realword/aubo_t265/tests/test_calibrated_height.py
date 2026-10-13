"""标定 Z 高度模式离线验证；不访问 SDK、相机或真实机械臂。"""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from diffusion_policy.envs.realword.aubo_t265.config import DEFAULT_CONFIG, load_config
from diffusion_policy.envs.realword.aubo_t265.calibration import load_calibration
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_t265.tests import test_height_recovery as recovery_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_strict_orientation as strict_tests


class CalibratedHeightTests(unittest.TestCase):
    def workspace(self):
        config = load_config()
        # 故意让标定 Z 与重力不同，防止测试因当前文件的两者恰好相同而漏检。
        matrices = [Rotation.from_euler('x', .3).as_matrix(),
                    Rotation.from_euler('y', -.4).as_matrix()]
        records = [{'operator_basis_base': matrix, 'gravity': [0., 0., -9.81]}
                   for matrix in matrices]
        with patch('diffusion_policy.envs.realword.aubo_t265.workspace.load_calibration',
                   side_effect=records) as loader:
            workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='calibrated_base'))
        self.assertEqual(loader.call_count, 2)
        return workspace, matrices

    def test_both_default_configs_select_calibrated_z_and_use_configured_limits(self):
        for path in (DEFAULT_CONFIG, DEFAULT_CONFIG.with_name('dual_no_smoothing.json')):
            config = load_config(path)
            self.assertEqual(config['workspace']['reference'], 'calibrated_base')
            w = DualWorkspace(config['arms'], config['workspace'])
            # 现场高度由所选 JSON 决定，不要求两份配置相同或固定数值。
            for side, distance in config['workspace']['max_downward_distance'].items():
                self.assertEqual(w.minimum_height(side), -distance)
            for arm in config['arms']:
                saved = load_calibration(arm['calibration'])
                np.testing.assert_array_equal(w.up[arm['side']], np.asarray(saved['operator_basis_base'])[:, 2])

    def test_height_uses_matrix_column_not_gravity_or_matrix_row(self):
        w, matrices = self.workspace()
        pose = np.array([.2, .4, -.3, .7, .8, .9])
        for side, matrix in zip(('left', 'right'), matrices):
            self.assertAlmostEqual(w.height(side, pose), (matrix.T @ pose[:3])[2])
            self.assertNotAlmostEqual(w.height(side, pose), pose[2])
            self.assertNotAlmostEqual(w.height(side, pose), matrix[2] @ pose[:3])

    def test_live_gravity_cannot_overwrite_calibrated_z(self):
        w, _ = self.workspace()
        original = w.up['left'].copy()
        with self.assertRaisesRegex(RuntimeError, 'fixed by calibration'):
            w.update_gravity('left', [0., 9.81, 0.])
        np.testing.assert_array_equal(w.up['left'], original)

    def test_each_projection_follows_calibrated_z_then_locks_attitude(self):
        w, _ = self.workspace()
        locked = strict_tests.StrictOrientationTests.locked
        for side in ('left', 'right'):
            w.set_locked_orientation(side, locked)
            axis = w.up[side]
            horizontal = np.cross(axis, [1., 0., 0.]) * .2
            pose = np.r_[horizontal + (w.minimum_height(side) - .001) * axis, [.7, .8, .9]]
            original = pose.copy()
            goal = w.recovery_target(side, pose)
            np.testing.assert_allclose(goal[:3] - pose[:3], .001 * axis, atol=1e-12)
            self.assertAlmostEqual(w.height(side, goal), w.minimum_height(side))
            np.testing.assert_array_equal(goal[3:], locked)
            np.testing.assert_array_equal(pose, original)
            w.validate_feedback(side, pose)
            with self.assertRaisesRegex(RuntimeError, 'height limit'):
                w.validate_target(side, pose)
            self.assertIsNone(w.recovery_target(side, goal))

    def test_v2_v3_recover_along_each_saved_axis_before_sdk_send(self):
        helper = strict_tests.StrictOrientationTests()
        for version in (2, 3):
            for side in ('left', 'right'):
                c, r, loop = recovery_tests.HeightRecoveryTests().rig(version)
                c.workspace, _ = self.workspace()
                c.workspace_side = side
                c.workspace.set_locked_orientation(side, helper.locked)
                axis = c.workspace.up[side]
                actual = np.r_[(c.workspace.minimum_height(side) - .001) * axis, [.7, .8, .9]]
                r.get_state.return_value['eef_pose'] = actual.copy()
                space, target = helper.sample(version, c, r, loop)
                helper.send(version, c, r, loop, space, target)
                goal = (r.inverse_kinematics.call_args.args[1] if version == 2
                        else r.servo_cartesian.call_args.args[0])
                np.testing.assert_allclose(goal[:3] - actual[:3], .001 * axis, atol=1e-12)
                np.testing.assert_array_equal(goal[3:], helper.locked)
                np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], actual)

    def test_legacy_gravity_mode_still_updates_height_axis(self):
        config = load_config()
        w = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base'))
        w.update_gravity('left', [0., 9.81, 0.])
        self.assertAlmostEqual(w.height('left', [.1, .4, -.2, 0., 0., 0.]), -.4)

    def test_invalid_rotation_rejected_before_control(self):
        config = load_config()
        with patch('diffusion_policy.envs.realword.aubo_t265.workspace.load_calibration',
                   return_value={'operator_basis_base': np.zeros((3, 3))}), \
             self.assertRaisesRegex(ValueError, 'proper 3x3 rotation'):
            DualWorkspace(config['arms'], dict(config['workspace'], reference='calibrated_base'))


if __name__ == '__main__':
    unittest.main()
