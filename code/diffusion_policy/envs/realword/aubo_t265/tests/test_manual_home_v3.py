"""v3 启动保持 → 首次 H 关节归位 → 固定姿态遥操；全部离线。"""
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import numpy as np
from diffusion_policy.envs.realword.aubo_homing import JointHome, LockedCartesianHome
from diffusion_policy.envs.realword.aubo_t265.tests import test_servo_cartesian_v3 as cartesian_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_strict_orientation as strict_tests
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
from diffusion_policy.envs.realword.dual_aubo_real_env_v3 import DualAuboRealEnvV3


class ManualHomeV3Tests(unittest.TestCase):
    def rig(self):
        c, r, loop = strict_tests.StrictOrientationTests().rig(3)
        # 启动种子来自首帧实测关节；保持测试夹具与真实启动路径一致。
        r.get_state.return_value["joint_angles"] = loop.q_seed.copy()
        c.manual_home = True
        reference = c.workspace
        c._prepare_manual_home()
        return c, r, loop, reference

    def test_start_hold_keeps_actual_joints_without_rotation_jump(self):
        c, r, loop, reference = self.rig()
        original = r.get_state.return_value['eef_pose'].copy()
        self.assertTrue(c._manual_home_pending)
        self.assertFalse(c._orientation_locked())
        self.assertTrue(reference.orientation_locked('left'))
        space, target = c._target_command(r, loop, 10.02, .02)
        self.assertEqual(space, 'joint')
        np.testing.assert_array_equal(target, loop.q_seed)
        c._send_command(r, loop, space, target, .02)
        r.servo_cartesian.assert_not_called()
        r.servo_joint.assert_called_once()
        np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], original)

    def home_command(self, c, r, loop):
        target = np.ones(6) * .3
        c.input_queue.get_all.return_value = cartesian_tests.batch(
            dict(cmd=cartesian_tests.Command.HOME.value, target_pose=target))
        c._process_commands(r, loop, 10.01, .02)
        return target

    def test_first_h_uses_gradual_joint_home(self):
        c, r, loop, _ = self.rig()
        target = self.home_command(c, r, loop)
        self.assertIs(type(loop.home_plan), JointHome)
        space, sample = c._target_command(r, loop, 10.02, .02)
        self.assertEqual(space, 'joint')
        self.assertLess(np.max(np.abs(sample - loop.q_seed)), np.max(np.abs(target - loop.q_seed)))
        c._send_command(r, loop, space, sample, .02)
        self.assertEqual(loop.home_plan.elapsed, .02)
        r.servo_cartesian.assert_not_called()
        r.inverse_kinematics.assert_not_called()

    def test_completed_first_home_restores_lock_without_changing_parent(self):
        c, r, loop, reference = self.rig()
        target = self.home_command(c, r, loop)
        loop.home_plan.elapsed = loop.home_plan.duration
        loop.home_plan.settled = 4
        loop.q_seed = target.copy()
        state = r.get_state.return_value
        state['joint_angles'] = target.copy()
        state['eef_pose'][3:] = reference.locked_orientations['left']
        space, sample = c._target_command(r, loop, 10.02, .02)
        self.assertEqual(c.home_state.value, 2)
        self.assertFalse(c._manual_home_pending)
        self.assertIs(c.workspace, reference)
        self.assertTrue(c._orientation_locked())
        c._send_command(r, loop, space, sample, .02)
        # 后续 H 继续使用原固定姿态 XYZ 归位。
        self.home_command(c, r, loop)
        self.assertIsInstance(loop.home_plan, LockedCartesianHome)
        np.testing.assert_array_equal(loop.home_plan.sample(.02)[3:], reference.locked_orientations['left'])

    def test_existing_low_height_recovery_still_runs_before_first_h(self):
        c, r, loop, reference = self.rig()
        r.get_state.return_value['eef_pose'][1] = .551
        original = r.get_state.return_value['eef_pose'].copy()
        space, sample = c._target_command(r, loop, 10.02, .02)
        self.assertEqual(space, 'cartesian')
        self.assertAlmostEqual(sample[1], .55)
        np.testing.assert_array_equal(sample[3:], original[3:])
        c._send_command(r, loop, space, sample, .02)
        self.assertTrue(c._manual_home_pending)
        self.assertTrue(reference.orientation_locked('left'))

    def test_library_default_retains_its_existing_fixed_reference_mode(self):
        c, r, loop = strict_tests.StrictOrientationTests().rig(3)
        c.manual_home = False
        original = c.workspace
        c._prepare_manual_home()
        self.assertIs(c.workspace, original)
        self.assertTrue(c._orientation_locked())
        self.assertFalse(c._manual_home_pending)

    def test_environment_assigns_manual_home_to_both_controllers(self):
        def fake_init(env, *args, **kwargs):
            env.workspace = object()
            env.left_robot = SimpleNamespace()
            env.right_robot = SimpleNamespace()
        with patch.object(DualAuboRealEnv, '__init__', fake_init):
            env = DualAuboRealEnvV3(manual_home=True)
        for robot in (env.left_robot, env.right_robot):
            self.assertTrue(robot.manual_home)
            self.assertIs(robot.workspace, env.workspace)

    def test_minus5_reports_target_and_actual_without_motion_retry(self):
        c, r, loop = cartesian_tests.CartesianV3Tests().rig()
        c.servo_t = .02
        r.servo_cartesian.return_value = -5
        with self.assertRaisesRegex(RuntimeError, 'servoCartesian.*-5.*target_xyz_rpy=.*actual_xyz_rpy=.*delta_orientation_deg='):
            c._send_cartesian(r, loop, np.array([.1, .2, .3, .4, .5, .6]))
        r.servo_cartesian.assert_called_once()
        r.servo_joint.assert_not_called()
        r.inverse_kinematics.assert_not_called()
        r.motion.setServoModeSelect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
