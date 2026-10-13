"""严格姿态锁定离线回归：SDK 使用替身，不连接机械臂。"""
import unittest
from unittest.mock import patch
import numpy as np
from diffusion_policy.envs.realword.aubo_homing import LockedCartesianHome
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_t265.tests import test_height_recovery as recovery_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_latest_teleop as realtime_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_servo_cartesian_v3 as cartesian_tests
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
from diffusion_policy.envs.realword.dual_aubo_real_env_v2 import DualAuboRealEnvV2
from diffusion_policy.envs.realword.dual_aubo_real_env_v3 import DualAuboRealEnvV3


class StrictOrientationTests(unittest.TestCase):
    locked = np.array([.2, -.3, .4])

    def rig(self, version, low=False):
        c, r, loop = (realtime_tests.RealtimeTests().rig() if version == 2
                      else cartesian_tests.CartesianV3Tests().rig())
        c.workspace = recovery_tests.HeightRecoveryTests().workspace()
        c.workspace_side = 'left'
        c.workspace.set_locked_orientation('left', self.locked)
        state = r.get_state.return_value
        state['eef_pose'] = np.array([.1, .551 if low else .2, -.1, .7, .8, .9])
        state['eef_speed'] = np.zeros(6)
        loop.target_pose = state['eef_pose'].copy()
        loop.joint_hold = True
        r.forward_kinematics.side_effect = lambda _: (loop.target_pose.copy(), 0)
        r.servo_joint.return_value = r.servo_cartesian.return_value = 0
        return c, r, loop

    def sample(self, version, c, r, loop):
        if version == 2:
            return 'joint', c._target_joints(r, loop, 10.02, .02)
        return c._target_command(r, loop, 10.02, .02)

    def send(self, version, c, r, loop, space, target):
        if version == 2:
            c._send_joints(r, loop, target, .02)
        else:
            c._send_command(r, loop, space, target, .02)

    def test_tilted_height_then_rotation_keeps_raw_feedback(self):
        w = recovery_tests.HeightRecoveryTests().workspace()
        w.update_gravity('left', [4., 8., 4.])
        w.set_locked_orientation('left', self.locked)
        up = w.up['left']
        pose = np.r_[np.cross(up, [0., 0., 1.]) * .2 + up * -.551, [.7, .8, .9]]
        original = pose.copy()
        goal = w.recovery_target('left', pose)
        np.testing.assert_allclose(goal[:3] - pose[:3], .001 * up, atol=1e-12)
        np.testing.assert_array_equal(goal[3:], self.locked)
        np.testing.assert_array_equal(pose, original)
        self.assertAlmostEqual(w.height('left', goal), -.55)
        w.validate_feedback('left', pose)
        with self.assertRaisesRegex(RuntimeError, 'locked TCP orientation'):
            w.validate_target('left', np.r_[goal[:3], pose[3:]])

    def test_lock_independent_of_height_switch(self):
        w = DualWorkspace([], {'enabled': False})
        w.set_locked_orientation('left', self.locked)
        source = np.ones(6)
        target = w.limit_target('left', source)
        np.testing.assert_array_equal(target[:3], source[:3])
        np.testing.assert_array_equal(target[3:], self.locked)
        np.testing.assert_array_equal(source, np.ones(6))

    def test_paused_startup_enforces_lock(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            space, target = self.sample(version, c, r, loop)
            self.send(version, c, r, loop, space, target)
            np.testing.assert_array_equal(loop.target_pose[3:], self.locked)
            if version == 2:
                np.testing.assert_array_equal(r.inverse_kinematics.call_args.args[1][3:], self.locked)
                r.forward_kinematics.assert_called_once()
                r.servo_joint.assert_called_once()
            else:
                self.assertEqual(space, 'cartesian')
                np.testing.assert_array_equal(r.servo_cartesian.call_args.args[0][3:], self.locked)
                r.inverse_kinematics.assert_not_called()

    def test_recovery_uses_home_rotation_without_editing_feedback(self):
        for version in (2, 3):
            c, r, loop = self.rig(version, low=True)
            original = r.get_state.return_value['eef_pose'].copy()
            space, target = self.sample(version, c, r, loop)
            self.send(version, c, r, loop, space, target)
            np.testing.assert_array_equal(loop.height_recovery_pose[3:], self.locked)
            self.assertAlmostEqual(loop.target_pose[1], .55)
            np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], original)

    def test_latest_and_interpolated_targets_receive_final_lock(self):
        from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
        for version in (2, 3):
            for active in (False, True):
                c, r, loop = self.rig(version)
                loop.joint_hold = False
                loop.teleop_active = active
                loop.target_pose = np.array([.12, .551, -.2, .9, .8, .7])
                loop.interpolator = PoseTrajectoryInterpolator([10.], [loop.target_pose])
                self.sample(version, c, r, loop)
                np.testing.assert_array_equal(loop.target_pose[3:], self.locked)
                self.assertAlmostEqual(loop.target_pose[1], .55)

    def test_ik_failure_cannot_fallback_to_tilted_joints(self):
        c, r, loop = self.rig(2)
        r.inverse_kinematics.return_value = (loop.q_seed.copy(), -1)
        with self.assertRaisesRegex(RuntimeError, 'Locked orientation inverseKinematics'):
            self.sample(2, c, r, loop)
        r.servo_joint.assert_not_called()

    def test_wrong_fk_rotation_rejected_before_servo(self):
        c, r, loop = self.rig(2)
        space, target = self.sample(2, c, r, loop)
        r.forward_kinematics.side_effect = None
        r.forward_kinematics.return_value = (np.zeros(6), 0)
        with self.assertRaisesRegex(RuntimeError, 'locked TCP orientation'):
            self.send(2, c, r, loop, space, target)
        r.servo_joint.assert_not_called()

    def test_v3_sdk_boundary_finalizes_height_then_rotation(self):
        c, r, loop = self.rig(3)
        source = np.array([.12, .551, -.2, .9, .8, .7])
        c._send_cartesian(r, loop, source)
        goal = r.servo_cartesian.call_args.args[0]
        self.assertAlmostEqual(goal[1], .55)
        np.testing.assert_array_equal(goal[3:], self.locked)
        np.testing.assert_array_equal(source[3:], [.9, .8, .7])

    def test_home_xyz_advances_only_after_sdk_success(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            r.forward_kinematics.side_effect = None
            r.forward_kinematics.return_value = (np.array([.2, .3, -.2, .7, .8, .9]), 0)
            c.input_queue.get_all.return_value = realtime_tests.batch(
                dict(cmd=realtime_tests.Command.HOME.value, target_pose=np.ones(6) * .3))
            c._process_commands(r, loop, 10.01, .02)
            self.assertIsInstance(loop.home_plan, LockedCartesianHome)
            np.testing.assert_array_equal(loop.home_plan.sample(.02)[3:], self.locked)
            r.forward_kinematics.side_effect = lambda _: (loop.target_pose.copy(), 0)
            r.servo_joint.return_value = r.servo_cartesian.return_value = 2
            space, target = self.sample(version, c, r, loop)
            self.send(version, c, r, loop, space, target)
            self.assertEqual(loop.home_plan.elapsed, 0.)
            r.servo_joint.return_value = r.servo_cartesian.return_value = 0
            self.send(version, c, r, loop, space, target)
            self.assertEqual(loop.home_plan.elapsed, .02)
            np.testing.assert_array_equal(loop.target_pose[3:], self.locked)
            if version == 3:
                self.assertEqual(space, 'cartesian')

    def test_recovery_preserves_locked_home_class_and_endpoint(self):
        for version in (2, 3):
            c, r, loop = self.rig(version, low=True)
            goal = np.r_[[.2, .3, -.2], self.locked]
            original = LockedCartesianHome(loop.q_seed, np.ones(6) * .3,
                                           r.get_state.return_value['eef_pose'], goal)
            loop.home_plan = original
            c.home_state.value = 1
            space, target = self.sample(version, c, r, loop)
            self.send(version, c, r, loop, space, target)
            self.assertEqual(original.elapsed, 0.)
            r.get_state.return_value['eef_pose'][1] = .55
            self.sample(version, c, r, loop)
            self.assertIsInstance(loop.home_plan, LockedCartesianHome)
            self.assertIsNot(loop.home_plan, original)
            np.testing.assert_array_equal(loop.home_plan.target_pose, goal)

    def test_home_rotation_fixed_and_arrival_samples_fresh(self):
        goal = np.r_[[.2, .3, .1], self.locked]
        plan = LockedCartesianHome(np.zeros(6), np.ones(6) * .3, np.ones(6), goal)
        for fraction in (0., .2, .5, .9, 1.):
            plan.elapsed = plan.duration * fraction
            np.testing.assert_array_equal(plan.sample(.02)[3:], self.locked)
        for _ in range(10):
            self.assertFalse(plan.check_pose(goal, np.zeros(6), 1))
        self.assertEqual(plan.settled, 1)
        for received in (2, 3, 4):
            self.assertFalse(plan.check_pose(goal, np.zeros(6), received))
        self.assertTrue(plan.check_pose(goal, np.zeros(6), 5))

    def test_policy_copy_preserves_io_timestamps_and_stages(self):
        for cls in (DualAuboRealEnvV2, DualAuboRealEnvV3):
            env = object.__new__(cls)
            env.workspace = recovery_tests.HeightRecoveryTests().workspace()
            env.workspace.set_locked_orientation('left', self.locked)
            env.workspace.set_locked_orientation('right', -self.locked)
            env.gripper_dims = {'left': 2, 'right': 1}
            env.action_dim = 15
            actions = np.array([[.1, .551, .2, .9, .8, .7, 1, 0, .2, .552, .1, .9, .8, .7, 1.]])
            original = actions.copy()
            times, stages = np.array([123.]), np.array([2])
            with patch.object(DualAuboRealEnv, 'exec_actions') as parent:
                env.exec_actions(actions, times, stages)
                output, sent_times, sent_stages = parent.call_args.args
            np.testing.assert_array_equal(actions, original)
            np.testing.assert_array_equal(output[0, 3:6], self.locked)
            np.testing.assert_array_equal(output[0, 11:14], -self.locked)
            np.testing.assert_array_equal(output[0, [6, 7, 14]], [1, 0, 1])
            np.testing.assert_allclose(output[0, [1, 9]], [.55, .55])
            self.assertIs(sent_times, times)
            self.assertIs(sent_stages, stages)


if __name__ == '__main__':
    unittest.main()
