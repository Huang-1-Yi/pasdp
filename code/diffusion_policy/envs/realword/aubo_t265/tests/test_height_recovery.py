"""Actual low TCP recovery: no SDK connection or physical motion."""
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diffusion_policy.envs.realword import aubo_interpolation_controller_v2 as v2
from diffusion_policy.envs.realword import aubo_interpolation_controller_v3 as v3
from diffusion_policy.envs.realword.aubo_homing import JointHome
from diffusion_policy.envs.realword.aubo_t265.config import load_config
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_t265.tests import test_latest_teleop as teleop_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_servo_cartesian_v3 as cartesian_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_integration as integration_tests
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv


class HeightRecoveryTests(unittest.TestCase):
    def workspace(self):
        config = load_config()
        w = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        for side in ('left', 'right'):
            w.update_gravity(side, [0., 9.81, 0.])
        return w

    def rig(self, version):
        c, r, loop = (teleop_tests.RealtimeTests().rig() if version == 2
                      else cartesian_tests.CartesianV3Tests().rig())
        c.workspace, c.workspace_side = self.workspace(), 'left'
        pose = np.array([.12, .551, -.23, .1, .2, .3])
        r.get_state.return_value['eef_pose'] = pose.copy()
        r.get_state.return_value['received'] = time.time()
        r.servo_joint.return_value = r.servo_cartesian.return_value = 0
        r.forward_kinematics.side_effect = lambda _: (loop.target_pose.copy(), 0)
        loop.joint_hold = True
        return c, r, loop

    def test_tilted_recovery_only_changes_gravity_height(self):
        w = self.workspace()
        w.update_gravity('left', [4., 8., 4.])
        up = w.up['left']
        horizontal = np.cross(up, [0., 0., 1.]) * .2
        pose = np.r_[horizontal + up * -.551, [.1, .2, .3]]
        original = pose.copy()
        recovery = w.recovery_target('left', pose)
        np.testing.assert_allclose(recovery[:3] - pose[:3], up * .001, atol=1e-12)
        self.assertAlmostEqual(w.height('left', recovery), -.55)
        np.testing.assert_array_equal(recovery[3:], pose[3:])
        np.testing.assert_array_equal(pose, original)
        w.validate_feedback('left', pose)
        with self.assertRaises(RuntimeError):
            w.validate_target('left', pose)

    def test_v2_recovers_paused_actual_pose_using_ik_and_servo_joint(self):
        c, r, loop = self.rig(2)
        original = r.get_state.return_value['eef_pose'].copy()
        joints = c._target_joints(r, loop, 10.02, .02)
        c._send_joints(r, loop, joints, .02)
        np.testing.assert_allclose(r.inverse_kinematics.call_args.args[1], [.12, .55, -.23, .1, .2, .3])
        r.servo_joint.assert_called_once()
        np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], original)

    def test_v3_recovers_paused_actual_pose_using_native_tcp_servo(self):
        c, r, loop = self.rig(3)
        original = r.get_state.return_value['eef_pose'].copy()
        space, pose = c._target_command(r, loop, 10.02, .02)
        c._send_command(r, loop, space, pose, .02)
        self.assertEqual(space, 'cartesian')
        np.testing.assert_allclose(r.servo_cartesian.call_args.args[0], [.12, .55, -.23, .1, .2, .3])
        r.inverse_kinematics.assert_not_called()
        r.forward_kinematics.assert_not_called()
        np.testing.assert_array_equal(r.get_state.return_value['eef_pose'], original)

    def test_recovery_goal_is_latched_and_takes_priority_over_human_goal(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            loop.teleop_active = True
            loop.target_pose = np.zeros(6)
            sample = c._target_joints if version == 2 else c._target_command
            sample(r, loop, 10.02, .02)
            first_goal = loop.height_recovery_pose.copy()
            r.get_state.return_value['eef_pose'][0] += .003
            loop.target_pose = np.ones(6)
            sample(r, loop, 10.04, .02)
            np.testing.assert_array_equal(loop.target_pose, first_goal)

    def test_reaching_boundary_releases_recovery_and_keeps_pause(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            if version == 2:
                joints = c._target_joints(r, loop, 10.02, .02)
                c._send_joints(r, loop, joints, .02)
            else:
                space, pose = c._target_command(r, loop, 10.02, .02)
                c._send_command(r, loop, space, pose, .02)
            r.get_state.return_value['eef_pose'][1] = .55
            if version == 2:
                joints = c._target_joints(r, loop, 10.04, .02)
                np.testing.assert_array_equal(joints, loop.q_seed)
            else:
                space, pose = c._target_command(r, loop, 10.04, .02)
                self.assertEqual(space, 'cartesian')
                self.assertAlmostEqual(pose[1], .55)
            self.assertIsNone(loop.height_recovery_pose)
            self.assertTrue(loop.joint_hold)

    def test_home_freezes_during_lift_and_restarts_from_actual_joints(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            loop.home_plan = JointHome(loop.q_seed, np.ones(6) * .3)
            previous_plan = loop.home_plan
            c.home_state.value = 1
            if version == 2:
                joints = c._target_joints(r, loop, 10.02, .02)
                c._send_joints(r, loop, joints, .02)
            else:
                space, pose = c._target_command(r, loop, 10.02, .02)
                c._send_command(r, loop, space, pose, .02)
            self.assertEqual(previous_plan.elapsed, 0.)
            r.get_state.return_value['eef_pose'][1] = .55
            r.get_state.return_value['joint_angles'] = np.ones(6) * .15
            sample = c._target_joints if version == 2 else c._target_command
            sample(r, loop, 10.04, .02)
            self.assertIsNot(loop.home_plan, previous_plan)
            np.testing.assert_array_equal(loop.home_plan.start, np.ones(6) * .15)
            np.testing.assert_array_equal(loop.home_plan.target, np.ones(6) * .3)

    def test_ik_failure_stops_recovery_without_sending_old_low_hold(self):
        c, r, loop = self.rig(2)
        r.inverse_kinematics.return_value = (np.zeros(6), -1)
        with self.assertRaisesRegex(RuntimeError, 'Height recovery inverseKinematics failed'):
            c._target_joints(r, loop, 10.02, .02)
        r.servo_joint.assert_not_called()

    def test_stop_preempts_recovery_in_both_controller_loops(self):
        for version, module in ((2, v2), (3, v3)):
            c, r, _ = self.rig(version)
            c.frequency, c.soft_real_time = 50., False
            c.startup_ok = SimpleNamespace(value=False)
            c.ready_event = Mock()
            c._publish_state = Mock()
            r.start_servo.return_value = r.get_state.return_value
            c.input_queue.get_all.return_value = teleop_tests.batch(dict(cmd=v2.Command.STOP.value))
            with patch.object(module, 'require_pyaubo_sdk'), patch.object(module, 'AuboInterface', return_value=r):
                c.run()
            r.servo_joint.assert_not_called()
            r.servo_cartesian.assert_not_called()
            r.inverse_kinematics.assert_not_called()

    def test_stale_feedback_still_blocks_height_recovery(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            r.get_state.return_value['received'] = time.time() - 1.
            sample = c._target_joints if version == 2 else c._target_command
            with self.assertRaisesRegex(RuntimeError, 'RTDE'):
                sample(r, loop, 10.02, .02)
            r.servo_joint.assert_not_called()
            r.servo_cartesian.assert_not_called()

    def test_io_events_keep_order_during_recovery(self):
        for version, module in ((2, v2), (3, v3)):
            c, r, _ = self.rig(version)
            c.frequency, c.soft_real_time = 50., False
            c.startup_ok = SimpleNamespace(value=False)
            c.ready_event = Mock()
            c._publish_state = Mock()
            r.start_servo.return_value = r.get_state.return_value
            r.forward_kinematics.side_effect = None
            r.forward_kinematics.return_value = ([.12, .55, -.23, .1, .2, .3], 0)
            c.input_queue.get_all.side_effect = [teleop_tests.batch(
                dict(cmd=v2.GRIPPER_COMMAND, gripper=1.), dict(cmd=v2.GRIPPER_COMMAND, gripper=0.)),
                teleop_tests.batch(dict(cmd=v2.Command.STOP.value))]
            with patch.object(module, 'require_pyaubo_sdk'), patch.object(module, 'AuboInterface', return_value=r), \
                    patch.object(module.time, 'sleep'):
                c.run()
            self.assertEqual([entry.args[0] for entry in r.gripper.command.call_args_list], [1., 0.])
            servo = r.servo_joint if version == 2 else r.servo_cartesian
            servo.assert_called_once()

    def test_recovery_does_not_refresh_expired_command_watchdog(self):
        for version in (2, 3):
            c, r, loop = self.rig(version)
            c.input_queue.get_all.return_value = teleop_tests.batch()
            with self.assertRaisesRegex(RuntimeError, 'watchdog'):
                c._process_commands(r, loop, 10.6, .02)
            r.servo_joint.assert_not_called()
            r.servo_cartesian.assert_not_called()

    def test_environment_preserves_low_actual_feedback_for_recovery_controllers(self):
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.AuboInterpolationController', integration_tests.FakeRobot):
            env = DualAuboRealEnv('left', 'right', shm_manager=object(), workspace=self.workspace())
        env.start()
        self.addCleanup(env.stop)
        for robot in (env.left_robot, env.right_robot):
            robot.supports_height_recovery = True
            robot.pose[1] = .551
        state = env.get_state()
        self.assertEqual(state['left']['ActualTCPPose'][1], .551)
        self.assertEqual(state['right']['ActualTCPPose'][1], .551)


if __name__ == '__main__':
    unittest.main()
