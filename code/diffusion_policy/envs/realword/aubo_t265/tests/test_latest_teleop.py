"""Realtime goals and reliable events, with no robot/network/camera access."""
import multiprocessing as mp
import time
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, call, patch

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.envs.realword.aubo_interpolation_controller_v2 import (
    AuboInterpolationControllerV2, Command, GRIPPER_COMMAND, _LatestPose, _LoopState,
)
from diffusion_policy.envs.realword.aubo_t265.pose_mapper import QuaternionPoseMapper
from diffusion_policy.envs.realword.dual_aubo_real_env_v2 import DualAuboRealEnvV2
from demo_dual_teleop_aubo_v2 import (
    _target_from_anchor, _toggle_output, _advance_left_output, _update_encoder_outputs,
)


def _publish_in_child(slot):
    for i in range(1000):
        slot.put(np.ones(6) * i, 10. + i * .001)


def batch(*commands):
    examples = dict(cmd=0, target_pose=np.zeros(6), duration=.02,
                    target_time=0., gripper=-1.)
    return {key: np.asarray([command.get(key, value) for command in commands])
            for key, value in examples.items()}


class RealtimeTests(unittest.TestCase):
    def rig(self, commands=None):
        controller = object.__new__(AuboInterpolationControllerV2)
        controller.latest_teleop_pose = _LatestPose()
        controller.input_queue = Mock()
        controller.input_queue.get_all.return_value = batch()
        if commands is not None:
            controller.input_queue.get_all.return_value = batch(*commands)
        controller.stop_event = Event()
        controller.home_state = SimpleNamespace(value=0)
        controller.command_timeout = .5
        controller.max_pos_speed, controller.max_rot_speed = .25, .2
        controller.max_joint_step, controller.max_joint_speed = .001, .001
        controller.max_following_error = None
        controller.reject_joint_jumps = True
        controller.robot_ip = 'offline'
        controller.gripper_config = dict(type='standard_digital', mode='independent', pins=[0, 1])
        pose = np.zeros(6)
        loop = _LoopState(PoseTrajectoryInterpolator([10.], [pose]), pose.copy(),
                          np.ones(6) * .1, 10., 10.)
        state = dict(eef_pose=pose.copy(), joint_angles=loop.q_seed.copy(),
                     joint_speeds=np.zeros(6), received=time.time())
        robot = Mock()
        robot.get_state.return_value = state
        robot.inverse_kinematics.return_value = (np.ones(6) * .8, 0)
        return controller, robot, loop

    def process(self, controller, robot, loop, now):
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.time.monotonic', return_value=now), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.time.time', return_value=990. + now):
            return controller._process_commands(robot, loop, now, .02)

    def test_slot_is_shared_and_never_accumulates_old_poses(self):
        slot = _LatestPose()
        child = mp.get_context().Process(target=_publish_in_child, args=(slot,))
        child.start()
        child.join(timeout=10)
        if child.is_alive():
            child.terminate()
            child.join()
            self.fail('Shared pose writer did not finish')
        self.assertEqual(child.exitcode, 0)
        revision, pose, received = slot.get()
        self.assertEqual(revision, 1000)
        np.testing.assert_array_equal(pose, np.ones(6) * 999)
        self.assertAlmostEqual(received, 10.999)

    def test_pose_publication_does_not_enqueue_motion(self):
        controller, _, _ = self.rig()
        with patch.object(AuboInterpolationControllerV2, 'is_ready', new_callable=PropertyMock, return_value=True):
            for i in range(300):
                controller.set_teleop_target(np.ones(6) * i)
            controller.set_gripper(1)
            controller.set_gripper(0)
        self.assertEqual(controller.input_queue.put.call_args_list,
                         [call({'cmd': GRIPPER_COMMAND, 'gripper': 1.}),
                          call({'cmd': GRIPPER_COMMAND, 'gripper': 0.})])
        np.testing.assert_array_equal(controller.latest_teleop_pose.get()[1], np.ones(6) * 299)

    def test_latest_goal_used_without_joint_speed_or_step_limit(self):
        controller, robot, loop = self.rig()
        controller.latest_teleop_pose.put(np.ones(6) * .2, 10.01)
        controller.latest_teleop_pose.put(np.ones(6) * .3, 10.02)
        self.process(controller, robot, loop, 10.05)
        np.testing.assert_array_equal(loop.target_pose, np.ones(6) * .3)
        joints = controller._target_joints(robot, loop, 10.05, .02)
        np.testing.assert_array_equal(robot.inverse_kinematics.call_args.args[1], np.ones(6) * .3)
        np.testing.assert_array_equal(joints, np.ones(6) * .8)
        self.assertEqual(loop.last_command_at, 10.02)

    def test_same_snapshot_does_not_refresh_watchdog(self):
        controller, robot, loop = self.rig()
        controller.latest_teleop_pose.put(np.ones(6), 10.01)
        self.process(controller, robot, loop, 10.02)
        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            self.process(controller, robot, loop, 10.52)

    def test_stale_snapshot_cannot_resume_motion(self):
        controller, robot, loop = self.rig()
        controller.latest_teleop_pose.put(np.ones(6), 9.)
        loop.joint_hold = True
        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            self.process(controller, robot, loop, 10.6)
        self.assertTrue(loop.joint_hold)
        self.assertFalse(loop.teleop_active)

    def test_hold_discards_goal_but_preserves_all_immediate_io(self):
        controller, robot, loop = self.rig([
            dict(cmd=GRIPPER_COMMAND, gripper=1.), dict(cmd=Command.HOLD.value),
            dict(cmd=GRIPPER_COMMAND, gripper=0.)])
        controller.latest_teleop_pose.put(np.ones(6), 10.01)
        self.process(controller, robot, loop, 10.02)
        self.assertTrue(loop.joint_hold)
        self.assertFalse(loop.teleop_active)
        self.assertEqual(list(loop.immediate_gripper_events), [1., 0.])
        controller.input_queue.get_all.return_value = batch()
        self.process(controller, robot, loop, 10.04)
        self.assertTrue(loop.joint_hold)
        controller.latest_teleop_pose.put(np.ones(6) * .5, 10.05)
        self.process(controller, robot, loop, 10.06)
        self.assertFalse(loop.joint_hold)
        np.testing.assert_array_equal(loop.target_pose, np.ones(6) * .5)

    def test_stop_preempts_goal_and_io(self):
        controller, robot, loop = self.rig([
            dict(cmd=GRIPPER_COMMAND, gripper=1.), dict(cmd=Command.STOP.value)])
        controller.latest_teleop_pose.put(np.ones(6), 10.01)
        self.assertFalse(self.process(controller, robot, loop, 10.02))
        self.assertFalse(loop.immediate_gripper_events)
        self.assertFalse(loop.teleop_active)

    def test_home_cancels_pending_realtime_goal(self):
        controller, robot, loop = self.rig([dict(cmd=Command.HOME.value, target_pose=np.ones(6) * .12)])
        controller.latest_teleop_pose.put(np.ones(6), 10.01)
        self.process(controller, robot, loop, 10.02)
        self.assertIsNotNone(loop.home_plan)
        self.assertFalse(loop.teleop_active)
        self.assertEqual(controller.latest_teleop_pose.get()[2], 0.)

    def test_future_waypoint_survives_21ms_queue_delay(self):
        pose = np.zeros(6); pose[0] = .001
        controller, robot, loop = self.rig([dict(cmd=Command.SCHEDULE_WAYPOINT.value,
                                               target_pose=pose, target_time=1000.04)])
        self.process(controller, robot, loop, 10.021)
        np.testing.assert_allclose(loop.interpolator.poses[-1], pose)
        self.assertAlmostEqual(loop.interpolator.times[-1], 10.04)

    def test_replanning_does_not_jump_current_sample_into_future(self):
        pose = np.zeros(6); pose[0] = .006
        controller, robot, loop = self.rig([dict(cmd=Command.SCHEDULE_WAYPOINT.value,
                                               target_pose=pose, target_time=1000.06)])
        initial = np.zeros((2, 6)); initial[1, 0] = .004
        loop.interpolator = PoseTrajectoryInterpolator([10., 10.04], initial)
        loop.last_waypoint_time = 10.04
        expected = loop.interpolator(10.01)
        self.process(controller, robot, loop, 10.01)
        np.testing.assert_allclose(loop.interpolator(10.01), expected, atol=1e-12)

    def test_run_delivers_each_io_event_and_only_latest_goal_before_stop(self):
        controller, robot, _ = self.rig()
        controller.frequency = 50.
        controller.soft_real_time = False
        controller.ready_event = Mock()
        controller.startup_ok = SimpleNamespace(value=False)
        controller._publish_state = Mock()
        robot.start_servo.return_value = robot.get_state.return_value
        robot.servo_joint.return_value = 0
        controller.latest_teleop_pose.put(np.ones(6) * .2, time.monotonic())
        controller.latest_teleop_pose.put(np.ones(6) * .3, time.monotonic())
        controller.input_queue.get_all.side_effect = [batch(
            dict(cmd=GRIPPER_COMMAND, gripper=1.), dict(cmd=GRIPPER_COMMAND, gripper=0.),
            dict(cmd=GRIPPER_COMMAND, gripper=1.)), batch(dict(cmd=Command.STOP.value))]
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.require_pyaubo_sdk'), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.AuboInterface', return_value=robot), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.time.sleep'):
            controller.run()
        self.assertEqual(robot.gripper.command.call_args_list, [call(1.), call(0.), call(1.)])
        robot.servo_joint.assert_called_once()
        np.testing.assert_array_equal(robot.inverse_kinematics.call_args.args[1], np.ones(6) * .3)
        robot.close.assert_called_once()

    def test_stop_event_during_io_prevents_remaining_io_and_servo(self):
        controller, robot, _ = self.rig()
        controller.frequency = 50.
        controller.soft_real_time = False
        controller.ready_event = Mock()
        controller.startup_ok = SimpleNamespace(value=False)
        controller._publish_state = Mock()
        robot.start_servo.return_value = robot.get_state.return_value
        controller.input_queue.get_all.return_value = batch(
            dict(cmd=GRIPPER_COMMAND, gripper=1.), dict(cmd=GRIPPER_COMMAND, gripper=0.))
        controller.latest_teleop_pose.put(np.ones(6) * .2, time.monotonic())
        robot.gripper.command.side_effect = lambda value: controller.stop_event.set()
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.require_pyaubo_sdk'), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.AuboInterface', return_value=robot):
            controller.run()
        robot.gripper.command.assert_called_once_with(1.)
        robot.servo_joint.assert_not_called()
        robot.close.assert_called_once()

    def test_policy_chunk_keeps_ordered_waypoints_and_timed_io(self):
        first = np.zeros(6); first[0] = .001
        second = first * 2
        controller, robot, loop = self.rig([
            dict(cmd=Command.SCHEDULE_WAYPOINT.value, target_pose=first, target_time=1000.04, gripper=1.),
            dict(cmd=Command.SCHEDULE_WAYPOINT.value, target_pose=second, target_time=1000.06, gripper=0.)])
        self.process(controller, robot, loop, 10.01)
        np.testing.assert_allclose(loop.interpolator(10.04), first)
        np.testing.assert_allclose(loop.interpolator(10.06), second)
        np.testing.assert_allclose(list(loop.gripper_events), [[10.04, 1.], [10.06, 0.]])
        self.assertFalse(loop.teleop_active)


class EntryAndEnvironmentTests(unittest.TestCase):
    def rig(self):
        env = object.__new__(DualAuboRealEnvV2)
        env.left_robot, env.right_robot = Mock(), Mock()
        for robot in (env.left_robot, env.right_robot):
            robot.home_state = SimpleNamespace(value=0)
        env.grippers_enabled = True
        env.gripper_dims = dict(left=2, right=1)
        env.gripper_configs = dict(left=dict(mode='independent', pins=[0, 1]),
                                   right=dict(mode='independent', pins=[0]))
        env.action_dim = 15
        env.assert_healthy, env.stop = Mock(), Mock()
        env.workspace = None
        env._recording = True
        env.action_accumulator, env.stage_accumulator = Mock(), Mock()
        return env

    def test_target_records_outputs_but_only_sends_poses(self):
        env = self.rig()
        action = np.concatenate((np.ones(6) * .2, [1., 0.], np.ones(6) * .3, [1.]))
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env_v2.time.time', return_value=1234.):
            env.set_teleop_target(action, stage=2)
        np.testing.assert_array_equal(env.left_robot.set_teleop_target.call_args.args[0], action[:6])
        np.testing.assert_array_equal(env.right_robot.set_teleop_target.call_args.args[0], action[8:14])
        env.left_robot.set_gripper.assert_not_called()
        env.right_robot.set_gripper.assert_not_called()
        np.testing.assert_array_equal(env.action_accumulator.put.call_args.args[1], [1234.])
        np.testing.assert_array_equal(env.stage_accumulator.put.call_args.args[0], [2])

    def test_repeated_key_toggles_are_separate_io_events(self):
        env = self.rig()
        outputs = dict(left=np.zeros(2), right=np.zeros(1))
        _toggle_output(env, outputs, 'left', 0, True)
        _toggle_output(env, outputs, 'left', 0, True)
        self.assertEqual(env.left_robot.set_gripper.call_args_list, [call(1.), call(0.)])
        env.right_robot.set_gripper.assert_not_called()

    def test_encoder_edge_is_an_independent_event(self):
        env = self.rig()
        env.gripper_configs['right']['mode'] = 'binary'
        outputs = dict(left=np.zeros(2), right=np.zeros(1))
        sample = dict(right=SimpleNamespace(encoder_angle=40.))
        toggled = dict(right=False)
        _update_encoder_outputs(env, outputs, sample, dict(right=0.), toggled, 30.)
        _update_encoder_outputs(env, outputs, sample, dict(right=0.), toggled, 30.)
        env.right_robot.set_gripper.assert_called_once_with(1.)

    def test_left_four_step_sequence_keeps_every_transition(self):
        env = self.rig()
        outputs = dict(left=np.zeros(2), right=np.zeros(1))
        for _ in range(4):
            _advance_left_output(env, outputs, True)
        self.assertEqual(env.left_robot.set_gripper.call_args_list,
                         [call(2.), call(3.), call(1.), call(0.)])
        np.testing.assert_array_equal(outputs['left'], np.zeros(2))
        env.right_robot.set_gripper.assert_not_called()

    def test_realtime_mapper_ignores_configured_speed_limits(self):
        mapper = QuaternionPoseMapper(np.eye(3), np.zeros(6), scale=1., speed=.001,
                                      angular_speed=.001, tau=0., radius=None, angle=None, deadzone=0.)
        mapper.anchor(np.zeros(3), Rotation.identity(), np.zeros(6))
        sample = SimpleNamespace(position=np.array([.1, 0., 0.]), orientation=Rotation.identity())
        pose = _target_from_anchor(mapper, sample, .02)
        self.assertAlmostEqual(pose[0], .1)

    def test_invalid_right_output_blocks_both_position_submissions(self):
        env = self.rig()
        action = np.zeros(15); action[-1] = .5
        with self.assertRaises(ValueError):
            env.set_teleop_target(action)
        env.left_robot.set_teleop_target.assert_not_called()
        env.right_robot.set_teleop_target.assert_not_called()


if __name__ == '__main__':
    unittest.main()
