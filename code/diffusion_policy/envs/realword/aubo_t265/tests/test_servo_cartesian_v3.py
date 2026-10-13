"""Offline v3 SDK boundary, mode transitions, workspace and event tests."""
import time
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.envs.realword import aubo_interpolation_controller_v3 as module
from diffusion_policy.envs.realword.aubo_interpolation_controller_v3 import (
    AuboInterface, AuboInterpolationControllerV3, Command, GRIPPER_COMMAND,
    _LatestPose, _LoopState,
)
from diffusion_policy.envs.realword.aubo_t265.config import load_config
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.dual_aubo_real_env_v3 import DualAuboRealEnvV3


def batch(*commands):
    defaults = dict(cmd=0, target_pose=np.zeros(6), duration=.02,
                    target_time=0., gripper=-1.)
    return {key: np.asarray([command.get(key, value) for command in commands])
            for key, value in defaults.items()}


class CartesianV3Tests(unittest.TestCase):
    def rig(self):
        c = object.__new__(AuboInterpolationControllerV3)
        c.latest_teleop_pose = _LatestPose()
        c.input_queue = Mock()
        c.input_queue.get_all.return_value = batch()
        c.stop_event = Event()
        c.home_state = SimpleNamespace(value=0)
        c.command_timeout = .5
        c.max_pos_speed, c.max_rot_speed = .25, .2
        c.max_following_error = None
        c.robot_ip = 'offline'
        c.workspace = None
        pose = np.zeros(6)
        loop = _LoopState(PoseTrajectoryInterpolator([10.], [pose]), pose.copy(),
                          np.ones(6) * .1, 10., 10., accepted_pose=pose.copy())
        r = Mock()
        r.get_state.return_value = dict(eef_pose=pose.copy(), joint_angles=np.ones(6)*.4,
                                       joint_speeds=np.zeros(6), received=time.time())
        r.servo_cartesian.return_value = r.servo_joint.return_value = 0
        r.sdk.returnValue2Str.side_effect = lambda ret: str(ret)
        return c, r, loop

    def process(self, c, r, loop, now=10.02):
        with patch.object(module.time, 'monotonic', return_value=now), \
                patch.object(module.time, 'time', return_value=1000.):
            r.get_state.return_value['received'] = 1000.
            return c._process_commands(r, loop, now, .02)

    def test_sdk_boundary_converts_rotvec_and_preserves_parameters(self):
        owner = SimpleNamespace(servo_a=3.5, servo_v=2., servo_t=.02,
                                lookahead_time=.05, gain=120)
        interface = AuboInterface(owner, Mock())
        interface.motion = Mock()
        interface.motion.servoCartesian.return_value = 0
        rotation = Rotation.from_euler('xyz', [.4, -.3, .2])
        pose = np.r_[[.2, -.1, .3], rotation.as_rotvec()]
        self.assertEqual(interface.servo_cartesian(pose), 0)
        args = interface.motion.servoCartesian.call_args.args
        np.testing.assert_allclose(args[0][:3], pose[:3])
        np.testing.assert_allclose(Rotation.from_euler('xyz', args[0][3:]).as_matrix(),
                                   rotation.as_matrix())
        self.assertEqual(args[1:], (3.5, 2., .02, .05, 120))
        interface.motion.servoJoint.assert_not_called()

    def test_invalid_pose_never_reaches_sdk(self):
        interface = AuboInterface(SimpleNamespace(), Mock())
        interface.motion = Mock()
        for pose in ([0.]*5, [0., 0., 0., 0., 0., np.nan]):
            with self.assertRaises(ValueError):
                interface.servo_cartesian(pose)
        interface.motion.servoCartesian.assert_not_called()

    def test_latest_goal_uses_tcp_without_explicit_ik_or_fk(self):
        c, r, loop = self.rig()
        c.latest_teleop_pose.put(np.ones(6)*.2, 10.01)
        self.process(c, r, loop)
        r.get_state.return_value['received'] = time.time()
        space, pose = c._target_command(r, loop, 10.02, .02)
        c._send_command(r, loop, space, pose, .02)
        self.assertEqual(space, 'cartesian')
        np.testing.assert_array_equal(r.servo_cartesian.call_args.args[0], np.ones(6)*.2)
        r.inverse_kinematics.assert_not_called()
        r.forward_kinematics.assert_not_called()
        r.servo_joint.assert_not_called()
        np.testing.assert_array_equal(loop.accepted_pose, pose)

    def test_policy_interpolation_uses_tcp_too(self):
        c, r, loop = self.rig()
        poses = np.zeros((2, 6)); poses[1, 0] = .02
        loop.interpolator = PoseTrajectoryInterpolator([10., 10.2], poses)
        space, pose = c._target_command(r, loop, 10.1, .02)
        self.assertEqual(space, 'cartesian')
        self.assertAlmostEqual(pose[0], .01)
        r.inverse_kinematics.assert_not_called()

    def test_rejected_tcp_is_not_used_for_hold(self):
        c, r, loop = self.rig()
        accepted = np.ones(6)*.05
        c._send_cartesian(r, loop, accepted)
        r.servo_cartesian.return_value = 2
        c._send_cartesian(r, loop, np.ones(6)*.4)
        c.latest_teleop_pose.put(np.ones(6)*.5, 10.01)
        c.input_queue.get_all.return_value = batch(dict(cmd=Command.HOLD.value))
        self.process(c, r, loop)
        r.get_state.return_value['received'] = time.time()
        space, pose = c._target_command(r, loop, 10.02, .02)
        self.assertEqual(space, 'cartesian')
        np.testing.assert_array_equal(pose, accepted)
        self.assertFalse(loop.teleop_active)

    def test_startup_hold_still_uses_original_joint_hold(self):
        c, r, loop = self.rig()
        loop.joint_hold = True
        space, q = c._target_command(r, loop, 10.02, .02)
        c._send_command(r, loop, space, q, .02)
        self.assertEqual(space, 'joint')
        r.servo_joint.assert_called_once()
        r.servo_cartesian.assert_not_called()

    def test_cartesian_to_home_starts_from_fresh_actual_joints(self):
        c, r, loop = self.rig()
        loop.last_space = 'cartesian'
        c.input_queue.get_all.return_value = batch(dict(cmd=Command.HOME.value,
                                                       target_pose=np.ones(6)*.5))
        self.process(c, r, loop)
        np.testing.assert_array_equal(loop.home_plan.start, np.ones(6)*.4)
        loop.home_plan.deadline = time.monotonic() + 30.
        r.get_state.return_value['received'] = time.time()
        space, q = c._target_command(r, loop, 10.02, .02)
        r.servo_joint.return_value = 2
        c._send_command(r, loop, space, q, .02)
        self.assertEqual(loop.home_plan.elapsed, 0.)
        r.servo_joint.return_value = 0
        c._send_command(r, loop, space, q, .02)
        self.assertEqual(loop.home_plan.elapsed, .02)
        self.assertEqual(loop.last_space, 'joint')
        r.servo_cartesian.assert_not_called()

    def test_home_release_accepts_new_cartesian_goal(self):
        c, r, loop = self.rig()
        loop.home_plan = Mock()
        c.home_state.value = 2
        c.latest_teleop_pose.put(np.ones(6)*.3, 10.01)
        self.process(c, r, loop)
        self.assertIsNone(loop.home_plan)
        self.assertEqual(c.home_state.value, 0)
        r.get_state.return_value['received'] = time.time()
        self.assertEqual(c._target_command(r, loop, 10.02, .02)[0], 'cartesian')

    def test_internal_ik_error_is_reported_and_not_accepted(self):
        c, r, loop = self.rig()
        r.servo_cartesian.return_value = -23
        with self.assertRaisesRegex(RuntimeError, 'servoCartesian.*-23'):
            c._send_cartesian(r, loop, np.ones(6))
        np.testing.assert_array_equal(loop.accepted_pose, np.zeros(6))
        self.assertEqual(loop.last_space, 'joint')

    def test_not_in_servo_recovery_has_a_limit(self):
        c, r, loop = self.rig()
        r.servo_cartesian.return_value = -13
        with patch.object(module.time, 'sleep'):
            for _ in range(3):
                c._send_cartesian(r, loop, np.ones(6))
            with self.assertRaisesRegex(RuntimeError, 'servoCartesian.*-13'):
                c._send_cartesian(r, loop, np.ones(6))
        self.assertEqual(r.motion.setServoModeSelect.call_count, 3)

    def test_stop_prevents_both_servo_modes(self):
        c, r, loop = self.rig()
        c.stop_event.set()
        c._send_cartesian(r, loop, np.ones(6))
        c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_cartesian.assert_not_called()
        r.servo_joint.assert_not_called()

    def test_io_events_survive_hold_and_keep_order(self):
        c, r, loop = self.rig()
        c.input_queue.get_all.return_value = batch(
            dict(cmd=GRIPPER_COMMAND, gripper=1.), dict(cmd=Command.HOLD.value),
            dict(cmd=GRIPPER_COMMAND, gripper=0.))
        self.process(c, r, loop)
        self.assertEqual(list(loop.immediate_gripper_events), [1., 0.])

    def test_stale_rtde_or_goal_stops_control(self):
        c, r, loop = self.rig()
        r.get_state.return_value['received'] = time.time()-1.
        with self.assertRaisesRegex(RuntimeError, 'RTDE'):
            c._target_command(r, loop, 10.02, .02)
        c.latest_teleop_pose.put(np.ones(6), 9.)
        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            self.process(c, r, loop, 10.6)

    def test_tilted_height_limits_goal_and_actual_feedback(self):
        c, r, loop = self.rig()
        config = load_config()
        c.workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        c.workspace_side = 'left'
        c.workspace.update_gravity('left', [0., 9.81, 0.])
        loop.teleop_active = True
        loop.target_pose = np.array([.1, .8, -.3, 0., 0., 0.])
        space, pose = c._target_command(r, loop, 10.02, .02)
        self.assertAlmostEqual(pose[1], .55)
        c._send_command(r, loop, space, pose, .02)
        r.forward_kinematics.assert_not_called()
        r.get_state.return_value['eef_pose'][1] = .551
        space, recovery = c._target_command(r, loop, 10.04, .02)
        self.assertEqual(space, 'cartesian')
        self.assertAlmostEqual(recovery[1], .55)
        self.assertAlmostEqual(r.get_state.return_value['eef_pose'][1], .551)
        r.servo_cartesian.reset_mock()
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c._send_cartesian(r, loop, [0., .551, 0., 0., 0., 0.])
        r.servo_cartesian.assert_not_called()

    def test_controller_loop_dispatches_tcp_and_cleans_up(self):
        c, r, loop = self.rig()
        c.frequency, c.soft_real_time = 50., False
        c.startup_ok = SimpleNamespace(value=False)
        c.ready_event = Event()
        c.ring_buffer = Mock()
        r.start_servo.return_value = r.get_state.return_value
        r.get_state.return_value['eef_speed'] = np.zeros(6)
        r.gripper.state, r.gripper.enabled = -1., False
        c.latest_teleop_pose.put(np.ones(6)*.2, time.monotonic())
        def send(pose):
            c.stop_event.set()
            return 0
        r.servo_cartesian.side_effect = send
        with patch.object(module, 'require_pyaubo_sdk', return_value=Mock()), \
                patch.object(module, 'AuboInterface', return_value=r), \
                patch.object(module.time, 'sleep'):
            c.run()
        self.assertTrue(c.startup_ok.value)
        r.servo_cartesian.assert_called_once()
        r.inverse_kinematics.assert_not_called()
        r.servo_joint.assert_not_called()
        r.close.assert_called_once()

    def test_environment_uses_its_own_v3_factory(self):
        from diffusion_policy.envs.realword import dual_aubo_real_env_v3 as env_module
        observed = {}
        def initialize(env, *args, **kwargs):
            observed.update(kwargs)
            env.workspace = None
            env.left_robot, env.right_robot = Mock(), Mock()
        with patch.object(env_module._DualAuboRealEnv, '__init__', initialize):
            env = DualAuboRealEnvV3()
        self.assertIs(observed['controller_class'], AuboInterpolationControllerV3)
        self.assertEqual(env.left_robot.workspace_side, 'left')
        with self.assertRaisesRegex(ValueError, 'own v3 controller'):
            DualAuboRealEnvV3(controller_class=Mock())

    def test_minus_550_boundary_is_inclusive_without_margin(self):
        c, r, loop = self.rig()
        config = load_config()
        c.workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        c.workspace_side = 'left'
        c.workspace.update_gravity('left', [0., 9.81, 0.])
        self.assertEqual(c.workspace.target_margin, 0.)
        for depth in (.550, .551, .600):
            requested = np.array([.1, depth, -.3, .1, .2, .3])
            limited = c.workspace.limit_target('left', requested)
            self.assertAlmostEqual(limited[1], .550)
            self.assertAlmostEqual(c.workspace.height('left', limited), -.550)
            np.testing.assert_array_equal(limited[[0, 2, 3, 4, 5]], requested[[0, 2, 3, 4, 5]])
            c.workspace.validate('left', limited)
            c.workspace.validate_target('left', limited)
        r.get_state.return_value['eef_pose'][1] = .54984
        loop.target_pose = r.get_state.return_value['eef_pose'].copy()
        loop.interpolator = PoseTrajectoryInterpolator([10.], [loop.target_pose])
        space, pose = c._target_command(r, loop, 10.02, .02)
        c._send_command(r, loop, space, pose, .02)
        r.servo_cartesian.assert_called_once()
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c.workspace.validate('left', [0., .551, 0., 0., 0., 0.])

    def test_home_fk_height_and_stop_checks_remain(self):
        c, r, loop = self.rig()
        config = load_config()
        c.workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        c.workspace_side = 'left'
        c.workspace.update_gravity('left', [0., 9.81, 0.])
        r.forward_kinematics.return_value = ([0., .551, 0., 0., 0., 0.], 0)
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_joint.assert_not_called()
        def fk(q):
            c.stop_event.set()
            return np.zeros(6), 0
        r.forward_kinematics.side_effect = fk
        c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_joint.assert_not_called()


if __name__ == '__main__':
    unittest.main()
