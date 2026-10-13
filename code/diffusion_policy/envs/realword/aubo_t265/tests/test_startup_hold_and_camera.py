"""Regressions for paused startup near the floor and short camera histories."""
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diffusion_policy.envs.realword.aubo_interpolation_controller_v2 import Command
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
from diffusion_policy.envs.realword.aubo_t265.tests import test_gravity_height as height_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_latest_teleop as teleop_tests
from diffusion_policy.envs.realword.aubo_t265.tests import test_integration as integration_tests


class InitialHoldTests(unittest.TestCase):
    def rig(self):
        controller, robot, loop = height_tests.HeightTests().rig()
        pose = np.array([0., .54984, 0., 0., 0., 0.])
        robot.get_state.return_value['eef_pose'] = pose
        robot.forward_kinematics.return_value = (pose.copy(), 0)
        robot.servo_joint.return_value = 0
        loop.target_pose = pose.copy()
        loop.joint_hold = True
        return controller, robot, loop

    def test_startup_holds_without_ik_inside_physical_boundary(self):
        c, robot, _ = self.rig()
        c.frequency = 50.
        c.soft_real_time = False
        c.startup_ok = SimpleNamespace(value=False)
        c.ready_event = Mock()
        c._publish_state = Mock()
        robot.start_servo.return_value = robot.get_state.return_value
        c.input_queue.get_all.side_effect = [teleop_tests.batch(
            dict(cmd=Command.HEARTBEAT.value)), teleop_tests.batch(dict(cmd=Command.STOP.value))]
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.require_pyaubo_sdk'), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.AuboInterface', return_value=robot), \
                patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.time.sleep'):
            c.run()
        self.assertTrue(c.startup_ok.value)
        robot.inverse_kinematics.assert_not_called()
        robot.servo_joint.assert_called_once()
        np.testing.assert_array_equal(robot.servo_joint.call_args.args[0], robot.get_state.return_value['joint_angles'])
        robot.close.assert_called_once()

    def test_hold_keeps_physical_floor_check(self):
        c, robot, loop = self.rig()
        robot.forward_kinematics.return_value = ([0., .551, 0., 0., 0., 0.], 0)
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c._send_joints(robot, loop, loop.q_seed.copy(), .02)
        robot.servo_joint.assert_not_called()

    def test_changed_joint_target_still_checks_550_mm_boundary(self):
        c, robot, loop = self.rig()
        robot.forward_kinematics.return_value = ([0., .551, 0., 0., 0., 0.], 0)
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c._send_joints(robot, loop, loop.q_seed + .001, .02)
        robot.servo_joint.assert_not_called()

    def test_motion_target_does_not_apply_five_mm_margin(self):
        c, robot, loop = self.rig()
        loop.joint_hold = False
        c._send_joints(robot, loop, loop.q_seed.copy(), .02)
        robot.servo_joint.assert_called_once()


class CameraHistoryTests(unittest.TestCase):
    def env(self, counts):
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.AuboInterpolationController', integration_tests.FakeRobot):
            env = DualAuboRealEnv('left', 'right', shm_manager=object())
        env.start()
        self.addCleanup(env.stop)
        camera = integration_tests.FakeCamera(time.time())
        camera.cameras = {str(i): SimpleNamespace(
            pid=None, ring_buffer=SimpleNamespace(count=count, get_max_k=30))
            for i, count in enumerate(counts)}
        original_get = camera.get

        def bounded_get(k):
            assert k <= min(counts), 'Requested unwritten camera frames'
            return original_get(k)

        camera.get = Mock(side_effect=bounded_get)
        env.realsense = camera
        return env

    def test_first_frame_produces_complete_observation_shape(self):
        env = self.env([1])
        obs = env.get_obs()
        env.realsense.get.assert_called_once_with(k=1)
        self.assertEqual(obs['camera_0'].shape, (2, 2, 2, 3))
        np.testing.assert_array_equal(obs['camera_0'][0], obs['camera_0'][1])

    def test_each_camera_count_bounds_shared_history_request(self):
        env = self.env([12, 3])
        env.get_obs()
        env.realsense.get.assert_called_once_with(k=3)

    def test_full_history_retains_requested_window(self):
        env = self.env([20])
        env.get_obs()
        env.realsense.get.assert_called_once_with(k=8)

    def test_empty_buffer_reports_error_before_camera_get(self):
        env = self.env([0])
        with self.assertRaisesRegex(RuntimeError, 'no buffered frames'):
            env.get_obs()
        env.realsense.get.assert_not_called()


class StartupWaitTests(unittest.TestCase):
    def env(self):
        env = object.__new__(DualAuboRealEnv)
        env._started = True
        env.left_robot = SimpleNamespace(pid=None, is_ready=True,
                                        heartbeat=Mock(), is_alive=Mock(return_value=True))
        env.right_robot = SimpleNamespace(pid=None, is_ready=True,
                                         heartbeat=Mock(), is_alive=Mock(return_value=True))
        env.realsense = SimpleNamespace(is_ready=False, cameras={})
        env.multi_cam_vis = None
        env.print_ready_status = Mock()
        return env

    def test_ready_arms_receive_heartbeats_while_camera_starts(self):
        env = self.env()
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.time.sleep',
                   side_effect=lambda _: setattr(env.realsense, 'is_ready', True)):
            self.assertTrue(env.wait_until_ready(1.))
        env.left_robot.heartbeat.assert_called_once()
        env.right_robot.heartbeat.assert_called_once()

    def test_not_ready_arm_is_not_sent_a_heartbeat(self):
        env = self.env()
        env.left_robot.is_ready = False
        env.realsense.is_ready = True
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.time.sleep',
                   side_effect=lambda _: setattr(env.left_robot, 'is_ready', True)):
            self.assertTrue(env.wait_until_ready(1.))
        env.left_robot.heartbeat.assert_not_called()
        env.right_robot.heartbeat.assert_called_once()


if __name__ == '__main__':
    unittest.main()
