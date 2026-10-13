"""Offline regression for delayed cabinet output latch feedback, without IO hardware."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
from diffusion_policy.envs.realword.aubo_interpolation_controller_v2 import AuboInterpolationControllerV2
from diffusion_policy.envs.realword.aubo_interpolation_controller_v3 import AuboInterpolationControllerV3
from demo_dual_teleop_aubo_v3 import _initialize_output_state


class IoFeedbackTests(unittest.TestCase):
    def device(self, levels):
        io = Mock()
        io.getStandardDigitalOutputNum.return_value = len(levels)
        io.getStandardDigitalOutput.side_effect = lambda pin: levels[pin]
        # Setter accepts the request, but the latch remains unchanged until the
        # test explicitly advances the cabinet. This reproduces the old race.
        io.setStandardDigitalOutput.return_value = 0
        device = AuboDigitalGripper(io, dict(type='standard_digital', mode='independent',
                                             pins=list(range(len(levels)))))
        device.start()
        return device, io

    def test_delayed_low_is_confirmed_without_repeating_writes(self):
        levels = [True]
        device, io = self.device(levels)
        with patch('diffusion_policy.envs.realword.aubo_gripper.time.monotonic', return_value=10.) as now:
            self.assertEqual(device.command(0), 1)
            self.assertEqual(device.last_command, 0)
            levels[0] = False
            now.return_value = 10.101
            self.assertEqual(device.poll_feedback(), 0)
        io.setStandardDigitalOutput.assert_called_once_with(0, False)

    def test_polling_is_throttled_and_stops_after_confirmation(self):
        levels = [True]
        device, io = self.device(levels)
        with patch('diffusion_policy.envs.realword.aubo_gripper.time.monotonic', return_value=10.) as now:
            device.command(0)
            io.getStandardDigitalOutput.reset_mock()
            now.return_value = 10.05
            device.poll_feedback()
            io.getStandardDigitalOutput.assert_not_called()
            now.return_value = 10.101
            self.assertEqual(device.poll_feedback(), 1)
            now.return_value = 10.15
            device.poll_feedback()
            self.assertEqual(io.getStandardDigitalOutput.call_count, 1)
            levels[0] = False
            now.return_value = 10.202
            self.assertEqual(device.poll_feedback(), 0)
            now.return_value = 11.
            device.poll_feedback()
            self.assertEqual(io.getStandardDigitalOutput.call_count, 2)
        io.setStandardDigitalOutput.assert_called_once_with(0, False)

    def test_latest_independent_event_is_the_confirmation_target(self):
        levels = [True, True]
        device, io = self.device(levels)
        with patch('diffusion_policy.envs.realword.aubo_gripper.time.monotonic', return_value=10.) as now:
            device.command(1)
            device.command(2)
            self.assertEqual(device.last_command, 2)
            self.assertEqual(device.state, 3)
            levels[:] = [False, True]
            now.return_value = 10.101
            self.assertEqual(device.poll_feedback(), 2)
        self.assertEqual(io.setStandardDigitalOutput.call_count, 4)
        self.assertEqual(io.setStandardDigitalOutput.call_args_list[-2].args, (0, False))
        self.assertEqual(io.setStandardDigitalOutput.call_args_list[-1].args, (1, True))

    def test_repeated_request_reads_feedback_without_repeating_output_write(self):
        levels = [True]
        device, io = self.device(levels)
        device.command(0)
        levels[0] = False
        self.assertEqual(device.command(0), 0)
        io.setStandardDigitalOutput.assert_called_once_with(0, False)

    def test_disabled_feedback_never_acquires_io(self):
        io = Mock()
        device = AuboDigitalGripper(io, {'type': 'disabled'})
        self.assertEqual(device.poll_feedback(), -1)
        self.assertEqual(io.mock_calls, [])

    def test_controller_publishes_updated_latch_and_unchanged_rtde_pose(self):
        for controller_type in (AuboInterpolationControllerV2, AuboInterpolationControllerV3):
            with self.subTest(controller=controller_type.__name__):
                levels = [True]
                device, io = self.device(levels)
                controller = controller_type.__new__(controller_type)
                controller.ring_buffer = Mock()
                pose = np.array([.2, .3, .4, .1, .2, .3])
                rtde = dict(eef_pose=pose.copy(), eef_speed=np.zeros(6),
                            joint_angles=np.zeros(6), joint_speeds=np.zeros(6), received=8.)
                robot = SimpleNamespace(gripper=device, get_state=lambda: rtde)
                loop = SimpleNamespace(target_pose=np.zeros(6))
                with patch('diffusion_policy.envs.realword.aubo_gripper.time.monotonic', return_value=10.) as now:
                    device.command(0)
                    levels[0] = False
                    now.return_value = 10.101
                    controller._publish_state(robot, loop)
                published = controller.ring_buffer.put.call_args.args[0]
                self.assertEqual(published['GripperState'], 0)
                self.assertEqual(published['robot_receive_timestamp'], 8.)
                np.testing.assert_array_equal(published['ActualTCPPose'], pose)
                np.testing.assert_array_equal(rtde['eef_pose'], pose)

    def startup_env(self, settle, clock):
        left, left_io = self.device([False, False])
        right_levels = [False]
        right, right_io = self.device(right_levels)
        devices = dict(left=left, right=right)
        env = Mock()
        env.grippers_enabled = True
        env.gripper_configs = {side: dict(mode='independent', pins=device.pins)
                               for side, device in devices.items()}
        def state():
            if settle and clock[0] >= .12:
                right_levels[0] = True
            for device in devices.values():
                device.poll_feedback()
            return {side: {'GripperState': device.state} for side, device in devices.items()}
        env.get_state.side_effect = state
        env.decode_gripper.side_effect = lambda side, mask: (
            np.array([float(bool(int(mask) & (1 << i))) for i in range(len(devices[side].pins))]),)
        env.set_gripper.side_effect = lambda side, values: devices[side].command(
            sum(int(v) << i for i, v in enumerate(values)))
        return env, right_io

    def test_startup_confirms_delayed_right_high_and_keeps_heartbeat(self):
        clock = [0.]
        env, right_io = self.startup_env(True, clock)
        def sleep(seconds):
            clock[0] += seconds
        with patch('time.monotonic', side_effect=lambda: clock[0]), patch('time.sleep', side_effect=sleep):
            actual = _initialize_output_state(env)
        np.testing.assert_array_equal(actual['right'], [1.])
        np.testing.assert_array_equal(actual['left'], [0., 0.])
        self.assertGreaterEqual(clock[0], .12)
        self.assertGreater(env.heartbeat.call_count, 1)
        right_io.setStandardDigitalOutput.assert_called_once_with(0, True)

    def test_unconfirmed_right_high_still_fails_with_pin_level_details(self):
        clock = [0.]
        env, right_io = self.startup_env(False, clock)
        def sleep(seconds):
            clock[0] += seconds
        with patch('time.monotonic', side_effect=lambda: clock[0]), patch('time.sleep', side_effect=sleep):
            with self.assertRaisesRegex(RuntimeError, "right.*DO00.*actual.*0.0.*target.*1.0"):
                _initialize_output_state(env)
        right_io.setStandardDigitalOutput.assert_called_once_with(0, True)


if __name__ == '__main__':
    unittest.main()
