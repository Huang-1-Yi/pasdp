"""Offline v3 double-speed home and restoration of startup output levels."""
import unittest
from unittest.mock import Mock, patch
import numpy as np
from diffusion_policy.envs.realword.aubo_homing import JointHome, LockedCartesianHome
from diffusion_policy.envs.realword.aubo_t265.tests import test_servo_cartesian_v3 as cartesian_tests
from diffusion_policy.envs.realword.aubo_t265.tests.test_servo_cartesian_v3 import batch, Command, GRIPPER_COMMAND
from diffusion_policy.envs.realword.aubo_t265.tests import test_strict_orientation as strict_tests
from demo_dual_teleop_aubo_v3 import _start_home, _confirm_home_outputs


class HomeSpeedIoV3Tests(unittest.TestCase):
    def test_double_speed_halves_duration_and_preserves_joint_path(self):
        for distance in (.01, .8):
            normal = JointHome(np.zeros(6), np.ones(6) * distance)
            fast = JointHome(normal.start, normal.target, speed_scale=2.)
            self.assertAlmostEqual(fast.duration, normal.duration / 2.)
            # At corresponding fractions of time, position is identical and
            # the finite-difference velocity is exactly twice the original.
            normal.elapsed = normal.duration * .3
            fast.elapsed = fast.duration * .3
            a, b = normal.sample(.01), normal.sample(.03)
            c, d = fast.sample(.005), fast.sample(.015)
            np.testing.assert_allclose(a, c)
            np.testing.assert_allclose(b, d)
            np.testing.assert_allclose((d-c)/.01, 2.*(b-a)/.02)

    def test_double_speed_keeps_recorded_orientation_and_restart_scale(self):
        start, goal = np.zeros(6), np.array([.3, -.1, .2, .1, .2, .3])
        q, target_q = np.zeros(6), np.ones(6) * .4
        normal = LockedCartesianHome(q, target_q, start, goal)
        fast = LockedCartesianHome(q, target_q, start, goal, speed_scale=2.)
        self.assertAlmostEqual(fast.duration, normal.duration / 2.)
        np.testing.assert_allclose(normal.sample(.04), fast.sample(.02))
        np.testing.assert_array_equal(fast.sample(.02)[3:], goal[3:])
        restarted = fast.restart(q + .1, start + .01)
        reference = LockedCartesianHome(q + .1, target_q, start + .01, goal)
        self.assertEqual(restarted.speed_scale, 2.)
        self.assertAlmostEqual(restarted.duration, reference.duration / 2.)

    def test_invalid_speed_scale_is_rejected(self):
        for scale in (0., -1., float('inf'), float('nan')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                JointHome(np.zeros(6), np.zeros(6), speed_scale=scale)

    def test_v3_joint_home_selects_double_speed_and_keeps_io_events(self):
        c, r, loop = cartesian_tests.CartesianV3Tests().rig()
        target = np.ones(6) * .8
        c.input_queue.get_all.return_value = batch(
            dict(cmd=GRIPPER_COMMAND, gripper=1.),
            dict(cmd=Command.HOME.value, target_pose=target))
        c._process_commands(r, loop, 10.01, .02)
        normal = JointHome(loop.q_seed, target)
        self.assertEqual(loop.home_plan.speed_scale, 2.)
        self.assertAlmostEqual(loop.home_plan.duration, normal.duration / 2.)
        self.assertEqual(list(loop.immediate_gripper_events), [1.])

    def test_v3_locked_home_selects_double_speed(self):
        c, r, loop = strict_tests.StrictOrientationTests().rig(3)
        target = np.ones(6) * .8
        c.input_queue.get_all.return_value = batch(dict(cmd=Command.HOME.value, target_pose=target))
        c._process_commands(r, loop, 10.01, .02)
        plan = loop.home_plan
        normal = LockedCartesianHome(plan.start, plan.target, plan.start_pose, plan.target_pose)
        self.assertIsInstance(plan, LockedCartesianHome)
        self.assertEqual(plan.speed_scale, 2.)
        self.assertAlmostEqual(plan.duration, normal.duration / 2.)

    def test_height_recovery_joint_restart_keeps_double_speed(self):
        c, r, loop = strict_tests.StrictOrientationTests().rig(3)
        loop.home_plan = JointHome(loop.q_seed, np.ones(6) * .8, speed_scale=2.)
        loop.height_recovery_pose = r.get_state.return_value['eef_pose'].copy()
        c._height_recovery(loop, r.get_state.return_value, 10.02)
        self.assertEqual(loop.home_plan.speed_scale, 2.)
        normal = JointHome(loop.home_plan.start, loop.home_plan.target)
        self.assertAlmostEqual(loop.home_plan.duration, normal.duration / 2.)

    def env(self, enabled=True, masks=None):
        env = Mock()
        env.grippers_enabled = enabled
        # A third left output demonstrates that unrelated pins are preserved.
        env.gripper_configs = dict(left=dict(mode='independent', pins=[0, 1, 2]),
                                   right=dict(mode='independent', pins=[0]))
        masks = dict(left=7, right=0) if masks is None else masks
        env.get_state.return_value = {side: {'GripperState': mask} for side, mask in masks.items()}
        env.decode_gripper.side_effect = lambda side, mask: (
            np.array([float(bool(int(mask) & (1 << i)))
                      for i in range(len(env.gripper_configs[side]['pins']))]),)
        return env

    def test_h_requests_startup_levels_without_waiting_and_updates_cache(self):
        env = self.env()
        homes = dict(left=np.zeros(6), right=np.ones(6))
        with patch('time.sleep', side_effect=AssertionError('H must not block on IO')):
            outputs = _start_home(env, homes)
        np.testing.assert_array_equal(outputs['left'], [0., 0., 1.])
        np.testing.assert_array_equal(outputs['right'], [1.])
        for call, side in zip(env.set_gripper.call_args_list, ('left', 'right')):
            self.assertEqual(call.args[0], side)
            np.testing.assert_array_equal(call.args[1], outputs[side])
        env.return_home.assert_called_once_with(homes['left'], homes['right'])
        self.assertEqual([call[0] for call in env.method_calls][-3:],
                         ['set_gripper', 'set_gripper', 'return_home'])
        env.heartbeat.assert_not_called()

    def test_home_completion_uses_confirmed_actual_io(self):
        env = self.env(masks=dict(left=4, right=1))
        targets = dict(left=np.array([0., 0., 1.]), right=np.array([1.]))
        actual = _confirm_home_outputs(env, targets)
        for side in targets:
            np.testing.assert_array_equal(actual[side], targets[side])
        env.set_gripper.assert_not_called()

    def test_home_completion_rejects_unconfirmed_right_io(self):
        env = self.env(masks=dict(left=4, right=0))
        targets = dict(left=np.array([0., 0., 1.]), right=np.array([1.]))
        with self.assertRaisesRegex(RuntimeError, "Home output levels.*right"):
            _confirm_home_outputs(env, targets)

    def test_home_with_io_disabled_only_requests_motion(self):
        env = self.env(enabled=False)
        homes = dict(left=np.zeros(6), right=np.ones(6))
        outputs = _start_home(env, homes)
        env.get_state.assert_not_called()
        env.set_gripper.assert_not_called()
        env.return_home.assert_called_once_with(homes['left'], homes['right'])
        _confirm_home_outputs(env, outputs)


if __name__ == '__main__':
    unittest.main()
