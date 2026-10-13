"""Offline height checks: no robot connection or actual digital output writes."""
import unittest
from unittest.mock import Mock, patch

import numpy as np

from diffusion_policy.envs.realword.aubo_t265.config import load_config
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_t265.tests import test_latest_teleop as _teleop_tests

batch = _teleop_tests.batch
from diffusion_policy.envs.realword.aubo_interpolation_controller_v2 import Command
from diffusion_policy.envs.realword.dual_aubo_real_env_v2 import DualAuboRealEnvV2


class HeightTests(unittest.TestCase):
    def workspace(self):
        config = load_config()
        workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        # Tilted base: gravity points along BASE +Y, not BASE -Z.
        for side in ('left', 'right'):
            workspace.update_gravity(side, [0., 9.81, 0.])
        return workspace

    def rig(self):
        controller, robot, loop = _teleop_tests.RealtimeTests().rig()
        controller.workspace = self.workspace()
        controller.workspace_side = 'left'
        robot.forward_kinematics.return_value = (np.zeros(6), 0)
        return controller, robot, loop

    def test_height_uses_gravity_projection_not_base_z_or_start_pose(self):
        w = self.workspace()
        w.validate('left', [2., .4, -3., 0., 0., 0.])
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            w.validate('right', [0., .551, 2., 0., 0., 0.])
        w.validate('left', [0., .55, 0., 0., 0., 0.])

    def test_targets_and_feedback_share_inclusive_550_mm_boundary(self):
        w = self.workspace()
        for side in ('left', 'right'):
            for target in (False, True):
                for downward in (.545, .54984, .55):
                    w.validate(side, [0., downward, 0., 0., 0., 0.], target=target)
                with self.assertRaisesRegex(RuntimeError, 'limit -550.00 mm'):
                    w.validate(side, [0., .551, 0., 0., 0., 0.], target=target)

    def test_551_mm_teleop_goal_is_projected_to_550_mm(self):
        w = self.workspace()
        for side in ('left', 'right'):
            pose = np.array([.2, .551, -.3, .1, .2, .3])
            limited = w.limit_target(side, pose)
            np.testing.assert_allclose(limited, [.2, .55, -.3, .1, .2, .3])
            w.validate(side, limited)
            w.validate_target(side, limited)

    def test_nonzero_margin_cannot_restore_545_mm_boundary(self):
        config = load_config()
        settings = dict(config['workspace'], target_margin=.005)
        with self.assertRaisesRegex(ValueError, 'target_margin must be 0'):
            DualWorkspace(config['arms'], settings)

    def test_limit_preserves_horizontal_motion_and_orientation(self):
        w = self.workspace()
        pose = np.array([.4, .8, -.9, .1, .2, .3])
        original = pose.copy()
        limited = w.limit_target('left', pose)
        np.testing.assert_allclose(limited, [.4, .55, -.9, .1, .2, .3])
        np.testing.assert_array_equal(pose, original)
        w.validate_target('left', limited)
        for side in ('left', 'right'):
            np.testing.assert_allclose(w.limit_target(side, limited), limited)

    def test_live_gravity_update_changes_plane_and_invalid_gravity_rejected(self):
        w = self.workspace()
        w.update_gravity('left', [0., 0., -9.81])
        self.assertAlmostEqual(w.height('left', [0., .8, -.4, 0., 0., 0.]), -.4)
        with self.assertRaises(ValueError):
            w.update_gravity('left', [0., 0., 0.])

    def test_controller_clamps_latest_goal_before_ik(self):
        c, r, loop = self.rig()
        c.latest_teleop_pose.put([0., .8, -2., 0., 0., 0.], 10.01)
        _teleop_tests.RealtimeTests().process(c, r, loop, 10.02)
        c._target_joints(r, loop, 10.02, .02)
        self.assertAlmostEqual(r.inverse_kinematics.call_args.args[1][1], .55)

    def test_low_actual_feedback_selects_upward_ik_target(self):
        c, r, loop = self.rig()
        r.get_state.return_value['eef_pose'][1] = .551
        c._target_joints(r, loop, 10.02, .02)
        self.assertAlmostEqual(r.inverse_kinematics.call_args.args[1][1], .55)
        self.assertAlmostEqual(r.get_state.return_value['eef_pose'][1], .551)

    def test_invalid_fk_or_low_joint_command_never_reaches_servo(self):
        c, r, loop = self.rig()
        r.forward_kinematics.return_value = ([0., .551, 0., 0., 0., 0.], 0)
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_joint.assert_not_called()
        r.forward_kinematics.return_value = ([], -1)
        with self.assertRaisesRegex(RuntimeError, 'forwardKinematics failed'):
            c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_joint.assert_not_called()

    def test_stop_received_during_fk_prevents_servo(self):
        c, r, loop = self.rig()
        def fk(joints):
            c.stop_event.set()
            return np.zeros(6), 0
        r.forward_kinematics.side_effect = fk
        c._send_joints(r, loop, loop.q_seed, .02)
        r.servo_joint.assert_not_called()

    def test_home_endpoint_and_intermediate_point_checked(self):
        c, r, loop = self.rig()
        c.input_queue.get_all.return_value = batch(dict(cmd=Command.HOME.value, target_pose=np.ones(6)))
        r.forward_kinematics.return_value = ([0., .6, 0., 0., 0., 0.], 0)
        with self.assertRaisesRegex(RuntimeError, 'height limit'):
            _teleop_tests.RealtimeTests().process(c, r, loop, 10.02)
        self.assertIsNone(loop.home_plan)
        r.forward_kinematics.return_value = (np.zeros(6), 0)
        _teleop_tests.RealtimeTests().process(c, r, loop, 10.04)
        with patch('diffusion_policy.envs.realword.aubo_interpolation_controller_v2.time.monotonic', return_value=10.04):
            q = c._target_joints(r, loop, 10.04, .02)
        r.forward_kinematics.return_value = ([0., .6, 0., 0., 0., 0.], 0)
        with self.assertRaises(RuntimeError):
            c._send_joints(r, loop, q, .02)
        self.assertEqual(loop.home_plan.elapsed, 0.)
        r.servo_joint.assert_not_called()

    def test_environment_limits_both_goals_and_records_limited_action(self):
        env = object.__new__(DualAuboRealEnvV2)
        env.grippers_enabled = False
        env.gripper_dims = {'left': 0, 'right': 0}
        env.action_dim = 12
        env.workspace = self.workspace()
        env.assert_healthy = Mock()
        env.left_robot, env.right_robot = Mock(), Mock()
        env.left_robot.home_state.value = env.right_robot.home_state.value = 0
        env._recording = True
        env.action_accumulator, env.stage_accumulator = Mock(), Mock()
        action = np.array([0., .8, 0., 0., 0., 0.] * 2)
        env.set_teleop_target(action)
        for robot in (env.left_robot, env.right_robot):
            self.assertAlmostEqual(robot.set_teleop_target.call_args.args[0][1], .55)
        self.assertAlmostEqual(env.action_accumulator.put.call_args.args[0][0, 1], .55)
        self.assertEqual(action[1], .8)
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.DualAuboRealEnv.exec_actions') as submit:
            with self.assertRaises(RuntimeError):
                env.exec_actions([action], [1.])
            submit.assert_not_called()

    def test_both_saved_home_tcp_poses_fit_the_new_limit(self):
        import json
        config = load_config()
        workspace = DualWorkspace(config['arms'], dict(config['workspace'], reference='gravity_base', max_downward_distance=.55))
        with open(config['home_file']) as f:
            home = json.load(f)
        for arm in home['arms']:
            workspace.validate_target(arm['side'], arm['tcp_pose_aubo_rpy'])


if __name__ == '__main__':
    unittest.main()
