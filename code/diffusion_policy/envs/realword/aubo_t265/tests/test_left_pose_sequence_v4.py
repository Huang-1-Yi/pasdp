"""Offline P-key path/recording checks; hardware, keyboard and clocks are replaced."""
from contextlib import ExitStack, nullcontext, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.spatial.transform import Rotation

import demo_dual_teleop_aubo_v4 as demo
from diffusion_policy.common.timestamp_accumulator import TimestampActionAccumulator, TimestampObsAccumulator
from diffusion_policy.envs.realword.dual_aubo_real_env_v3 import DualAuboRealEnvV3
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
from diffusion_policy.envs.realword.aubo_interpolation_controller_v3 import AuboInterpolationControllerV3
from diffusion_policy.envs.realword.aubo_t265.calibration import load_calibration
from diffusion_policy.envs.realword.aubo_t265 import sdk_paths, teleop_input, keyboard


class LeftPoseSequenceV4Tests(unittest.TestCase):
    def setUp(self):
        self.config = demo.load_config()
        self.arms = {arm['side']: arm for arm in self.config['arms']}
        self.workspace = demo.DualWorkspaceV4(self.config['arms'], self.config['workspace'])
        self.rotations = demo._load_home_rotvecs(self.config['home_file'], self.arms)
        for side, rotation in self.rotations.items():
            self.workspace.set_locked_orientation(side, rotation)
        self.points = demo._load_left_waypoints(demo.DEFAULT_LEFT_POSE_FILES, self.arms['left'], self.workspace)
        self.home_point = demo._load_home_waypoint(self.config['home_file'], self.arms['left'], self.workspace)
        self.right_home_point = demo._load_home_waypoint(self.config['home_file'], self.arms['right'], self.workspace)

    def state(self):
        q, pose = self.points[0]
        return dict(ActualQ=q.copy(), ActualTCPPose=pose.copy(), ActualTCPSpeed=np.zeros(6),
                    robot_receive_timestamp=1000., GripperState=0.)

    def test_points_use_recorded_order_and_reject_wrong_arm_units_or_height(self):
        np.testing.assert_allclose(self.points[0][1][:3], [.45550606885430067, .22979023325578915, .7291831425705789])
        np.testing.assert_allclose(self.points[1][1][:3], [.4526778483170883, .30939805328126113, .7547990340214871])
        original = json.loads(Path(demo.DEFAULT_LEFT_POSE_FILES[0]).read_text())
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'pose.json'
            for field, value in (('serial', 'wrong'), ('tcp_pose_aubo_rpy', [0.] * 5)):
                data = copy.deepcopy(original)
                data['arms'][0][field] = value
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    demo._load_left_waypoints([path, demo.DEFAULT_LEFT_POSE_FILES[1]], self.arms['left'], self.workspace)
            data = copy.deepcopy(original)
            data['units'] = 'degrees'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                demo._load_left_waypoints([path, demo.DEFAULT_LEFT_POSE_FILES[1]], self.arms['left'], self.workspace)
            data = copy.deepcopy(original)
            data['arms'][0]['tcp_pose_aubo_rpy'][:3] = (-2. * self.workspace.up['left']).tolist()
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'height limit'):
                demo._load_left_waypoints([path, demo.DEFAULT_LEFT_POSE_FILES[1]], self.arms['left'], self.workspace)
            data = copy.deepcopy(original)
            data['arms'][0]['tcp_pose_aubo_rpy'][3:] = [0., 0., 0.]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'orientation'):
                demo._load_left_waypoints([path, demo.DEFAULT_LEFT_POSE_FILES[1]], self.arms['left'], self.workspace)

    def test_elapsed_time_and_repeated_feedback_cannot_skip_first_point(self):
        state = self.state()
        with patch('time.monotonic', return_value=0.):
            sequence = demo._LeftPoseSequence(self.points, state, .1)
            sequence.plan.elapsed = sequence.plan.duration
            state['ActualTCPPose'][:3] += .02
            for index in range(20):
                state['robot_receive_timestamp'] += .02
                sequence.tick(state, .02)
            self.assertEqual(sequence.index, 0)
            state['ActualTCPPose'] = self.points[0][1].copy()
            for _ in range(20):
                sequence.tick(state, .02)
            self.assertEqual(sequence.index, 0)
            for _ in range(5):
                state['robot_receive_timestamp'] += .02
                sequence.tick(state, .02)
            self.assertEqual(sequence.index, 1)
            self.assertFalse(sequence.done)

    def test_unsettled_speed_does_not_advance_and_timeout_is_reported(self):
        state = self.state()
        clock = [0.]
        with patch('time.monotonic', side_effect=lambda: clock[0]):
            sequence = demo._LeftPoseSequence(self.points, state, .1)
            sequence.plan.elapsed = sequence.plan.duration
            state['ActualTCPSpeed'][0] = .1
            for _ in range(10):
                state['robot_receive_timestamp'] += .02
                sequence.tick(state, .02)
            self.assertEqual(sequence.index, 0)
            clock[0] = sequence.plan.deadline + 1.
            with self.assertRaisesRegex(RuntimeError, 'timeout'):
                sequence.tick(state, .02)

    def test_first_segment_obeys_configured_translation_speed(self):
        state = self.state()
        state['ActualTCPPose'][:3] += [.2, .1, .3]
        with patch('time.monotonic', return_value=0.):
            sequence = demo._LeftPoseSequence(self.points, state, .1)
            positions = [state['ActualTCPPose'][:3].copy()]
            while sequence.plan.elapsed < sequence.plan.duration:
                positions.append(sequence.tick(state, .02)[:3])
            speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / .02
            self.assertLessEqual(speed.max(), .1 + 1e-8)
            np.testing.assert_allclose(positions[-1], self.points[0][1][:3])

    def test_p_speed_scales_shorten_each_segment_and_preserve_corresponding_positions(self):
        with patch('time.monotonic', return_value=0.):
            for index, start, target, scale in ((0, self.home_point, self.points[0], 2.),
                                                (1, self.points[0], self.points[1], 2.),
                                                (2, self.points[1], self.home_point, 4.)):
                state = dict(ActualQ=start[0], ActualTCPPose=start[1])
                normal = demo._LeftPoseSequence(self.points, state, .1, self.home_point)
                fast = demo._LeftPoseSequence(self.points, state, .1, self.home_point,
                                              point_speed_scale=demo.P_POINT_SPEED_SCALE,
                                              home_speed_scale=demo.P_RETURN_HOME_SPEED_SCALE)
                for sequence in (normal, fast):
                    sequence.index = index
                    sequence._start_segment(state)
                self.assertAlmostEqual(fast.plan.duration, normal.plan.duration / scale)
                normal.plan.elapsed = normal.plan.duration * .3
                fast.plan.elapsed = fast.plan.duration * .3
                np.testing.assert_allclose(normal.plan.sample(.04), fast.plan.sample(.04 / scale))
                distance = np.linalg.norm(target[1][:3] - start[1][:3])
                self.assertLessEqual(1.875 * distance / fast.plan.duration, .1 * scale + 1e-10)
                self.assertEqual(fast.plan.speed_scale, scale)

    def test_v4_h_speed_override_is_per_controller_instance(self):
        def fake_init(env, *args, **kwargs):
            env.workspace = object()
            env.left_robot = SimpleNamespace(HOME_SPEED_SCALE=2.)
            env.right_robot = SimpleNamespace(HOME_SPEED_SCALE=2.)
        with patch.object(DualAuboRealEnv, '__init__', fake_init):
            env = DualAuboRealEnvV3(home_speed_scale=demo.V4_HOME_SPEED_SCALE)
            normal = DualAuboRealEnvV3()
            for side in ('left_robot', 'right_robot'):
                self.assertEqual(getattr(env, side).HOME_SPEED_SCALE, 4.)
                self.assertEqual(getattr(normal, side).HOME_SPEED_SCALE, 2.)
            self.assertEqual(AuboInterpolationControllerV3.HOME_SPEED_SCALE, 2.)
            for invalid in (0., -1., float('inf'), float('nan')):
                with self.assertRaises(ValueError):
                    DualAuboRealEnvV3(home_speed_scale=invalid)

    def test_final_state_is_checked_at_the_common_saved_row_not_unaligned_latest_rows(self):
        env = MagicMock()
        env._recording = True
        env.gripper_configs = dict(left=dict(pins=[0, 1]))
        env.obs_accumulator.__len__.return_value = 3
        env.action_accumulator.__len__.return_value = 5
        env.stage_accumulator.__len__.return_value = 5
        observed = np.zeros((3, 2))
        actions = np.zeros((5, 15))
        env.obs_accumulator.data = dict(left_gripper_state=observed)
        env.action_accumulator.actions = actions
        actions[2, 6:8] = [1., 0.]
        self.assertFalse(demo._left_output_levels_recorded(env, (0., 0.)))
        actions[2, 6:8] = [0., 0.]
        self.assertTrue(demo._left_output_levels_recorded(env, (0., 0.)))
        observed[2] = [1., 0.]
        self.assertFalse(demo._left_output_levels_recorded(env, (0., 0.)))
        env._recording = False
        self.assertTrue(demo._left_output_levels_recorded(env, (0., 0.)))

    def test_confirmed_left_mapping_and_reversed_right_startup_home_and_cycle(self):
        for left_pins in ([0, 1, 2], [1, 0, 2]):
            with self.subTest(pins=left_pins):
                env = MagicMock()
                env.grippers_enabled = True
                env.gripper_configs = dict(left=dict(mode='independent', pins=left_pins),
                                           right=dict(mode='independent', pins=[0, 2]))
                env.gripper_dims = dict(left=3, right=2)
                masks = dict(left=7., right=3.)
                env.get_state.side_effect = lambda: {side: {'GripperState': mask}
                                                     for side, mask in masks.items()}
                env.decode_gripper.side_effect = lambda side, mask: DualAuboRealEnvV3.decode_gripper(env, side, mask)
                env.set_gripper.side_effect = lambda side, values: masks.update(
                    {side: DualAuboRealEnvV3.encode_gripper(env, side, values)})
                with redirect_stdout(io.StringIO()):
                    outputs = demo._initialize_output_state(env)
                indices = [left_pins.index(0), left_pins.index(1)]
                np.testing.assert_array_equal(outputs['left'][indices], [0., 0.])
                np.testing.assert_array_equal(outputs['right'], [0., 1.])
                # DO00=fold, DO01=suction: on suction, fold, off suction, unfold.
                for expected in ((0., 1.), (1., 1.), (1., 0.), (0., 0.)):
                    _, levels = demo._advance_left_output(env, outputs, True)
                    self.assertEqual(levels, expected)
                    actual = demo._read_output_state(env)
                    np.testing.assert_array_equal(actual['left'][indices], expected)
                    self.assertEqual(actual['left'][left_pins.index(2)], 1.)
                self.assertEqual(demo._toggle_output(env, outputs, 'right', 0, True), 1.)
                self.assertEqual(demo._toggle_output(env, outputs, 'right', 0, True), 0.)
                # Change both latches before H; home restores flat/suction off and right LOW.
                masks.update(left=7., right=3.)
                homes = dict(left=np.zeros(6), right=np.ones(6))
                restored = demo._start_home(env, homes)
                np.testing.assert_array_equal(restored['left'][indices], [0., 0.])
                self.assertEqual(restored['right'][0], 0.)
                env.return_home.assert_called_once_with(homes['left'], homes['right'])

    def simulate(self, cancel=None, input_loss=False, io_delay=0., stuck_state=None, slow_right=False,
                 resume_key=None):
        clock = [0.]
        state = {'left': self.state(), 'right': self.state()}
        state['left']['ActualTCPPose'][:3] += [.01, -.02, .01]
        state['right']['ActualTCPPose'] = self.right_home_point[1].copy()
        state['right']['ActualTCPPose'][:3] += [.04, .02, .01]
        state['right']['ActualQ'] = self.right_home_point[0] + (1.5 if slow_right else .1)
        state['right']['GripperState'] = 1.
        env = MagicMock()
        env.assert_healthy = MagicMock()
        env.__enter__.return_value = env
        env.grippers_enabled = True
        env.gripper_configs = {side: dict(mode='independent', pins=([0, 1] if side == 'left' else [0]))
                               for side in state}
        env.gripper_dims = dict(left=2, right=1)
        env.action_dim = 15
        env.workspace = self.workspace
        env.home_state = 'holding'
        env._recording = False
        env.realsense = None
        commands = []
        sequence_instances = []
        started = []
        saved = []
        pending_io = []

        def feedback():
            ready = [event for event in pending_io if clock[0] >= event[0]]
            for event in ready:
                _, side, mask = event
                if not (side == 'left' and mask == stuck_state):
                    state[side]['GripperState'] = mask
                pending_io.remove(event)
            for sample in state.values():
                sample['robot_receive_timestamp'] = 1000. + clock[0]
            return copy.deepcopy(state)

        def observations():
            query = np.array([1000. + clock[0]])
            values = {side + '_robot_eef_pose': sample['ActualTCPPose'][None] for side, sample in state.items()}
            values.update({side + '_gripper_state': env.decode_gripper(side, sample['GripperState'])
                           for side, sample in state.items()})
            if env._recording:
                env.obs_accumulator.put(values, query)
            return {'timestamp': query, **values}

        def record(start_time):
            started.append(start_time)
            env._recording = True
            env.obs_accumulator = TimestampObsAccumulator(start_time, .1)
            env.action_accumulator = TimestampActionAccumulator(start_time, .1)
            env.stage_accumulator = TimestampActionAccumulator(start_time, .1)

        def save():
            saved.append(env.action_accumulator.actions.copy())
            env.saved_at = clock[0]
            env._recording = False

        def send(action, stage=0):
            DualAuboRealEnvV3.set_teleop_target(env, action, stage)
            commands.append((clock[0], action.copy()))

        def sequence(*args, **kwargs):
            result = self.sequence_class(*args, **kwargs)
            sequence_instances.append(result)
            return result

        def sleep(seconds):
            if sequence_instances:
                raise AssertionError('P must not sleep while waiting for IO or arrival')
            clock[0] += seconds  # Advance only the existing startup confirmation loop.

        def create_env(*args, **kwargs):
            env.requested_home_speed_scale = kwargs.get('home_speed_scale')
            return env

        env.get_state.side_effect = feedback
        env.get_obs.side_effect = observations
        env.start_episode.side_effect = record
        env.end_episode.side_effect = save
        env.set_teleop_target.side_effect = send
        env.decode_gripper.side_effect = lambda side, mask: DualAuboRealEnvV3.decode_gripper(env, side, mask)
        env.set_gripper.side_effect = lambda side, values: pending_io.append(
            (clock[0] + io_delay, side, DualAuboRealEnvV3.encode_gripper(env, side, values)))
        for side in state:
            robot = env.left_robot if side == 'left' else env.right_robot
            robot.set_teleop_target.side_effect = lambda pose, side=side: state[side].update(ActualTCPPose=pose.copy())

        class Keyboard:
            index = -1
            completed_ticks = 0

            def get_press_events(self):
                self.index += 1
                keys = {0: 'h', 1: 'r', 2: 'p', 3: 'p', 4: 'z', 5: 'ac'}
                if cancel is not None:
                    keys[10] = cancel
                    keys[11] = 'q'
                else:
                    if saved:
                        self.completed_ticks += 1
                        if self.completed_ticks == 3:
                            return [SimpleNamespace(char=char) for char in 'ztac']
                        if resume_key is not None and self.completed_ticks == 12:
                            env.resumed_at = clock[0]
                            return [SimpleNamespace(char=resume_key)]
                        if resume_key is not None and self.completed_ticks < 22:
                            return []
                        if self.completed_ticks >= 12:
                            return [SimpleNamespace(char='q')]
                    keys[3000] = 'q'  # Fail the completion assertion if the path gets stuck.
                return [SimpleNamespace(char=char) for char in keys.get(self.index, '')]

        def samples():
            if input_loss and clock[0] >= .18:
                raise RuntimeError('simulated lost tracking')
            return {side: SimpleNamespace(
                position=(np.array([.04, .02, .01]) if side == 'left' and clock[0] >= .06
                          else np.array([0., .001 * np.sin(clock[0]), 0.])),
                orientation=Rotation.identity(), encoder_angle=None) for side in state}

        keyboard_module = SimpleNamespace(Key=SimpleNamespace(esc=object(), backspace=object(), space=object()))
        args = demo.parser().parse_args(['--enable-motion', '--no-cameras', '--headless', '--output', '/tmp/offline-P'])
        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, {'diffusion_policy.envs.realword.keystroke_counter': keyboard_module}))
            replacements = {
                'diffusion_policy.envs.realword.aubo_t265.sdk_paths.bootstrap_sdk': lambda: None,
                'diffusion_policy.envs.realword.aubo_t265.sdk_paths.CabinetLocks': lambda arms: nullcontext(),
                'diffusion_policy.envs.realword.aubo_t265.calibration.read_installation': lambda ip, port: (
                    load_calibration(next(arm['calibration'] for arm in self.arms.values() if arm['ip'] == ip), ip,
                                     next(arm['serial'] for arm in self.arms.values() if arm['ip'] == ip))['gravity'], None),
                'diffusion_policy.envs.realword.aubo_interpolation_controller.query_robot_status': lambda ip, port: dict(
                    robot_mode='Running', safety_mode='Normal', tcp_pose_rotvec=state[next(side for side, arm in self.arms.items()
                    if arm['ip'] == ip)]['ActualTCPPose'].copy()),
                'diffusion_policy.envs.realword.dual_aubo_real_env_v3.DualAuboRealEnvV3': create_env,
                'diffusion_policy.envs.realword.aubo_t265.teleop_input.DualTeleopInput': lambda *args: nullcontext(
                    SimpleNamespace(get=samples)),
                'diffusion_policy.envs.realword.aubo_t265.keyboard.create_keyboard': lambda: nullcontext(Keyboard()),
                'demo_dual_teleop_aubo_v4.SharedMemoryManager': lambda: nullcontext(),
                'demo_dual_teleop_aubo_v4._close_preview': lambda: None,
                'demo_dual_teleop_aubo_v4.precise_wait': lambda target: clock.__setitem__(0, target),
                'time.monotonic': lambda: clock[0],
                'time.time': lambda: 1000. + clock[0],
                'time.sleep': sleep,
            }
            for name, replacement in replacements.items():
                stack.enter_context(patch(name, replacement))
            stack.enter_context(patch.object(demo, '_LeftPoseSequence', side_effect=sequence))
            with redirect_stdout(output):
                demo.run(args, self.config)
        return env, commands, sequence_instances, started, saved, output.getvalue()

    sequence_class = demo._LeftPoseSequence

    def test_p_records_full_path_right_home_hold_and_automatically_saves_like_s(self):
        env, commands, sequences, started, saved, output = self.simulate()
        self.assertEqual(len(sequences), 2)  # One left path and one right return; repeated P ignored.
        self.assertTrue(all(sequence.done for sequence in sequences))
        self.assertEqual(len(started), 1)
        self.assertEqual(len(saved), 1)
        self.assertFalse(env._recording)
        env.hold.assert_called_once()
        env.end_episode.assert_called_once()
        self.assertGreater(len(env.obs_accumulator), 30)
        self.assertGreater(len(env.action_accumulator), 30)
        self.assertEqual(sequences[0].point_speed_scale, 2.)
        self.assertEqual(sequences[0].home_speed_scale, 4.)
        self.assertEqual(env.requested_home_speed_scale, 4.)
        np.testing.assert_allclose(np.diff(env.action_accumulator.timestamps), .1, atol=1e-10)
        np.testing.assert_allclose(np.diff(env.obs_accumulator.timestamps), .1, atol=1e-10)
        recorded = env.action_accumulator.actions
        recorded_first = np.flatnonzero(np.linalg.norm(recorded[:, :3] - self.points[0][1][:3], axis=1) < 1e-8)
        recorded_second = np.flatnonzero(np.linalg.norm(recorded[:, :3] - self.points[1][1][:3], axis=1) < 1e-8)
        self.assertGreater(recorded_second[0], recorded_first[0])
        array = np.array([action for _, action in commands])
        first = np.flatnonzero(np.linalg.norm(array[:, :3] - self.points[0][1][:3], axis=1) < 1e-8)
        second = np.flatnonzero(np.linalg.norm(array[:, :3] - self.points[1][1][:3], axis=1) < 1e-8)
        self.assertGreaterEqual(len(first), 5)
        self.assertGreater(second[0], first[4])
        state3 = np.flatnonzero(np.all(array[:, 6:8] == [1., 1.], axis=1))
        state4 = np.flatnonzero(np.all(array[:, 6:8] == [1., 0.], axis=1))
        self.assertLess(state3[0], first[0])
        self.assertGreater(state4[0], second[4])
        np.testing.assert_array_equal(array[state3[0]:state4[0], 6:8], np.tile([1., 1.], (state4[0] - state3[0], 1)))
        np.testing.assert_allclose(array[state4[0], :3], self.points[1][1][:3])
        np.testing.assert_allclose(array[-5:, :3], np.tile(self.home_point[1][:3], (5, 1)), atol=1e-10)
        final_state1 = np.flatnonzero((np.linalg.norm(array[:, :3] - self.home_point[1][:3], axis=1) < 1e-8)
                                     & np.all(array[:, 6:8] == [0., 0.], axis=1))
        left_home = np.flatnonzero(np.linalg.norm(array[:, :3] - self.home_point[1][:3], axis=1) < 1e-8)
        self.assertGreater(final_state1[0], left_home[4])
        np.testing.assert_array_equal(array[-1, 6:8], [0., 0.])
        np.testing.assert_array_equal(recorded[-1, 6:8], [0., 0.])
        np.testing.assert_array_equal(env.obs_accumulator.data['left_gripper_state'][-1], [0., 0.])
        self.assertEqual(array[-1, 14], 1.)
        right_home = np.flatnonzero(np.linalg.norm(array[:, 8:11] - self.right_home_point[1][:3], axis=1) < 1e-8)
        self.assertLess(right_home[0], state4[0])
        np.testing.assert_allclose(array[right_home[0]:, 8:11],
                                   np.tile(self.right_home_point[1][:3], (len(array) - right_home[0], 1)), atol=1e-10)
        self.assertEqual(commands[-1][0], env.saved_at)  # No teleop targets after the automatic S effect.
        np.testing.assert_array_equal(saved[0], recorded)
        self.assertIn('P complete', output)
        self.assertIn('P complete in state 1', output)
        self.assertIn('left home reached and settled; state 1', output)
        self.assertIn('episode saved and teleoperation paused (same as S)', output)
        self.assertIn('manual tracking ignored', output)
        self.assertIn('point 1 reached and settled', output)
        self.assertIn('point 2 reached and settled; state 4', output)
        self.assertIn('P controls left IO automatically', output)
        self.assertEqual(env.return_home.call_count, 1)  # P's return did not enter H's stop-recording path.

    def test_p_waits_for_state3_and_state4_latches_without_blocking_capture(self):
        env, commands, sequences, started, saved, output = self.simulate(io_delay=.16)
        self.assertTrue(sequences[0].done)
        array = np.array([action for _, action in commands])
        times = np.array([when for when, _ in commands])
        state3 = np.flatnonzero(np.all(array[:, 6:8] == [1., 1.], axis=1))
        state4 = np.flatnonzero(np.all(array[:, 6:8] == [1., 0.], axis=1))
        state1 = np.flatnonzero((np.linalg.norm(array[:, :3] - self.home_point[1][:3], axis=1) < 1e-8)
                               & np.all(array[:, 6:8] == [0., 0.], axis=1))
        # During all three asynchronous IO transitions, left holds and capture still runs.
        for indices in (state3, state4, state1):
            start = indices[0]
            waiting = (times >= times[start]) & (times < times[start] + .15)
            np.testing.assert_allclose(array[waiting, :3], np.tile(array[start, :3], (waiting.sum(), 1)), atol=1e-10)
        self.assertEqual(len(started), 1)
        self.assertEqual(len(saved), 1)
        self.assertGreater(len(env.obs_accumulator), 30)
        self.assertGreaterEqual(env.saved_at, times[state1[0]] + .15)
        np.testing.assert_array_equal(saved[0][-1, 6:8], [0., 0.])
        np.testing.assert_array_equal(env.obs_accumulator.data['left_gripper_state'][-1], [0., 0.])
        self.assertIn('P complete', output)

    def test_p_waits_for_right_home_and_holds_left_home_until_both_settled(self):
        env, commands, sequences, started, saved, output = self.simulate(slow_right=True)
        self.assertEqual(len(saved), 1)
        self.assertTrue(all(sequence.done for sequence in sequences))
        self.assertEqual(sequences[1].point_speed_scale, 1.)
        array = np.array([action for _, action in commands])
        left_home = np.flatnonzero(np.linalg.norm(array[:, :3] - self.home_point[1][:3], axis=1) < 1e-8)
        right_home = np.flatnonzero(np.linalg.norm(array[:, 8:11] - self.right_home_point[1][:3], axis=1) < 1e-8)
        self.assertGreater(right_home[0], left_home[0] + 5)
        np.testing.assert_allclose(array[left_home[0]:, :3],
                                   np.tile(self.home_point[1][:3], (len(array) - left_home[0], 1)), atol=1e-10)
        self.assertGreater(len(array) - right_home[0], 4)
        self.assertEqual(commands[-1][0], env.saved_at)
        env.hold.assert_called_once()
        env.end_episode.assert_called_once()

    def test_missing_any_p_output_confirmation_stops_path(self):
        for stuck in (3., 1., 0.):  # Pins [0,1]: state 3, state 4, final state 1.
            with self.subTest(mask=stuck), self.assertRaisesRegex(RuntimeError, 'P output latch'):
                self.simulate(stuck_state=stuck)

    def test_after_auto_save_tracking_resumes_only_when_e_or_r_is_pressed(self):
        for key in ('e', 'r'):
            with self.subTest(key=key):
                env, commands, sequences, started, saved, output = self.simulate(resume_key=key)
                self.assertEqual(len(saved), 1)
                self.assertEqual(len(started), 2 if key == 'r' else 1)
                self.assertEqual(env._recording, key == 'r')
                env.hold.assert_called_once()
                between = [action for when, action in commands if env.saved_at < when < env.resumed_at]
                self.assertEqual(between, [])
                resumed = [action for when, action in commands if when >= env.resumed_at]
                self.assertGreater(len(resumed), 1)
                np.testing.assert_allclose(resumed[0][:3], self.home_point[1][:3], atol=1e-10)
                np.testing.assert_allclose(resumed[0][8:11], self.right_home_point[1][:3], atol=1e-10)
                np.testing.assert_array_equal(resumed[0][6:8], [0., 0.])

    def test_s_and_e_can_cancel_p_without_blocking_keyboard(self):
        for key in ('s', 'e'):
            with self.subTest(key=key):
                env, commands, sequences, started, saved, output = self.simulate(cancel=key)
                self.assertEqual(len(sequences), 2)
                self.assertFalse(sequences[0].done)
                self.assertEqual(len(started), 1)
                self.assertEqual(len(saved), 1)
                self.assertFalse(env._recording)
                env.hold.assert_called_once()
                self.assertNotIn('P complete', output)

    def test_h_can_replace_p_and_tracking_loss_cancels_it(self):
        env, _, sequences, _, saved, output = self.simulate(cancel='h')
        self.assertFalse(sequences[0].done)
        self.assertEqual(len(saved), 1)
        self.assertEqual(env.return_home.call_count, 2)
        self.assertFalse(env._recording)
        self.assertNotIn('P complete', output)
        env, _, sequences, _, saved, output = self.simulate(cancel='s', input_loss=True)
        self.assertFalse(sequences[0].done)
        self.assertEqual(len(saved), 1)
        self.assertFalse(env._recording)
        self.assertIn('Input lost: paused', output)
        self.assertNotIn('P complete', output)


if __name__ == '__main__':
    unittest.main()
