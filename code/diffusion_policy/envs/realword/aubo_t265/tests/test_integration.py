"""Offline frame, input, IO, environment and runtime acceptance tests."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys
from collections import defaultdict
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
from diffusion_policy.envs.realword.aubo_t265.calibration import load_calibration, base_basis, t265_operator_basis, validate_installation
from diffusion_policy.envs.realword.aubo_t265.config import load_config, recording_metadata, controller_kwargs
from diffusion_policy.envs.realword.aubo_t265.runtime import TeleopSession
from diffusion_policy.envs.realword.aubo_t265.teleop_input import decode_samples, HandSample
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
from diffusion_policy.envs.realword.udp_sender_dual_absolute_v3 import _make_teleop_sample


class FrameTests(unittest.TestCase):
    def test_config_rejects_invalid_filter_and_controller_settings(self):
        for section, key, value in [('motion', 'smooth_alpha', 0),
                                    ('motion', 'rotation_smooth_alpha', 1.1),
                                    ('controller', 'servo_t', .001),
                                    ('controller', 'max_joint_speed', -.5),
                                    ('controller', 'max_following_error', float('nan')),
                                    ('controller', 'reject_joint_jumps', 'false')]:
            config = load_config()
            config[section][key] = value
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'dual.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    load_config(path)

    def test_range_switch_controls_actual_mapper_output(self):
        config = load_config()
        config['motion'].update(smooth_alpha=1., rotation_smooth_alpha=1.,
                                max_speed=100., max_angular_speed_deg=10000.)
        poses = {side: np.zeros(6) for side in ('left', 'right')}
        for unlimited in (True, False):
            for arm in config['arms']:
                arm['no_range_limit'] = unlimited
            session = TeleopSession(config, poses, DualWorkspace(config['arms'], config['workspace']))
            mapper = session.mappers['left']
            mapper.anchor(np.zeros(3), Rotation.identity(), np.zeros(6))
            target = mapper.step([.25, 0, 0], Rotation.from_rotvec([0, 0, np.pi/2]), .02)
            if unlimited:
                self.assertGreater(np.linalg.norm(target[:3]), .1)
                self.assertGreater(np.linalg.norm(target[3:]), 1.)
            else:
                self.assertLessEqual(np.linalg.norm(target[:3]), .05 + 1e-12)
                self.assertLessEqual(np.linalg.norm(target[3:]), np.radians(10) + 1e-12)

    def test_default_metadata_embeds_calibrations(self):
        config = load_config()
        self.assertEqual(len(recording_metadata(config)['calibrations']), 2)
        self.assertTrue(Path(config['home_file']).is_absolute())

    def test_wrong_robot_or_camera_rejected(self):
        arm = load_config()['arms'][0]
        for ip, serial in [('wrong', arm['serial']), (arm['ip'], 'wrong')]:
            with self.assertRaises(ValueError):
                load_calibration(arm['calibration'], ip, serial)

    def test_gravity_does_not_determine_yaw(self):
        a = base_basis([0, 0, -9.81], [1, 0, 0])
        b = base_basis([0, 0, -9.81], [0, 1, 0])
        np.testing.assert_allclose(a[:, 2], b[:, 2])
        self.assertFalse(np.allclose(a, b))

    def test_changed_installation_rejected(self):
        arm = load_config()['arms'][0]
        calibration = load_calibration(arm['calibration'])
        with self.assertRaises(RuntimeError):
            validate_installation(calibration, [0, 0, -9.81])

    def test_gestures_reject_mirrored_basis(self):
        with self.assertRaises(ValueError):
            t265_operator_basis([0, 0, -1], [1, 0, 0])
        np.testing.assert_allclose(np.linalg.det(t265_operator_basis([0, 0, -1], [-1, 0, 0])), 1.)

    def test_world_height_is_not_base_z(self):
        config = load_config()
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler('y', 90, degrees=True).as_matrix()
        transform[2, 3] = 1.
        for arm in config['arms']:
            arm['world_from_base'] = transform.tolist()
        workspace = DualWorkspace(config['arms'], {'enabled': True, 'minimum_world_z': .3})
        workspace.validate('left', [.5, 0, -.8, 0, 0, 0])
        with self.assertRaises(RuntimeError):
            workspace.validate('right', [.8, 0, .8, 0, 0, 0])

    def test_missing_world_transform_not_guessed(self):
        with self.assertRaises(ValueError):
            DualWorkspace(load_config()['arms'], {'enabled': True, 'minimum_world_z': .2})


class InputTests(unittest.TestCase):
    def packet(self):
        return {'arms': [dict(index=i, pose=dict(valid=True, tracker_confidence=3,
                         sample_timestamp=100., position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1])) for i in range(2)]}

    def test_separate_sensor_time_detects_frozen_pose(self):
        data = _make_teleop_sample(self.packet())
        self.assertEqual(set(decode_samples(data, load_config()['arms'], now=100.1)), {'left', 'right'})
        data['receive_timestamp'] = 100.3
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            decode_samples(data, load_config()['arms'], now=100.3)

    def test_one_bad_hand_rejects_pair(self):
        data = _make_teleop_sample(self.packet())
        data['tracker_confidence'][1] = 1
        with self.assertRaises(RuntimeError):
            decode_samples(data, load_config()['arms'], now=100.)

    def test_encoder_has_its_own_age(self):
        packet = self.packet()
        packet['arms'][0]['gripper'] = dict(valid=True, sample_timestamp=99., angle_deg=30)
        arms = load_config()['arms']
        arms[0]['encoder_port'] = '/dev/ttyUSB0'
        with self.assertRaisesRegex(RuntimeError, 'encoder'):
            decode_samples(_make_teleop_sample(packet), arms, now=100., require_encoders=True)


class IoTests(unittest.TestCase):
    def io(self):
        levels = [False, False]
        io = Mock()
        io.getStandardDigitalOutputNum.return_value = 2
        io.getStandardDigitalOutput.side_effect = lambda pin: levels[pin]
        def write(pin, value):
            levels[pin] = value
            return 0
        io.setStandardDigitalOutput.side_effect = write
        return io, levels

    def test_disabled_never_touches_io(self):
        io, _ = self.io()
        device = AuboDigitalGripper(io, {'type': 'disabled'})
        device.start()
        self.assertEqual(io.mock_calls, [])
        with self.assertRaises(RuntimeError):
            device.command(0)

    def test_single_pin_active_high_close(self):
        io, levels = self.io()
        device = AuboDigitalGripper(io, {'type': 'standard_digital', 'pin': 0, 'open_level': False})
        device.start()
        io.setStandardDigitalOutput.assert_not_called()
        self.assertEqual(device.command(0), 0)
        self.assertTrue(levels[0])
        self.assertEqual(device.command(1), 1)

    def test_two_pins_require_explicit_truth_table(self):
        with self.assertRaises(ValueError):
            AuboDigitalGripper(None, {'type': 'standard_digital', 'pins': [0, 1], 'open_levels': [False, True]})

    def test_suction_unknown_state_and_off_before_on(self):
        io, levels = self.io()
        device = AuboDigitalGripper(io, {'type': 'standard_digital', 'pins': [0, 1],
                                   'open_levels': [False, True], 'close_levels': [True, False]})
        device.start()
        self.assertEqual(device.state, -1)
        self.assertEqual(device.command(0), 0)
        device.command(1)
        self.assertEqual(levels, [False, True])
        self.assertEqual(io.setStandardDigitalOutput.call_args_list[-2].args, (0, False))
        self.assertEqual(io.setStandardDigitalOutput.call_args_list[-1].args, (1, True))

    def test_output_failure_is_not_claimed_success(self):
        io, _ = self.io()
        io.setStandardDigitalOutput.side_effect = None
        io.setStandardDigitalOutput.return_value = 7
        device = AuboDigitalGripper(io, {'type': 'standard_digital', 'pin': 0, 'open_level': False})
        device.start()
        with self.assertRaises(RuntimeError):
            device.command(0)
        self.assertIsNone(device.last_command)

    def test_independent_outputs_preserve_other_channel(self):
        io, levels = self.io()
        device = AuboDigitalGripper(io, dict(type='standard_digital', mode='independent', pins=[0, 1]))
        device.start()
        self.assertEqual(device.command(1), 1)
        self.assertEqual(levels, [True, False])
        self.assertEqual(device.command(3), 3)
        self.assertEqual(levels, [True, True])
        self.assertEqual(device.command(2), 2)
        self.assertEqual(levels, [False, True])
        with self.assertRaises(ValueError): device.command(4)


class KeyboardTests(unittest.TestCase):
    def test_held_key_toggles_only_once_per_press(self):
        from diffusion_policy.envs.realword.aubo_t265.keyboard import create_keyboard
        class Counter:
            def __init__(self):
                self.lock = Lock()
                self.key_count_map = defaultdict(int)
                self.key_press_list = []
        with patch.dict(sys.modules, {'diffusion_policy.envs.realword.keystroke_counter': SimpleNamespace(KeystrokeCounter=Counter)}):
            keyboard = create_keyboard()
        keyboard.on_press('v')
        keyboard.on_press('v')
        self.assertEqual(keyboard.key_press_list, ['v'])
        keyboard.on_release('v')
        keyboard.on_press('v')
        self.assertEqual(keyboard.key_press_list, ['v', 'v'])


class FakeRobot:
    def __init__(self, **kwargs):
        self.settings = kwargs
        self.is_ready = True
        self.pid = 1
        self.alive = True
        self.home_state = SimpleNamespace(value=0)
        self.calls = []
        self.now = time.time()
        self.pose = np.array([.5, 0, 0, 0, 0, 0])

    def is_alive(self): return self.alive
    def start(self, wait=True): pass
    def start_wait(self): pass
    def stop(self, wait=True): self.alive = False
    def stop_wait(self): pass
    def heartbeat(self): pass
    def hold(self): self.calls.clear()
    def return_home(self, target): self.home_state.value = 1
    def schedule_waypoint(self, pose, target_time, **kwargs):
        self.calls.append((pose.copy(), target_time, kwargs))
    def get_state(self):
        return dict(ActualTCPPose=self.pose.copy(), ActualQ=np.zeros(6), GripperState=1., robot_receive_timestamp=self.now)
    def get_all_state(self):
        latest = self.get_state()
        result = {key: np.stack([value] * 20) for key, value in latest.items()}
        result['robot_receive_timestamp'] = self.now - np.arange(19, -1, -1) * .02
        return result


class FakeCamera:
    is_ready = True
    n_cameras = 1
    cameras = {}

    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.recorded_paths = []

    def get(self, k):
        return {0: dict(timestamp=self.timestamp-np.arange(k-1, -1, -1)/30.,
                        color=np.full((k, 2, 2, 3), 128, dtype=np.uint8))}
    def start_recording(self, paths, start_time): self.recorded_paths = paths
    def stop_recording(self): pass
    def stop(self, wait=True): pass
    def stop_wait(self): pass


class EnvironmentTests(unittest.TestCase):
    def test_configured_servo_parameters_reach_both_controllers(self):
        config = load_config()
        settings = controller_kwargs(config)
        env = self.env(**settings)
        for robot in (env.left_robot, env.right_robot):
            for name, value in settings.items():
                self.assertEqual(robot.settings[name], value, name)
        self.assertEqual(env.left_robot.settings['servo_t'], .02)
        self.assertEqual(env.left_robot.settings['servo_a'], 3.5)
        self.assertIsNone(env.left_robot.settings['max_joint_speed'])

    def env(self, **kwargs):
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.AuboInterpolationController', FakeRobot):
            env = DualAuboRealEnv('left', 'right', shm_manager=object(), **kwargs)
        env.start()
        self.addCleanup(env.stop)
        return env

    def test_strict_disabled_gripper_action_dimension(self):
        env = self.env()
        with self.assertRaises(ValueError): env.exec_actions(np.zeros(14), [time.time() + 1])
        env.exec_actions(np.zeros(12), [time.time() + 1])
        self.assertEqual(len(env.left_robot.calls), 1)
        self.assertEqual(len(env.right_robot.calls), 1)

    def test_both_targets_validate_before_either_is_sent(self):
        workspace = Mock()
        workspace.validate.side_effect = [None, RuntimeError('right outside')]
        env = self.env(workspace=workspace)
        with self.assertRaises(RuntimeError): env.exec_actions(np.zeros(12), [time.time() + 1])
        self.assertEqual(env.left_robot.calls, [])
        self.assertEqual(env.right_robot.calls, [])

    def test_nan_and_nonbinary_gripper_rejected(self):
        settings = {side: dict(type='standard_digital', pin=0, open_level=False) for side in ('left', 'right')}
        env = self.env(gripper_configs=settings)
        action = np.zeros(14)
        action[6] = .5
        with self.assertRaises(ValueError): env.exec_actions(action, [time.time() + 1])
        action[6] = float('nan')
        with self.assertRaises(ValueError): env.exec_actions(action, [time.time() + 1])

    def test_homing_blocks_cartesian_actions(self):
        env = self.env()
        env.return_home(np.zeros(6), np.zeros(6))
        self.assertEqual(env.home_state, 'moving')
        with self.assertRaises(RuntimeError): env.exec_actions(np.zeros(12), [time.time() + 1])

    def test_one_dead_robot_invalidates_pair(self):
        env = self.env()
        env.right_robot.alive = False
        with self.assertRaises(RuntimeError): env.get_state()

    def test_second_arm_start_failure_cleans_up_first(self):
        with patch('diffusion_policy.envs.realword.dual_aubo_real_env.AuboInterpolationController', FakeRobot):
            env = DualAuboRealEnv('left', 'right', shm_manager=object())
        env.right_robot.start = Mock(side_effect=RuntimeError('startup failed'))
        with self.assertRaisesRegex(RuntimeError, 'startup failed'): env.start()
        self.assertFalse(env.left_robot.alive)
        self.assertFalse(env.right_robot.alive)

    def test_rgb_observations_align_and_record_images(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.env(output_dir=directory, record_images=True)
            env.realsense = FakeCamera(env.left_robot.now - .02)
            env.start_episode(time.time() - .2)
            env.exec_actions(np.zeros(12), [time.time() + .05])
            obs = env.get_obs()
            self.assertEqual(obs['timestamp'][-1], env.realsense.timestamp)
            self.assertEqual(obs['camera_0'].shape, (2, 2, 2, 3))
            self.assertTrue(env.end_episode())
            self.assertIn('img_camera_0', env.replay_buffer.data)
            self.assertEqual(env.realsense.recorded_paths, [str(Path(directory) / 'videos/0/0.mp4')])

    def test_stale_rgb_frame_is_rejected(self):
        env = self.env()
        env.realsense = FakeCamera(time.time() - 1)
        with self.assertRaisesRegex(RuntimeError, 'frames are stale'): env.get_obs()

    def test_feedback_age_is_not_publication_age(self):
        env = self.env()
        env.left_robot.now -= 1
        with self.assertRaisesRegex(RuntimeError, 'stale'): env.get_obs()

    def test_preview_anchor_and_clutch(self):
        config = load_config()
        workspace = DualWorkspace(config['arms'], config['workspace'])
        poses = {side: np.zeros(6) for side in ('left', 'right')}
        samples = {side: HandSample(np.zeros(3), Rotation.identity(), time.time()) for side in poses}
        session = TeleopSession(config, poses, workspace)
        session.resume(samples)
        session.tick(samples, .02)
        for pose in session.poses.values(): np.testing.assert_allclose(pose, 0, atol=1e-12)
        session.pause()
        samples['left'].position[:] = 1
        session.resume(samples)
        session.tick(samples, .02)
        for pose in session.poses.values(): np.testing.assert_allclose(pose, 0, atol=1e-12)

    def test_recording_writes_real_samples_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.env(output_dir=directory)
            env.start_episode(time.time())
            env.exec_actions(np.zeros(12), [env.start_time + .05])
            env.left_robot.now = env.right_robot.now = env.start_time + .15
            with patch('diffusion_policy.envs.realword.dual_aubo_real_env.time.time', return_value=env.start_time+.15):
                env.get_obs()
            self.assertTrue(env.end_episode())
            self.assertEqual(env.replay_buffer.n_episodes, 1)
            self.assertEqual(env.replay_buffer.data['action'].shape[1:], (12,))
            metadata = json.loads((Path(directory) / 'videos/0/metadata.json').read_text())
            self.assertEqual(metadata['gripper_state_kind'], 'disabled')
            env.start_episode()
            self.assertFalse(env.end_episode())
            self.assertEqual(env.replay_buffer.n_episodes, 1)
            self.assertFalse((Path(directory) / 'videos/1').exists())
            env.start_episode()
            env.drop_episode()
            self.assertEqual(env.replay_buffer.n_episodes, 1)

    def test_schema_mismatch_preserves_existing_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.env(output_dir=directory)
            for index in range(2):
                env.start_episode(time.time())
                env.exec_actions(np.zeros(12), [env.start_time + .05])
                env.left_robot.now = env.right_robot.now = env.start_time + .15
                with patch('diffusion_policy.envs.realword.dual_aubo_real_env.time.time', return_value=env.start_time+.15):
                    env.get_obs()
                if index == 0:
                    env.end_episode()
                else:
                    before = {k: v[:].copy() for k, v in env.replay_buffer.data.items()}
                    env.action_accumulator.action_buffer = np.zeros((2, 14))
                    with self.assertRaises(ValueError): env.end_episode()
                    for key, value in before.items(): np.testing.assert_array_equal(env.replay_buffer.data[key][:], value)
                    self.assertEqual(env.replay_buffer.n_episodes, 1)
                    env._recording = False  # Keep the failed episode directory for recovery.

    def io_config(self):
        return {side: dict(type='standard_digital', mode='independent', pins=pins)
                for side, pins in [('left', [0, 1]), ('right', [0])]}

    def test_v_b_n_independent_toggle_and_15_dim_actions(self):
        config = load_config()
        settings = self.io_config()
        env = self.env(gripper_configs=settings)
        self.assertEqual(env.action_dim, 15)
        session = TeleopSession(config, {side: env.get_state()[side]['ActualTCPPose'] for side in settings},
                               DualWorkspace(config['arms'], config['workspace']), env)
        samples = {side: HandSample(np.zeros(3), Rotation.identity(), time.time()) for side in settings}
        with self.assertRaises(RuntimeError): session.toggle_output('left', 0)
        session.resume(samples)
        session.toggle_output('left', 0)
        session.toggle_output('left', 1)
        session.toggle_output('right', 0)
        session.tick(samples, .02)
        self.assertEqual(env.left_robot.calls[-1][2]['gripper'], 2)
        self.assertEqual(env.right_robot.calls[-1][2]['gripper'], 0)
        session.toggle_output('left', 0)
        session.tick(samples, .02)
        self.assertEqual(env.left_robot.calls[-1][2]['gripper'], 3)
        self.assertEqual(env.right_robot.calls[-1][2]['gripper'], 0)

    def test_recording_expands_independent_outputs_to_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.env(output_dir=directory, gripper_configs=self.io_config())
            env.start_episode(time.time())
            action = np.zeros(15)
            action[[6, 7, 14]] = [1, 1, 0]
            env.exec_actions(action, [env.start_time + .05])
            env.left_robot.now = env.right_robot.now = env.start_time + .15
            with patch('diffusion_policy.envs.realword.dual_aubo_real_env.time.time', return_value=env.start_time+.15):
                env.get_obs()
            env.end_episode()
            self.assertEqual(env.replay_buffer.data['left_gripper_state'].shape[1:], (2,))
            self.assertEqual(env.replay_buffer.data['right_gripper_state'].shape[1:], (1,))
            np.testing.assert_array_equal(env.replay_buffer.data['action'][0][[6, 7, 14]], [1, 1, 0])
            self.assertEqual(len(env.metadata['action_layout']), 15)


if __name__ == '__main__':
    unittest.main()
