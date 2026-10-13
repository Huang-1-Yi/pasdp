"""
Dual-arm Franka real environment.

Both arms are served by ONE two-arm zerorpc server
(launch_franka_interface_server_two_franka.py), which exposes the left arm
under the plain method names and the right arm under the same names with a
``1`` suffix. ``left_robot_ip``/``right_robot_ip`` therefore point to the same
server host (default port 4242); the left/right distinction is made by
``left_arm_suffix=''`` vs ``right_arm_suffix='1'`` on the controllers.

Shared cameras, unified 14-dim action. Mirrors DualPiperRealEnv's interface but
drives FrankaInterpolationController (Cartesian impedance control) on each arm.

Action format (14-dim):
    [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_gripper,
     right_x, right_y, right_z, right_rx, right_ry, right_rz, right_gripper]

Franka gripper is binary: 0.08 → open, anything else → close.

State keys (obs):
    left_robot_eef_pose, left_robot_joint, left_gripper_width,
    right_robot_eef_pose, right_robot_joint, right_gripper_width,
    timestamp
"""

import math
import time
import shutil
import pathlib
import numpy as np
from multiprocessing.managers import SharedMemoryManager
from diffusion_policy.envs.realword.franka_interpolation_controller import FrankaInterpolationController
from diffusion_policy.envs.realword.multi_realsense import MultiRealsense, SingleRealsense
from diffusion_policy.envs.realword.video_recorder import VideoRecorder
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    TimestampRGBpcAccumulator,
)
from diffusion_policy.envs.realword.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols


class DualFrankaRealEnv:
    """
    Dual-arm Franka environment with shared cameras and unified action space.

    Usage:
        env = DualFrankaRealEnv(
            output_dir='data/dual_franka_demo',
            left_robot_ip='172.16.0.1',     # same two-arm server host
            right_robot_ip='172.16.0.1',    # same two-arm server host
        )
        env.start()
        obs = env.get_obs()
        env.exec_actions(actions, timestamps)  # actions: (14,)
        env.start_episode(start_time)
        env.end_episode()
        env.stop()
    """

    def __init__(self,
            output_dir,
            left_robot_ip,
            right_robot_ip,
            left_robot_port=4242,
            right_robot_port=4242,
            left_arm_suffix='',
            right_arm_suffix='1',
            frequency=10,
            n_obs_steps=2,
            obs_image_resolution=(640, 480),
            max_obs_buffer_size=30,
            camera_serial_numbers=None,
            camera_labels=None,
            obs_float32=False,
            max_pos_speed=0.25,
            max_rot_speed=0.6,
            robot_obs_latency=0.0001,
            init_joints=False,
            left_joints_init=None,
            right_joints_init=None,
            video_capture_fps=30,
            video_capture_resolution=(1280, 720),
            record_raw_video=True,
            record_images=True,
            thread_per_video=2,
            video_crf=21,
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1280, 720),
            shm_manager=None,
            verbose=False):
        assert frequency <= video_capture_fps
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        try:
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')
        except (AssertionError, KeyError):
            shutil.rmtree(zarr_path, ignore_errors=True)
            replay_buffer = ReplayBuffer.create_from_path(
                zarr_path=zarr_path, mode='a')

        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if camera_serial_numbers is None:
            camera_serial_numbers = SingleRealsense.get_connected_devices_serial()

        # Support per-camera obs resolution: single tuple or list of tuples
        n_cameras = len(camera_serial_numbers)
        if isinstance(obs_image_resolution, (list, tuple)) and isinstance(
                obs_image_resolution[0], int):
            obs_res_list = [obs_image_resolution] * n_cameras
        else:
            obs_res_list = list(obs_image_resolution)

        def _make_transform(out_res):
            tf = get_image_transform(
                input_res=video_capture_resolution,
                output_res=out_res, bgr_to_rgb=True)
            if obs_float32:
                return lambda x, _tf=tf: _tf(x).astype(np.float32) / 255
            return tf

        color_transforms = [_make_transform(r) for r in obs_res_list]

        def _make_transform_fn(tf):
            def _fn(data):
                data['color'] = tf(data['color'])
                return data
            return _fn

        transform = [_make_transform_fn(tf) for tf in color_transforms]

        rw, rh, col, row = optimal_row_cols(
            n_cameras=n_cameras,
            in_wh_ratio=obs_res_list[0][0] / obs_res_list[0][1],
            max_resolution=multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transform = None
        recording_fps = video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not record_raw_video:
            recording_transform = transform
            recording_fps = frequency
            recording_pix_fmt = 'rgb24'

        video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=video_crf,
            thread_type='FRAME',
            thread_count=thread_per_video)

        realsense = MultiRealsense(
            serial_numbers=camera_serial_numbers,
            shm_manager=shm_manager,
            resolution=video_capture_resolution,
            capture_fps=video_capture_fps,
            put_fps=video_capture_fps,
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transform,
            video_recorder=video_recorder,
            verbose=False
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row, col=col,
                rgb_to_bgr=False,
                camera_labels=camera_labels or {
                    0: 'global', 1: 'left wrist', 2: 'right wrist'},
            )

        # ── Two Franka controllers (Cartesian impedance) ──
        # Joint-space init mirrors RealEnvFranka: when init_joints is True each
        # arm moves to its joints_init via move_to_joint_positions at startup.
        default_j_init = np.array(
            [-0.165, -0.059, 0.167, -1.693, 0.002, 1.642, 0.751])

        def _resolve_joints_init(js):
            if not init_joints:
                return None
            if js is not None:
                js = np.array(js, dtype=np.float64)
                assert js.shape == (7,)
                return js
            return default_j_init.copy()

        left_joints_init = _resolve_joints_init(left_joints_init)
        right_joints_init = _resolve_joints_init(right_joints_init)

        left_robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=left_robot_ip,
            robot_port=left_robot_port,
            frequency=200,
            Kx_scale=1.0,
            Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
            joints_init=left_joints_init,
            joints_init_duration=3.0,
            verbose=verbose,
            receive_latency=robot_obs_latency,
            arm_suffix=left_arm_suffix,
        )
        right_robot = FrankaInterpolationController(
            shm_manager=shm_manager,
            robot_ip=right_robot_ip,
            robot_port=right_robot_port,
            frequency=200,
            Kx_scale=1.0,
            Kxd_scale=np.array([2.0, 1.5, 2.0, 1.0, 1.0, 1.0]),
            joints_init=right_joints_init,
            joints_init_duration=3.0,
            verbose=verbose,
            receive_latency=robot_obs_latency,
            arm_suffix=right_arm_suffix,
        )

        self.realsense = realsense
        self.left_robot = left_robot
        self.right_robot = right_robot
        self.multi_cam_vis = multi_cam_vis
        self.video_capture_fps = video_capture_fps
        self.frequency = frequency
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.replay_buffer = replay_buffer
        self.record_images = record_images
        self.last_realsense_data = None
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        self.rgbpc_accumulator = None
        self.start_time = None

    # ======== start-stop API =============

    @property
    def is_ready(self):
        return (self.realsense.is_ready
                and self.left_robot.is_ready
                and self.right_robot.is_ready)

    def ready_status(self):
        return {
            'realsense_ready': bool(self.realsense.is_ready),
            'left_robot_ready': bool(self.left_robot.is_ready),
            'right_robot_ready': bool(self.right_robot.is_ready),
            'left_robot_alive': bool(self.left_robot.is_alive()),
            'right_robot_alive': bool(self.right_robot.is_alive()),
            'multi_cam_vis_alive': (
                bool(self.multi_cam_vis.is_alive())
                if self.multi_cam_vis is not None else True
            ),
            'env_ready': bool(self.is_ready),
        }

    def print_ready_status(self, prefix='[DualFrankaRealEnv]'):
        status = self.ready_status()
        missing = [
            name for name, ready in [
                ('realsense', status['realsense_ready']),
                ('left_robot', status['left_robot_ready']),
                ('right_robot', status['right_robot_ready']),
                ('left_robot_process', status['left_robot_alive']),
                ('right_robot_process', status['right_robot_alive']),
                ('multi_cam_vis', status['multi_cam_vis_alive']),
            ]
            if not ready
        ]
        print(
            f"{prefix} "
            f"realsense_ready={status['realsense_ready']} "
            f"left_robot_ready={status['left_robot_ready']} "
            f"right_robot_ready={status['right_robot_ready']} "
            f"left_robot_alive={status['left_robot_alive']} "
            f"right_robot_alive={status['right_robot_alive']} "
            f"multi_cam_vis_alive={status['multi_cam_vis_alive']} "
            f"env_ready={status['env_ready']} "
            f"missing={','.join(missing) if missing else 'none'}"
        )

    def wait_until_ready(self, timeout_s=10.0, print_interval_s=1.0):
        deadline = time.monotonic() + timeout_s
        next_print = 0.0
        while time.monotonic() < deadline:
            if self.is_ready:
                self.print_ready_status('[DualFrankaRealEnv] ready')
                return True
            now = time.monotonic()
            if now >= next_print:
                self.print_ready_status('[DualFrankaRealEnv] waiting')
                next_print = now + print_interval_s
            time.sleep(0.05)
        self.print_ready_status('[DualFrankaRealEnv] timeout')
        return False

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.left_robot.start(wait=False)
        self.right_robot.start(wait=False)
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        if self.is_ready or self.obs_accumulator is not None:
            self.end_episode()
        else:
            self.print_ready_status('[DualFrankaRealEnv] stop before ready')
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop(wait=False)
        self.left_robot.stop(wait=False)
        self.right_robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.left_robot.start_wait()
        self.right_robot.start_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        self.left_robot.stop_wait()
        self.right_robot.stop_wait()
        self.realsense.stop_wait()
        if self.multi_cam_vis is not None:
            self.multi_cam_vis.stop_wait()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========

    def get_obs(self) -> dict:
        """Return observation dict with left_/right_ prefixed robot state keys."""
        assert self.is_ready

        k = math.ceil(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        self.last_realsense_data = self.realsense.get(
            k=k, out=self.last_realsense_data)

        left_robot_data = self.left_robot.get_all_state()
        right_robot_data = self.right_robot.get_all_state()

        dt = 1 / self.frequency
        last_timestamp = np.max(
            [x['timestamp'][-1] for x in self.last_realsense_data.values()])
        obs_align_timestamps = last_timestamp - (
            np.arange(self.n_obs_steps)[::-1] * dt)

        # Camera obs
        camera_obs = dict()
        for camera_idx, value in self.last_realsense_data.items():
            this_timestamps = value['timestamp']
            this_idxs = []
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # Robot obs — align timestamps
        def _align_robot_data(robot_data, query_timestamps=obs_align_timestamps):
            """Resample robot_data onto query_timestamps.

            Each output row is the last sample received strictly before the
            corresponding query time (index 0 if none is that old yet).
            """
            robot_timestamps = robot_data['robot_receive_timestamp']
            # Ring buffer returns samples oldest -> newest, so it is sorted.
            this_idxs = np.searchsorted(
                robot_timestamps, query_timestamps, side='left') - 1
            this_idxs = np.clip(this_idxs, 0, len(robot_timestamps) - 1)
            return {k: v[this_idxs] for k, v in robot_data.items()}

        left_aligned = _align_robot_data(left_robot_data)
        right_aligned = _align_robot_data(right_robot_data)

        robot_obs = {}
        robot_obs['left_robot_eef_pose'] = left_aligned['ActualTCPPose']
        robot_obs['left_robot_joint'] = left_aligned['ActualQ']
        robot_obs['left_gripper_width'] = left_aligned['Actualgripper']
        robot_obs['right_robot_eef_pose'] = right_aligned['ActualTCPPose']
        robot_obs['right_robot_joint'] = right_aligned['ActualQ']
        robot_obs['right_gripper_width'] = right_aligned['Actualgripper']
        robot_obs['timestamp'] = left_aligned['robot_receive_timestamp']

        # Accumulate camera frames (all cameras, all k buffered frames)
        if self.rgbpc_accumulator is not None:
            data_dict = {}
            cam_timestamps = None
            for cam_idx in sorted(self.last_realsense_data.keys()):
                cdata = self.last_realsense_data[cam_idx]
                data_dict[f'camera_{cam_idx}'] = cdata['color']  # (k, H, W, 3)
                if cam_timestamps is None:
                    cam_timestamps = cdata['timestamp']
            self.rgbpc_accumulator.put(data_dict, cam_timestamps)

        # Accumulate for recording — raw-rate robot state, on a shared timeline.
        # The two arms run in separate processes with independent ring buffers,
        # so their sample counts and sample times never match exactly. Use the
        # left arm's timestamps as the reference and resample the right arm onto
        # them, so every recorded row is a consistent snapshot of both arms.
        if self.obs_accumulator is not None:
            ref_timestamps = left_robot_data['robot_receive_timestamp']
            right_on_ref = _align_robot_data(right_robot_data, ref_timestamps)
            robot_obs_raw = {}
            robot_obs_raw['left_robot_eef_pose'] = left_robot_data['ActualTCPPose']
            robot_obs_raw['left_robot_joint'] = left_robot_data['ActualQ']
            robot_obs_raw['left_gripper_width'] = left_robot_data['Actualgripper']
            robot_obs_raw['right_robot_eef_pose'] = right_on_ref['ActualTCPPose']
            robot_obs_raw['right_robot_joint'] = right_on_ref['ActualQ']
            robot_obs_raw['right_gripper_width'] = right_on_ref['Actualgripper']
            self.obs_accumulator.put(robot_obs_raw, ref_timestamps)

        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def exec_actions(self, actions, timestamps, stages=None):
        """
        Send 14-dim actions to both arms.

        Parameters
        ----------
        actions : np.ndarray
            Shape (N, 14): [left_xyz_rxyz_gripper, right_xyz_rxyz_gripper].
        timestamps : np.ndarray
            Target wall-clock times.
        stages : np.ndarray or None
        """
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_actions = actions[is_new]
        new_timestamps = timestamps[is_new]
        new_stages = stages[is_new]

        for i in range(len(new_actions)):
            action = new_actions[i]
            # Left arm: dims 0-6 (6 pose + gripper width)
            left_pose = action[0:7]
            self.left_robot.schedule_waypoint(
                pose=left_pose, target_time=new_timestamps[i])
            # Right arm: dims 7-13 (6 pose + gripper width)
            right_pose = action[7:14]
            self.right_robot.schedule_waypoint(
                pose=right_pose, target_time=new_timestamps[i])

        if self.action_accumulator is not None:
            self.action_accumulator.put(new_actions, new_timestamps)
        if self.stage_accumulator is not None:
            self.stage_accumulator.put(new_stages, new_timestamps)

    def get_robot_state(self):
        """Return dict with left/right arm states (latest values)."""
        left = self.left_robot.get_state()
        right = self.right_robot.get_state()
        return {
            'left_ActualTCPPose': left['ActualTCPPose'],
            'left_Actualgripper': left['Actualgripper'],
            'right_ActualTCPPose': right['ActualTCPPose'],
            'right_Actualgripper': right['Actualgripper'],
        }

    def move_to_joint_positions(self, left_joints, right_joints, time_to_go=5.0):
        """Joint-space move both arms to target joint positions (non-blocking).

        Fires a MOVE_TO_JOINT command to each arm's controller process and
        returns immediately. The controllers execute the move asynchronously
        while keeping their ring buffers streaming fresh state, so the teleop
        loop and recording stay responsive. Use :meth:`is_at_joint_positions`
        to poll for arrival.
        """
        left_joints = np.asarray(left_joints, dtype=np.float64)
        right_joints = np.asarray(right_joints, dtype=np.float64)
        assert left_joints.shape == (7,)
        assert right_joints.shape == (7,)

        self.left_robot.move_to_joint_positions(left_joints, time_to_go)
        self.right_robot.move_to_joint_positions(right_joints, time_to_go)

    def is_at_joint_positions(self, left_joints, right_joints, tol=0.02):
        """Return True once both arms have reached the target joint positions."""
        lq = self.left_robot.get_state()['ActualQ']
        rq = self.right_robot.get_state()['ActualQ']
        return (np.allclose(lq, left_joints, atol=tol) and
                np.allclose(rq, right_joints, atol=tol))

    # ========= recording API ===========

    def start_episode(self, start_time=None):
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time
        assert self.is_ready
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None
        self.rgbpc_accumulator = None

        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = [
            str(this_video_dir.joinpath(f'{i}.mp4').absolute())
            for i in range(n_cameras)]

        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths, start_time=start_time)

        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time, dt=1 / self.frequency)
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency)
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time, dt=1 / self.frequency)
        self.rgbpc_accumulator = TimestampRGBpcAccumulator(
            start_time=start_time, dt=1 / self.frequency) \
            if self.record_images else None
        print(f'Episode {episode_id} started!')

    def end_episode(self):
        if not self.is_ready:
            self.print_ready_status('[DualFrankaRealEnv] end_episode skipped')
            return
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps
            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            img_data = self.rgbpc_accumulator.data if self.rgbpc_accumulator else None

            # Truncate all to the same length (shortest accumulator)
            lens = [len(obs_timestamps), len(action_timestamps), len(stages)]
            if img_data:
                for v in img_data.values():
                    if len(v) > 0:
                        lens.append(len(v))
                        break
            n_steps = min(lens)
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                if img_data:
                    for key, value in img_data.items():
                        episode[f'img_{key}'] = value[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]

                try:
                    self.replay_buffer.add_episode(episode,
                                                   compressors='disk')
                except AssertionError:
                    zarr_path = str(self.output_dir.joinpath(
                        'replay_buffer.zarr').absolute())
                    self.replay_buffer = ReplayBuffer.create_from_path(
                        zarr_path=zarr_path, mode='w')
                    self.replay_buffer.add_episode(episode,
                                                   compressors='disk')

                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None
            self.rgbpc_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
