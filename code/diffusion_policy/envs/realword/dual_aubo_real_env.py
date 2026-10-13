"""Dual AUBO control, aligned observations, optional cameras and demonstrations.

Public TCP poses use metres and rotation vectors in each arm's own base frame.
Gripper actions, when explicitly configured on both arms, are binary 0/1.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import time
from multiprocessing.managers import SharedMemoryManager
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.envs.realword.aubo_interpolation_controller import AuboInterpolationController
from diffusion_policy.envs.realword.aubo_homing import joints
from diffusion_policy.common.timestamp_accumulator import TimestampObsAccumulator, TimestampActionAccumulator

ArmSelection = Literal["left", "right", "both"]


def _align(data, query):
    # 历史必须非空且按反馈时间排序，才能用时间戳重采样。
    timestamps = np.asarray(data["robot_receive_timestamp"], dtype=float)
    if timestamps.size == 0 or np.any(np.diff(timestamps) < 0):
        raise RuntimeError("Missing or unordered robot history")
    # 取不晚于查询时刻的最近帧；超出历史范围时夹到首尾帧。
    indices = np.clip(np.searchsorted(timestamps, query, side="right") - 1, 0, len(timestamps) - 1)
    return {key: np.asarray(value)[indices] for key, value in data.items()}


class DualAuboRealEnv:
    def __init__(self, left_robot_ip, right_robot_ip, left_robot_port=30004,
                 right_robot_port=30004, username="aubo", password="123456",
                 robot_frequency=100., state_frequency=20., servo_t=None,
                 max_pos_speed=.05, max_rot_speed=.3, speed_fraction=.2,
                 max_joint_step=.01, auto_power_on=False, launch_timeout=20.,
                 shm_manager=None, verbose=False, *, command_timeout=None,
                 frequency=10., n_obs_steps=2, output_dir=None,
                 camera_serial_numbers=None, obs_image_resolution=(224, 224),
                 video_capture_resolution=(640, 480), video_capture_fps=30,
                 record_images=False, enable_multi_cam_vis=False,
                 camera_labels=None, gripper_configs=None, workspace=None,
                 recording_metadata=None, reject_joint_jumps=False,
                 servo_a=.2, servo_v=.3, lookahead_time=.1, gain=200.,
                 max_joint_speed=.5, max_following_error=.2, controller_class=None):
        # 核对双柜地址与采样参数：观测频率不能高于机器人状态发布频率。
        if left_robot_ip == right_robot_ip:
            raise ValueError("Left and right AUBO cabinets need distinct IPs")
        if not np.isfinite(frequency) or not 0 < frequency <= state_frequency:
            raise ValueError("Require 0 < observation frequency <= state frequency")
        if not isinstance(n_obs_steps, int) or n_obs_steps < 1:
            raise ValueError("n_obs_steps must be positive")
        # 双臂同时启用或禁用末端输出；独立模式按配置的 DO 数量分配动作维数。
        gripper_configs = gripper_configs or {side: {"type": "disabled"} for side in ("left", "right")}
        enabled = [gripper_configs[side].get("type", "disabled") != "disabled" for side in ("left", "right")]
        if enabled[0] != enabled[1]:
            raise ValueError("Configure both grippers or disable both explicitly")
        self.grippers_enabled = all(enabled)
        self.gripper_configs = copy.deepcopy(gripper_configs)
        self.gripper_dims = {side: (len(gripper_configs[side]['pins'])
                            if gripper_configs[side].get('mode') == 'independent' else 1)
                            if self.grippers_enabled else 0 for side in ('left', 'right')}
        self.action_dim = 12 + sum(self.gripper_dims.values())
        self.frequency = float(frequency)
        self.n_obs_steps = n_obs_steps
        self.state_frequency = state_frequency
        self.max_pos_speed, self.max_rot_speed = max_pos_speed, max_rot_speed
        self.left_robot_ip, self.right_robot_ip = left_robot_ip, right_robot_ip
        self.launch_timeout = launch_timeout
        self.workspace = workspace
        self.record_images = record_images
        # 保存动作各列的含义与坐标约定，供示教数据读取端解释位姿和输出。
        self.metadata = copy.deepcopy(recording_metadata or {})
        self.metadata.update(action_dim=self.action_dim,
                             gripper_state_kind="digital_output_latch" if self.grippers_enabled else "disabled",
                             pose_convention="xyz_m+rotvec_rad_in_each_robot_base")
        self.metadata['action_layout'] = []
        for side in ('left', 'right'):
            self.metadata['action_layout'].extend(f'{side}_{axis}' for axis in ('x', 'y', 'z', 'rx', 'ry', 'rz'))
            if self.grippers_enabled:
                pins = gripper_configs[side].get('pins', [])
                names = ([f'{side}_DO{pin:02d}' for pin in pins]
                         if gripper_configs[side].get('mode') == 'independent' else [f'{side}_open'])
                self.metadata['action_layout'].extend(names)
        self.metadata['gripper_configs'] = copy.deepcopy(gripper_configs)
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir else None
        self.video_dir = self.output_dir / "videos" if self.output_dir else None
        self.replay_buffer = None
        self._recording = False
        self.obs_accumulator = self.action_accumulator = self.stage_accumulator = None
        self.start_time = None
        # 只负责释放自己创建的共享内存管理器；外部传入时由调用者管理。
        self._owns_shm_manager = shm_manager is None
        self.shm_manager = shm_manager or SharedMemoryManager()
        if self._owns_shm_manager:
            self.shm_manager.start()
        self._started = False
        self.realsense = self.multi_cam_vis = None
        try:
            # 左右控制器共用伺服参数，各自使用本臂 IP、端口及末端输出配置。
            common = dict(shm_manager=self.shm_manager, username=username, password=password,
                          frequency=robot_frequency, state_frequency=state_frequency, servo_t=servo_t,
                          max_pos_speed=max_pos_speed, max_rot_speed=max_rot_speed,
                          speed_fraction=speed_fraction, max_joint_step=max_joint_step,
                          auto_power_on=auto_power_on, launch_timeout=launch_timeout,
                          command_timeout=command_timeout, reject_joint_jumps=reject_joint_jumps,
                          servo_a=servo_a, servo_v=servo_v, lookahead_time=lookahead_time, gain=gain,
                          max_joint_speed=max_joint_speed, max_following_error=max_following_error,
                          verbose=verbose)
            # 默认保持 v1；v2 环境可显式注入控制器，避免改写模块全局变量。
            controller = AuboInterpolationController if controller_class is None else controller_class
            self.left_robot = controller(robot_ip=left_robot_ip,
                robot_port=left_robot_port, gripper_config=gripper_configs["left"], **common)
            self.right_robot = controller(robot_ip=right_robot_ip,
                robot_port=right_robot_port, gripper_config=gripper_configs["right"], **common)
            if self.output_dir:
                # 以追加模式打开数据集，不覆盖已经保存的 episode。
                from diffusion_policy.common.replay_buffer import ReplayBuffer
                self.output_dir.mkdir(parents=True, exist_ok=True)
                self.video_dir.mkdir(exist_ok=True)
                self.replay_buffer = ReplayBuffer.create_from_path(
                    str(self.output_dir / "replay_buffer.zarr"), mode="a")
            if camera_serial_numbers:
                # 相机是可选设备；启用时检查采集帧率能覆盖观测频率。
                if frequency > video_capture_fps:
                    raise ValueError("Observation frequency exceeds camera capture rate")
                self._create_cameras(camera_serial_numbers, obs_image_resolution,
                                     video_capture_resolution, video_capture_fps,
                                     enable_multi_cam_vis, camera_labels)
        except BaseException:
            # 构造失败也释放本对象创建的共享内存，再保留原始异常。
            if self._owns_shm_manager:
                self.shm_manager.shutdown()
                self._owns_shm_manager = False
            raise

    def _create_cameras(self, serials, obs_resolution, capture_resolution, fps, visualize, labels):
        # 延迟导入相机依赖，未配置相机时不加载采集和视频模块。
        from diffusion_policy.envs.realword.multi_realsense import MultiRealsense
        from diffusion_policy.envs.realword.video_recorder import VideoRecorder
        from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols
        from diffusion_policy.envs.realword.multi_camera_visualizer import MultiCameraVisualizer
        # 支持所有相机共用分辨率，或为每个序列号分别指定分辨率。
        if isinstance(obs_resolution[0], int):
            resolutions = [obs_resolution] * len(serials)
        else:
            resolutions = obs_resolution
        if len(resolutions) != len(serials):
            raise ValueError("Camera observation resolution count mismatch")
        transforms = []
        for resolution in resolutions:
            fn = get_image_transform(capture_resolution, resolution, bgr_to_rgb=True)
            def transform(data, fn=fn):
                # 默认参数固定本轮转换函数，避免闭包全部引用最后一台相机的配置。
                # 策略观测使用缩放后的 RGB 图像。
                data["color"] = fn(data["color"])
                return data
            transforms.append(transform)
        rw, rh, columns, rows = optimal_row_cols(len(serials), capture_resolution[0] / capture_resolution[1], (1280, 720))
        vis_fn = get_image_transform(capture_resolution, (rw, rh), bgr_to_rgb=False)
        def vis_transform(data):
            # 显示分支保留 BGR，并缩放到多相机窗口的单格尺寸。
            data["color"] = vis_fn(data["color"])
            return data
        # 历史帧用于时间对齐；视频录制器接收 BGR 原始采集图像。
        self.realsense = MultiRealsense(serial_numbers=serials, shm_manager=self.shm_manager,
            resolution=capture_resolution, capture_fps=fps, put_fps=fps, put_downsample=False,
            enable_color=True, enable_depth=False, get_max_k=max(30, self.n_obs_steps * 4),
            transform=transforms, vis_transform=vis_transform, verbose=False,
            video_recorder=VideoRecorder.create_h264(fps=fps, input_pix_fmt="bgr24", crf=21,
                                                    thread_type="FRAME", thread_count=3))
        if visualize:
            # 只有显式启用时才创建独立的多相机显示进程。
            self.multi_cam_vis = MultiCameraVisualizer(self.realsense, row=rows, col=columns,
                                                       rgb_to_bgr=False, camera_labels=labels)

    @property
    def is_ready(self):
        # 双臂及所有已配置的相机/显示组件都就绪，环境才允许使用。
        return bool(self._started and self.left_robot.is_ready and self.right_robot.is_ready
                    and (self.realsense is None or self.realsense.is_ready)
                    and (self.multi_cam_vis is None or self.multi_cam_vis.is_alive()))

    def ready_status(self):
        # 分别报告就绪标志与进程存活状态，便于定位启动卡在哪个组件。
        return dict(left_robot_ready=self.left_robot.is_ready, right_robot_ready=self.right_robot.is_ready,
                    left_robot_alive=self.left_robot.is_alive(), right_robot_alive=self.right_robot.is_alive(),
                    cameras_ready=self.realsense is None or self.realsense.is_ready)

    def print_ready_status(self, prefix="[DualAuboRealEnv]"):
        # 立即刷新输出，使启动等待期间也能看到当前状态。
        print(prefix, self.ready_status(), flush=True)

    def start(self, wait=True):
        # 重复启动直接返回；先标记已启动，便于部分启动失败时统一清理。
        if self._started:
            return
        self._started = True
        try:
            # 先异步启动所有组件，再按需统一等待，避免串行等待拖慢启动。
            if self.realsense:
                self.realsense.start(wait=False)
            self.left_robot.start(wait=False)
            self.right_robot.start(wait=False)
            if self.multi_cam_vis:
                self.multi_cam_vis.start(wait=False)
            if wait:
                self.start_wait()
        except BaseException:
            # 任一组件启动失败时停止已启动组件，不留下单臂控制进程。
            self.stop(wait=True)
            raise

    def start_wait(self):
        # Poll all components together so an already-ready arm receives
        # heartbeats while the other arm or RGB cameras are still starting.
        if not self.wait_until_ready(self.launch_timeout):
            raise TimeoutError(f"Dual AUBO startup timed out: {self.ready_status()}")

    def wait_until_ready(self, timeout_s=20., print_interval_s=1.):
        # 超时用单调时钟计量，避免系统时间调整影响等待期限。
        deadline = time.monotonic() + timeout_s
        printed = 0.
        while time.monotonic() < deadline:
            if self.is_ready:
                return True
            if self._started:
                # 等待中允许尚未就绪，但已退出的子进程应立即报错。
                self.assert_healthy(check_ready=False)
                for robot in (self.left_robot, self.right_robot):
                    if robot.is_ready:
                        robot.heartbeat()
            if time.monotonic() - printed > print_interval_s:
                self.print_ready_status()
                printed = time.monotonic()
            time.sleep(.02)
        return False

    def stop(self, wait=True):
        # 先结束当前录制；个别清理失败时仍继续停止其他组件。
        errors = []
        if self._recording:
            try:
                self.end_episode()
            except Exception as exc:
                errors.append(exc)
        if self._started:
            components = [self.left_robot, self.right_robot]
            if self.multi_cam_vis:
                components.append(self.multi_cam_vis)
            if self.realsense:
                components.append(self.realsense)
            # 先向全部组件发停止请求，再统一等待退出。
            for component in components:
                try:
                    component.stop(wait=False)
                except Exception as exc:
                    errors.append(exc)
            if wait:
                for component in components:
                    try:
                        component.stop_wait()
                    except Exception as exc:
                        errors.append(exc)
            self._started = False
        # 共享内存须在子进程停止后释放，外部管理器不在此关闭。
        if self._owns_shm_manager:
            self.shm_manager.shutdown()
            self._owns_shm_manager = False
        if errors:
            raise RuntimeError(f"Errors stopping dual AUBO: {errors}")

    def __enter__(self):
        # with/ExitStack 进入时启动环境，并返回可操作的环境对象。
        self.start()
        return self

    def __exit__(self, *_):
        # 正常退出或抛出异常时均执行环境清理。
        self.stop()

    def assert_healthy(self, check_ready=True):
        # 区分尚未启动与启动后异常退出，检查机器人及相机子进程。
        for robot in (self.left_robot, self.right_robot):
            if robot.pid is not None and not robot.is_alive():
                raise RuntimeError(f"An AUBO controller exited: {self.ready_status()}")
        if self.realsense:
            for camera in self.realsense.cameras.values():
                if camera.pid is not None and not camera.is_alive():
                    raise RuntimeError("RGB camera process exited")
        # 启动等待阶段可跳过就绪检查；运行中的调用默认必须全部就绪。
        if check_ready and not self.is_ready:
            raise RuntimeError(f"Dual AUBO is not ready: {self.ready_status()}")

    def get_state(self):
        # 返回真实反馈；使用实际接收时间判定新鲜度，不能用重复发布掩盖断流。
        self.assert_healthy()
        state = {"left": self.left_robot.get_state(), "right": self.right_robot.get_state()}
        for side, sample in state.items():
            age = time.time() - sample["robot_receive_timestamp"]
            if not np.isfinite(age) or not -.05 <= age <= .6:
                raise RuntimeError(f"{side} AUBO state is stale")
            if self.workspace:
                # 校验各臂基座系下的实际 TCP 是否仍在配置工作空间中。
                controller = self.left_robot if side == 'left' else self.right_robot
                if getattr(controller, 'supports_height_recovery', False):
                    self.workspace.validate_feedback(side, sample["ActualTCPPose"])
                else:
                    self.workspace.validate(side, sample["ActualTCPPose"])
        return state

    def get_robot_state(self):
        # 加上左右臂前缀，合并成原框架可读取的扁平状态字典。
        return {f"{side}_{key}": value for side, sample in self.get_state().items() for key, value in sample.items()}

    def get_obs(self):
        self.get_state()  # 重采样历史前，先确认当前反馈仍然有效。
        histories = {"left": self.left_robot.get_all_state(), "right": self.right_robot.get_all_state()}
        raw_cameras = {}
        if self.realsense:
            # Camera readiness means first frame, not a full history window.
            # Read only the history available in every camera; alignment below
            # already repeats the oldest frame when the requested time predates it.
            k = max(2, self.n_obs_steps * 4)
            for camera in self.realsense.cameras.values():
                k = min(k, camera.ring_buffer.count, camera.ring_buffer.get_max_k)
            if k < 1:
                raise RuntimeError("RGB camera has no buffered frames")
            raw_cameras = self.realsense.get(k=k)
        latest = [history["robot_receive_timestamp"][-1] for history in histories.values()]
        latest += [camera["timestamp"][-1] for camera in raw_cameras.values()]
        # 以最慢数据源的最新时刻为基准，按观测频率生成从旧到新的共同时间网格。
        reference = min(latest)
        query = reference - np.arange(self.n_obs_steps - 1, -1, -1) / self.frequency
        result = {"timestamp": query}
        for side, history in histories.items():
            # 两臂位姿、关节和输出状态使用同一组查询时间。
            sample = _align(history, query)
            result[side + "_robot_eef_pose"] = sample["ActualTCPPose"]
            result[side + "_robot_joint"] = sample["ActualQ"]
            result[side + "_gripper_state"] = self.decode_gripper(side, sample["GripperState"])
            result[side + "_robot_receive_timestamp"] = sample["robot_receive_timestamp"]
        for index, camera in raw_cameras.items():
            # 相机同样检查断流，并取不晚于对应查询时刻的最近图像。
            if time.time() - camera["timestamp"][-1] > .6:
                raise RuntimeError("RGB camera frames are stale")
            indices = np.clip(np.searchsorted(camera["timestamp"], query, side="right") - 1, 0, len(camera["timestamp"]) - 1)
            result[f"camera_{index}"] = camera["color"][indices]
        if self._recording:
            # 图像可仅录视频；启用 record_images 才把图像数组也写入观测数据集。
            values = {key: value for key, value in result.items() if key != "timestamp"
                      and (self.record_images or not key.startswith("camera_"))}
            values = {("img_" + key if key.startswith("camera_") else key): value for key, value in values.items()}
            self.obs_accumulator.put(values, query)
        return result

    def exec_actions(self, actions, timestamps, stages=None):
        # 单条动作统一转成批次；timestamps 是 Unix 秒，与观测/录制时间一致。
        actions = np.asarray(actions, dtype=float)
        timestamps = np.atleast_1d(np.asarray(timestamps, dtype=float))
        if actions.ndim == 1:
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] != self.action_dim or timestamps.ndim != 1 or len(actions) != len(timestamps):
            raise ValueError(f"Require (N,{self.action_dim}) actions and (N,) timestamps; grippers are {'enabled' if self.grippers_enabled else 'disabled'}")
        if not np.isfinite(actions).all() or not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) < 0):
            raise ValueError("Actions/timestamps must be finite and ordered")
        # 每条动作对应一个阶段标记；未提供时全部记为阶段 0。
        stages = np.zeros(len(actions), dtype=np.int64) if stages is None else np.asarray(stages)
        if stages.shape != (len(actions),) or not np.isfinite(stages).all():
            raise ValueError("Require one finite stage per action")
        self.assert_healthy()
        if self.home_state == "moving":
            raise RuntimeError("Cartesian targets are blocked during joint homing")
        # 动作布局为左位姿、左输出、右位姿、右输出，输出维数由配置决定。
        right_start = 6 + self.gripper_dims['left']
        poses = [(action[:6], action[right_start:right_start+6]) for action in actions]
        # 先校验整个批次的电平和工作空间，再开始向控制器提交目标。
        for action, (left, right) in zip(actions, poses):
            if self.grippers_enabled:
                for values in (action[6:right_start], action[right_start+6:]):
                    if not np.isin(values, [0., 1.]).all():
                        raise ValueError("Digital output actions must be 0 or 1")
            if self.workspace:
                self.workspace.validate("left", left)
                self.workspace.validate("right", right)
        # 过期目标不下发；有效目标按同一绝对时间分别进入左右控制器队列。
        future = timestamps > time.time()
        try:
            for action, (left, right), target_time, keep in zip(actions, poses, timestamps, future):
                if not keep:
                    continue
                if self.grippers_enabled:
                    self.left_robot.schedule_waypoint(left, float(target_time), gripper=self.encode_gripper('left', action[6:right_start]))
                    self.right_robot.schedule_waypoint(right, float(target_time), gripper=self.encode_gripper('right', action[right_start+6:]))
                else:
                    self.left_robot.schedule_waypoint(left, float(target_time))
                    self.right_robot.schedule_waypoint(right, float(target_time))
        except Exception:
            self.stop()
            raise
        # 只累计实际提交的未来动作，动作与 stage 使用相同时间网格。
        if self._recording and np.any(future):
            self.action_accumulator.put(actions[future], timestamps[future])
            self.stage_accumulator.put(stages[future], timestamps[future])

    @property
    def home_state(self):
        # 任一臂运动中即视为回零中；两臂都完成保持才视为回零完成。
        states = [self.left_robot.home_state.value, self.right_robot.home_state.value]
        if 1 in states:
            return "moving"
        return "holding" if states == [2, 2] else "idle"

    def decode_gripper(self, side, state):
        # 转成批次数组：binary 模式为单列，independent 模式按配置顺序拆位。
        # 位索引对应 pins 列表中的位置，不直接等于柜体 DO 引脚编号。
        values = np.asarray(state).reshape(-1)
        if self.gripper_configs[side].get('mode') != 'independent' or not self.grippers_enabled:
            return values[:, None]
        return np.stack([((values.astype(int) >> i) & 1).astype(float)
                         for i in range(self.gripper_dims[side])], axis=1)

    def encode_gripper(self, side, values):
        # 按 pins 的顺序把各路 0/1 打包为位掩码；binary 模式传单个状态值。
        if self.gripper_configs[side].get('mode') == 'independent':
            return float(sum(int(v) << i for i, v in enumerate(values)))
        return float(values[0])

    def return_home(self, left_joints, right_joints):
        # 验证左右各六个关节角后提交回零，轨迹由控制器按关节限速生成。
        left, right = joints(left_joints), joints(right_joints)
        self.assert_healthy()
        try:
            self.left_robot.return_home(left)
            self.right_robot.return_home(right)
        except Exception:
            # 任一臂提交失败时统一停止，避免另一臂单独继续回零。
            self.stop()
            raise

    def move_to_joint_positions(self, left_joints, right_joints, time_to_go=None):
        # 兼容原框架接口；time_to_go 不决定时长，时长由关节位移与回零限速计算。
        self.return_home(left_joints, right_joints)

    def is_at_joint_positions(self, left_joints, right_joints, tol=.01):
        # 两臂均须处于回零保持状态，且实际关节误差小于容差（弧度）。
        state = self.get_state()
        return self.home_state == "holding" and all(
            np.max(np.abs(state[side]["ActualQ"] - joints(target))) < tol
            for side, target in (("left", left_joints), ("right", right_joints)))

    def heartbeat(self):
        # 暂停/回零时没有笛卡尔目标，也需向两臂发送心跳以维持命令看门狗。
        self.assert_healthy()
        self.left_robot.heartbeat()
        self.right_robot.heartbeat()

    def hold(self):
        # 两臂进入保持并取消后续遥操轨迹；提交失败则停止整个环境。
        self.assert_healthy()
        try:
            self.left_robot.hold()
            self.right_robot.hold()
        except Exception:
            self.stop()
            raise

    def start_episode(self, start_time=None):
        # 必须配置数据目录且未在录制，才允许创建新 episode。
        if self.replay_buffer is None:
            raise RuntimeError("Recording requires output_dir")
        if self._recording:
            raise RuntimeError("An episode is already recording")
        self.assert_healthy()
        # 观测、动作与阶段共享起点和采样周期，录制频率独立于伺服频率。
        self.start_time = time.time() if start_time is None else float(start_time)
        self.obs_accumulator = TimestampObsAccumulator(self.start_time, 1 / self.frequency)
        self.action_accumulator = TimestampActionAccumulator(self.start_time, 1 / self.frequency)
        self.stage_accumulator = TimestampActionAccumulator(self.start_time, 1 / self.frequency)
        directory = self._episode_directory(self.replay_buffer.n_episodes)
        # 写入本段的安装/动作布局等元数据，目录已存在时拒绝覆盖。
        directory.mkdir(exist_ok=False)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if self.realsense:
            # 视频与数组累计使用同一个录制起点。
            self.realsense.start_recording([str(directory / f"{i}.mp4") for i in range(self.realsense.n_cameras)], self.start_time)
        self._recording = True

    def end_episode(self):
        # 未录制时直接返回；先停止视频，再整理本段有效数据。
        if not self._recording:
            return False
        if self.realsense:
            self.realsense.stop_recording()
        obs = self.obs_accumulator.data
        # 截取观测/动作/阶段共同拥有的长度，避免三者末尾错位。
        count = min(len(self.obs_accumulator), len(self.action_accumulator), len(self.stage_accumulator))
        directory = self._episode_directory(self.replay_buffer.n_episodes)
        if count:
            episode = {key: value[:count] for key, value in obs.items()}
            episode.update(timestamp=self.obs_accumulator.timestamps[:count],
                           action=self.action_accumulator.actions[:count],
                           stage=self.stage_accumulator.actions[:count])
            existing = self.replay_buffer.data
            if len(existing):
                # 追加前检查字段、形状和类型，防止混入不兼容的数据格式。
                if set(existing.keys()) != set(episode):
                    raise ValueError("Dataset keys changed; use a separate output directory")
                for key, value in episode.items():
                    if value.shape[1:] != existing[key].shape[1:] or value.dtype != existing[key].dtype:
                        raise ValueError(f"Dataset schema changed for {key}; use a separate output directory")
            # 格式校验失败时保留已有数据集；通过后才追加本段数据。
            self.replay_buffer.add_episode(episode, compressors="disk")
        else:
            # 没有共同有效帧时只清理本次创建的空 episode 目录。
            shutil.rmtree(directory)
        # 保存成功或空段清理完成后，复位录制状态及累计器。
        self._recording = False
        self.obs_accumulator = self.action_accumulator = self.stage_accumulator = None
        return bool(count)

    def drop_episode(self):
        # 录制中先结束本段；空段不应误删此前已经保存的 episode。
        if self._recording and not self.end_episode():
            return
        if self.replay_buffer is None or self.replay_buffer.n_episodes == 0:
            return
        # 撤销最后一段的数组记录，并删除对应视频/元数据目录。
        index = self.replay_buffer.n_episodes - 1
        directory = self._episode_directory(index)
        self.replay_buffer.drop_episode()
        if directory.is_dir():
            shutil.rmtree(directory)

    def _episode_directory(self, index):
        # 解析绝对路径并限定为 videos 的直接子目录，避免目录清理越界。
        root = self.video_dir.resolve()
        directory = (root / str(index)).resolve()
        if directory.parent != root:
            raise ValueError('Episode path escapes the video directory')
        return directory

    @staticmethod
    def _apply_delta(pose, translation, rotation):
        # 拷贝原位姿；平移为米，旋转向量为弧度，增量按该臂基座系解释。
        target = np.asarray(pose, dtype=float).copy()
        target[:3] += translation
        # 增量旋转左乘当前旋转，不能直接相加两个旋转向量。
        target[3:] = (Rotation.from_rotvec(rotation) * Rotation.from_rotvec(target[3:])).as_rotvec()
        return target

    def move_relative(self, arm: ArmSelection, translation=(0., 0., 0.), rotation=(0., 0., 0.),
                      duration=.5, max_translation_step=.01, max_rotation_step=np.deg2rad(5.)):
        # 校验选臂、有限三维增量、正时长和单步幅度，供小范围调试移动使用。
        if arm not in ("left", "right", "both"):
            raise ValueError("Unknown arm selection")
        translation, rotation = np.asarray(translation), np.asarray(rotation)
        if translation.shape != (3,) or rotation.shape != (3,) or not np.isfinite([translation, rotation]).all():
            raise ValueError("Translation/rotation must be finite 3-vectors")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be positive")
        if np.linalg.norm(translation) > max_translation_step or np.linalg.norm(rotation) > max_rotation_step:
            raise ValueError("Relative movement exceeds configured step")
        state = self.get_state()
        # 从实际反馈构造目标；未选中的臂保持当前实际位姿。
        poses = []
        for side in ("left", "right"):
            pose = state[side]["ActualTCPPose"]
            poses.append(self._apply_delta(pose, translation, rotation) if arm in (side, "both") else pose)
        if self.grippers_enabled:
            # 沿用当前输出锁存，避免移动接口附带改变末端执行器状态。
            action = np.concatenate((poses[0], self.decode_gripper('left', state['left']['GripperState'])[0],
                                     poses[1], self.decode_gripper('right', state['right']['GripperState'])[0]))
        else:
            action = np.concatenate(poses)
        # 复用统一动作入口，为目标保留至少 50 ms 的调度时间。
        self.exec_actions(action, time.time() + max(duration, .05))
        return tuple(poses)
