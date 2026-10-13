#!/usr/bin/env python3
"""双臂 Aubo 遥操 v4：按 demo_dual_udp_teleop_franka_v3.py 的流程组织。

初始化输入/环境 → 捕获锚点 → 读取观测 → 处理按键 → 计算左右目标
→ 软阻尼限位与抗饱和 → env.set_teleop_target → 等待下一周期。

相比 v3，v4 针对机械臂触及下限位时的抖动问题集成了两项核心优化：
1. 回差死区与恢复容差 (Hysteresis Deadband & Recovery Margin)：
   引入 recovery_margin（默认 3mm）与迟滞回线，正常物理跟随误差不触发 _height_recovery，
   彻底消除控制目标截断与底层保护强行抬升的高频乒乓振荡。
2. 软限位平滑阻尼与抗饱和 (Soft Boundary Damping & Anti-Windup)：
   在限位上方 15mm 缓冲区内对向下速度执行 S-curve 阻尼平滑衰减（零 Jerk 冲击）；
   并在限位处持续下压时同步截断映射器位姿，消除反向抬手时的延迟粘滞。

E 开关遥操，Z 重新锚定，T 锁定旋转，R 开始录制并开启遥操，
S 保存并暂停，H 六关节回零，P 左臂完成两点/I/O 路径、右臂缓慢回 home，双臂到位后自动保存并暂停，
Space 增加 stage，Q/Esc 退出。
A/B 推进左 DO00/DO01 四步循环，C 切换右 DO00，1/2/3 切换相机。
左 DO00：0 平铺、1 弯折；左 DO01：0 吸盘关闭、1 吸盘打开。
默认只读预览；--enable-motion 启动伺服，--enable-io 启用已配置的柜体输出。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.envs.realword.aubo_homing import load_home, LockedCartesianHome
from diffusion_policy.envs.realword.aubo_t265.config import (
    DEFAULT_CONFIG, load_config, controller_kwargs, recording_metadata,
)
from diffusion_policy.envs.realword.aubo_t265.runtime import make_mappers
from diffusion_policy.envs.realword.aubo_t265.workspace_v4 import DualWorkspaceV4


DEFAULT_LEFT_POSE_FILES = tuple(DEFAULT_CONFIG.parent / name for name in (
    'left_pose_20261012_144317_446652.json',
    'left_pose_20261012_144604_163695.json',
))
# P 路径倍率同时用于规划时长和 TCP 速度上限；右臂返回 home 保持默认 1 倍。
P_POINT_SPEED_SCALE = 2.
P_RETURN_HOME_SPEED_SCALE = 4.
V4_HOME_SPEED_SCALE = 4.  # H 原来使用 2 倍归位速度，本次再乘 2。


def _load_left_waypoints(paths, arm, workspace):
    """Validate both saved points before motion; retain the existing orientation lock."""
    if len(paths) != 2:
        raise ValueError('P requires exactly two recorded left poses')
    waypoints = []
    for path in paths:
        data = json.loads(Path(path).expanduser().read_text(encoding='utf-8'))
        if (data.get('units') != 'radians' or data.get('tcp_translation_units') != 'metres'
                or data.get('tcp_frame') != 'robot_base'):
            raise ValueError(f'Invalid recorded pose units/frame: {path}')
        matches = [entry for entry in data['arms'] if entry.get('side') == 'left'
                   and entry.get('ip') == arm['ip'] and entry.get('serial') == arm['serial']]
        if len(matches) != 1:
            raise ValueError(f'Recorded P pose does not match the left arm: {path}')
        q = np.asarray(matches[0]['joint_positions_rad'], dtype=float)
        rpy = np.asarray(matches[0]['tcp_pose_aubo_rpy'], dtype=float)
        if any(value.shape != (6,) or not np.isfinite(value).all() for value in (q, rpy)):
            raise ValueError(f'Recorded P pose must contain six finite joint/TCP values: {path}')
        pose = np.r_[rpy[:3], Rotation.from_euler('xyz', rpy[3:]).as_rotvec()]
        # Reject an out-of-bounds recorded position instead of silently moving to a clipped point.
        workspace.validate('left', pose, target=True)
        target = workspace.limit_target('left', pose)
        angle = (Rotation.from_rotvec(target[3:]) * Rotation.from_rotvec(pose[3:]).inv()).magnitude()
        if angle >= np.deg2rad(1.):
            raise ValueError(f'Recorded P orientation differs from the fixed orientation by >=1 degree: {path}')
        waypoints.append((q.copy(), target))
    return waypoints


def _load_home_waypoint(path, arm, workspace):
    """Use the selected arm's existing home TCP as a fixed-orientation XYZ target."""
    side = arm['side']
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('units') != 'radians':
        raise ValueError('Recorded home must be in radians')
    matches = [entry for entry in data['arms'] if entry['ip'] == arm['ip']
               and entry['serial'] == arm['serial']]
    if len(matches) != 1:
        raise ValueError(f'Recorded P return home does not match the {side} arm')
    rpy = np.asarray(matches[0]['tcp_pose_aubo_rpy'], dtype=float)
    if rpy.shape != (6,) or not np.isfinite(rpy).all():
        raise ValueError('Recorded P return home requires six finite TCP values')
    pose = np.r_[rpy[:3], Rotation.from_euler('xyz', rpy[3:]).as_rotvec()]
    workspace.validate(side, pose, target=True)
    return load_home(path, arm['ip'], arm['serial']), workspace.limit_target(side, pose)


class _LeftPoseSequence:
    """Tick a bounded XYZ path; advance only after five fresh, settled RTDE samples."""
    def __init__(self, waypoints, state, max_speed, home_waypoint=None,
                 point_speed_scale=1., home_speed_scale=1.):
        if not np.isfinite(max_speed) or max_speed <= 0:
            raise ValueError('P max speed must be finite and positive')
        if any(not np.isfinite(scale) or scale <= 0 for scale in (point_speed_scale, home_speed_scale)):
            raise ValueError('P speed scales must be finite and positive')
        self.waypoints = list(waypoints)
        self.returning_home = home_waypoint is not None
        if self.returning_home:
            self.waypoints.append(home_waypoint)
        self.max_speed = max_speed
        self.point_speed_scale = float(point_speed_scale)
        self.home_speed_scale = float(home_speed_scale)
        self.index = 0
        self.done = False
        self._start_segment(state)

    def _start_segment(self, state):
        q, pose = self.waypoints[self.index]
        speed_scale = self.home_speed_scale if self.returning_home and self.index == 2 else self.point_speed_scale
        self.plan = LockedCartesianHome(state['ActualQ'], q, state['ActualTCPPose'], pose,
                                        speed_scale=speed_scale)
        distance = np.linalg.norm(pose[:3] - state['ActualTCPPose'][:3])
        # Quintic peak speed is 1.875 * distance / duration; do not use unlimited teleop speed.
        self.plan.duration = max(self.plan.duration, 1.875 * distance / (self.max_speed * speed_scale))
        self.plan.deadline = time.monotonic() + self.plan.duration * 2. + 10.
        self.io_deadline = time.monotonic() + 2.
        self.last_target = self.plan.start_pose.copy()

    @property
    def output_levels(self):
        # State 3 through both points, state 4 on return, state 1 after left home settles.
        if self.returning_home and self.done:
            return (0., 0.)
        return (1., 0.) if self.returning_home and self.index == 2 else (1., 1.)

    @property
    def label(self):
        return 'home' if self.returning_home and self.index == 2 else f'point {self.index + 1}/2'

    def tick(self, state, dt, io_ready=True):
        if not io_ready:
            if time.monotonic() > self.io_deadline:
                raise RuntimeError('P output latch not confirmed within 2 seconds; stopping both arms')
            return self.last_target.copy()  # Keep recording/keyboard alive while IO settles.
        if not self.done and self.plan.check_pose(
                state['ActualTCPPose'], state['ActualTCPSpeed'], state['robot_receive_timestamp']):
            if self.index + 1 == len(self.waypoints):
                self.done = True
                if self.returning_home:
                    self.io_deadline = time.monotonic() + 2.
                    self.last_target = self.plan.target_pose.copy()
            else:
                previous_target = self.plan.target_pose.copy()
                self.index += 1
                self._start_segment(state)
                if self.returning_home and self.index == 2:
                    # Let the caller request state 4 first; no home motion until the next latch check.
                    self.last_target = previous_target
                    return previous_target
        if self.done:
            return self.plan.target_pose.copy()
        target = self.plan.sample(dt)
        self.plan.accepted(dt)
        self.last_target = target.copy()
        return target


# ===========================================================================
# 锚点与目标映射（对应 Franka 的 _capture_teleop_anchor / _target_from_anchor）
# ===========================================================================
def _capture_teleop_anchor(poses, samples, mappers):
    """把两只手的当前姿态与各臂的实际 TCP 对齐，不改变安装标定。"""
    anchors = {}
    for side in ('left', 'right'):
        # 独立拷贝本臂 TCP，以此作为手部相对运动映射到机器人目标的起点。
        sample, pose = samples[side], np.asarray(poses[side], dtype=float).copy()
        # anchor 同步重置映射器内部的参考姿态及滤波/限速状态。
        mappers[side].anchor(sample.position, sample.orientation, pose)
        # 保存便于查看的锚点快照；后续计算实际使用 mapper 内部的锚点。
        anchors[side] = dict(robot=pose, hand_position=sample.position.copy(),
                             hand_orientation=sample.orientation)
    return anchors


def _target_from_anchor(mapper, sample, dt, lock_rot=False):
    """映射器执行坐标变换、轴符号与滤波，实时人控不做速度约束。

    锁旋转时用锚点的手部朝向，目标保持在锚点朝向；
    主循环在锁定/解锁时重新锚定，避免切换时向旧手部旋转目标追赶。
    """
    # 锁旋转只替换朝向输入；保留配置中的低通滤波和空间范围处理。
    orientation = mapper.hand_rotation if lock_rot else sample.orientation
    return mapper.step(sample.position, orientation, dt, limit_speed=False)


def _load_home_rotvecs(home_file, arms):
    """Load each arm's recorded TCP orientation as an action rotation vector."""
    data = json.loads(Path(home_file).read_text(encoding='utf-8'))
    if data.get('units') != 'radians':
        raise ValueError('Recorded home orientations must be in radians')
    rotations = {}
    for side, arm in arms.items():
        matches = [entry for entry in data['arms']
                   if entry['ip'] == arm['ip'] and entry['serial'] == arm['serial']]
        if len(matches) != 1:
            raise ValueError(f'{side} home orientation does not match its IP and T265 serial')
        pose = np.asarray(matches[0]['tcp_pose_aubo_rpy'], dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f'{side} home TCP pose must have six finite values')
        rotations[side] = Rotation.from_euler('xyz', pose[3:]).as_rotvec()
    return rotations


# ===========================================================================
# Aubo 输出与动作布局（Franka 的夹爪宽度在这里替换为柜体数字输出）
# ===========================================================================
def _read_output_state(env):
    # 只读预览或未启用 I/O 时返回占位值，动作构造不会把它下发。
    if env is None or not env.grippers_enabled:
        return {side: np.array([1.]) for side in ('left', 'right')}
    # 读取控制器维护的输出锁存，再按配置顺序拆成各路 DO 电平。
    # 锁存表示输出状态，不代表吸取成功或夹爪到位的传感器反馈。
    state = env.get_state()
    outputs = {side: env.decode_gripper(side, state[side]['GripperState'])[0].copy()
               for side in ('left', 'right')}
    # 未知组合不能作为切换起点，避免把不确定状态直接取反。
    if any(not np.isin(value, [0., 1.]).all() for value in outputs.values()):
        raise RuntimeError('End effector outputs are in an unknown combination')
    return outputs


def _request_initial_output_state(env):
    """Request startup/home output levels as independent events; do not wait."""
    outputs = _read_output_state(env)
    if env is None or not env.grippers_enabled:
        return outputs
    targets = {}
    # 现场确认：左 DO00=LOW 平铺、DO01=LOW 关闭吸盘；右夹爪 DO00 保留反转后的 LOW 初始值。
    for side, required_pins, levels in (('left', (0, 1), (0., 0.)), ('right', (0,), (0.,))):
        settings = env.gripper_configs[side]
        pins = settings.get('pins', [])
        if settings.get('mode') != 'independent' or not all(pin in pins for pin in required_pins):
            raise RuntimeError(f'{side} startup requires independent outputs {required_pins}')
        target = outputs[side].copy()
        target[[pins.index(pin) for pin in required_pins]] = levels
        targets[side] = target
    for side, target in targets.items():
        env.set_gripper(side, target)
    return targets


def _initialize_output_state(env):
    """Initialize left DO00/DO01 LOW (flat, suction off) and right DO00 LOW."""
    targets = _request_initial_output_state(env)
    if env is None or not env.grippers_enabled:
        return targets
    # Confirm both asynchronous writes while maintaining the servo watchdogs.
    deadline = time.monotonic() + 2.
    while True:
        env.heartbeat()
        state = env.get_state()
        actual = {side: env.decode_gripper(side, state[side]['GripperState'])[0].copy()
                  for side in targets}
        pending = [side for side in targets if not np.array_equal(actual[side], targets[side])]
        if not pending:
            print('IO initialized: left DO00=LOW, DO01=LOW (flat, suction off); right DO00=LOW')
            return actual
        if time.monotonic() >= deadline:
            details = {
                side: {f'DO{pin:02d}': dict(actual=float(value), target=float(target))
                       for pin, value, target in zip(env.gripper_configs[side]['pins'],
                                                      actual[side], targets[side])}
                for side in pending
            }
            raise RuntimeError(
                f'Startup output levels were not confirmed within 2 seconds: {pending}; '
                f'output latch feedback: {details}')
        time.sleep(.02)


def _start_home(env, homes):
    # 与启动共用电平定义，先提交独立 I/O 事件再提交 HOME；不阻塞键盘/停止处理。
    # HOME 清空的是带时间的动作事件，独立 I/O 事件不会被轨迹切换合并或删除。
    outputs = _request_initial_output_state(env)
    env.return_home(homes['left'], homes['right'])
    return outputs


def _confirm_home_outputs(env, targets):
    # 两臂运动完成后再核对真实锁存，避免把请求电平当作归位后的实际状态。
    actual = _read_output_state(env)
    if env.grippers_enabled:
        pending = [side for side in targets if not np.array_equal(actual[side], targets[side])]
        if pending:
            raise RuntimeError(f'Home output levels were not confirmed: {pending}; '
                               f'actual={actual}, target={targets}')
    return actual


def _toggle_output(env, outputs, side, pin, teleop_enabled):
    # 仅在启用 I/O 且遥操运行时接受按键；暂停时不改缓存电平。
    if env is None or not env.grippers_enabled or not teleop_enabled:
        raise RuntimeError('Enable I/O and resume teleoperation before toggling outputs')
    settings = env.gripper_configs[side]
    if settings.get('mode') != 'independent' or pin not in settings['pins']:
        raise RuntimeError('This key requires an explicitly configured independent cabinet output')
    # 每次按键都提交独立事件，连续两次取反也不会被位置覆盖或合并。
    index = settings['pins'].index(pin)
    outputs[side][index] = 1. - outputs[side][index]
    env.set_gripper(side, outputs[side])
    return outputs[side][index]


def _advance_left_output(env, outputs, teleop_enabled):
    """One A press advances DO00/DO01 through 00 -> 01 -> 11 -> 10 -> 00."""
    if env is None or not env.grippers_enabled or not teleop_enabled:
        raise RuntimeError('Enable I/O and resume teleoperation before toggling outputs')
    settings = env.gripper_configs['left']
    pins = settings.get('pins', [])
    if settings.get('mode') != 'independent' or not all(pin in pins for pin in (0, 1)):
        raise RuntimeError('Left sequence requires independent DO00 and DO01')
    indices = [pins.index(0), pins.index(1)]
    # 现场确认：DO00 控制弯折（0 平铺、1 弯折），DO01 控制吸盘（0 关、1 开）。
    # [DO00, DO01]：00 平铺/吸盘关 → 01 平铺/吸盘开 → 11 弯折/吸盘开 → 10 弯折/吸盘关。
    cycle = ((0., 0.), (0., 1.), (1., 1.), (1., 0.))
    current = tuple(outputs['left'][indices])
    if current not in cycle:
        raise RuntimeError('Left output levels are unknown; cannot advance sequence')
    next_index = (cycle.index(current) + 1) % len(cycle)
    target = outputs['left'].copy()
    target[indices] = cycle[next_index]
    env.set_gripper('left', target)
    outputs['left'] = target
    return next_index, cycle[next_index]


def _set_left_output_levels(env, outputs, levels):
    """Request absolute DO00/DO01 levels for P, preserving any other outputs."""
    if env is None or not env.grippers_enabled:
        raise RuntimeError('P requires enabled I/O')
    settings = env.gripper_configs['left']
    pins = settings.get('pins', [])
    if settings.get('mode') != 'independent' or not all(pin in pins for pin in (0, 1)):
        raise RuntimeError('P requires independent left DO00 and DO01')
    target = outputs['left'].copy()
    target[[pins.index(0), pins.index(1)]] = levels
    env.set_gripper('left', target)
    outputs['left'] = target


def _left_output_levels_recorded(env, levels):
    """Check the final row that end_episode will save, keeping the existing time alignment."""
    if not env._recording:
        return True
    count = min(len(env.obs_accumulator), len(env.action_accumulator), len(env.stage_accumulator))
    if not count:
        return False
    pins = env.gripper_configs['left']['pins']
    indices = np.array([pins.index(0), pins.index(1)])
    observed = env.obs_accumulator.data['left_gripper_state'][count - 1][indices]
    requested = env.action_accumulator.actions[count - 1, 6 + indices]
    return np.array_equal(observed, levels) and np.array_equal(requested, levels)


def _update_encoder_outputs(env, outputs, samples, anchors, toggled, threshold):
    """保留 v1 对可选 binary 编码器夹爪的支持；V/B/C 的 independent 模式不走这里。"""
    if env is None or not env.grippers_enabled:
        return
    for side, sample in samples.items():
        # 独立 DO 只由 V/B/C 控制，不参与编码器角度切换。
        if env.gripper_configs[side].get('mode') == 'independent':
            continue
        anchor = anchors.get(side)
        if sample.encoder_angle is None or anchor is None:
            continue
        # 将角差折回 [-180, 180)，避免编码器跨越零度时误判为大幅转动。
        delta = abs((sample.encoder_angle - anchor + 180.) % 360. - 180.)
        if delta > threshold and not toggled[side]:
            outputs[side][0] = 1. - outputs[side][0]
            env.set_gripper(side, outputs[side])
            toggled[side] = True
        elif delta < threshold * .5:
            # 回到半阈值以内才允许再次触发，形成滞回以避免反复切换。
            toggled[side] = False


def _build_action(left_target, right_target, outputs, env):
    # 禁用 I/O 时 12 维；当前三路 I/O 启用后 15 维，各 DO 都是 0/1。
    # 字段顺序须与 DualAuboRealEnv 的 action_layout 和拆分偏移一致。
    if env is not None and env.grippers_enabled:
        return np.concatenate((left_target, outputs['left'], right_target, outputs['right']))
    return np.concatenate((left_target, right_target))


def _close_preview():
    """只关闭本入口的图像窗口；未打开窗口时不加载 OpenCV。"""
    cv2 = sys.modules.get('cv2')
    if cv2 is not None:
        try:
            # ExitStack 的清理回调只关闭本入口窗口，不影响其他 OpenCV 窗口。
            cv2.destroyWindow('AUBO teleop v4')
        except Exception:
            # 窗口未创建或已经关闭时，继续完成其他资源的退出清理。
            pass


# ===========================================================================
# 参数（仍读取同一 JSON，不复制 Franka 的频率、home 或夹爪默认值）
# ===========================================================================
def parser():
    result = argparse.ArgumentParser(description=__doc__)
    # 配置与硬件授权：默认只读，运动及柜体输出各自需要显式启用。
    result.add_argument('--config', default=str(DEFAULT_CONFIG))
    result.add_argument('--left-pose-files', nargs=2, default=DEFAULT_LEFT_POSE_FILES,
                        metavar=('FIRST', 'SECOND'), help='P 键左臂两段路径的姿态记录 JSON，按顺序执行')
    result.add_argument('--enable-motion', action='store_true')
    result.add_argument('--enable-io', action='store_true')
    result.add_argument('--check-config', action='store_true')
    # 安装坐标标定可单独执行，也可使用已启动的 T265 输入在运行前执行。
    result.add_argument('--calibrate', choices=['left', 'right'])
    result.add_argument('--calibrate-on-start', action='store_true')
    result.add_argument('--forward-base', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    # 限位防抖优化选项 (v4 新增)
    result.add_argument('--soft-zone', type=float, default=0.015,
                        help='软限位阻尼减速缓冲带厚度 (米)，默认 0.015 (15mm)')
    result.add_argument('--recovery-margin', type=float, default=0.003,
                        help='底层实际高度恢复容差死区 (米)，默认 0.003 (3mm)')
    # 录制与相机选项：数组图像存储和视频采集分别由环境处理。
    result.add_argument('--output', '-o', help='Aubo 示教数据保存目录')
    result.add_argument('--no-cameras', action='store_true')
    result.add_argument('--record-images', action='store_true')
    result.add_argument('--headless', action='store_true', help='仅关闭图像窗口，键盘仍需桌面会话')
    result.add_argument('--vis-camera-idx', type=int, default=1)
    # 可选运行时长用于有限时间的预览/调试；不提供则持续到退出按键。
    result.add_argument('--duration', type=float)
    return result


# ===========================================================================
# 主流程（按 Franka v3 的初始化、观测、按键、目标、exec_actions、等待顺序）
# ===========================================================================
def run(args, config):
    # 离线 --check-config 不进入这里；硬件模块须在 SDK 路径初始化之后导入。
    from diffusion_policy.envs.realword.aubo_t265.sdk_paths import bootstrap_sdk, CabinetLocks
    bootstrap_sdk()
    from diffusion_policy.envs.realword.aubo_t265.calibration import (
        calibrate_arm, calibrate_live_arm, load_calibration, read_installation, validate_installation,
    )
    if args.calibrate:
        # 独立标定流程完成即返回，不继续进入伺服或录制主循环。
        if args.enable_motion or args.output:
            raise ValueError('Run calibration separately from motion/recording')
        arm = next(arm for arm in config['arms'] if arm['side'] == args.calibrate)
        calibrate_arm(arm, args.forward_base)
        print(f"Calibration saved: {arm['calibration']}")
        return
    from diffusion_policy.envs.realword.aubo_interpolation_controller import query_robot_status
    from diffusion_policy.envs.realword.dual_aubo_real_env_v3 import DualAuboRealEnvV3
    from diffusion_policy.envs.realword.aubo_t265.teleop_input import DualTeleopInput
    from diffusion_policy.envs.realword.aubo_t265.keyboard import create_keyboard
    from diffusion_policy.envs.realword.keystroke_counter import Key

    motion = config['motion']
    arms = {arm['side']: arm for arm in config['arms']}
    homes = {side: load_home(config['home_file'], arm['ip'], arm['serial']) for side, arm in arms.items()}
    home_rotvecs = _load_home_rotvecs(config['home_file'], arms)
    workspace_settings = copy.deepcopy(config['workspace'])
    if args.soft_zone is not None:
        workspace_settings['soft_zone'] = args.soft_zone
    if args.recovery_margin is not None:
        workspace_settings['recovery_margin'] = args.recovery_margin
    workspace = DualWorkspaceV4(config['arms'], workspace_settings)
    # 子进程启动前固定各臂归位姿态，恢复/暂停/归位都共用同一参考。
    for side, orientation in home_rotvecs.items():
        workspace.set_locked_orientation(side, orientation)
    left_waypoints = _load_left_waypoints(args.left_pose_files, arms['left'], workspace)
    home_waypoints = {side: _load_home_waypoint(config['home_file'], arm, workspace)
                      for side, arm in arms.items()}
    # 按进入顺序注册资源，退出时反向清理，异常路径也停止设备进程。
    with ExitStack() as stack:
        # ---- 1. 初始化设备：柜体锁、安装核对、共享内存 sender ----
        if args.enable_motion:
            stack.enter_context(CabinetLocks(config['arms']))
        statuses, gravities = {}, {}
        for side, arm in arms.items():
            # 对照现场重力方向核验安装标定；不匹配时拒绝继续遥操。
            gravity, _ = read_installation(arm['ip'], arm['port'])
            # calibrated_base 保留保存的 Z 轴；安装重力仅作核对，不覆盖高度方向。
            if workspace.enabled and workspace.reference == 'gravity_base':
                workspace.update_gravity(side, gravity)
            gravities[side] = gravity
            validate_installation(load_calibration(arm['calibration'], arm['ip'], arm['serial']), gravity)
            status = query_robot_status(arm['ip'], arm['port'])
            if args.enable_motion and (status['robot_mode'] != 'Running' or status['safety_mode'] != 'Normal'):
                raise RuntimeError(f'{side} must be Running/Normal on the pendant: {status}')
            statuses[side] = status
            workspace.validate_feedback(side, status['tcp_pose_rotvec'])
        shm = stack.enter_context(SharedMemoryManager())
        receiver = stack.enter_context(DualTeleopInput(shm, config))
        if args.calibrate_on_start:
            for side, arm in arms.items():
                # 固定本轮 side，标定与之后遥操复用同一个输入实例。
                calibrate_live_arm(arm, lambda side=side: receiver.get()[side], gravities[side])

        # ---- 2. 初始化双臂环境：v3 左右控制器 + 原框架相机/录制接口 ----
        env = None
        if args.enable_motion:
            cameras = config['cameras']
            env = stack.enter_context(DualAuboRealEnvV3(
                arms['left']['ip'], arms['right']['ip'], arms['left']['port'], arms['right']['port'],
                shm_manager=shm, robot_frequency=motion['robot_frequency'], state_frequency=motion['state_frequency'],
                frequency=motion['record_frequency'], **controller_kwargs(config),
                command_timeout=motion['command_timeout'], output_dir=args.output,
                camera_serial_numbers=[] if args.no_cameras else cameras['serial_numbers'],
                video_capture_resolution=tuple(cameras['capture_resolution']), video_capture_fps=cameras['capture_fps'],
                obs_image_resolution=tuple(cameras['obs_resolution']), record_images=args.record_images,
                gripper_configs={side: arm['gripper'] for side, arm in arms.items()}, workspace=workspace,
                manual_home=True, home_speed_scale=V4_HOME_SPEED_SCALE,
                recording_metadata=recording_metadata(config)))
        key_counter = stack.enter_context(create_keyboard())
        stack.callback(_close_preview)

        # ---- 3. 初始化目标、映射器及显式状态变量 ----
        poses = ({side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()} if env else
                 {side: state['tcp_pose_rotvec'].copy() for side, state in statuses.items()})
        prev_targets = {side: poses[side].copy() for side in ('left', 'right')}
        mappers = make_mappers(config, poses)
        outputs = _initialize_output_state(env)
        teleop_anchor, encoder_anchors, encoder_toggled = {}, {}, {}
        teleop_enabled = homing = is_recording = False
        left_sequence = None
        right_home_sequence = None
        lock_rot = True  # 强制锁定；T 只确认锁定并重锚定 XYZ。
        manual_home_pending = env is not None  # 启动保持，首次 H 完成后再开放固定姿态遥操。
        stage, vis_camera_idx = 0, args.vis_camera_idx

        def refresh_anchor(samples):
            # 与 Franka 的 _refresh_anchor 对应：从实际 TCP 重新锚定并同步输出锁存。
            nonlocal poses, prev_targets, teleop_anchor, outputs, encoder_anchors, encoder_toggled
            if manual_home_pending:
                print('启动保持当前位置；请按 H 归位，完成后再按 E 开始遥操。')
                return False
            if samples is None:
                print('Cannot resume: tracking is not valid')
                return False
            if env:
                # 重新启用时从实际反馈锚定，避免沿用暂停前尚未到达的目标。
                poses = {side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()}
            prev_targets = {side: poses[side].copy() for side in ('left', 'right')}
            outputs = _read_output_state(env)
            teleop_anchor = _capture_teleop_anchor(poses, samples, mappers)
            # 编码器切换也以当前读数重新开始，清除上一段的触发标记。
            encoder_anchors = {side: sample.encoder_angle for side, sample in samples.items()}
            encoder_toggled = {side: False for side in samples}
            return True

        def pause():
            nonlocal teleop_enabled, is_recording, left_sequence, right_home_sequence
            left_sequence = None
            right_home_sequence = None
            if env:
                # 先取消运动目标，再保存录制；下一次 E/R 必须重新锚定。
                env.hold()
                if env._recording:
                    env.end_episode()
            teleop_enabled = is_recording = False

        print('E: teleop; Z: anchor; T: confirm orientation lock; R/S: record/save; H: home; P: path + right home -> state 1 -> save/pause; Space: stage; Q/Esc: quit')
        print('A(+B), or V: left IO cycle 00 -> 01 -> 11 -> 10 -> 00; B: release ignored; C(+D): toggle right DO00; 1/2/3: camera')
        print('Left IO mapping: DO00 LOW=flat/HIGH=bent; DO01 LOW=suction off/HIGH=suction on')
        print(f'Speed scales: P points={P_POINT_SPEED_SCALE:g}x, P return home={P_RETURN_HOME_SPEED_SCALE:g}x, H={V4_HOME_SPEED_SCALE:g}x base home')
        print('MOTION ENABLED, initially paused' if env else 'READ-ONLY PREVIEW, initially paused')
        print('Realtime mode: latest XYZ target; independent I/O events; no arrival deadline or software speed limits')
        if env:
            print('启动保持当前关节；按 H 归位，完成后按 E 开始固定姿态遥操。')
        period = 1. / motion['teleop_frequency']
        # 周期/超时用单调时钟；下发环境和录制时才转换为 Unix 时间。
        t_start = previous = time.monotonic()
        next_cycle = t_start
        next_obs = next_report = t_start
        stop, obs = False, None
        while not stop and (args.duration is None or time.monotonic() - t_start < args.duration):
            # 实时目标没有未来到达时间；周期使用单调时钟。
            t_cycle_start = time.monotonic()
            actual_dt, previous = max(1e-6, t_cycle_start - previous), t_cycle_start

            # ---- 4. 获取共享内存输入和观测：任何手部失效都暂停整对机械臂 ----
            try:
                samples = receiver.get()
            except RuntimeError as exc:
                samples = None
                if homing:
                    raise RuntimeError('Input lost during home; stopping both arms') from exc
                if teleop_enabled:
                    pause()
                    print(f'Input lost: paused; press E after tracking recovers. {exc}')
            obs_updated = False
            if env:
                env.get_state()  # 即使暂停且本轮不采样观测，也检查控制器与反馈是否正常。
                if t_cycle_start >= next_obs:
                    obs = env.get_obs()
                    obs_updated = True
                    next_obs = t_cycle_start + 1. / motion['record_frequency']

            # ---- 5. 按键：E/Z/T/R/S/Space 管理遥操和录制，V/B/C 管理 Aubo 输出 ----
            for key in key_counter.get_press_events():
                character = getattr(key, 'char', '')
                if key == Key.esc or character == 'q':
                    stop = True
                    break
                elif character == 'e':
                    if homing:
                        raise RuntimeError('Operator aborted home; stopping both arms')
                    if teleop_enabled:
                        pause()
                    else:
                        teleop_enabled = refresh_anchor(samples)
                elif character == 'p':
                    if left_sequence is not None:
                        print('P path already running; repeat ignored')
                    elif env is None or not env.grippers_enabled or homing or not teleop_enabled or samples is None:
                        print('P requires motion/I/O enabled and active teleoperation after H/E')
                    else:
                        p_state = env.get_state()
                        left_sequence = _LeftPoseSequence(
                            left_waypoints, p_state['left'], motion['max_speed'], home_waypoints['left'],
                            point_speed_scale=P_POINT_SPEED_SCALE, home_speed_scale=P_RETURN_HOME_SPEED_SCALE)
                        # Right XYZ moves at the original 1x planning speed, then holds home; T265 is ignored.
                        right_home_sequence = _LeftPoseSequence(
                            [home_waypoints['right']], p_state['right'], motion['max_speed'])
                        _set_left_output_levels(env, outputs, left_sequence.output_levels)
                        print('P: left state 3 -> two points -> state 4 -> home; right slow return/hold; auto-save when both settled')
                elif character in ('z', 't', 'r') and left_sequence is not None:
                    print('Wait for P path to finish before re-anchoring or starting a new recording')
                elif character == 'z' and not homing:
                    if refresh_anchor(samples):
                        print('Teleoperation anchors updated')
                elif character == 't' and not homing:
                    # 在切换点重锚定，锁定与解锁都不会跳向旧朝向。
                    if samples is not None and refresh_anchor(samples):
                        lock_rot = True
                        print('Orientation strictly locked to recorded home; XYZ anchor refreshed')
                elif character == 'r' and env and not homing:
                    # 录制从重锚定后的阶段 0 开始，起点换算成环境所用的 Unix 秒。
                    if not args.output:
                        print('Recording requires --output')
                    elif not env._recording and refresh_anchor(samples):
                        stage = 0
                        teleop_enabled = True
                        env.start_episode(time.time())
                        is_recording = True
                elif character == 's' and not homing:
                    pause()  # 保存当前 episode，并像 Franka 的 S 一样关闭遥操。
                elif character == 'h' and env and not homing:
                    # 回零先结束录制并关闭遥操，避免笛卡尔目标与关节回零混用。
                    if env._recording:
                        env.end_episode()
                    teleop_enabled = is_recording = False
                    left_sequence = None
                    right_home_sequence = None
                    outputs = _start_home(env, homes)
                    homing = True
                elif key == Key.backspace and env:
                    env.drop_episode()
                    is_recording = False
                elif key == Key.space:
                    stage += 1  # Space 是录制 stage 标记，不再是遥操开关。
                elif character in ('a', 'v', 'b', 'c'):
                    # 数采设备每次按下会连续发 C、D：只响应 C 取反一次，D 不处理。
                    try:
                        if character in ('a', 'v'):
                            if left_sequence is not None:
                                print('P controls left IO automatically; A/V ignored until complete')
                                continue
                            step, levels = _advance_left_output(env, outputs, teleop_enabled)
                            print(f'left IO step {step or 4}/4 requested: '
                                  f'DO00={"HIGH" if levels[0] else "LOW"}, '
                                  f'DO01={"HIGH" if levels[1] else "LOW"}')
                        elif character == 'b':
                            print('Left IO: B ignored; A advances the four-step sequence; B is the release event')
                        else:
                            side, pin = 'right', 0
                            level = _toggle_output(env, outputs, side, pin, teleop_enabled)
                            print(f'{side} DO{pin:02d} requested: {"HIGH" if level else "LOW"}')
                    except RuntimeError as exc:
                        print(exc)
                elif character in ('1', '2', '3'):
                    vis_camera_idx = int(character) - 1
            if stop:
                break

            # ---- 6. 回零/暂停/遥操目标：状态判断与动作构造直接写在主循环 ----
            if homing:
                # 回零由控制器推进；主循环只续心跳并等待两臂共同完成。
                env.heartbeat()
                if env.home_state == 'holding':
                    outputs = _confirm_home_outputs(env, outputs)
                    # 回零后重建映射器并保持暂停，下一次 E 使用新的实际 TCP 锚点。
                    poses = {side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()}
                    prev_targets = {side: poses[side].copy() for side in ('left', 'right')}
                    mappers = make_mappers(config, poses)
                    teleop_anchor = {}
                    homing = False
                    manual_home_pending = False
                    print('Home complete; press E to re-anchor and resume')
            elif not teleop_enabled:
                if env:
                    # 暂停不提交新目标，但维持控制器的命令心跳。
                    env.heartbeat()
            elif samples is not None:
                # 各臂独立按安装坐标映射；双方目标通过工作空间检查后才统一下发。
                if left_sequence is None:
                    left_raw = _target_from_anchor(mappers['left'], samples['left'], actual_dt, lock_rot)
                    left_target = workspace.apply_soft_damping('left', left_raw, prev_targets['left'])
                else:
                    p_state = env.get_state()
                    left_state = p_state['left']
                    previous_segment = left_sequence.index
                    was_left_done = left_sequence.done
                    pins = env.gripper_configs['left']['pins']
                    actual_levels = env.decode_gripper('left', left_state['GripperState'])[0][
                        [pins.index(0), pins.index(1)]]
                    # Advance one controller period at most; a delayed UI/recording tick cannot jump the path.
                    left_target = left_sequence.tick(left_state, min(actual_dt, period),
                                                     io_ready=np.array_equal(actual_levels, left_sequence.output_levels))
                    workspace.validate_target('left', left_target)
                    if left_sequence.index != previous_segment:
                        if left_sequence.index == 1:
                            print('P: point 1 reached and settled; moving to left point 2/2 in state 3')
                        else:
                            _set_left_output_levels(env, outputs, left_sequence.output_levels)
                            print('P: point 2 reached and settled; state 4 (DO00=HIGH, DO01=LOW) requested; return home after confirmation')
                    if left_sequence.done and not was_left_done:
                        _set_left_output_levels(env, outputs, left_sequence.output_levels)
                        print('P: left home reached and settled; state 1 (DO00=LOW, DO01=LOW) requested')
                if right_home_sequence is None:
                    right_raw = _target_from_anchor(mappers['right'], samples['right'], actual_dt, lock_rot)
                    right_target = workspace.apply_soft_damping('right', right_raw, prev_targets['right'])
                else:
                    was_done = right_home_sequence.done
                    right_target = right_home_sequence.tick(p_state['right'], min(actual_dt, period))
                    workspace.validate_target('right', right_target)
                    if right_home_sequence.done and not was_done:
                        print('P: right home reached and settled; holding, manual tracking ignored')

                # 遥操仅沿 XYZ 平移；姿态固定为各臂归位记录的 TCP 朝向。
                left_target[3:] = home_rotvecs['left']
                right_target[3:] = home_rotvecs['right']

                # 优化二补充：映射器抗积分饱和（Anti-Windup），消除反向抬手时的延迟粘滞
                workspace.sync_mapper_anti_windup('left', mappers['left'], left_target)
                workspace.sync_mapper_anti_windup('right', mappers['right'], right_target)

                _update_encoder_outputs(env, outputs, samples, encoder_anchors, encoder_toggled,
                                        motion.get('gripper_toggle_deg', 30.))
                action = _build_action(left_target, right_target, outputs, env)
                # ---- 7. 覆盖最新位置；输出已由独立事件入口提交 ----
                if env:
                    env.set_teleop_target(action, stage=stage)
                poses = {'left': left_target.copy(), 'right': right_target.copy()}
                prev_targets = {'left': left_target.copy(), 'right': right_target.copy()}
                if (left_sequence is not None and left_sequence.done
                        and right_home_sequence is not None and right_home_sequence.done
                        and np.array_equal(actual_levels, left_sequence.output_levels)
                        and _left_output_levels_recorded(env, (0., 0.))):
                    # Record confirmed state 1 before S's hold/save path; only E/R can resume tracking.
                    pause()
                    print('P complete in state 1: both arms at home; episode saved and teleoperation paused (same as S); press E/R to resume')

            # ---- 8. 可视化与诊断：与观测一起降频，避免拖慢 50 Hz 遥操 ----
            mode = ('homing' if homing else (f'P {left_sequence.label}' if left_sequence is not None
                                          else ('teleop' if teleop_enabled else 'paused')))
            if obs_updated and env.realsense and not args.headless:
                import cv2
                names = sorted(key for key in obs if key.startswith('camera_'))
                if names:
                    # 观测是 RGB；显示前取最新一帧并转换为 OpenCV 使用的 BGR。
                    frame = np.ascontiguousarray(obs[names[vis_camera_idx % len(names)]][-1][..., ::-1])
                    cv2.putText(frame, f'{mode} Stage:{stage} REC:{is_recording} ROTLOCK:{lock_rot}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    cv2.imshow('AUBO teleop v4', frame)
                    cv2.waitKey(1)
            if t_cycle_start >= next_report:
                print(mode, {side: np.round(pose, 4).tolist() for side, pose in poses.items()}, flush=True)
                next_report = t_cycle_start + 2.

            # ---- 9. 等待下一周期：与 Franka 的 precise_wait 调度方式相同 ----
            next_cycle += period
            finished = time.monotonic()
            if next_cycle <= finished:
                next_cycle += (int((finished - next_cycle) / period) + 1) * period
            precise_wait(next_cycle)
        # 离开 with 后自动保存有效 episode、停止控制器和 sender，并释放共享内存/锁。


# ===========================================================================
# 离线检查与启动选择
# ===========================================================================
def main(argv=None):
    # 先校验纯命令行参数和 JSON 配置，失败时不进入硬件初始化。
    args = parser().parse_args(argv)
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        raise ValueError('--duration must be finite and positive')
    if args.vis_camera_idx < 0:
        raise ValueError('--vis-camera-idx must be nonnegative')
    config = load_config(args.config, validate_calibrations=not bool(args.calibrate))
    if args.enable_io:
        # 显式启用配置中的柜体输出；这里仅验证配置，不创建 SDK 连接。
        if not args.enable_motion and not args.check_config:
            raise ValueError('--enable-io requires --enable-motion')
        if args.calibrate:
            raise ValueError('I/O cannot be enabled during calibration')
        from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
        for arm in config['arms']:
            arm['gripper']['type'] = 'standard_digital'
            AuboDigitalGripper(None, arm['gripper'])
    if args.check_config:
        # 离线校验配置、回零文件及工作空间后返回，不调用 run 的设备初始化。
        for arm in config['arms']:
            load_home(config['home_file'], arm['ip'], arm['serial'])
        _load_home_rotvecs(config['home_file'], {arm['side']: arm for arm in config['arms']})
        workspace_settings = copy.deepcopy(config['workspace'])
        if args.soft_zone is not None:
            workspace_settings['soft_zone'] = args.soft_zone
        if args.recovery_margin is not None:
            workspace_settings['recovery_margin'] = args.recovery_margin
        workspace = DualWorkspaceV4(config['arms'], workspace_settings)
        arms = {arm['side']: arm for arm in config['arms']}
        for side, orientation in _load_home_rotvecs(config['home_file'], arms).items():
            workspace.set_locked_orientation(side, orientation)
        _load_left_waypoints(args.left_pose_files, arms['left'], workspace)
        for arm in arms.values():
            _load_home_waypoint(config['home_file'], arm, workspace)
        print('Configuration valid; no SDK, robot, camera, or I/O accessed.')
        return
    # 录制依赖运动环境提供的反馈与动作累计器，纯预览不创建示教数据集。
    if args.output and not args.enable_motion:
        raise ValueError('Recording requires --enable-motion')
    run(args, config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # run 中的 ExitStack 已负责资源释放；Ctrl+C 退出无需再打印异常栈。
        pass
