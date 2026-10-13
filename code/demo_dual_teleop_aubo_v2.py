#!/usr/bin/env python3
"""双臂 Aubo 遥操 v2：按 demo_dual_udp_teleop_franka_v3.py 的流程组织。

初始化输入/环境 → 捕获锚点 → 读取观测 → 处理按键 → 计算左右目标
→ env.set_teleop_target → 等待下一周期。位置目标覆盖更新，I/O 独立排队。
沿用 Aubo 安装标定、四元数映射、滤波与伺服参数；实时路径不做软件限速。
遥操目标姿态始终采用各臂归位记录，仅 XYZ 跟随手部运动。

E 开关遥操，Z 重新锚定，T 锁定旋转，R 开始录制并开启遥操，
S 保存并暂停，H 六关节回零，Space 增加 stage，Q/Esc 退出。
A/B 按下和松开推进左 DO00/DO01 四步循环一次（只响应 A，V 同效），数采按钮的 C/D 组合信号切换右 DO00，1/2/3 切换相机。
默认只读预览；--enable-motion 启动伺服，--enable-io 启用已配置的柜体输出。
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.envs.realword.aubo_homing import load_home
from diffusion_policy.envs.realword.aubo_t265.config import (
    DEFAULT_CONFIG, load_config, controller_kwargs, recording_metadata,
)
from diffusion_policy.envs.realword.aubo_t265.runtime import make_mappers
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace


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


def _initialize_output_state(env):
    """Initialize left DO00/DO01 and right DO00 low before accepting input."""
    outputs = _read_output_state(env)
    if env is None or not env.grippers_enabled:
        return outputs
    targets = {}
    for side, required_pins in (('left', (0, 1)), ('right', (0,))):
        settings = env.gripper_configs[side]
        pins = settings.get('pins', [])
        if settings.get('mode') != 'independent' or not all(pin in pins for pin in required_pins):
            raise RuntimeError(f'{side} startup requires independent outputs {required_pins}')
        target = outputs[side].copy()
        target[[pins.index(pin) for pin in required_pins]] = 0.
        targets[side] = target
    for side, target in targets.items():
        env.set_gripper(side, target)
    # Confirm both asynchronous writes while maintaining the servo watchdogs.
    deadline = time.monotonic() + 2.
    while True:
        env.heartbeat()
        state = env.get_state()
        actual = {side: env.decode_gripper(side, state[side]['GripperState'])[0].copy()
                  for side in targets}
        pending = [side for side in targets if not np.array_equal(actual[side], targets[side])]
        if not pending:
            outputs.update(actual)
            print('IO initialized: left DO00=LOW, DO01=LOW; right DO00=LOW')
            return outputs
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Startup LOW was not confirmed within 2 seconds: {pending}')
        time.sleep(.02)


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
            cv2.destroyWindow('AUBO teleop v2')
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
    result.add_argument('--enable-motion', action='store_true')
    result.add_argument('--enable-io', action='store_true')
    result.add_argument('--check-config', action='store_true')
    # 安装坐标标定可单独执行，也可使用已启动的 T265 输入在运行前执行。
    result.add_argument('--calibrate', choices=['left', 'right'])
    result.add_argument('--calibrate-on-start', action='store_true')
    result.add_argument('--forward-base', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    # 录制与相机选项：数组图像存储和视频采集分别由环境处理。
    result.add_argument('--output', '-o', help='Aubo 示教数据保存目录')
    result.add_argument('--no-cameras', action='store_true')
    result.add_argument('--record-images', action='store_true')
    result.add_argument('--headless', action='store_true', help='仅关闭图像窗口，键盘仍需桌面会话')
    result.add_argument('--vis-camera-idx', type=int, default=0)
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
    from diffusion_policy.envs.realword.dual_aubo_real_env_v2 import DualAuboRealEnvV2
    from diffusion_policy.envs.realword.aubo_t265.teleop_input import DualTeleopInput
    from diffusion_policy.envs.realword.aubo_t265.keyboard import create_keyboard
    from diffusion_policy.envs.realword.keystroke_counter import Key

    motion = config['motion']
    arms = {arm['side']: arm for arm in config['arms']}
    homes = {side: load_home(config['home_file'], arm['ip'], arm['serial']) for side, arm in arms.items()}
    home_rotvecs = _load_home_rotvecs(config['home_file'], arms)
    workspace = DualWorkspace(config['arms'], config['workspace'])
    # 子进程启动前固定各臂归位姿态，恢复/暂停/归位都共用同一参考。
    for side, orientation in home_rotvecs.items():
        workspace.set_locked_orientation(side, orientation)
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

        # ---- 2. 初始化双臂环境：v2 左右控制器 + 原框架相机/录制接口 ----
        env = None
        if args.enable_motion:
            cameras = config['cameras']
            env = stack.enter_context(DualAuboRealEnvV2(
                arms['left']['ip'], arms['right']['ip'], arms['left']['port'], arms['right']['port'],
                shm_manager=shm, robot_frequency=motion['robot_frequency'], state_frequency=motion['state_frequency'],
                frequency=motion['record_frequency'], **controller_kwargs(config),
                command_timeout=motion['command_timeout'], output_dir=args.output,
                camera_serial_numbers=[] if args.no_cameras else cameras['serial_numbers'],
                video_capture_resolution=tuple(cameras['capture_resolution']), video_capture_fps=cameras['capture_fps'],
                obs_image_resolution=tuple(cameras['obs_resolution']), record_images=args.record_images,
                gripper_configs={side: arm['gripper'] for side, arm in arms.items()}, workspace=workspace,
                recording_metadata=recording_metadata(config)))
        key_counter = stack.enter_context(create_keyboard())
        stack.callback(_close_preview)

        # ---- 3. 初始化目标、映射器及显式状态变量 ----
        poses = ({side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()} if env else
                 {side: state['tcp_pose_rotvec'].copy() for side, state in statuses.items()})
        mappers = make_mappers(config, poses)
        outputs = _initialize_output_state(env)
        teleop_anchor, encoder_anchors, encoder_toggled = {}, {}, {}
        teleop_enabled = homing = is_recording = False
        lock_rot = True  # 强制锁定；T 只确认锁定并重锚定 XYZ。
        stage, vis_camera_idx = 0, args.vis_camera_idx

        def refresh_anchor(samples):
            # 与 Franka 的 _refresh_anchor 对应：从实际 TCP 重新锚定并同步输出锁存。
            nonlocal poses, teleop_anchor, outputs, encoder_anchors, encoder_toggled
            if samples is None:
                print('Cannot resume: tracking is not valid')
                return False
            if env:
                # 重新启用时从实际反馈锚定，避免沿用暂停前尚未到达的目标。
                poses = {side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()}
            outputs = _read_output_state(env)
            teleop_anchor = _capture_teleop_anchor(poses, samples, mappers)
            # 编码器切换也以当前读数重新开始，清除上一段的触发标记。
            encoder_anchors = {side: sample.encoder_angle for side, sample in samples.items()}
            encoder_toggled = {side: False for side in samples}
            return True

        def pause():
            nonlocal teleop_enabled, is_recording
            if env:
                # 先取消运动目标，再保存录制；下一次 E/C 必须重新锚定。
                env.hold()
                if env._recording:
                    env.end_episode()
            teleop_enabled = is_recording = False

        print('E: teleop; Z: anchor; T: confirm orientation lock; R/S: record/save; H: home; Space: stage; Q/Esc: quit')
        print('A(+B), or V: left IO cycle 00 -> 01 -> 11 -> 10 -> 00; B: release ignored; C(+D): toggle right DO00; 1/2/3: camera')
        print('MOTION ENABLED, initially paused' if env else 'READ-ONLY PREVIEW, initially paused')
        print('Realtime mode: latest XYZ target; independent I/O events; no arrival deadline or software speed limits')
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
                    env.return_home(homes['left'], homes['right'])
                    teleop_enabled = is_recording = False
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
                    # 回零后重建映射器并保持暂停，下一次 E 使用新的实际 TCP 锚点。
                    poses = {side: state['ActualTCPPose'].copy() for side, state in env.get_state().items()}
                    mappers = make_mappers(config, poses)
                    teleop_anchor = {}
                    homing = False
                    print('Home complete; press E to re-anchor and resume')
            elif not teleop_enabled:
                if env:
                    # 暂停不提交新目标，但维持控制器的命令心跳。
                    env.heartbeat()
            elif samples is not None:
                # 各臂独立按安装坐标映射；双方目标通过工作空间检查后才统一下发。
                left_target = _target_from_anchor(mappers['left'], samples['left'], actual_dt, lock_rot)
                right_target = _target_from_anchor(mappers['right'], samples['right'], actual_dt, lock_rot)
                # 遥操仅沿 XYZ 平移；姿态固定为各臂归位记录的 TCP 朝向。
                # 固定处理顺序：先沿所选高度轴修正 XYZ，最后覆盖各臂归位记录姿态。
                left_target = workspace.limit_target('left', left_target)
                right_target = workspace.limit_target('right', right_target)
                left_target[3:] = home_rotvecs['left']
                right_target[3:] = home_rotvecs['right']
                _update_encoder_outputs(env, outputs, samples, encoder_anchors, encoder_toggled,
                                        motion.get('gripper_toggle_deg', 30.))
                action = _build_action(left_target, right_target, outputs, env)
                # ---- 7. 覆盖最新位置；输出已由独立事件入口提交 ----
                if env:
                    env.set_teleop_target(action, stage=stage)
                poses = {'left': left_target.copy(), 'right': right_target.copy()}

            # ---- 8. 可视化与诊断：与观测一起降频，避免拖慢 50 Hz 遥操 ----
            mode = 'homing' if homing else ('teleop' if teleop_enabled else 'paused')
            if obs_updated and env.realsense and not args.headless:
                import cv2
                names = sorted(key for key in obs if key.startswith('camera_'))
                if names:
                    # 观测是 RGB；显示前取最新一帧并转换为 OpenCV 使用的 BGR。
                    frame = np.ascontiguousarray(obs[names[vis_camera_idx % len(names)]][-1][..., ::-1])
                    cv2.putText(frame, f'{mode} Stage:{stage} REC:{is_recording} ROTLOCK:{lock_rot}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                    cv2.imshow('AUBO teleop v2', frame)
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
        DualWorkspace(config['arms'], config['workspace'])
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
