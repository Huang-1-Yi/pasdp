#!/usr/bin/env python3
"""Dual T265 teleoperation for the tilted AUBO installation.

Preview is read-only. --enable-motion explicitly starts both servo controllers.
--check-config validates frames and recorded homes without SDKs or hardware.
"""
from __future__ import annotations

# 整体数据流：双 T265 样本 → 标定坐标映射与滤波限速 → 左右 TCP 目标
# → 环境按时间戳下发 → 各臂插值、IK 与 servoJoint；反馈另经 get_obs 进入录制。
# motion 配置手部映射，controller 配置机器人插值/伺服，arms 绑定设备与标定。
# 新框架公开姿态使用旋转向量，SDK 所需的 RPY 转换由机器人控制器处理。

# 标准库负责命令行、计时和资源释放；数值运算用于姿态与图像处理。
import argparse
from contextlib import ExitStack
import time

import numpy as np

# 顶层仅导入可离线使用的模块，保证 --check-config 无需加载硬件 SDK。
# 参数、映射和控制逻辑位于 realword 模块，本文件负责连接各模块及交互调度。
from diffusion_policy.envs.realword.aubo_t265.config import DEFAULT_CONFIG, load_config, recording_metadata, controller_kwargs
from diffusion_policy.envs.realword.aubo_t265.workspace import DualWorkspace
from diffusion_policy.envs.realword.aubo_homing import load_home


# 构建命令行：默认进入只读预览，运动、I/O 和数据录制分别显式启用。
def parser():
    result = argparse.ArgumentParser(description=__doc__)
    # 配置与运行模式：JSON 指定左右设备、标定、遥操限速和机器人伺服参数。
    result.add_argument("--config", default=str(DEFAULT_CONFIG))
    result.add_argument("--enable-motion", action="store_true")
    result.add_argument("--enable-io", action="store_true", help="Enable configured cabinet outputs after wiring")
    result.add_argument("--check-config", action="store_true")
    # 标定：单臂独立标定用于安装设置，启动时标定用于当前 T265 跟踪会话。
    # --forward-base 是机器人自身基坐标系中的“人的前方”方向向量。
    result.add_argument("--calibrate", choices=["left", "right"])
    result.add_argument("--calibrate-on-start", action="store_true", help="Calibrate both T265s in the live sender before starting servo")
    result.add_argument("--forward-base", type=float, nargs=3, metavar=("X", "Y", "Z"))
    # 数据与显示：录制目录、RGB 相机开关、Zarr 图像保存和窗口显示开关。
    # --headless 只关闭图像窗口，键盘监听仍依赖桌面会话。
    result.add_argument("--output", help="Demonstration output directory (use a new AUBO dataset)")
    result.add_argument("--no-cameras", action="store_true")
    result.add_argument("--record-images", action="store_true")
    result.add_argument("--headless", action="store_true", help="Disable camera windows; keyboard still needs a desktop")
    # 未指定时持续运行，指定后按经过的实际时间结束。
    result.add_argument("--duration", type=float, help="Stop after this many seconds")
    return result


# 执行硬件流程：检查安装 → 启动双 T265 → 可选标定/伺服 → 键盘与遥操循环。
# 正常退出和异常退出均通过 ExitStack 释放已成功注册的资源。
def run(args, config):
    from multiprocessing.managers import SharedMemoryManager
    from diffusion_policy.envs.realword.aubo_t265.sdk_paths import bootstrap_sdk, CabinetLocks
    # 先把随包 SDK 加入加载路径，再导入可能引用 Aubo/RealSense SDK 的模块。
    bootstrap_sdk()  # Must precede all imports of hardware controllers/cameras.
    from diffusion_policy.envs.realword.aubo_t265.calibration import calibrate_arm, calibrate_live_arm, load_calibration, read_installation, validate_installation
    # 独立标定只处理选中的臂；保存后返回，不进入双臂运动或录制流程。
    if args.calibrate:
        if args.enable_motion or args.output:
            raise ValueError("Run calibration separately from motion/recording")
        arm = next(arm for arm in config["arms"] if arm["side"] == args.calibrate)
        calibrate_arm(arm, args.forward_base)
        print(f"Calibration saved: {arm['calibration']}")
        return
    # 延迟加载实机环境、T265 共享内存输入和键盘监听，保持离线检查入口独立。
    from diffusion_policy.envs.realword.aubo_interpolation_controller import query_robot_status
    from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv
    from diffusion_policy.envs.realword.aubo_t265.teleop_input import DualTeleopInput
    from diffusion_policy.envs.realword.aubo_t265.runtime import TeleopSession
    from diffusion_policy.envs.realword.keystroke_counter import Key
    from diffusion_policy.envs.realword.aubo_t265.keyboard import create_keyboard
    # 用 side 绑定机器人与 T265；回零记录按 IP/序列号校验，防止左右臂错配。
    # workspace 仅在配置启用且提供基座外参时检查共同世界坐标中的 TCP 边界。
    motion = config["motion"]
    arms = {arm["side"]: arm for arm in config["arms"]}
    homes = {side: load_home(config["home_file"], arm["ip"], arm["serial"]) for side, arm in arms.items()}
    workspace = DualWorkspace(config["arms"], config["workspace"])
    # 按创建的逆序关闭键盘、机器人环境、输入进程、共享内存和控制柜锁。
    with ExitStack() as stack:
        # 运动模式持有控制柜文件锁，避免本启动器的多个实例争用同一机器人。
        if args.enable_motion:
            stack.enter_context(CabinetLocks(config["arms"]))
        statuses, gravities = {}, {}
        # 预览和运动都会读取机器人：先核对重力与保存的安装标定，再读取当前 TCP。
        # 运动模式额外要求示教器处于 Running/Normal；此入口不自动上电。
        for side, arm in arms.items():
            gravity, _ = read_installation(arm["ip"], arm["port"])
            gravities[side] = gravity
            validate_installation(load_calibration(arm["calibration"], arm["ip"], arm["serial"]), gravity)
            status = query_robot_status(arm["ip"], arm["port"])
            if args.enable_motion and (status["robot_mode"] != "Running" or status["safety_mode"] != "Normal"):
                raise RuntimeError(f"{side} must be Running/Normal on the pendant: {status}")
            statuses[side] = status
            workspace.validate(side, status["tcp_pose_rotvec"])
        # 双 T265 sender 与机器人控制器共用共享内存管理器，输入按设备时间判定有效性。
        shm = stack.enter_context(SharedMemoryManager())
        inputs = stack.enter_context(DualTeleopInput(shm, config))
        # 在本次 sender 跟踪会话中更新手部方向标定，随后继续使用同一会话。
        # lambda 的默认参数固定当前 side，确保分别读取左、右 T265。
        if args.calibrate_on_start:
            for side, arm in arms.items():
                calibrate_live_arm(arm, lambda side=side: inputs.get()[side], gravities[side])
        # env=None 表示只读预览：仍计算目标，但不会启动伺服或发送运动目标。
        env = None
        if args.enable_motion:
            cameras = config["cameras"]
            # 三个频率分别控制机器人伺服、状态反馈、观测/录制采样。
            # controller_kwargs 从 controller 配置读取插值/伺服参数并转换角度单位；
            # 它独立于 motion 中的手部目标限速，避免把两层速度约束混用。
            # 相机仅使用 RGB 序列号；metadata 保存配置和实际标定内容用于回溯。
            env = stack.enter_context(DualAuboRealEnv(
                arms["left"]["ip"], arms["right"]["ip"], arms["left"]["port"], arms["right"]["port"],
                shm_manager=shm, robot_frequency=motion["robot_frequency"], state_frequency=motion["state_frequency"],
                frequency=motion["record_frequency"], **controller_kwargs(config),
                command_timeout=motion["command_timeout"], output_dir=args.output,
                camera_serial_numbers=[] if args.no_cameras else cameras["serial_numbers"],
                video_capture_resolution=tuple(cameras["capture_resolution"]), video_capture_fps=cameras["capture_fps"],
                obs_image_resolution=tuple(cameras["obs_resolution"]), record_images=args.record_images,
                gripper_configs={side: arm["gripper"] for side, arm in arms.items()}, workspace=workspace,
                recording_metadata=recording_metadata(config)))
        # 初始 TCP 来自实测反馈，公开位姿为各臂基坐标系中的 xyz(m)+rotvec(rad)。
        # 会话初始暂停；按 Space 后才把当前手部姿态与实际 TCP 建立对应锚点。
        poses = ({side: state["ActualTCPPose"] for side, state in env.get_state().items()} if env
                 else {side: status["tcp_pose_rotvec"] for side, status in statuses.items()})
        session = TeleopSession(config, poses, workspace, env)
        # 键盘适配器过滤按住按键时的重复事件，使 V/B/N 每次按下只切换一次。
        keyboard = stack.enter_context(create_keyboard())
        print("Space: resume/pause; H: joint home; C/S: start/save episode; Backspace: drop last; Q/Esc: quit")
        print("1/2/3: camera view; V: left DO00; B: left DO01; N: right DO00 (toggle high/low)")
        print("MOTION ENABLED, initially paused" if env else "READ-ONLY PREVIEW, initially paused")
        # 控制循环使用单调时钟计算 dt 和运行时长；观测与日志单独降频。
        # 发送给环境的目标时间和录制时间则使用 time.time() 的绝对时间。
        period = 1. / motion["teleop_frequency"]
        started = previous = time.monotonic()
        next_obs = next_report = started
        samples, view = None, 0
        while args.duration is None or time.monotonic() - started < args.duration:
            # 采用实际经过时间驱动映射滤波与限速，而非假定每轮严格等于 period。
            cycle = time.monotonic()
            dt, previous = max(1e-6, cycle - previous), cycle
            # 一次获取两只手的有效输入；任一设备过期或追踪失效时拒绝整对样本。
            try:
                samples = inputs.get()
            except RuntimeError as exc:
                samples = None
                # 遥操中输入丢失：保持当前位置并结束录制；恢复后需要手动重新锚定。
                if session.mode == "teleop":
                    session.pause()
                    if env and env._recording:
                        env.end_episode()
                    print(f"Input lost: paused; press Space after tracking recovers. {exc}")
                # 回零期间输入丢失直接退出，资源清理流程停止左右控制器。
                elif session.mode == "homing":
                    raise RuntimeError("Input lost during home; stopping both arms") from exc
            quit_requested = False
            # 先处理本轮按键，再更新目标，保证暂停/回零/退出及时影响当前周期。
            for key in keyboard.get_press_events():
                character = getattr(key, "char", "")
                # Q/Esc 跳出主循环；环境退出时结束当前录制并停止控制器。
                if key == Key.esc or character == "q":
                    quit_requested = True
                    break
                if key == Key.space:
                    # Space：暂停时保持当前关节；恢复时用新样本重新锚定，避免目标跳变。
                    # 回零中按 Space 视为操作员中止，退出并停止双臂。
                    if session.mode == "homing":
                        raise RuntimeError("Operator aborted home; stopping both arms")
                    if session.mode == "teleop":
                        session.pause()
                    elif samples is not None:
                        session.resume(samples)
                    else:
                        print("Cannot resume: tracking is not valid")
                elif character == "h" and env and session.mode != "homing":
                    # H：进入六关节回零状态，回零结束后停留在暂停状态。
                    session.home(homes)
                elif character == "c" and env:
                    # C：仅在已指定目录、处于遥操且未录制时新建 episode。
                    # 开始时间与目标调度提前量对齐，后续 get_obs/exec_actions 累积数据。
                    if not args.output:
                        print("Recording requires --output")
                    elif session.mode != "teleop":
                        print("Resume teleoperation before starting an episode")
                    elif not env._recording:
                        env.start_episode(time.time() + motion["command_latency"])
                elif character == "s" and env:
                    # S：保存并结束当前 episode，遥操本身继续。
                    env.end_episode()
                elif key == Key.backspace and env:
                    # Backspace：交给环境结束当前录制并删除相应/最近的 episode。
                    env.drop_episode()
                elif character in ('v', 'b', 'n'):
                    # V/B 分别切换左 DO00/DO01，N 切换右 DO00，通道间独立。
                    # 会话只接受已启用 I/O 且正在遥操的切换，实际输出随目标动作发送。
                    side, pin = {'v': ('left', 0), 'b': ('left', 1), 'n': ('right', 0)}[character]
                    try:
                        session.toggle_output(side, pin)
                    except RuntimeError as exc:
                        print(exc)
                elif character in ("1", "2", "3"):
                    # 1/2/3：选择 RGB 预览，显示时按实际相机数量取模。
                    view = int(character) - 1
            if quit_requested:
                break
            # tick 在遥操中执行手部映射→边界校验→带时间戳的双臂目标下发；
            # 暂停中仅维护心跳，回零中检查完成状态。暂停允许 samples=None。
            if samples is not None or session.mode == "paused":
                session.tick(samples, dt)
            if env and cycle >= next_obs:
                # 按 record_frequency 对齐机器人/RGB 观测；录制中由环境累积这些样本。
                obs = env.get_obs()
                next_obs = cycle + 1. / motion["record_frequency"]
                if env.realsense and not args.headless:
                    # OpenCV 显示最新 RGB 帧：转换为连续 BGR 数组并叠加会话状态。
                    import cv2
                    names = sorted(key for key in obs if key.startswith("camera_"))
                    if names:
                        frame = np.ascontiguousarray(obs[names[view % len(names)]][-1][..., ::-1])
                        cv2.putText(frame, session.mode, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2)
                        cv2.imshow("AUBO teleop", frame)
                        cv2.waitKey(1)
            if cycle >= next_report:
                # 每两秒输出当前状态和目标位姿；这里打印的是目标，不是实测反馈。
                print(session.mode, {side: np.round(pose, 4).tolist() for side, pose in session.poses.items()}, flush=True)
                next_report = cycle + 2.
            # 扣除本轮处理耗时后补足周期；超时周期直接进入下一轮。
            time.sleep(max(0., period - (time.monotonic() - cycle)))


# 统一入口：先校验参数与配置，再决定离线检查、标定或硬件遥操。
def main(argv=None):
    args = parser().parse_args(argv)
    # 拒绝非法运行时长；独立标定允许尚未存在有效标定文件。
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        raise ValueError("--duration must be finite and positive")
    config = load_config(args.config, validate_calibrations=not bool(args.calibrate))
    # --enable-io 将两臂预留输出切换到控制柜标准数字输出接口。
    # AuboDigitalGripper(None, ...) 此处仅校验通道配置，不写入任何电平。
    if args.enable_io:
        if not args.enable_motion and not args.check_config:
            raise ValueError('--enable-io requires --enable-motion')
        if args.calibrate:
            raise ValueError('I/O cannot be enabled during calibration')
        from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
        for arm in config['arms']:
            arm['gripper']['type'] = 'standard_digital'
            AuboDigitalGripper(None, arm['gripper'])
    # 离线检查校验绑定的回零记录和工作空间配置，然后返回，绕过 run 的硬件加载。
    if args.check_config:
        for arm in config["arms"]:
            load_home(config["home_file"], arm["ip"], arm["serial"])
        DualWorkspace(config["arms"], config["workspace"])
        print("Configuration valid; no SDK, robot, camera, or I/O accessed.")
        return
    # 录制必须来自已启用运动的环境，不能把只读预览目标当作示教数据保存。
    if args.output and not args.enable_motion:
        raise ValueError("Recording requires --enable-motion")
    run(args, config)


# 直接运行脚本时启动主入口；Ctrl+C 经上下文管理器清理资源后安静退出。
# 其他异常保持可见，便于定位配置、输入或机器人故障。
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
