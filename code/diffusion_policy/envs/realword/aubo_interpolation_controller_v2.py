"""按 Franka 控制器的阅读顺序组织 Aubo：Interface → Process → 命令 → 控制循环。

FrankaInterface 对接 zerorpc/笛卡尔阻抗；这里的 AuboInterface 对接
RPC + RTDE，执行 xyz/rotvec → SDK RPY → IK → servoJoint。
保留已验证的 Aubo 命令顺序、回零、HOLD、看门狗和伺服返回码处理。
共享内存、进程生命周期及 SDK/RTDE 启动细节复用原 Aubo 控制器；
本文件显式实现新的 Interface 和主循环，不调用原控制器的 run()。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import multiprocessing as mp
import os
import time

import numpy as np

from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
from diffusion_policy.envs.realword.aubo_homing import JointHome, LockedCartesianHome
from diffusion_policy.envs.realword.aubo_interpolation_controller import (
    AuboInterpolationController as _AuboController,
    Command, Empty, require_pyaubo_sdk, rotvec_to_aubo, aubo_to_rotvec,
    SharedMemoryQueue, SharedMemoryRingBuffer,
)

# v2-only reliable event; position updates use a separate overwriteable slot.
GRIPPER_COMMAND = 6


class _LatestPose:
    """One atomic pose snapshot, shared by the parent and servo process."""

    def __init__(self):
        self.data = mp.Array('d', 7, lock=False)  # xyz/rotvec + monotonic receipt time
        self.revision = mp.Value('Q', 0)

    def put(self, pose, received):
        with self.revision.get_lock():
            self.data[:6] = pose
            self.data[6] = received
            self.revision.value += 1

    def get(self):
        with self.revision.get_lock():
            return self.revision.value, np.array(self.data[:6]), self.data[6]

    def clear(self):
        with self.revision.get_lock():
            self.data[6] = 0.
            self.revision.value += 1
            return self.revision.value


# ========= robot interface（对应 FrankaInterface） =========
class AuboInterface:
    """只在控制子进程中创建：持有 RPC、RTDE 和柜体 I/O，读取反馈不占用 RPC。"""

    def __init__(self, controller, sdk):
        # 保存控制器参数和 SDK；这里只初始化句柄，连接在子进程 connect() 中建立。
        self.controller, self.sdk = controller, sdk
        self.rpc = self.rtde = self.rtde_topic = self.motion = None
        self.servo_enabled = False

    def connect(self):
        # 每个 Interface 对应一个控制柜 IP，左右臂由环境分别创建。
        owner = self.controller
        # 建立 RPC 通道并核对连接结果，避免登录/运动指令发往无效连接。
        self.rpc = self.sdk.RpcClient()
        self.rpc.setRequestTimeout(1000)
        self.rpc.connect(owner.robot_ip, owner.robot_port)
        if not self.rpc.hasConnected():
            raise RuntimeError(f"cannot connect to AUBO RPC {owner.robot_ip}:{owner.robot_port}")
        # 登录后检查状态；柜体用户名、密码由控制器构造参数传入。
        self.rpc.login(owner.username, owner.password)
        if not self.rpc.hasLogined():
            raise RuntimeError(f"cannot login to AUBO RPC {owner.robot_ip}")
        # 选取该柜体报告的机器人，并取得运动控制和 IK 算法接口。
        names = list(self.rpc.getRobotNames())
        if not names:
            raise RuntimeError(f"no robot reported by {owner.robot_ip}")
        self.robot = self.rpc.getRobotInterface(names[0])
        self.motion = self.robot.getMotionControl()
        self.algorithm = self.robot.getRobotAlgorithm()

    def start_servo(self):
        """与 v1 相同：状态检查 → startMove → RTDE 首帧 → runtime → 关节保持预热。"""
        owner = self.controller
        # 复用原控制器的上电/示教器状态检查；是否自动上电由 auto_power_on 决定。
        owner._power_on(self.robot)
        # 恢复运动接收状态，检查 startMove 返回码后设置柜体速度比例。
        ret = self.motion.startMove()
        if ret not in (None, 0):
            raise RuntimeError(f"startMove rejected: {ret}")
        self.motion.setSpeedFraction(owner.speed_fraction)
        # I/O 适配器启动时读取当前输出锁存；禁用模式不获取柜体 I/O 接口。
        self.gripper = AuboDigitalGripper(
            self.robot.getIoControl() if owner.gripper_config['type'] != 'disabled' else None,
            owner.gripper_config)
        self.gripper.start()
        # 启动独立 RTDE 反馈订阅，主伺服循环从缓存读状态。
        self.rtde, self.rtde_topic, self.rtde_lock, self.rtde_latest = owner._start_rtde()
        deadline = time.monotonic() + 3.
        # 等待首个有效反馈，用实测关节作为预热保持点；三秒无反馈则退出。
        while True:
            state = self.get_state()
            if state is not None:
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"no valid RTDE state received from {owner.robot_ip}")
            time.sleep(.02)
        # 启动 runtime，再以首帧关节进入 servo 模式并预热保持。
        owner._check_pose(state['eef_pose'])  # Low feedback enters vertical recovery.
        owner._start_runtime(self.rpc.getRuntimeMachine(), self.sdk)
        self.servo_enabled = True  # 即使预热失败，finally 也需要退出 servo 模式。
        owner._warmup_servo(self.motion, state['joint_angles'].copy(), self.sdk)
        return state

    def get_state(self):
        # 在 RTDE 锁保护下复制最新缓存；尚未收到数据时可能返回 None。
        return self.controller._copy_rtde_state(self.rtde_lock, self.rtde_latest)

    def get_ee_pose(self):
        # 对外返回各机器人基坐标系中的 xyz(m)+rotvec(rad)，复制后可独立修改。
        return self.get_state()['eef_pose'].copy()

    def get_joint_positions(self):
        # Aubo 为六关节，单位 rad；读取的是实际反馈而非上次命令值。
        return self.get_state()['joint_angles'].copy()

    def get_joint_velocities(self):
        # 返回实际六关节速度 rad/s，回零完成判断也使用此反馈。
        return self.get_state()['joint_speeds'].copy()

    def inverse_kinematics(self, seed, pose):
        # 主框架使用 rotvec，仅在 SDK 边界转为 Aubo 的 RPY。
        # seed 是上一条成功接收的关节目标，用于维持 IK 解的连续性。
        return self.algorithm.inverseKinematics(list(map(float, seed)), rotvec_to_aubo(pose))

    def forward_kinematics(self, joints):
        pose, ret = self.algorithm.forwardKinematics(list(map(float, joints)))
        ret = int(ret)
        return (aubo_to_rotvec(pose) if ret == 0 else np.asarray(pose)), ret

    def servo_joint(self, joints):
        # 复用已验证的 SDK 下发格式，a/v/t/lookahead/gain 均来自控制器参数。
        return self.controller._servo_joint(self.motion, joints)

    def close(self):
        # 输出电平沿用 v1：退出不自动切换吸取/夹爪状态。
        # 只对已经进入的 servo 模式执行退出；某一步失败也继续尝试释放其他资源。
        if self.motion is not None and self.servo_enabled:
            try:
                self.controller._set_servo_mode(self.motion, False)
            except Exception:
                pass
        if self.rtde is not None:
            # 撤销订阅、注销并断开 RTDE 通道。
            try:
                if self.rtde_topic is not None:
                    self.rtde.removeTopic(False, self.rtde_topic)
                self.rtde.logout()
                self.rtde.disconnect()
            except Exception:
                pass
        if self.rpc is not None:
            # RPC 登录状态与连接分别清理，登录失败时也能尝试断开连接。
            try:
                if self.rpc.hasLogined():
                    self.rpc.logout()
            except Exception:
                pass
            try:
                self.rpc.disconnect()
            except Exception:
                pass


@dataclass
class _LoopState:
    """子进程本地状态；父进程通过队列传命令，通过 ring_buffer 收反馈。"""
    # 当前 TCP 插值轨迹、目标位姿与最后成功接收的关节种子。
    interpolator: PoseTrajectoryInterpolator
    target_pose: np.ndarray
    q_seed: np.ndarray
    # 两个时间均使用单调时钟：一个维持轨迹排程，一个供看门狗判断。
    last_waypoint_time: float
    last_command_at: float
    # 回零轨迹和关节保持状态决定是否绕过普通笛卡尔 IK 分支。
    home_plan: JointHome | None = None
    joint_hold: bool = False
    # 每个实例单独维护待执行 I/O 队列；计数器用于持续 IK 失败和 servo 恢复判断。
    gripper_events: deque = field(default_factory=deque)
    ik_failed: int = 0
    servo_ignored: int = 0
    teleop_revision: int = 0
    teleop_active: bool = False
    immediate_gripper_events: deque = field(default_factory=deque)
    height_recovery_pose: np.ndarray | None = None


# ========= interpolation controller（对应 FrankaInterpolationController） =========
class AuboInterpolationControllerV2(_AuboController):
    """独立 v2 主循环，沿用 v1 的共享内存协议和公开控制接口。

    阅读顺序与 Franka 一致：
      __init__ → start/stop → servoL/schedule_waypoint → get_state → run。
    start/stop/is_ready、get_state/get_all_state 和命令入队方法由父类实现；
    HOME 仍是六关节轨迹，不能套用 Franka 的七关节、阻抗或夹爪宽度接口。
    """

    # ========= initialization / launch / receive APIs =========
    def __init__(self, *args, **kwargs):
        # 父类负责参数校验、命令队列、反馈缓冲区和进程同步对象。
        super().__init__(*args, **kwargs)
        # 标记进程版本，便于终端和进程列表区分左右 v2 控制器。
        self.name = f'AuboControllerV2-{self.robot_ip}'
        self.latest_teleop_pose = _LatestPose()
        self.workspace = None
        self.workspace_side = None

    def set_teleop_target(self, pose):
        """Replace the realtime goal, without an arrival deadline or speed limit."""
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError('Teleop pose must be finite xyz+rotvec with shape (6,)')
        if not self.is_ready or self.stop_event.is_set():
            raise RuntimeError('Controller is not ready for teleoperation')
        if self.home_state.value == 1:
            raise RuntimeError('Cartesian targets are blocked during joint homing')
        workspace = getattr(self, 'workspace', None)
        if workspace is not None:
            pose = workspace.limit_target(self.workspace_side, pose)
        self.latest_teleop_pose.put(pose, time.monotonic())

    def set_gripper(self, value):
        """Enqueue every I/O transition independently of position updates."""
        AuboDigitalGripper(None, self.gripper_config).validate_command(value)
        if not self.is_ready or self.stop_event.is_set():
            raise RuntimeError('Controller is not ready for I/O')
        self.input_queue.put({'cmd': GRIPPER_COMMAND, 'gripper': float(value)})

    def _discard_teleop(self, loop):
        loop.teleop_revision = self.latest_teleop_pose.clear()
        loop.teleop_active = False

    def _consume_teleop(self, loop, now):
        revision, pose, received = self.latest_teleop_pose.get()
        if not received or revision == loop.teleop_revision:
            return
        loop.teleop_revision = revision
        if self.command_timeout is not None and now - received > self.command_timeout:
            return  # An old mailbox snapshot cannot revive the watchdog.
        if loop.home_plan is not None:
            if self.home_state.value == 1:
                return
            loop.home_plan = None
            self.home_state.value = 0
        loop.last_command_at = max(loop.last_command_at, received)
        loop.joint_hold = False
        loop.teleop_active = True
        loop.target_pose = pose
        loop.interpolator = PoseTrajectoryInterpolator([now], [pose])
        loop.last_waypoint_time = now

    # ========= command APIs =========
    # servoL、schedule_waypoint、return_home、hold、heartbeat 沿用同一队列协议。
    # 下面把子进程里的命令消费从 run 中拆出，主循环可以按功能顺序直接阅读。
    def _process_commands(self, robot, loop, now, dt):
        # 一次取出父进程累积的整批命令；队列为空时继续保持当前轨迹。
        try:
            commands = self.input_queue.get_all()
            count = len(commands['cmd'])
        except Empty:
            count = 0
        motion_blocked = False
        if count:
            # 收到任何命令（包括心跳）都刷新看门狗时间。
            loop.last_command_at = now
            if np.any(commands['cmd'] == Command.STOP.value):
                self._discard_teleop(loop)
                return False  # STOP 抢占本批所有目标和回零命令。
            if np.any(commands['cmd'] == Command.HOLD.value):
                # 回零中收到 HOLD 视为中止；普通暂停先核对反馈仍然有效。
                if loop.home_plan is not None and self.home_state.value == 1:
                    raise RuntimeError('Homing aborted; stopping servo')
                state = robot.get_state()
                if state is None or time.time() - state['received'] > .6:
                    raise RuntimeError('Stale state at hold')
                # 冻结 TCP 插值轨迹并保持最后成功接收的关节，不回退到滞后的反馈关节。
                # 清空未来排程 I/O 和位置目标；独立即时 I/O 仍逐条处理。
                loop.target_pose = loop.interpolator(now)
                loop.interpolator = PoseTrajectoryInterpolator([now], [loop.target_pose])
                loop.gripper_events.clear()
                self._discard_teleop(loop)
                loop.home_plan, loop.joint_hold = None, True
                self.home_state.value = 0
                motion_blocked = True
        for index in range(count):
            if (index + 1 < count and commands['cmd'][index] == Command.SERVOL.value
                    and commands['cmd'][index + 1] == Command.SERVOL.value):
                continue  # 连续 SERVOL 合并，带时间戳的 waypoint 逐个保留。
            command = {key: value[index] for key, value in commands.items()}
            kind = int(command['cmd'])
            if kind == GRIPPER_COMMAND:
                loop.immediate_gripper_events.append(float(command['gripper']))
                continue
            if motion_blocked:
                continue  # HOLD cancels motion, but does not coalesce I/O events.
            if kind == Command.HOME.value:
                # HOME 的 target_pose 字段承载六关节目标，建立受速度/加速度约束的回零轨迹。
                if self._orientation_locked():
                    # FK 只确定归位 XYZ；记录姿态最后覆盖，归位途中不插值旋转。
                    state = robot.get_state()
                    if state is None or time.time() - state['received'] > .6:
                        raise RuntimeError('Stale state at locked home transition')
                    pose, ret = robot.forward_kinematics(command['target_pose'])
                    if int(ret) != 0:
                        raise RuntimeError(f'Home forwardKinematics failed: {ret}')
                    goal = self.workspace.limit_target(self.workspace_side, pose)
                    loop.home_plan = LockedCartesianHome(
                        state['joint_angles'], command['target_pose'], state['eef_pose'], goal)
                    loop.q_seed = state['joint_angles'].copy()
                else:
                    self._check_joint_target(robot, command['target_pose'])
                loop.joint_hold = False
                if not self._orientation_locked():
                    loop.home_plan = JointHome(loop.q_seed, command['target_pose'])
                loop.gripper_events.clear()
                self._discard_teleop(loop)
                self.home_state.value = 1
                continue
            if kind == Command.HEARTBEAT.value:
                # 心跳仅维持连接，不解除保持，也不插入新的运动目标。
                continue
            loop.joint_hold = False
            if loop.home_plan is not None:
                # 回零执行中忽略笛卡尔目标；完成后用真实 TCP 重建插值器再释放回零状态。
                if self.home_state.value == 1:
                    continue
                state = robot.get_state()
                if state is None or time.time() - state['received'] > .6:
                    raise RuntimeError('Stale state at home release')
                loop.interpolator = PoseTrajectoryInterpolator([now], [state['eef_pose']])
                loop.home_plan = None
                self.home_state.value = 0
            if kind == Command.SERVOL.value:
                self._discard_teleop(loop)
                # Update and sample at the same time; no future-start clipping.
                insert_time = now + float(command['duration'])
                loop.interpolator = loop.interpolator.drive_to_waypoint(
                    pose=command['target_pose'], time=insert_time, curr_time=now,
                    max_pos_speed=self.max_pos_speed, max_rot_speed=self.max_rot_speed)
                loop.last_waypoint_time = insert_time
            elif kind == Command.SCHEDULE_WAYPOINT.value:
                self._discard_teleop(loop)
                # 父进程传入 Unix 绝对时间，这里转换到插值器使用的单调时钟。
                target_time = time.monotonic() - time.time() + float(command['target_time'])
                loop.interpolator = loop.interpolator.schedule_waypoint(
                    pose=command['target_pose'], time=target_time, curr_time=now,
                    last_waypoint_time=loop.last_waypoint_time,
                    max_pos_speed=self.max_pos_speed, max_rot_speed=self.max_rot_speed)
                loop.last_waypoint_time = target_time
                if float(command.get('gripper', -1.)) >= 0:
                    # 非负 gripper 值是输出命令，与该 waypoint 的目标时间绑定；-1 表示无命令。
                    loop.gripper_events.append((target_time, float(command['gripper'])))
            else:
                # 未识别的协议命令直接报错，由 run 的 finally 退出伺服并释放连接。
                raise RuntimeError(f'unknown controller command {kind}')
        if not motion_blocked:
            self._consume_teleop(loop, now)
        # Mailbox receipt times, queue events and heartbeats all feed the watchdog.
        if self.command_timeout is not None and now - loop.last_command_at > self.command_timeout:
            raise RuntimeError('teleop command watchdog expired')
        return True

    supports_height_recovery = True

    def _orientation_locked(self):
        # 姿态锁与高度开关独立；关闭高度检查也保持固定姿态。
        workspace = getattr(self, 'workspace', None)
        return workspace is not None and workspace.orientation_locked(self.workspace_side)

    def _check_pose(self, pose, target=False):
        workspace = getattr(self, 'workspace', None)
        if workspace is not None:
            if target:
                workspace.validate_target(self.workspace_side, pose)
            else:
                workspace.validate_feedback(self.workspace_side, pose)

    def _height_recovery(self, robot, loop, state, now):
        workspace = getattr(self, 'workspace', None)
        recovery = (workspace.recovery_target(self.workspace_side, state['eef_pose'])
                    if workspace is not None and workspace.enabled else None)
        if recovery is not None:
            if loop.height_recovery_pose is None:
                # 恢复只锁存实测水平位置；recovery 已先抬高、再覆盖归位姿态。
                loop.height_recovery_pose = recovery.copy()
                print(f'[{self.robot_ip}] TCP below height boundary; lifting to '
                      f'{workspace.minimum_height(self.workspace_side) * 1000:.2f} mm', flush=True)
            loop.target_pose = loop.height_recovery_pose.copy()
            self._check_pose(loop.target_pose, target=True)
            joints, ret = robot.inverse_kinematics(loop.q_seed, loop.target_pose)
            joints = np.asarray(joints, dtype=float)
            if int(ret) != 0 or joints.shape != (6,) or not np.isfinite(joints).all():
                raise RuntimeError(f'Height recovery inverseKinematics failed: {ret}')
            return joints
        if loop.height_recovery_pose is not None:
            loop.interpolator = PoseTrajectoryInterpolator([now], [loop.height_recovery_pose])
            loop.last_waypoint_time = now
            if loop.home_plan is not None:
                if isinstance(loop.home_plan, LockedCartesianHome):
                    # 保持笛卡尔归位类型，不退回会改变末端姿态的关节轨迹。
                    loop.home_plan = loop.home_plan.restart(state['joint_angles'], state['eef_pose'])
                else:
                    loop.home_plan = JointHome(state['joint_angles'], loop.home_plan.target)
                loop.q_seed = state['joint_angles'].copy()
            loop.height_recovery_pose = None
            print(f'[{self.robot_ip}] Height recovery complete', flush=True)
        return None

    def _check_joint_target(self, robot, joints, *, target=True):
        workspace = getattr(self, 'workspace', None)
        if workspace is not None and (workspace.enabled or self._orientation_locked()):
            pose, ret = robot.forward_kinematics(joints)
            if int(ret) != 0:
                raise RuntimeError(f'Height check forwardKinematics failed: {ret}')
            self._check_pose(pose, target=True)

    def _target_joints(self, robot, loop, now, dt):
        # 每轮先检查 RTDE 新鲜度，避免用停止更新的反馈继续求解和控制。
        state = robot.get_state()
        if state is None or time.time() - state['received'] > .6:
            raise RuntimeError('RTDE state is stale; stopping servo')
        self._check_pose(state['eef_pose'])
        if (self.max_following_error is not None
                and np.max(np.abs(state['joint_angles'] - loop.q_seed)) > self.max_following_error):
            raise RuntimeError('Joint following error exceeded')
        recovery_joints = self._height_recovery(robot, loop, state, now)
        if recovery_joints is not None:
            return recovery_joints
        if loop.home_plan is not None:
            if isinstance(loop.home_plan, LockedCartesianHome):
                # 归位：XYZ 曲线 → 高度修正 → 最后覆盖姿态 → IK。
                if self.home_state.value == 1 and loop.home_plan.check_pose(
                        state['eef_pose'], state['eef_speed'], state['received']):
                    self.home_state.value = 2
                loop.target_pose = self.workspace.limit_target(
                    self.workspace_side, loop.home_plan.sample(dt))
                joints, ret = robot.inverse_kinematics(loop.q_seed, loop.target_pose)
                joints = np.asarray(joints, dtype=float)
                if int(ret) != 0 or joints.shape != (6,) or not np.isfinite(joints).all():
                    raise RuntimeError(f'Locked home inverseKinematics failed: {ret}')
                return joints
            # 回零直接生成关节目标；TCP 目标字段记录当前实测 TCP，供父进程观察。
            loop.target_pose = state['eef_pose'].copy()
            if self.home_state.value == 1 and loop.home_plan.check(
                    state['joint_angles'], state['joint_speeds'], state['received'], loop.q_seed):
                self.home_state.value = 2
            # 轨迹进度只有在 _send_joints 成功后推进，这里仅采样本轮候选目标。
            return loop.home_plan.sample(dt)
        if loop.joint_hold and not self._orientation_locked():
            # 暂停时重复最后成功接收的关节目标，绕过 IK。
            return loop.q_seed.copy()
        if not loop.teleop_active and not loop.joint_hold:
            loop.target_pose = loop.interpolator(now)
        workspace = getattr(self, 'workspace', None)
        if workspace is not None and (loop.teleop_active or self._orientation_locked()):
            loop.target_pose = workspace.limit_target(self.workspace_side, loop.target_pose)
        self._check_pose(loop.target_pose, target=True)
        # 实时路径用最新 TCP；排程路径用当前采样，二者均用连续关节种子求 IK。
        joints, ret = robot.inverse_kinematics(loop.q_seed, loop.target_pose)
        joints, ret = np.asarray(joints), int(ret)
        if ret:
            if self._orientation_locked():
                # 不能回退旧关节或按关节缩步：旧姿态可能倾斜，不符合严格锁定。
                raise RuntimeError(f'Locked orientation inverseKinematics failed: {ret}')
            # 短暂 IK 失败保持上一关节目标；连续超过 15 次才终止会话。
            loop.ik_failed += 1
            if loop.ik_failed > 15:
                raise RuntimeError(f'inverseKinematics failed on {self.robot_ip}: {ret} ({robot.sdk.returnValue2Str(ret)})')
            joints = loop.q_seed
        else:
            loop.ik_failed = 0
        if loop.teleop_active or self._orientation_locked():
            # Direct teleop has no Python Cartesian/joint speed or step limiter.
            if joints.shape != (6,) or not np.isfinite(joints).all():
                raise ValueError('Invalid IK joint vector')
            return joints
        # 旧排程路径仍按原配置限制关节步长/速度。
        limit = self.max_joint_step
        if self.max_joint_speed is not None:
            limit = min(limit, self.max_joint_speed * dt)
        if self.reject_joint_jumps and np.max(np.abs(joints - loop.q_seed)) > limit:
            # 启用拒绝模式时，越界 IK 解直接报错；其余情况按最大关节变化同比缩小。
            raise RuntimeError('IK joint jump exceeded configured limit')
        return self._limit_joint_step(loop.q_seed, joints, limit)

    def _send_joints(self, robot, loop, joints, dt):
        # 返回码决定是否接受候选关节，不能把发送尝试等同于柜体已经接收。
        # Check the final joint command after IK/limits, including every home point.
        # Every sent joint goal must satisfy the floor, including recovery and hold.
        self._check_joint_target(robot, joints)
        if self.stop_event.is_set():
            return  # A stop received during the FK RPC must prevent the servo call.
        ret = robot.servo_joint(joints)
        if ret == 0:
            # 成功后更新 IK 种子，并通知回零轨迹推进一个周期。
            loop.servo_ignored = 0
            loop.q_seed = joints
            if loop.home_plan is not None and loop.height_recovery_pose is None:
                loop.home_plan.accepted(dt)
        elif ret == 2:
            loop.servo_ignored = 0  # 队列满：不推进关节种子和回零时间。
        elif ret == -13 and loop.servo_ignored < 3:
            # 连续最多恢复三次：重新选择 servo 模式，短暂等待后进入下一轮。
            loop.servo_ignored += 1
            robot.motion.setServoModeSelect(1)
            time.sleep(.02)
        else:
            # 其他错误或恢复次数耗尽时退出，由上层统一执行连接清理。
            raise RuntimeError(f'servoJoint failed on {self.robot_ip}: {ret} ({robot.sdk.returnValue2Str(ret)})')

    def _publish_state(self, robot, loop):
        # DO 写入后的首次读取可能仍是旧电平；未确认期间以最多 10 Hz
        # 继续读取输出锁存，确认后停止轮询。运动反馈仍走 RTDE。
        robot.gripper.poll_feedback()
        # 反馈字段用 RTDE 实际接收时间，避免把重复发布旧样本误认为新反馈。
        state = robot.get_state()
        if state is not None:
            # 同时发布实际位姿/关节、当前目标和 I/O 锁存缓存，父进程据此对齐观测。
            self.ring_buffer.put(dict(
                ActualTCPPose=state['eef_pose'], ActualTCPSpeed=state['eef_speed'],
                ActualQ=state['joint_angles'], ActualQd=state['joint_speeds'],
                TargetTCPPose=loop.target_pose, robot_receive_timestamp=state['received'],
                GripperState=robot.gripper.state, GripperEnabled=robot.gripper.enabled))

    # ========= main loop in process（与 Franka 相同：接口、插值器、循环、finally） =========
    def run(self):
        # 接口在子进程中创建，父进程只通过共享内存队列/缓冲区交换数据。
        robot = None
        try:
            sdk = require_pyaubo_sdk()
            if self.soft_real_time:
                # 可选实时调度需系统权限；默认不启用。
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))
            robot = AuboInterface(self, sdk)
            robot.connect()
            state = robot.start_servo()
            # 用实际首帧初始化单点保持轨迹，所有插值时间使用单调时钟。
            dt, now = 1. / self.frequency, time.monotonic()
            loop = _LoopState(
                PoseTrajectoryInterpolator([now], [state['eef_pose'].copy()]),
                state['eef_pose'].copy(), state['joint_angles'].copy(), now, now,
                joint_hold=True)
            self._publish_state(robot, loop)  # 先发布真实首帧，再通知父进程 ready。
            self.startup_ok.value = True
            self.ready_event.set()
            next_cycle = time.monotonic()
            # 独立 stop_event 即使在命令队列饱和时也能请求停止。
            while not self.stop_event.is_set():
                now = time.monotonic()
                # 1. 消费命令：先处理 STOP/HOLD，再更新插值器或六关节回零轨迹。
                if not self._process_commands(robot, loop, now, dt):
                    break
                # 2. 插值得到 TCP，再用 Aubo IK 转为六关节目标；回零/保持有独立分支。
                joints = self._target_joints(robot, loop, now, dt)
                if self.stop_event.is_set():
                    break
                # 3. 到达目标时间的 I/O 与关节伺服一起处理。
                while loop.immediate_gripper_events and not self.stop_event.is_set():
                    robot.gripper.command(loop.immediate_gripper_events.popleft())
                while (loop.gripper_events and loop.gripper_events[0][0] <= now
                       and not self.stop_event.is_set()):
                    robot.gripper.command(loop.gripper_events.popleft()[1])
                if self.stop_event.is_set():
                    break
                self._send_joints(robot, loop, joints, dt)
                # 4. 通过 ring_buffer 发布真实 RTDE 反馈和当前目标，不用 RPC 轮询状态。
                self._publish_state(robot, loop)
                # 5. 扣除工作耗时，维持配置中的控制频率（现场为 50 Hz）。
                next_cycle += dt
                finished = time.monotonic()
                if next_cycle <= finished:
                    # Skip missed slots instead of bursting queued servo calls.
                    next_cycle += (int((finished - next_cycle) / dt) + 1) * dt
                time.sleep(max(0., next_cycle - time.monotonic()))
        except Exception as exc:
            # 保留异常原因并重新抛出，让父进程能识别控制子进程退出。
            print(f'[AuboInterpolationControllerV2:{self.robot_ip}] ERROR: {exc}', flush=True)
            raise
        finally:
            # 启动失败也唤醒等待者；startup_ok 用于区分失败与真正就绪。
            if not self.ready_event.is_set():
                self.ready_event.set()
            if robot is not None:
                robot.close()


# 与框架原模块同名的导出，方便环境/离线回放按统一 API 使用。
AuboInterpolationController = AuboInterpolationControllerV2
