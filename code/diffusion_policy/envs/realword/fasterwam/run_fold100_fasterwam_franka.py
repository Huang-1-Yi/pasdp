#!/usr/bin/env python3
"""
双臂 Franka + FasterWAM 真机脚本。

本文件直接复制自 ``demo_dual_udp_teleop_franka_v3.py``。原有遥操实现和
辅助函数保留在文件中供对照；实际入口是文件末尾的 ``fasterwam_main``，
它只把动作来源由 T265/编码器替换为 FasterWAM action chunk。

本脚本接收 T265/夹爪编码器数据，并用 DualFrankaRealEnv 控制两台
Franka。默认自动启动内置 sender，并通过共享内存把遥操数据送到主程序。
只有显式传入 --start_udp_sender 时才启用本机 UDP 兼容路径。
当前版本已经改成“锚点式相对位姿控制”：

    记录点 = 当前机器人末端位姿 + 当前 UDP 位姿
    遥操目标 = 记录点机器人位姿 + (当前 UDP 位姿 - 记录点 UDP 位姿)

这样可以配套 sender 端发送“基于启动位置的累计相对位移”，避免机器人端
继续逐帧积分误差，导致越操作越偏。

v2 使用固定倍率映射，不再使用空间鼠标风格的 Hamilton 非线性滤波：

    机器人平移增量 = 手持端平移增量 * pos_scale
    机器人旋转增量 = 手持端旋转增量 * rot_scale

两个倍率默认都是 1.0。例如 --pos_scale 1.1 --rot_scale 1.5 表示手持端
移动 10 cm 时机器人目标移动 11 cm，手持端旋转 10 度时机器人目标旋转
15 度。最大平移和旋转速度限制仍然生效，因此快速动作可能短暂滞后，但
最终目标位姿保持上述倍率关系。

重要操作流程：
1. 默认运行时会自动启动共享内存 sender 子进程。启动后会先倒计时 5 秒，
   这段时间内请手动把 T265/夹爪摆到初始姿态。

   需要测试旧 UDP 路径时，增加：

       --start_udp_sender --udp_host 0.0.0.0 --udp_port 5005

2. 启动本脚本后，等待终端出现“遥操数据已就绪”。
3. 按 E 开启遥操。倒计时结束后脚本会重新记录一次锚点，然后开始控制。
4. 操作过程中如果手柄/T265 和机器人姿态感觉不对应，先停止大幅动作，
   按 Z 重新记录锚点。
5. 双臂 sender 按设备编号读取 T265 和编码器，不在采集层绑定物理左右，
   也不会复制单臂数据；编号 0/1 分别发布为 arm index 0/1。
6. 启动 sender 前，请把 T265/夹爪摆到和机器人期望朝向一致；sender v3
   的姿态不再归零，位置以 sender 启动第一帧为 (0,0,0)。
7. --swap_grippers 只交换左右编码器/夹爪输入来源，不交换 T265 位姿。

按键说明：
    E：开启/关闭遥操
    Z：重新记录遥操锚点
    T：锁定/解锁旋转，只控制平移
    G：同时切换两个夹爪开/合
    C：开始录制，并自动开启遥操
    S：停止录制，并关闭遥操
    R：关节空间回到返回位
    H：关节空间回到起始位
    P：末端笛卡尔空间回到 home 位姿
    Backspace：删除上一段 episode
    Q：退出
    1/2/3：切换可视化相机画面
    Space：记录 stage 标记

相机说明：
    本脚本会通过 DualFrankaRealEnv 打开 RealSense D400 相机。推荐显式传入
    --camera_serial_numbers，并按 base,left_wrist,right_wrist 的顺序填写。
    程序内的索引是：
        camera_0 = base/global 基座全局相机
        camera_1 = left wrist 左腕相机
        camera_2 = right wrist 右腕相机

查询相机序列号：

       python -c "from diffusion_policy.envs.realword.single_realsense import SingleRealsense; print(SingleRealsense.get_connected_devices_serial())"

    或者：

       rs-enumerate-devices

    自动检测得到的序列号通常是排序后的列表，不一定对应物理安装位置。
    建议遮挡或拔插相机逐个确认，再显式传入。

期望遥操数据结构，沿用 udp_sender.py / lightumi.teleop_sender：
    {
      "type": "lightumi.teleop_sender",
      "version": 1,
      "sequence": ...,
      "arms": [
        {
          "index": 0,
          "pose": {"valid": 1, "tracker_confidence": 3,
                   "position_m": [x,y,z], "quaternion_xyzw": [x,y,z,w]},
          "gripper": {"valid": 1, "angle_deg": 10.0, "confidence": 1.0}
        },
        {
          "index": 1,
          "pose": {"valid": 1, "tracker_confidence": 3,
                   "position_m": [x,y,z], "quaternion_xyzw": [x,y,z,w]},
          "gripper": {"valid": 0, "angle_deg": 0.0, "confidence": 0.0}
        }
      ]
    }

运行示例：
    sudo chmod 666 /dev/ttyUSB*

    python demo_dual_udp_teleop_franka_v3.py -o data/dual_franka_demo \
        --left_robot_ip 192.168.0.168 --right_robot_ip 192.168.0.168 \
        --camera_serial_numbers <base>,<left_wrist>,<right_wrist> \
        --swap_grippers

    如果只是临时测试原始初始化流程，可以加 --skip_device_check 跳过启动前
    硬件枚举；正常真机实验建议保留自检。

    默认共享内存模式不创建 UDP socket。只有 --start_udp_sender 模式才会
    监听 --udp_host/--udp_port，并让内置 sender 向该端口发送 UDP 数据。

    当前双臂 Franka server 的用法是：left_robot_ip 和 right_robot_ip 都指向
    同一个双臂服务端主机，默认端口 4242。服务端左臂使用普通 RPC 方法名，
    右臂使用带 `1` 后缀的方法名，DualFrankaRealEnv 内部负责区分。
"""

from __future__ import annotations

import json
import socket
import threading
import time
import contextlib
import atexit
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from diffusion_policy.envs.realword.dual_franka_real_env import DualFrankaRealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.envs.realword.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode)
from diffusion_policy.envs.realword.udp_sender_dual_absolute_v3 import (
    ENCODER_PORT0,
    ENCODER_PORT1,
    T265_SERIAL0,
    T265_SERIAL1,
    DualTeleopSenderProcess,
    check_input_devices,
)


# ===========================================================================
# UDP 数据结构与解析器（来自 udp_sender.py）
# ===========================================================================

@dataclass
class PoseSample:
    valid: bool = False
    tracker_confidence: int = 0
    position_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)


@dataclass
class GripperSample:
    valid: bool = False
    angle_deg: float = 0.0
    confidence: float = 0.0


@dataclass
class ArmSample:
    index: int = 0
    pose: Optional[PoseSample] = None
    gripper: Optional[GripperSample] = None


@dataclass
class TeleopPacket:
    packet_type: str = ""
    version: int = 0
    sequence: int = 0
    arms: List[ArmSample] = field(default_factory=list)


def _format_gripper_encoder(sample: Optional[ArmSample]) -> str:
    """Format raw teleop encoder angle for runtime diagnostics."""
    if sample is None:
        return "N/A"
    if sample.gripper is None:
        return "no-grip"
    gripper = sample.gripper
    if not gripper.valid:
        return f"invalid:{gripper.angle_deg:.1f}deg"
    return f"{gripper.angle_deg:.1f}deg,c={gripper.confidence:.2f}"


def parse_packet(data: bytes) -> Optional[TeleopPacket]:
    """解析 UDP 遥操数据包；数据格式错误时返回 None。"""
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None

    if payload.get("type", "") != "lightumi.teleop_sender":
        return None
    if int(payload.get("version", 0)) != 1:
        return None

    arms: List[ArmSample] = []
    for arm_data in payload.get("arms", []):
        arm = ArmSample(index=int(arm_data.get("index", 0)))
        pose_dict = arm_data.get("pose")
        if pose_dict is not None:
            pos = pose_dict.get("position_m", [0.0, 0.0, 0.0])
            quat = pose_dict.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
            arm.pose = PoseSample(
                valid=bool(pose_dict.get("valid", 0)),
                tracker_confidence=int(pose_dict.get("tracker_confidence", 0)),
                position_m=(float(pos[0]), float(pos[1]), float(pos[2])),
                quaternion_xyzw=(float(quat[0]), float(quat[1]),
                                 float(quat[2]), float(quat[3])),
            )
        grip_dict = arm_data.get("gripper")
        if grip_dict is not None:
            arm.gripper = GripperSample(
                valid=bool(grip_dict.get("valid", 0)),
                angle_deg=float(grip_dict.get("angle_deg", 0.0)),
                confidence=float(grip_dict.get("confidence", 0.0)),
            )
        arms.append(arm)
    return TeleopPacket(
        packet_type=str(payload.get("type", "")),
        version=int(payload.get("version", 0)),
        sequence=int(payload.get("sequence", 0)),
        arms=arms,
    )


# ===========================================================================
# UDP 接收线程
# ===========================================================================

class UDPPoseReceiver(threading.Thread):
    """在后台线程中接收 UDP 遥操数据包。

    外部可通过 :meth:`get_latest` 获取每条手臂的最新数据。
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5005):
        super().__init__(daemon=True)
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self._latest: Dict[int, TeleopPacket] = {}  # 手臂索引 -> 最新数据包
        self._running = False
        self._sock = None
        self._seq = 0
        self._dropped = 0

    def get_latest(self, arm_index: int = 0) -> Optional[ArmSample]:
        """返回 arm_index 对应的最新 ArmSample；没有数据时返回 None。"""
        with self._lock:
            pkt = self._latest.get(arm_index)
            if pkt is None:
                return None
            for arm in pkt.arms:
                if arm.index == arm_index:
                    return arm
            return None

    @property
    def stats(self) -> Tuple[int, int]:
        return self._seq, self._dropped

    def stop(self, wait=False):
        self._running = False
        sock = self._sock
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if wait:
            self.stop_wait()

    def stop_wait(self):
        self.stop(wait=False)
        if self.is_alive():
            self.join(timeout=2.0)

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock = sock
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            sock.bind((self._host, self._port))
            sock.settimeout(0.5)

            self._running = True
            print(f"[UDP] 正在监听 {self._host}:{self._port}")

            while self._running:
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except OSError:
                    break

                pkt = parse_packet(data)
                if pkt is None:
                    continue

                with self._lock:
                    for arm in pkt.arms:
                        self._latest[arm.index] = pkt
                    self._seq = pkt.sequence
        finally:
            self._running = False
            self._sock = None
            try:
                sock.close()
            except OSError:
                pass









class SharedMemoryTeleopReceiver:
    """从 sender 子进程共享内存读取遥操输入，并模拟 UDPPoseReceiver 接口。"""

    def __init__(self, sender: DualTeleopSenderProcess):
        self._sender = sender
        self._latest: Dict[int, ArmSample] = {}
        self._seq = 0
        self._dropped = 0

    @property
    def stats(self) -> Tuple[int, int]:
        self._refresh()
        return self._seq, self._dropped

    def get_latest(self, arm_index: int = 0) -> Optional[ArmSample]:
        self._refresh()
        return self._latest.get(arm_index)

    def stop_wait(self):
        pass

    def _refresh(self):
        if self._sender.teleop_ring_buffer.count <= 0:
            return

        data = self._sender.get_teleop()
        seq = int(data["sequence"][0])
        if seq == self._seq and self._latest:
            return
        # 共享内存接收器每次只读最新一帧，序列号跳号是正常覆盖行为，
        # 不是网络丢包；因此丢包计数保持为 0，避免误导诊断。
        self._dropped = 0
        self._seq = seq

        latest = {}
        arm_count = int(data["arm_count"][0])
        for i in range(min(arm_count, len(data["arm_indices"]))):
            arm_index = int(data["arm_indices"][i])
            if arm_index < 0:
                continue
            pose = PoseSample(
                valid=bool(data["pose_valid"][i]),
                tracker_confidence=int(data["tracker_confidence"][i]),
                position_m=tuple(float(x) for x in data["position_m"][i]),
                quaternion_xyzw=tuple(
                    float(x) for x in data["quaternion_xyzw"][i]),
            )
            gripper = GripperSample(
                valid=bool(data["gripper_valid"][i]),
                angle_deg=float(data["gripper_angle_deg"][i]),
                confidence=float(data["gripper_confidence"][i]),
            )
            latest[arm_index] = ArmSample(
                index=arm_index,
                pose=pose,
                gripper=gripper,
            )
        self._latest = latest


# ===========================================================================
# 常量
# ===========================================================================

# Franka 夹爪在控制器中按二值控制：0.08 表示打开，其余值表示闭合。
GRIPPER_OPEN = 0.08
GRIPPER_CLOSE = 0.0
# UDP 夹爪角度的开合阈值，单位为度。LightUMI 编码器报告的夹爪角度
# 范围约为 0（闭合）到 20（打开）。高于阈值时打开，低于阈值时闭合。
# 默认阈值可通过 --gripper_toggle_deg 修改。
GRIPPER_TOGGLE_DEG = 20.0

# 双臂末端允许的最小 z 高度，单位为米。
MIN_Z_HEIGHT = 0.26

# ---- 关节空间归位目标（弧度）----
# H 键回到起始位，R 键在关节空间回到返回位。
HOME_JOINTS_START = {
    'left':  np.array([-0.1382, -0.4586,  0.0709, -2.3744,  0.0180,  1.9361,  0.6942]),
    'right': np.array([-0.1563, -0.4644,  0.0964, -2.3666,  0.0143,  1.9256,  0.6956]),
}
HOME_JOINTS_RETURN = {
    'left':  np.array([ 0.0633, -0.6177, -0.1771, -2.7240,  0.6087,  2.0252,  0.4907]),
    'right': np.array([-0.1842, -0.6493,  0.2949, -2.7437, -0.6790,  2.0836,  1.2106]),
}
JOINT_HOME_DURATION = 4.0

# ---- 末端位姿归位目标（xyz + 旋转向量）----
# P 键在笛卡尔末端位姿空间回到 home 位姿。
DEFAULT_HOME_POSE_RETURN = {
    'left':  np.array([0.34, 0.0, 0.41, -2.5, 1.0, 0.1], dtype=np.float64),
    'right': np.array([0.34, 0.0, 0.41,  2.5, -1.0, 0.36], dtype=np.float64),
}
POSE_HOME_EPSILON = 0.01
POSE_HOME_STEP_SIZE = 0.015
POSE_HOME_EPSILON_ROT = 0.02
POSE_HOME_MAX_STEP_ANGLE = 0.03

# ---- T265 到机器人坐标系的变换 ----
# 实测轴向:
#   机器人 +x 方向移动时，sent_pos_t265 的 z 减少  -> robot_x = -t265_z
#   机器人 +y 方向移动时，sent_pos_t265 的 x 减少  -> robot_y = -t265_x
#   机器人 +z 方向移动时，sent_pos_t265 的 y 增加  -> robot_z = +t265_y
LEFT_T265_TO_ROBOT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=np.float32)

RIGHT_T265_TO_ROBOT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=np.float32)


# ===========================================================================
# 辅助函数
# ===========================================================================


def _cartesian_return_step(target, init_position, epsilon, step_size,
                           epsilon_rot, max_step_angle):
    """在笛卡尔空间中让六维目标位姿逐步靠近初始位姿。

    返回更新后的目标位姿，以及是否已经到达。
    """
    target = target.copy()
    arrived = True

    for i in range(3):
        error = init_position[i] - target[i]
        if abs(error) > epsilon:
            target[i] += np.sign(error) * min(step_size, abs(error))
            arrived = False
        else:
            target[i] = init_position[i]

    cur_rot = st.Rotation.from_rotvec(target[3:6])
    target_rot = st.Rotation.from_rotvec(init_position[3:6])
    rel_rot = target_rot * cur_rot.inv()
    rel_angle = rel_rot.magnitude()
    if rel_angle > epsilon_rot:
        step_angle = min(rel_angle, max_step_angle)
        adjust_rot = st.Rotation.from_rotvec(
            rel_rot.as_rotvec() * (step_angle / rel_angle))
        target[3:6] = (adjust_rot * cur_rot).as_rotvec()
        arrived = False
    else:
        target[3:6] = init_position[3:6]

    return target, arrived


def _latest_pose(receiver: UDPPoseReceiver, arm_index: int):
    """返回最新有效的位置和四元数；无有效数据时返回两个 None。"""
    s = receiver.get_latest(arm_index)
    if s and s.pose and s.pose.valid:
        return (np.array(s.pose.position_m, dtype=np.float64),
                np.array(s.pose.quaternion_xyzw, dtype=np.float64))
    return None, None


def _capture_teleop_anchor(env, receiver, left_arm_index, right_arm_index):
    """把当前机器人位姿和当前遥操输入位姿记录为遥操锚点。

    sender 位姿已经相对于 sender 启动位置，因此机器人端不应持续累加
    相邻帧之间的增量。这里同时记录“机器人当前在哪里”和“当前遥操位姿
    对应什么状态”，后续遥操目标都相对于这个记录点计算。
    """
    lp, lq = _latest_pose(receiver, left_arm_index)
    rp, rq = _latest_pose(receiver, right_arm_index)
    if lp is None or rp is None:
        return None

    state = env.get_robot_state()
    left_robot_anchor = np.array(
        state['left_ActualTCPPose'], dtype=np.float64).copy()
    right_robot_anchor = np.array(
        state['right_ActualTCPPose'], dtype=np.float64).copy()

    return {
        'left_robot': left_robot_anchor,
        'right_robot': right_robot_anchor,
        'left_udp_pos': lp.copy(),
        'left_udp_quat': lq.copy(),
        'right_udp_pos': rp.copy(),
        'right_udp_quat': rq.copy(),
    }


def _target_from_anchor(robot_anchor, udp_pos_anchor, udp_quat_anchor,
                        udp_pos, udp_quat, t265_to_robot, pos_scale,
                        rot_scale, keep_rotvec=None):
    """根据最新遥操位姿和锚点计算末端目标位姿。

    位置：
        target_xyz = robot_anchor_xyz
                   + R_t265_to_robot * ((udp_pos - udp_pos_anchor) * scale)

    旋转：
        delta_q = udp_quat * inverse(udp_quat_anchor)
        target_rot = mapped_delta_q * robot_anchor_rot

    如果提供 keep_rotvec，则把旋转锁定为该值。
    """
    target = robot_anchor.copy()
    target[:3] += t265_to_robot @ ((udp_pos - udp_pos_anchor) * pos_scale)

    if keep_rotvec is not None:
        target[3:6] = keep_rotvec
    else:
        cur_rot = st.Rotation.from_quat(udp_quat)
        anchor_rot = st.Rotation.from_quat(udp_quat_anchor)
        delta_rot = cur_rot * anchor_rot.inv()
        mapped_rotvec = t265_to_robot @ (delta_rot.as_rotvec() * rot_scale)
        target[3:6] = (
            st.Rotation.from_rotvec(mapped_rotvec)
            * st.Rotation.from_rotvec(robot_anchor[3:6])
        ).as_rotvec()

    return target


def _limit_target_step(current, desired, max_dpos, max_drot):
    """在平移和旋转步长限制内，让当前目标逐步靠近期望目标。"""
    target = current.copy()

    dpos = desired[:3] - current[:3]
    dpos_norm = np.linalg.norm(dpos)
    if dpos_norm > max_dpos:
        dpos *= max_dpos / dpos_norm
    target[:3] = current[:3] + dpos

    cur_rot = st.Rotation.from_rotvec(current[3:6])
    desired_rot = st.Rotation.from_rotvec(desired[3:6])
    rel_rot = desired_rot * cur_rot.inv()
    rel_angle = rel_rot.magnitude()
    if rel_angle > max_drot:
        step_rot = st.Rotation.from_rotvec(
            rel_rot.as_rotvec() * (max_drot / rel_angle))
        target[3:6] = (step_rot * cur_rot).as_rotvec()
    else:
        target[3:6] = desired[3:6]

    return target


# ===========================================================================
# 遥操输入测试模式（不控制机器人）
# ===========================================================================

def _run_teleop_test(receiver, host, port, left_arm_index, right_arm_index):
    """打印接收到的遥操采样，不执行机器人控制。"""
    if isinstance(receiver, SharedMemoryTeleopReceiver):
        print("\n遥操输入测试模式：从共享内存读取 sender 数据。")
    else:
        print(f"\n遥操输入测试模式：监听 UDP {host}:{port}")
    print("按 Ctrl+C 停止。\n")
    last_seq = None
    while True:
        try:
            time.sleep(0.5)
        except KeyboardInterrupt:
            break
        l = receiver.get_latest(left_arm_index)
        r = receiver.get_latest(right_arm_index)
        seq, dropped = receiver.stats
        status = []
        if seq != last_seq:
            status.append(f"seq={seq}" + (f" dropped={dropped}" if dropped else ""))
            last_seq = seq
        for arm, label in [(l, left_arm_index), (r, right_arm_index)]:
            if arm:
                if arm.pose and arm.pose.valid:
                    p = arm.pose.position_m
                    q = arm.pose.quaternion_xyzw
                    status.append(
                        f"Arm[{label}]: pos=({p[0]:.4f},{p[1]:.4f},{p[2]:.4f}) "
                        f"quat=({q[0]:.4f},{q[1]:.4f},{q[2]:.4f},{q[3]:.4f})")
                else:
                    status.append(f"Arm[{label}]: 没有有效位姿")
                if arm.gripper and arm.gripper.valid:
                    status.append(f"grip={arm.gripper.angle_deg:.1f}deg")
                elif arm.gripper:
                    status.append("夹爪无效")
            else:
                status.append(f"Arm[{label}]: 没有数据")
        if status:
            print("  ".join(status))
    receiver.stop_wait()
    print("\n遥操输入测试结束。")


def _wait_for_teleop_ready(receiver, left_arm_index, right_arm_index,
                           timeout_s=15.0, require_gripper=True):
    """等待左右臂 pose/gripper 遥操数据有效，并打印连接状态。"""
    deadline = time.monotonic() + timeout_s
    next_print = 0.0
    last_status = None

    def sample_flags(sample):
        pose_ok = bool(sample and sample.pose and sample.pose.valid)
        gripper_ok = bool(sample and sample.gripper and sample.gripper.valid)
        return pose_ok, gripper_ok

    while time.monotonic() < deadline:
        left_sample = receiver.get_latest(left_arm_index)
        right_sample = receiver.get_latest(right_arm_index)
        left_pose_ok, left_gripper_ok = sample_flags(left_sample)
        right_pose_ok, right_gripper_ok = sample_flags(right_sample)
        pose_ready = left_pose_ok and right_pose_ok
        gripper_ready = left_gripper_ok and right_gripper_ok
        ready = pose_ready and (gripper_ready or not require_gripper)
        seq, dropped = receiver.stats

        status = (
            left_pose_ok, left_gripper_ok,
            right_pose_ok, right_gripper_ok,
            seq, dropped,
        )
        now = time.monotonic()
        if now >= next_print or status != last_status:
            print(
                "[Teleop Check] "
                f"seq={seq} dropped={dropped} "
                f"L(pose={left_pose_ok},grip={left_gripper_ok}) "
                f"R(pose={right_pose_ok},grip={right_gripper_ok}) "
                f"Enc[{left_arm_index}]={_format_gripper_encoder(left_sample)} "
                f"Enc[{right_arm_index}]={_format_gripper_encoder(right_sample)}"
            )
            next_print = now + 1.0
            last_status = status

        if ready:
            print("[Teleop Check] pose/gripper 遥操数据已就绪。")
            lp, lq = _latest_pose(receiver, left_arm_index)
            rp, rq = _latest_pose(receiver, right_arm_index)
            return lp, lq, rp, rq

        time.sleep(0.05)

    seq, dropped = receiver.stats
    raise TimeoutError(
        "等待遥操输入数据超时："
        f"seq={seq} dropped={dropped} "
        f"left_arm={left_arm_index} right_arm={right_arm_index} "
        f"require_gripper={require_gripper}"
    )


def _check_tcp_endpoint(label, host, port, timeout_s=1.0) -> bool:
    """检查机器人控制服务端口是否能建立 TCP 连接。"""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            pass
        print(f"[Device Check] {label} robot {host}:{port}: OK")
        return True
    except OSError as exc:
        print(f"[Device Check] {label} robot {host}:{port}: MISSING ({exc})")
        return False


def _run_startup_device_checks(camera_serials,
                               t265_serial0, t265_serial1,
                               encoder_port0, encoder_port1,
                               left_robot_ip, left_robot_port,
                               right_robot_ip, right_robot_port) -> None:
    """启动机器人环境前做第一层硬件自检并打印结果。"""
    print("[Device Check] 启动前硬件自检开始 ...")
    ok = True

    ok = check_input_devices(
        t265_serial0=t265_serial0,
        t265_serial1=t265_serial1,
        encoder_port0=encoder_port0,
        encoder_port1=encoder_port1,
        verbose=True,
    ) and ok

    try:
        from diffusion_policy.envs.realword.single_realsense import SingleRealsense

        detected_serials = SingleRealsense.get_connected_devices_serial()
        print(f"[Device Check] D400/普通 RealSense 相机数量: {len(detected_serials)}")
        for idx, serial in enumerate(detected_serials):
            print(f"  - camera[{idx}] serial={serial}")

        if camera_serials is not None:
            missing = [s for s in camera_serials if s not in detected_serials]
            if missing:
                ok = False
                print(f"[Device Check] 指定相机缺失: {missing}")
            else:
                print("[Device Check] 指定 camera_serial_numbers: OK")
        elif len(detected_serials) == 0:
            ok = False
            print("[Device Check] 没有检测到 RealSense 采集相机。")
    except Exception as exc:
        ok = False
        print(f"[Device Check] RealSense 采集相机枚举失败: {exc}")

    checked = set()
    for label, host, port in [
            ("left", left_robot_ip, left_robot_port),
            ("right", right_robot_ip, right_robot_port)]:
        endpoint = (host, int(port))
        if endpoint in checked:
            print(f"[Device Check] {label} robot {host}:{port}: same endpoint, skipped")
            continue
        checked.add(endpoint)
        ok = _check_tcp_endpoint(label, host, port) and ok

    if not ok:
        raise RuntimeError(
            "启动前硬件自检失败：请检查 T265、编码器串口、RealSense 相机和 "
            "Franka controller 服务端是否已经连接/启动。"
        )

    print("[Device Check] 启动前硬件自检通过。")


# ===========================================================================
# 主程序
# ===========================================================================

@click.command()
@click.option('--output', '-o', required=True,
              help="保存 demonstration 数据集的目录。")
@click.option('--left_robot_ip', '-li', required=True,
              help="双臂 Franka 服务端主机地址，左臂使用，例如 172.16.0.1。")
@click.option('--right_robot_ip', '-ri', required=True,
              help="双臂 Franka 服务端主机地址，右臂使用，通常和左臂相同。")
@click.option('--left_robot_port', default=4242, type=int, help="左臂端口。")
@click.option('--right_robot_port', default=4242, type=int, help="右臂端口。")
@click.option('--init_joints', '-j', is_flag=True, default=False,
              help="启动时先把双臂移动到指定初始关节位置。")
@click.option('--left_init_joints', default=None,
              help="左臂初始关节角，7 个逗号分隔的弧度值。")
@click.option('--right_init_joints', default=None,
              help="右臂初始关节角，7 个逗号分隔的弧度值。")
@click.option('--udp_host', default='0.0.0.0', help="UDP 监听地址。")
@click.option('--udp_port', default=5005, type=int, help="UDP 监听端口。")
@click.option('--start_udp_sender', is_flag=True,
              help="启用 UDP 兼容模式并自动启动本机 UDP sender；默认使用共享内存。")
@click.option('--t265_serial0', default=None,
              help="arm index 0 使用的 T265 序列号；空字符串表示不读取。")
@click.option('--t265_serial1', default=None,
              help="arm index 1 使用的 T265 序列号；空字符串表示不读取。")
@click.option('--encoder_port0', default=None,
              help="arm index 0 使用的编码器串口；空字符串表示不读取。")
@click.option('--encoder_port1', default=None,
              help="arm index 1 使用的编码器串口；空字符串表示不读取。")
@click.option('--sender_delay', default=5.0, type=float,
              help="内置 sender 启动前倒计时秒数，用于手动摆正 T265/夹爪。")
@click.option('--sender_ready_timeout', default=20.0, type=float,
              help="等待内置 sender 读到 T265 和编码器有效数据的超时时间。")
@click.option('--teleop_ready_timeout', default=15.0, type=float,
              help="等待左右臂遥操 pose/gripper 都有效的超时时间。")
@click.option('--env_ready_timeout', default=15.0, type=float,
              help="等待 RealSense 相机和左右 Franka controller ready 的超时时间。")
@click.option('--skip_device_check', is_flag=True,
              help="跳过启动前硬件自检，直接进入原始初始化流程。")
@click.option('--left_arm_index', default=0, type=int, help="遥操数据中左臂使用的 arm index。")
@click.option('--right_arm_index', default=1, type=int, help="遥操数据中右臂使用的 arm index。")
@click.option('--camera_serial_numbers', default=None,
              help="RealSense 序列号，逗号分隔，顺序为 base,left_wrist,right_wrist。")
@click.option('--vis_camera_idx', default=0, type=int,
              help="要显示的相机编号：0=base，1=左腕，2=右腕。")
@click.option('--camera_resolution', '-cr', default='640x480',
              help="相机采集分辨率。")
@click.option('--obs_resolution', '-or', default='640x480,224x224,224x224',
              help="观测图像分辨率。可给单个 WxH，或按相机分别给逗号分隔的 WxH。")
@click.option('--frequency', '-f', default=10, type=float,
              help="机器人控制频率，单位 Hz。")
@click.option('--command_latency', '-cl', default=0.01, type=float,
              help="采样到执行之间的指令延迟，单位秒。")
@click.option('--max_pos_speed', default=0.2, type=float,
              help="末端最大平移速度，单位 m/s。")
@click.option('--max_rot_speed', default=0.6, type=float,
              help="末端最大旋转速度，单位 rad/s。")
@click.option('--pos_scale', default=1.0, show_default=True, type=float,
              help="位移倍率，例如 1.1 表示手持端移动 10 cm，机器人目标移动 11 cm。")
@click.option('--rot_scale', default=1.0, show_default=True, type=float,
              help="旋转倍率，例如 1.5 表示手持端旋转 10 度，机器人目标旋转 15 度。")
@click.option('--gripper_toggle_deg', default=GRIPPER_TOGGLE_DEG, type=float,
              help="遥操夹爪角度开/合阈值，单位度。")
@click.option('--swap_grippers', is_flag=True,
              help="交换左右夹爪输入来源；用于 USB 编码器左右顺序和相机/手臂顺序不一致时。")
@click.option('--record_images', is_flag=True,
              help="把相机图像也写入 replay_buffer；默认只录视频。")
@click.option('--left_init_pose', default=None,
              help="左臂 home 位姿，6 个逗号分隔数值，xyz+rotvec。")
@click.option('--right_init_pose', default=None,
              help="右臂 home 位姿，6 个逗号分隔数值，xyz+rotvec。")
@click.option('--exposure', default=None, type=int, help="手动曝光值。")
@click.option('--teleop_test', '--udp_test', is_flag=True,
              help="遥操输入测试模式：只接收并打印数据，不初始化机器人。")
def main(output, left_robot_ip, right_robot_ip, left_robot_port, right_robot_port,
         init_joints, left_init_joints, right_init_joints,
         udp_host, udp_port, start_udp_sender,
         t265_serial0, t265_serial1, encoder_port0, encoder_port1,
         sender_delay, sender_ready_timeout, teleop_ready_timeout,
         env_ready_timeout, skip_device_check, left_arm_index, right_arm_index,
         camera_serial_numbers, vis_camera_idx, camera_resolution, obs_resolution,
         frequency, command_latency, max_pos_speed, max_rot_speed,
         pos_scale, rot_scale, gripper_toggle_deg, swap_grippers,
         left_init_pose, right_init_pose, exposure, teleop_test, record_images):
    # 默认由内置 sender 通过共享内存发布遥操数据。
    # 只有显式传入 --start_udp_sender 时才启用本机 UDP 兼容路径。
    use_udp = bool(start_udp_sender)
    receiver = None
    receiver_cleanup = None
    if use_udp:
        receiver = UDPPoseReceiver(host=udp_host, port=udp_port)
        receiver.start()
        receiver_cleanup = receiver.stop_wait
        atexit.register(receiver_cleanup)
        time.sleep(0.5)

    dt = 1.0 / frequency
    cam_w, cam_h = [int(x) for x in camera_resolution.split('x')]
    # 解析观测分辨率：单个 "宽x高"，或按相机给出三个逗号分隔的分辨率。
    if ',' in obs_resolution:
        obs_res_list = [
            tuple(int(x) for x in r.split('x'))
            for r in obs_resolution.split(',')]
    else:
        obs_w, obs_h = [int(x) for x in obs_resolution.split('x')]
        obs_res_list = (obs_w, obs_h)

    camera_serials = None
    if camera_serial_numbers is not None:
        camera_serials = [s.strip() for s in camera_serial_numbers.split(',')]

    if t265_serial0 is None:
        t265_serial0 = T265_SERIAL0
    if t265_serial1 is None:
        t265_serial1 = T265_SERIAL1
    if encoder_port0 is None:
        encoder_port0 = ENCODER_PORT0
    if encoder_port1 is None:
        encoder_port1 = ENCODER_PORT1

    def _parse_init_pose(s):
        if s is None:
            return None
        arr = np.array([float(x) for x in s.split(',')], dtype=np.float64)
        assert arr.shape == (6,)
        return arr

    left_init_position = _parse_init_pose(left_init_pose)
    right_init_position = _parse_init_pose(right_init_pose)

    def _parse_joints(s):
        if s is None:
            return None
        arr = np.array([float(x) for x in s.split(',')], dtype=np.float64)
        assert arr.shape == (7,)
        return arr

    left_joints_init = _parse_joints(left_init_joints)
    right_joints_init = _parse_joints(right_init_joints)

    if skip_device_check:
        print("[Device Check] 已按 --skip_device_check 跳过启动前硬件自检。")
    else:
        _run_startup_device_checks(
            camera_serials=camera_serials,
            t265_serial0=t265_serial0,
            t265_serial1=t265_serial1,
            encoder_port0=encoder_port0,
            encoder_port1=encoder_port1,
            left_robot_ip=left_robot_ip,
            left_robot_port=left_robot_port,
            right_robot_ip=right_robot_ip,
            right_robot_port=right_robot_port,
        )

    with SharedMemoryManager() as shm_manager, contextlib.ExitStack() as sender_stack:
        if use_udp:
            print(
                "[Teleop Sender] UDP 兼容模式："
                f"监听 {udp_host}:{udp_port}。"
            )
        else:
            print(
                "[Teleop Sender] 默认共享内存模式："
                "自动启动本机 sender 子进程。"
            )

        print(
            f"[Teleop Sender] 接下来 {sender_delay:.1f}s 内，"
            "请手动把 T265/夹爪摆到初始姿态。"
        )
        udp_target_ip = (
            '127.0.0.1' if udp_host in ('0.0.0.0', '::') else udp_host
        )
        teleop_sender = sender_stack.enter_context(DualTeleopSenderProcess(
            shm_manager=shm_manager,
            target_ip=udp_target_ip,
            target_port=udp_port,
            delay_s=sender_delay,
            ready_timeout_s=sender_ready_timeout,
            send_udp=use_udp,
            t265_serial0=t265_serial0,
            t265_serial1=t265_serial1,
            encoder_port0=encoder_port0,
            encoder_port1=encoder_port1,
        ))
        if not use_udp:
            receiver = SharedMemoryTeleopReceiver(teleop_sender)

        status = teleop_sender.get_status()
        print(
            "[Teleop Sender] 子进程已完成初始化 "
            f"pid={teleop_sender.pid} "
            f"samples={int(status['sent_packets'][0])} "
            f"errors={int(status['send_errors'][0])}"
        )

        if teleop_test:
            _run_teleop_test(
                receiver, udp_host, udp_port,
                left_arm_index, right_arm_index)
            return

        print("启动机器人环境前检查 T265/编码器遥操数据 ...")
        _wait_for_teleop_ready(
            receiver,
            left_arm_index,
            right_arm_index,
            timeout_s=teleop_ready_timeout,
            require_gripper=True,
        )

        with KeystrokeCounter() as key_counter, \
            DualFrankaRealEnv(
                output_dir=output,
                left_robot_ip=left_robot_ip,
                right_robot_ip=right_robot_ip,
                left_robot_port=left_robot_port,
                right_robot_port=right_robot_port,
                frequency=frequency,
                obs_image_resolution=obs_res_list,
                video_capture_resolution=(cam_w, cam_h),
                video_capture_fps=30,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                init_joints=init_joints,
                left_joints_init=left_joints_init,
                right_joints_init=right_joints_init,
                camera_serial_numbers=camera_serials,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                record_images=record_images,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                camera_labels={0: 'global', 1: 'left wrist', 2: 'right wrist'},
            ) as env:
            if not env.wait_until_ready(timeout_s=env_ready_timeout):
                raise RuntimeError("DualFrankaRealEnv 初始化超时，请检查相机和左右 Franka controller。")

            cv2.setNumThreads(1)

            if exposure is not None:
                env.realsense.set_exposure(exposure=exposure, gain=0)
                env.realsense.set_white_balance(white_balance=5900)
            else:
                env.realsense.set_exposure()
                env.realsense.set_white_balance()

            time.sleep(1.0)

            # 使用机器人当前实际末端位姿初始化控制目标。
            state = env.get_robot_state()
            left_target = np.array(state['left_ActualTCPPose'], dtype=np.float64).copy()
            right_target = np.array(state['right_ActualTCPPose'], dtype=np.float64).copy()
            if left_init_position is None:
                left_init_position = DEFAULT_HOME_POSE_RETURN['left'].copy()
            if right_init_position is None:
                right_init_position = DEFAULT_HOME_POSE_RETURN['right'].copy()
            left_gripper = GRIPPER_OPEN
            right_gripper = GRIPPER_OPEN

            print(f"左臂初始末端位姿: {np.array2string(left_target, separator=',')}")
            print(f"右臂初始末端位姿: {np.array2string(right_target, separator=',')}")
            print(f"左臂 home 位姿:    {np.array2string(left_init_position, separator=',')}")
            print(f"右臂 home 位姿:    {np.array2string(right_init_position, separator=',')}")

            # 等待首帧有效遥操数据，用于记录初始遥操锚点。
            print("等待遥操输入数据 ...")
            while True:
                lp, lq = _latest_pose(receiver, left_arm_index)
                rp, rq = _latest_pose(receiver, right_arm_index)
                if lp is not None and rp is not None:
                    break
                time.sleep(0.05)
            teleop_anchor = _capture_teleop_anchor(
                env, receiver, left_arm_index, right_arm_index)
            if teleop_anchor is None:
                raise RuntimeError("未能记录初始遥操锚点")

            def _refresh_anchor(label=None, sync_targets=True):
                nonlocal teleop_anchor, left_target, right_target
                anchor = _capture_teleop_anchor(
                    env, receiver, left_arm_index, right_arm_index)
                if anchor is None:
                    if label:
                        print(f'{label}: 没有有效遥操位姿，锚点保持不变')
                    return False
                teleop_anchor = anchor
                if sync_targets:
                    left_target = anchor['left_robot'].copy()
                    right_target = anchor['right_robot'].copy()
                if label:
                    print(label)
                return True

            print("遥操输入数据已收到，可以开始。")

            teleop_transport = (
                'shared_memory'
                if isinstance(receiver, SharedMemoryTeleopReceiver)
                else f'udp://{udp_host}:{udp_port}'
            )
            print('准备完成：双臂 Franka 遥操。')
            print(f'  Teleop: {teleop_transport} (左臂={left_arm_index}, 右臂={right_arm_index})')
            print(f'  位移倍率={pos_scale:.3f} 旋转倍率={rot_scale:.3f}')
            print(f'  left_robot.is_alive={env.left_robot.is_alive()} '
                  f'right_robot.is_alive={env.right_robot.is_alive()} '
                  f'is_ready={env.is_ready}')
            print('  E = 开启/关闭遥操')
            print('  T = 锁定/解锁旋转')
            print('  Z = 重新记录锚点')
            print('  G = 同时切换两个夹爪开/合')
            print('  C = 开始录制')
            print('  S = 停止录制')
            print('  R = 关节返回位, P = 末端 home 位姿, H = 关节起始位, Q = 退出')
            print('  1/2/3 = 切换相机画面')

            # 运行状态
            teleop_enabled = False
            lock_rot = False
            pose_homing = False
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            # H/R 键触发的非阻塞关节空间归位状态。
            homing = False
            homing_joints = None
            homing_deadline = 0.0

            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                obs = env.get_obs()

                # ---- 键盘输入 ----
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        env.start_episode(
                            t_start + (iter_idx + 2) * dt
                            - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        teleop_enabled = True
                        _refresh_anchor('录制开始前已记录遥操锚点。')
                        print('开始录制！')
                    elif key_stroke == KeyCode(char='s'):
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        teleop_enabled = False
                        print('已停止录制。')
                    elif key_stroke == Key.backspace:
                        if click.confirm('删除上一段 episode？'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                    elif key_stroke == KeyCode(char='r'):
                        key_counter.clear()
                        print('双臂关节空间回到返回位 ...')
                        pose_homing = False
                        env.move_to_joint_positions(
                            HOME_JOINTS_RETURN['left'],
                            HOME_JOINTS_RETURN['right'],
                            time_to_go=JOINT_HOME_DURATION)
                        homing_joints = (HOME_JOINTS_RETURN['left'],
                                         HOME_JOINTS_RETURN['right'])
                        homing_deadline = time.monotonic() + JOINT_HOME_DURATION + 1.0
                        homing = True
                    elif key_stroke == KeyCode(char='h'):
                        key_counter.clear()
                        print('双臂关节空间回到起始位 ...')
                        pose_homing = False
                        env.move_to_joint_positions(
                            HOME_JOINTS_START['left'],
                            HOME_JOINTS_START['right'],
                            time_to_go=JOINT_HOME_DURATION)
                        homing_joints = (HOME_JOINTS_START['left'],
                                         HOME_JOINTS_START['right'])
                        homing_deadline = time.monotonic() + JOINT_HOME_DURATION + 1.0
                        homing = True
                    elif key_stroke == KeyCode(char='p'):
                        key_counter.clear()
                        if homing:
                            print('关节空间归位正在执行，请完成后再按 P。')
                        else:
                            state = env.get_robot_state()
                            left_target = np.array(
                                state['left_ActualTCPPose'], dtype=np.float64).copy()
                            right_target = np.array(
                                state['right_ActualTCPPose'], dtype=np.float64).copy()
                            pose_homing = True
                            print('双臂末端笛卡尔空间回到 home 位姿 ...')
                    elif key_stroke == KeyCode(char='z'):
                        key_counter.clear()
                        _refresh_anchor('已重新记录遥操锚点！')
                    elif key_stroke == KeyCode(char='t'):
                        lock_rot = not lock_rot
                        print(f'旋转锁定: {"开启" if lock_rot else "关闭"}')
                    elif key_stroke == KeyCode(char='e'):
                        teleop_enabled = not teleop_enabled
                        if teleop_enabled:
                            key_counter.clear()
                            print('遥操将在 ', end='', flush=True)
                            for sec in range(5, 0, -1):
                                print(f'{sec}... ', end='', flush=True)
                                time.sleep(1.0)
                            _refresh_anchor()
                            print('开始！')
                        else:
                            print('遥操：关闭')
                            key_counter.clear()
                    elif key_stroke == KeyCode(char='g'):
                        left_gripper = GRIPPER_CLOSE if left_gripper == GRIPPER_OPEN \
                            else GRIPPER_OPEN
                        right_gripper = GRIPPER_CLOSE if right_gripper == GRIPPER_OPEN \
                            else GRIPPER_OPEN
                        print(f'夹爪 -> 左:{left_gripper:.3f} 右:{right_gripper:.3f}')
                    elif key_stroke == KeyCode(char='1'):
                        vis_camera_idx = 0
                    elif key_stroke == KeyCode(char='2'):
                        vis_camera_idx = 1
                    elif key_stroke == KeyCode(char='3'):
                        vis_camera_idx = 2
                stage = key_counter[Key.space]

                # ---- 可视化 ----
                camera_labels = {0: 'global', 1: 'left wrist', 2: 'right wrist'}
                cam_label = camera_labels.get(vis_camera_idx, f'cam_{vis_camera_idx}')
                vis_img = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = (f'Ep:{episode_id} St:{stage} '
                        f'LG:{left_gripper:.3f} RG:{right_gripper:.3f}')
                if is_recording:
                    text += ' REC!'
                if pose_homing or homing:
                    text += ' HOME'
                cv2.putText(vis_img, text, (10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.7, thickness=2, color=(255, 255, 255))
                mode_text = (
                    f'{"TRANS" if lock_rot else "6DOF"} | '
                    f'{"SHM" if isinstance(receiver, SharedMemoryTeleopReceiver) else "UDP"}'
                )
                if not teleop_enabled:
                    mode_text += ' | OFF'
                cv2.putText(vis_img, mode_text, (10, 55),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, color=(255, 255, 255))
                cv2.putText(vis_img, cam_label.upper(), (10, 75),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, color=(0, 255, 255))
                cv2.imshow('dual_udp_teleop_franka', vis_img)
                cv2.pollKey()

                # # 实时打印关节角，每秒一次。
                # if iter_idx % frequency == 0:
                #     lq = obs['left_robot_joint'][-1]
                #     rq = obs['right_robot_joint'][-1]
                #     lq_s = '[' + ' '.join(f'{v:+.4f}' for v in lq) + ']'
                #     rq_s = '[' + ' '.join(f'{v:+.4f}' for v in rq) + ']'
                #     print(f'[joints {iter_idx:4d}] '
                #           f'L={lq_s} R={rq_s}')

                precise_wait(t_sample)

                # ---- 机器人控制 ----
                if homing:
                    # H/R 键触发非阻塞关节空间归位。归位期间让笛卡尔目标与
                    # 实际末端位姿保持同步，并持续向录制累加器写入数据，确保
                    # 归位动作能够被完整记录，同时不阻塞主循环。
                    state = env.get_robot_state()
                    left_target = np.array(
                        state['left_ActualTCPPose'], dtype=np.float64).copy()
                    right_target = np.array(
                        state['right_ActualTCPPose'], dtype=np.float64).copy()
                    _refresh_anchor(sync_targets=False)
                    if (time.monotonic() >= homing_deadline or
                            env.is_at_joint_positions(homing_joints[0],
                                                      homing_joints[1])):
                        homing = False
                        print('归位完成 (joint-space)')
                elif pose_homing:
                    # 在笛卡尔空间回到 home 位姿，不进行关节空间插值。
                    left_gripper = GRIPPER_OPEN
                    right_gripper = GRIPPER_OPEN
                    left_target, l_arrived = _cartesian_return_step(
                        left_target, left_init_position, POSE_HOME_EPSILON,
                        POSE_HOME_STEP_SIZE, POSE_HOME_EPSILON_ROT,
                        POSE_HOME_MAX_STEP_ANGLE)
                    right_target, r_arrived = _cartesian_return_step(
                        right_target, right_init_position, POSE_HOME_EPSILON,
                        POSE_HOME_STEP_SIZE, POSE_HOME_EPSILON_ROT,
                        POSE_HOME_MAX_STEP_ANGLE)
                    _refresh_anchor(sync_targets=False)
                    if l_arrived and r_arrived:
                        print('归位完成 (EEF pose)')
                        pose_homing = False

                elif not teleop_enabled:
                    _refresh_anchor()

                else:
                    # 读取遥操数据，并相对于最近记录的锚点计算目标位姿。
                    # sender 位姿以其启动位置为基准，因此机器人端使用当前值
                    # 与锚点之差，不再持续累加相邻帧增量。
                    l_sample = receiver.get_latest(left_arm_index)
                    r_sample = receiver.get_latest(right_arm_index)

                    left_ok = l_sample and l_sample.pose and l_sample.pose.valid
                    right_ok = r_sample and r_sample.pose and r_sample.pose.valid

                    if left_ok:
                        raw_pos_l = np.array(l_sample.pose.position_m, dtype=np.float64)
                        raw_quat_l = np.array(l_sample.pose.quaternion_xyzw, dtype=np.float64)
                        desired_left_target = _target_from_anchor(
                            teleop_anchor['left_robot'],
                            teleop_anchor['left_udp_pos'],
                            teleop_anchor['left_udp_quat'],
                            raw_pos_l,
                            raw_quat_l,
                            LEFT_T265_TO_ROBOT,
                            pos_scale,
                            rot_scale,
                            keep_rotvec=left_target[3:6].copy() if lock_rot else None)
                    else:
                        desired_left_target = left_target.copy()

                    if right_ok:
                        raw_pos_r = np.array(r_sample.pose.position_m, dtype=np.float64)
                        raw_quat_r = np.array(r_sample.pose.quaternion_xyzw, dtype=np.float64)
                        desired_right_target = _target_from_anchor(
                            teleop_anchor['right_robot'],
                            teleop_anchor['right_udp_pos'],
                            teleop_anchor['right_udp_quat'],
                            raw_pos_r,
                            raw_quat_r,
                            RIGHT_T265_TO_ROBOT,
                            pos_scale,
                            rot_scale,
                            keep_rotvec=right_target[3:6].copy() if lock_rot else None)
                    else:
                        desired_right_target = right_target.copy()

                    # ---- 根据遥操输入二值控制夹爪开合 ----
                    # --swap_grippers 只交换左右编码器/夹爪输入来源，不交换
                    # T265 位姿，也不交换键盘 G/home 产生的最终夹爪命令。
                    left_gripper_sample = r_sample if swap_grippers else l_sample
                    right_gripper_sample = l_sample if swap_grippers else r_sample
                    if (left_gripper_sample and left_gripper_sample.gripper and
                            left_gripper_sample.gripper.valid):
                        left_gripper = GRIPPER_OPEN \
                            if left_gripper_sample.gripper.angle_deg > gripper_toggle_deg \
                            else GRIPPER_CLOSE
                    if (right_gripper_sample and right_gripper_sample.gripper and
                            right_gripper_sample.gripper.valid):
                        right_gripper = GRIPPER_OPEN \
                            if right_gripper_sample.gripper.angle_deg > gripper_toggle_deg \
                            else GRIPPER_CLOSE

                    # 按照原有速度限制，让当前目标逐步靠近基于锚点计算的目标。
                    max_dpos = max_pos_speed * dt
                    max_drot = max_rot_speed * dt
                    prev_left_target = left_target.copy()
                    prev_right_target = right_target.copy()
                    left_target = _limit_target_step(
                        left_target, desired_left_target, max_dpos, max_drot)
                    right_target = _limit_target_step(
                        right_target, desired_right_target, max_dpos, max_drot)
                    left_dpos = left_target[:3] - prev_left_target[:3]
                    right_dpos = right_target[:3] - prev_right_target[:3]

                    # 定期打印遥操数据。
                    if iter_idx % frequency == 0:
                        print(f'[iter {iter_idx:4d}] '
                              f'L_dpos=[{left_dpos[0]:+.4f},{left_dpos[1]:+.4f},{left_dpos[2]:+.4f}] '
                              f'R_dpos=[{right_dpos[0]:+.4f},{right_dpos[1]:+.4f},{right_dpos[2]:+.4f}] '
                              f'Enc[{left_arm_index}]={_format_gripper_encoder(l_sample)} '
                              f'Enc[{right_arm_index}]={_format_gripper_encoder(r_sample)} '
                              f'L_grip={left_gripper:.3f} R_grip={right_gripper:.3f}')

                    # 左右目标已经在上方朝基于锚点计算的期望位姿更新。

                # 高度限制：双臂末端的 z 坐标不得低于 MIN_Z_HEIGHT。
                left_target[2] = max(left_target[2], MIN_Z_HEIGHT)
                right_target[2] = max(right_target[2], MIN_Z_HEIGHT)

                action = np.concatenate([
                    left_target, [left_gripper],
                    right_target, [right_gripper],
                ])
                env.exec_actions(
                    actions=[action],
                    timestamps=[t_command_target - time.monotonic() + time.time()],
                    stages=[stage])

                precise_wait(t_cycle_end)
                iter_idx += 1

    if receiver is not None:
        receiver.stop_wait()
    if receiver_cleanup is not None:
        atexit.unregister(receiver_cleanup)
    cv2.destroyAllWindows()
    print("[Teleop] 遥操结束")


def _wam_instance_lock():
    """Keep the original real loop single-instance without changing its internals."""
    import fcntl

    path = Path('/tmp/dp_pro6000_fasterwam_real.lock')
    handle = path.open('a+', encoding='utf-8')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or 'unknown'
        handle.close()
        raise RuntimeError(
            f'已有 FasterWAM 真机进程正在运行，PID={owner}。请勿重复启动。'
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def _check_wam_devices(camera_serials, left_ip, left_port, right_ip, right_port):
    """Original device check, with T265/encoder requirements removed."""
    from diffusion_policy.envs.realword.single_realsense import SingleRealsense

    print('[Device Check] FasterWAM 真机启动前检查 ...')
    detected = SingleRealsense.get_connected_devices_serial()
    print(f'[Device Check] RealSense detected={detected}')
    if camera_serials is None or len(camera_serials) != 3:
        raise RuntimeError(
            '--camera_serial_numbers 必须按 base,left_wrist,right_wrist 提供三个序列号。'
        )
    missing = [serial for serial in camera_serials if serial not in detected]
    if missing:
        raise RuntimeError(f'RealSense 缺失，serial={missing}')
    for role, serial in zip(('base', 'left_wrist', 'right_wrist'), camera_serials):
        print(f'[Device Check] camera role={role} serial={serial}: OK')

    checked = set()
    for role, host, port in (
        ('left', left_ip, left_port), ('right', right_ip, right_port)
    ):
        endpoint = (host, int(port))
        if endpoint in checked:
            print(f'[Device Check] {role} robot {host}:{port}: same endpoint, skipped')
            continue
        checked.add(endpoint)
        if not _check_tcp_endpoint(role, host, port):
            raise RuntimeError(f'Franka controller 不可达：{host}:{port}')
    print('[Device Check] FasterWAM 真机启动前检查通过。')


def _copy_observation(obs):
    return {key: np.array(value, copy=True) for key, value in obs.items()}


@click.command()
@click.option('--output', '-o', required=True,
              help='真机运行数据和视频输出目录。')
@click.option('--left_robot_ip', '-li', required=True)
@click.option('--right_robot_ip', '-ri', required=True)
@click.option('--left_robot_port', default=4242, type=int)
@click.option('--right_robot_port', default=4242, type=int)
@click.option('--init_joints', '-j', is_flag=True, default=False)
@click.option('--left_init_joints', default=None)
@click.option('--right_init_joints', default=None)
@click.option('--env_ready_timeout', default=20.0, type=float)
@click.option('--skip_device_check', is_flag=True)
@click.option('--camera_serial_numbers', required=True,
              help='base,left_wrist,right_wrist 三个 RealSense 序列号。')
@click.option('--vis_camera_idx', default=0, type=int)
@click.option('--camera_resolution', '-cr', default='640x480')
@click.option('--obs_resolution', '-or', default='640x480,640x480,640x480')
@click.option('--frequency', '-f', default=10.0, type=float)
@click.option('--command_latency', '-cl', default=0.20, type=float)
@click.option('--max_pos_speed', default=0.2, type=float)
@click.option('--max_rot_speed', default=0.6, type=float)
@click.option('--record_images', is_flag=True)
@click.option('--left_init_pose', default=None)
@click.option('--right_init_pose', default=None)
@click.option('--exposure', default=None, type=int)
@click.option('--fasterwam_root', default='/home/hy/FasterWAM', show_default=True)
@click.option('--checkpoint',
              default='/home/hy/FasterWAM/checkpoints/fold100_action/step_010000_merged.pt',
              show_default=True)
@click.option('--dataset_stats',
              default='/home/hy/FasterWAM/checkpoints/fold100_action/dataset_stats.json',
              show_default=True)
@click.option('--text_embedding',
              default='/home/hy/FasterWAM/checkpoints/fold100_action/text_embeddings/c5e39e32e704fe7a533eb4296adeffb7e910f86911713503866679c5afc646d6.t5_len128.wan22ti2v5b.pt',
              show_default=True)
@click.option('--instruction', default='fold the T-shirt', show_default=True)
@click.option('--device', default='cuda:0', show_default=True)
@click.option('--mixed_precision',
              type=click.Choice(['bf16', 'fp16', 'no']), default='bf16')
@click.option('--num_inference_steps', default=10, type=int)
@click.option('--action_horizon', default=32, type=int)
@click.option('--replan_steps', default=8, type=int,
              help='执行多少个 action step 后提交下一次异步推理。')
@click.option('--gripper_open_threshold', default=0.04, type=float)
def fasterwam_main(
        output, left_robot_ip, right_robot_ip, left_robot_port, right_robot_port,
        init_joints, left_init_joints, right_init_joints, env_ready_timeout,
        skip_device_check, camera_serial_numbers, vis_camera_idx,
        camera_resolution, obs_resolution, frequency, command_latency,
        max_pos_speed, max_rot_speed, record_images, left_init_pose,
        right_init_pose, exposure, fasterwam_root, checkpoint, dataset_stats,
        text_embedding, instruction, device, mixed_precision,
        num_inference_steps, action_horizon, replan_steps,
        gripper_open_threshold):
    """原 dp_pro6000 真机循环，仅把遥操动作源替换为 FasterWAM。"""
    lock_handle = _wam_instance_lock()
    try:
        camera_serials = [s.strip() for s in camera_serial_numbers.split(',') if s.strip()]
        if len(camera_serials) != 3:
            raise click.ClickException('必须提供三个相机序列号。')
        if not skip_device_check:
            _check_wam_devices(
                camera_serials, left_robot_ip, left_robot_port,
                right_robot_ip, right_robot_port)

        dt = 1.0 / frequency
        cam_w, cam_h = [int(x) for x in camera_resolution.split('x')]
        if ',' in obs_resolution:
            obs_res_list = [
                tuple(int(x) for x in item.split('x'))
                for item in obs_resolution.split(',')]
        else:
            obs_w, obs_h = [int(x) for x in obs_resolution.split('x')]
            obs_res_list = (obs_w, obs_h)

        def parse_vector(raw, size):
            if raw is None:
                return None
            value = np.asarray([float(x) for x in raw.split(',')], dtype=np.float64)
            if value.shape != (size,):
                raise click.ClickException(f'期望 {size} 个数，实际为 {value.shape}')
            return value

        left_joints = parse_vector(left_init_joints, 7)
        right_joints = parse_vector(right_init_joints, 7)
        left_home = parse_vector(left_init_pose, 6)
        right_home = parse_vector(right_init_pose, 6)

        # 与原脚本相同：硬件子进程先启动；CUDA 模型在 env ready 后才加载。
        with SharedMemoryManager() as shm_manager, KeystrokeCounter() as key_counter, \
            DualFrankaRealEnv(
                output_dir=output,
                left_robot_ip=left_robot_ip,
                right_robot_ip=right_robot_ip,
                left_robot_port=left_robot_port,
                right_robot_port=right_robot_port,
                frequency=frequency,
                obs_image_resolution=obs_res_list,
                video_capture_resolution=(cam_w, cam_h),
                video_capture_fps=30,
                max_pos_speed=max_pos_speed,
                max_rot_speed=max_rot_speed,
                init_joints=init_joints,
                left_joints_init=left_joints,
                right_joints_init=right_joints,
                camera_serial_numbers=camera_serials,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                record_images=record_images,
                thread_per_video=3,
                video_crf=21,
                shm_manager=shm_manager,
                camera_labels={0: 'global', 1: 'left wrist', 2: 'right wrist'},
            ) as env:
            if not env.wait_until_ready(timeout_s=env_ready_timeout):
                raise RuntimeError('DualFrankaRealEnv 初始化超时。')

            cv2.setNumThreads(1)
            if exposure is not None:
                env.realsense.set_exposure(exposure=exposure, gain=0)
                env.realsense.set_white_balance(white_balance=5900)
            else:
                env.realsense.set_exposure()
                env.realsense.set_white_balance()
            time.sleep(1.0)

            state = env.get_robot_state()
            left_target = np.asarray(
                state['left_ActualTCPPose'], dtype=np.float64).copy()
            right_target = np.asarray(
                state['right_ActualTCPPose'], dtype=np.float64).copy()
            if left_home is None:
                left_home = DEFAULT_HOME_POSE_RETURN['left'].copy()
            if right_home is None:
                right_home = DEFAULT_HOME_POSE_RETURN['right'].copy()
            left_gripper = GRIPPER_OPEN
            right_gripper = GRIPPER_OPEN

            from .policy_adapter import FasterWAMPolicyAdapter
            policy = FasterWAMPolicyAdapter(
                fasterwam_root=fasterwam_root,
                checkpoint=checkpoint,
                dataset_stats=dataset_stats,
                text_embedding=text_embedding,
                instruction=instruction,
                device=device,
                mixed_precision=mixed_precision,
                num_inference_steps=num_inference_steps,
                action_horizon=action_horizon,
                gripper_open_threshold=gripper_open_threshold,
            )

            print('准备完成：dp_pro6000 双臂真机框架 + FasterWAM。')
            print('  E = 开启/暂停 FasterWAM')
            print('  C/S = 开始/停止录制，R/H/P = 原归位操作，Q = 退出')
            print('  1/2/3 = 切换相机画面')
            print('  启动后默认暂停，不会执行模型动作。')

            wam_enabled = False
            is_recording = False
            pose_homing = False
            homing = False
            homing_joints = None
            homing_deadline = 0.0
            action_buffer = deque()
            infer_future = None
            inference_epoch = 0
            steps_since_replan = replan_steps
            last_infer_seconds = 0.0
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            with ThreadPoolExecutor(max_workers=1, thread_name_prefix='fasterwam') as executor:
                while not stop:
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    obs = env.get_obs()

                    for key_stroke in key_counter.get_press_events():
                        if key_stroke == KeyCode(char='q'):
                            stop = True
                        elif key_stroke == KeyCode(char='e'):
                            wam_enabled = not wam_enabled
                            inference_epoch += 1
                            action_buffer.clear()
                            steps_since_replan = replan_steps
                            if wam_enabled:
                                policy.reset()
                                print('FasterWAM：开启')
                            else:
                                print('FasterWAM：暂停')
                        elif key_stroke == KeyCode(char='c'):
                            env.start_episode(time.time() + 2 * dt)
                            key_counter.clear()
                            is_recording = True
                            wam_enabled = True
                            inference_epoch += 1
                            policy.reset()
                            action_buffer.clear()
                            steps_since_replan = replan_steps
                            print('开始录制并开启 FasterWAM。')
                        elif key_stroke == KeyCode(char='s'):
                            env.end_episode()
                            key_counter.clear()
                            is_recording = False
                            wam_enabled = False
                            inference_epoch += 1
                            action_buffer.clear()
                            print('停止录制并暂停 FasterWAM。')
                        elif key_stroke == Key.backspace:
                            if click.confirm('删除上一段 episode？'):
                                env.drop_episode()
                                key_counter.clear()
                                is_recording = False
                        elif key_stroke == KeyCode(char='r'):
                            env.move_to_joint_positions(
                                HOME_JOINTS_RETURN['left'], HOME_JOINTS_RETURN['right'],
                                time_to_go=JOINT_HOME_DURATION)
                            homing_joints = (
                                HOME_JOINTS_RETURN['left'], HOME_JOINTS_RETURN['right'])
                            homing_deadline = time.monotonic() + JOINT_HOME_DURATION + 1.0
                            homing = True
                            pose_homing = False
                            wam_enabled = False
                            inference_epoch += 1
                            action_buffer.clear()
                        elif key_stroke == KeyCode(char='h'):
                            env.move_to_joint_positions(
                                HOME_JOINTS_START['left'], HOME_JOINTS_START['right'],
                                time_to_go=JOINT_HOME_DURATION)
                            homing_joints = (
                                HOME_JOINTS_START['left'], HOME_JOINTS_START['right'])
                            homing_deadline = time.monotonic() + JOINT_HOME_DURATION + 1.0
                            homing = True
                            pose_homing = False
                            wam_enabled = False
                            inference_epoch += 1
                            action_buffer.clear()
                        elif key_stroke == KeyCode(char='p'):
                            state = env.get_robot_state()
                            left_target = np.asarray(
                                state['left_ActualTCPPose'], dtype=np.float64).copy()
                            right_target = np.asarray(
                                state['right_ActualTCPPose'], dtype=np.float64).copy()
                            pose_homing = True
                            homing = False
                            wam_enabled = False
                            inference_epoch += 1
                            action_buffer.clear()
                        elif key_stroke == KeyCode(char='g'):
                            left_gripper = GRIPPER_CLOSE if left_gripper == GRIPPER_OPEN else GRIPPER_OPEN
                            right_gripper = GRIPPER_CLOSE if right_gripper == GRIPPER_OPEN else GRIPPER_OPEN
                        elif key_stroke == KeyCode(char='1'):
                            vis_camera_idx = 0
                        elif key_stroke == KeyCode(char='2'):
                            vis_camera_idx = 1
                        elif key_stroke == KeyCode(char='3'):
                            vis_camera_idx = 2
                    stage = key_counter[Key.space]

                    camera_labels = {0: 'global', 1: 'left wrist', 2: 'right wrist'}
                    vis_img = obs[f'camera_{vis_camera_idx}'][-1, :, :, ::-1].copy()
                    status = 'WAM ON' if wam_enabled else 'WAM PAUSED'
                    text = (
                        f'Ep:{env.replay_buffer.n_episodes} St:{stage} {status} '
                        f'Buf:{len(action_buffer)} Infer:{last_infer_seconds:.3f}s')
                    if is_recording:
                        text += ' REC!'
                    if pose_homing or homing:
                        text += ' HOME'
                    cv2.putText(vis_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 255), 2)
                    cv2.putText(
                        vis_img, camera_labels[vis_camera_idx].upper(), (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    cv2.imshow('dual_franka_fasterwam', vis_img)
                    cv2.pollKey()

                    precise_wait(t_sample)
                    if homing:
                        state = env.get_robot_state()
                        left_target = np.asarray(
                            state['left_ActualTCPPose'], dtype=np.float64).copy()
                        right_target = np.asarray(
                            state['right_ActualTCPPose'], dtype=np.float64).copy()
                        if (time.monotonic() >= homing_deadline or
                                env.is_at_joint_positions(
                                    homing_joints[0], homing_joints[1])):
                            homing = False
                            print('归位完成 (joint-space)')
                    elif pose_homing:
                        left_gripper = GRIPPER_OPEN
                        right_gripper = GRIPPER_OPEN
                        left_target, left_arrived = _cartesian_return_step(
                            left_target, left_home, POSE_HOME_EPSILON,
                            POSE_HOME_STEP_SIZE, POSE_HOME_EPSILON_ROT,
                            POSE_HOME_MAX_STEP_ANGLE)
                        right_target, right_arrived = _cartesian_return_step(
                            right_target, right_home, POSE_HOME_EPSILON,
                            POSE_HOME_STEP_SIZE, POSE_HOME_EPSILON_ROT,
                            POSE_HOME_MAX_STEP_ANGLE)
                        if left_arrived and right_arrived:
                            pose_homing = False
                            print('归位完成 (EEF pose)')
                    elif wam_enabled:
                        if infer_future is not None and infer_future[2].done():
                            started_at, result_epoch, future = infer_future
                            try:
                                chunk = future.result()
                            except Exception as exc:
                                if result_epoch == inference_epoch:
                                    wam_enabled = False
                                    action_buffer.clear()
                                    print(f'FASTERWAM_INFERENCE_FAIL: {exc}')
                                else:
                                    print(
                                        'FASTERWAM_STALE_INFERENCE_FAIL '
                                        f'ignored=true error={exc}')
                            else:
                                last_infer_seconds = time.perf_counter() - started_at
                                if result_epoch == inference_epoch:
                                    action_buffer = deque(chunk)
                                    steps_since_replan = 0
                                    print(
                                        f'FASTERWAM_CHUNK_READY shape={chunk.shape} '
                                        f'seconds={last_infer_seconds:.3f}')
                                else:
                                    print('FASTERWAM_CHUNK_DISCARDED stale_mode=true')
                            infer_future = None

                        if (infer_future is None and
                                (not action_buffer or steps_since_replan >= replan_steps)):
                            snapshot = _copy_observation(obs)
                            started_at = time.perf_counter()
                            future = executor.submit(policy.predict_action, snapshot)
                            infer_future = (started_at, inference_epoch, future)

                        if action_buffer:
                            desired = np.asarray(action_buffer.popleft(), dtype=np.float64)
                            desired_left = desired[:6]
                            desired_right = desired[7:13]
                            max_dpos = max_pos_speed * dt
                            max_drot = max_rot_speed * dt
                            left_target = _limit_target_step(
                                left_target, desired_left, max_dpos, max_drot)
                            right_target = _limit_target_step(
                                right_target, desired_right, max_dpos, max_drot)
                            left_gripper = float(desired[6])
                            right_gripper = float(desired[13])
                            steps_since_replan += 1

                    left_target[2] = max(left_target[2], MIN_Z_HEIGHT)
                    right_target[2] = max(right_target[2], MIN_Z_HEIGHT)
                    action = np.concatenate([
                        left_target, [left_gripper], right_target, [right_gripper]])
                    env.exec_actions(
                        actions=[action],
                        timestamps=[time.time() + max(command_latency, dt)],
                        stages=[stage])
                    precise_wait(t_cycle_end)
                    iter_idx += 1

            cv2.destroyAllWindows()
            print('[FasterWAM] 真机程序结束。')
    finally:
        lock_handle.close()


if __name__ == '__main__':
    fasterwam_main()
