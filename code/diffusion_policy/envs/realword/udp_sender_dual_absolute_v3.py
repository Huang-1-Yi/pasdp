#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""双臂遥操 sender v3：位置归零，姿态保持 VIO/world 原始方向。

本文件直接复制自 ``udp_sender_left_absolute_v2.py``，只把单套设备读取扩展
为按编号配置的两套设备。配置了几个 T265/编码器就读取几路，不绑定物理
左右；arm index 始终与设备编号 0/1 一致。

位姿定义保持不变：
    position_m：当前绝对 TCP 位置减去该 T265 启动时的 TCP 位置。
    quaternion_xyzw：当前 TCP 的原始 world 姿态，不相对启动姿态归零。

嵌入 ``demo_dual_udp_teleop_franka_v3.py`` 时默认通过共享内存发布；直接
运行本文件时仍可通过 UDP 发布，供兼容测试使用。
"""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import socket
import threading
import time
from typing import Optional, Tuple

import numpy as np
import serial
from serial.tools import list_ports

from diffusion_policy.envs.realword.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)


# ---------------------------------------------------------------------------
# 双臂设备配置。编号只表示数据顺序，不在 sender 内绑定物理左右。
# TODO(device-config): 真机配置稳定后，把每路 T265 序列号、编码器端口和
# TCP offset 收拢为显式 DeviceConfig；目前保留常量默认值，便于命令行覆盖。
# ---------------------------------------------------------------------------

TARGET_IP = "localhost"
TARGET_PORT = 5005
SEND_RATE_HZ = 200.0
DELAY_S = 5.0

TELEOP_MAX_ARMS = 2

T265_SERIAL0 = "224622111506"
T265_SERIAL1 = "121222110870"
T265_READ_TIMEOUT_MS = 1000
OUTPUT_FRAME = "zero_position_raw_orientation"

TCP_OFFSET_ENABLED = True
TCP_OFFSET_M = (-0.01068, 0.13763, 0.10759)
TCP_OFFSET_QUAT_XYZW = (0.0, 1.0, 0.0, 0.0)

SMOOTHING_ENABLED = False
SMOOTHING_RATE_HZ = 200.0
SMOOTHING_MIN_CUTOFF = 2.0
SMOOTHING_BETA = 0.007

ENCODER_PORT0 = "/dev/ttyUSB0"
ENCODER_PORT1 = "/dev/ttyUSB1"
ENCODER_BAUDRATE = 115200
ENCODER_REPORT_INTERVAL_MS = 5

SOF = b"\xAA\x55"
CMD_ANGLE = 0x01
CMD_SET_PARAM = 0x02
MODE_AUTO_REPORT = 0x00

SENDER_STAGE_STARTING = 0
SENDER_STAGE_DELAY = 1
SENDER_STAGE_RUNNING = 2
SENDER_STAGE_STOPPED = 3
SENDER_STAGE_ERROR = 4


def quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """归一化四元数，保证后续旋转计算稳定。

    T265 输出的四元数理论上已经接近单位四元数，但在实时读取和浮点
    计算中仍可能有微小误差。这里把 (x, y, z, w) 缩放到长度为 1，
    避免旋转矩阵、四元数乘法和坐标变换出现累计误差。

    如果输入长度异常接近 0，返回单位旋转 (0, 0, 0, 1)，防止除 0。
    """
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-15:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_to_rot(q: Tuple[float, float, float, float]) -> Tuple[float, ...]:
    """把四元数转换成 3x3 旋转矩阵，按行展开为 9 个数。

    输入和输出都沿用原双臂 sender 的处理方式：
    - 输入四元数顺序是 xyzw，和 RealSense / scipy 常用格式一致。
    - 输出是 row-major 的旋转矩阵:
      (r00, r01, r02, r10, r11, r12, r20, r21, r22)。

    v2 里这个矩阵只用于 TCP offset：把写在 T265 自身坐标系里的 TCP
    平移偏移量旋转到当前 VIO/world 坐标系。
    """
    x, y, z, w = quat_normalize(q)
    return (
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w),
        2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w),
        2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y),
    )


def quat_mul(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """计算两个四元数的乘积 a * b。

    四元数乘法表示旋转的组合，顺序不能交换。这里沿用原始 sender 的
    数据流：用 q_pose * q_offset 得到工具中心点的姿态。

    输入和输出的顺序都是 (x, y, z, w)。
    """
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def xor8(data: bytes) -> int:
    """计算编码器串口协议使用的 xor8 校验值。

    LightUMI 编码器帧格式是:
        AA 55 <len> <cmd> <data...> <xor8>

    校验方式是把最后一个校验字节之前的所有字节逐个异或。发送配置
    命令时需要生成这个值；接收角度帧时也用同样方式检查帧是否完整。
    """
    value = 0
    for b in data:
        value ^= b
    return value


class SharedState:
    """T265 线程、编码器线程和发布主循环之间共享的最新数据。

    两个后台线程分别更新 pose 和 gripper；主线程以固定频率读取这两份
    最新值并打包发布。这里用一个 Lock 保护字典替换过程，避免主
    线程正好读到一半更新的数据。
    """

    def __init__(self):
        """初始化共享状态。

        初始值都标记为 invalid。这样即使发布主循环先启动、T265 或
        编码器还没来得及给出第一帧，机器人端也能收到格式完整但
        valid=0 的数据，而不是解析失败。
        """
        self.lock = threading.Lock()
        self.pose = {
            "valid": 0,
            "tracker_confidence": 0,
            "position_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        self.gripper = {
            "valid": 0,
            "angle_deg": 0.0,
            "confidence": 0.0,
        }


class OneEuroFilter:
    """One Euro 低通滤波器，和原始 udp_sender.py 的平滑算法一致。

    这个滤波器的特点是：
    - 手基本不动、只有微小抖动时，滤波更强，减少 T265 噪声。
    - 手快速移动时，滤波自动变弱，减少延迟。

    每个标量单独滤波。位置的 x/y/z 各一个滤波器，四元数的 x/y/z/w
    也各一个滤波器，最后再把四元数归一化。
    """

    def __init__(self, rate_hz: float, min_cutoff: float, beta: float,
                 d_cutoff: float = 1.0):
        """初始化滤波器参数。

        rate_hz 是期望输入频率；min_cutoff 控制静止时平滑强度；
        beta 控制速度越快时滤波放开的程度；d_cutoff 是速度估计的滤波频率。
        """
        self.rate_hz = float(rate_hz)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.initialized = False

    def alpha(self, cutoff: float) -> float:
        """把截止频率转换成一阶低通滤波系数 alpha。"""
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.rate_hz
        return 1.0 / (1.0 + tau / te)

    def reset(self) -> None:
        """清空历史值。

        v2 第一次建立位置零点时需要重置滤波器，避免零点切换那一瞬间
        的跳变被滤波器记住。
        """
        self.x_prev = None
        self.dx_prev = None
        self.initialized = False

    def __call__(self, x: float) -> float:
        """输入一个标量，输出滤波后的标量。"""
        if not self.initialized:
            self.x_prev = x
            self.dx_prev = 0.0
            self.initialized = True
            return x

        dx = (x - self.x_prev) * self.rate_hz
        alpha_d = self.alpha(self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


class PoseSmoother:
    """对 3D 位置和 xyzw 四元数做和原始 sender 一样的 One Euro 平滑。"""

    def __init__(self, rate_hz: float, min_cutoff: float, beta: float):
        """为位置 3 个分量和四元数 4 个分量分别创建滤波器。"""
        self.pos_filters = [
            OneEuroFilter(rate_hz, min_cutoff, beta) for _ in range(3)
        ]
        self.quat_filters = [
            OneEuroFilter(rate_hz, min_cutoff, beta) for _ in range(4)
        ]

    def reset(self) -> None:
        """重置所有分量滤波器。"""
        for filt in self.pos_filters + self.quat_filters:
            filt.reset()

    def smooth(self, pos: Tuple[float, float, float],
               quat: Tuple[float, float, float, float]) -> Tuple[
                   Tuple[float, float, float],
                   Tuple[float, float, float, float]]:
        """输入当前 pose，返回平滑后的 pose。"""
        smooth_pos = tuple(
            filt(value) for filt, value in zip(self.pos_filters, pos)
        )
        smooth_quat = tuple(
            filt(value) for filt, value in zip(self.quat_filters, quat)
        )
        return smooth_pos, quat_normalize(smooth_quat)


def read_t265(stop: threading.Event, state: SharedState,
              t265_serial=T265_SERIAL0, arm_label="arm0",
              tcp_offset_enabled=TCP_OFFSET_ENABLED,
              tcp_offset_m=TCP_OFFSET_M,
              tcp_offset_quat=TCP_OFFSET_QUAT_XYZW) -> None:
    """后台读取一路 T265，并把最新 TCP pose 写入共享状态。

    v2 的位姿定义和 v1 不一样：
    1. 位置仍然以启动时为零点，即 position_m = 当前绝对位置 - 第一帧位置。
    2. 位置差值保持在 T265 的 VIO/world 坐标系里，不再用启动姿态转到 body 系。
    3. 姿态直接使用当前 T265/TCP 的原始 world 姿态，不再做 q_initial^-1 * q_raw。
    4. 默认关闭发送端 One Euro smoothing，避免滤波造成手势和机器人端不同步。

    使用方式上，启动前请把 T265/夹爪朝向摆到和机器人期望朝向一致。
    T265 的 roll/pitch 会跟随重力方向稳定；yaw 没有磁力计约束，所以 yaw
    的零方向就是设备启动时的朝向。
    """
    import pyrealsense2 as rs

    # RealSense pipeline 是 T265 pose 数据的入口。这里不打开彩色/深度流，
    # 只打开 pose 流，减少不必要的数据读取。
    pipeline = rs.pipeline()
    cfg = rs.config()
    if t265_serial:
        cfg.enable_device(t265_serial)
    cfg.enable_stream(rs.stream.pose, rs.format.six_dof, 200)
    pipeline.start(cfg)

    # 只保存启动瞬间的位置零点；姿态不保存为零点，因为 v2 要发原始姿态。
    initial_tcp_pos: Optional[Tuple[float, float, float]] = None
    smoother: Optional[PoseSmoother] = None
    if SMOOTHING_ENABLED:
        smoother = PoseSmoother(
            SMOOTHING_RATE_HZ,
            SMOOTHING_MIN_CUTOFF,
            SMOOTHING_BETA,
        )

    print(
        f"[T265:{arm_label}] Started serial={t265_serial or 'auto'} "
        f"frame={OUTPUT_FRAME}"
    )
    try:
        while not stop.is_set():
            try:
                frames = pipeline.wait_for_frames(T265_READ_TIMEOUT_MS)
            except RuntimeError:
                continue

            pose_frame = frames.get_pose_frame()
            if not pose_frame:
                continue

            # T265 原始输出：translation 单位是米，rotation 是 xyzw 四元数。
            data = pose_frame.get_pose_data()
            raw_pos = (data.translation.x, data.translation.y, data.translation.z)
            raw_quat = quat_normalize((
                data.rotation.x,
                data.rotation.y,
                data.rotation.z,
                data.rotation.w,
            ))

            # v2 先保留 T265 的 VIO/world 绝对位姿。这里不做启动姿态
            # 姿态归零，因此启动时怎么摆放 T265，就决定了发送出去的绝对朝向。
            absolute_pos = raw_pos
            absolute_quat = raw_quat

            if tcp_offset_enabled:
                # TCP 平移偏移量写在 T265 自身坐标系里。这里用原始 world
                # 姿态把 offset 转到 VIO/world 系，再得到绝对 TCP 位置。
                r = quat_to_rot(absolute_quat)
                ox, oy, oz = tcp_offset_m
                absolute_pos = (
                    absolute_pos[0] + r[0] * ox + r[1] * oy + r[2] * oz,
                    absolute_pos[1] + r[3] * ox + r[4] * oy + r[5] * oz,
                    absolute_pos[2] + r[6] * ox + r[7] * oy + r[8] * oz,
                )
                # 姿态同样保留“绝对”含义，只叠加固定的 T265->TCP 安装姿态。
                absolute_quat = quat_normalize(
                    quat_mul(absolute_quat, tcp_offset_quat)
                )

            if initial_tcp_pos is None:
                # 第一帧只用来定义位置零点，不改变姿态零点。
                initial_tcp_pos = absolute_pos
                if smoother is not None:
                    smoother.reset()
                print(
                    f"[T265:{arm_label}] Position zero set: "
                    f"tcp_pos={initial_tcp_pos}, "
                    f"raw_world_quat=({absolute_quat[0]:.4f},"
                    f"{absolute_quat[1]:.4f},{absolute_quat[2]:.4f},"
                    f"{absolute_quat[3]:.4f})"
                )

            pos = (
                absolute_pos[0] - initial_tcp_pos[0],
                absolute_pos[1] - initial_tcp_pos[1],
                absolute_pos[2] - initial_tcp_pos[2],
            )
            quat = absolute_quat

            if data.tracker_confidence > 0 and smoother is not None:
                # v2 默认关闭这个分支；保留开关仅用于必要时对比调试。
                pos, quat = smoother.smooth(pos, quat)

            with state.lock:
                # 保持和接收端期待的 JSON 字段一致，单位和顺序不变。
                state.pose = {
                    "sample_timestamp": time.time(),
                    "valid": int(data.tracker_confidence > 0),
                    "tracker_confidence": int(data.tracker_confidence),
                    "position_m": list(pos),
                    "quaternion_xyzw": list(quat),
                }
    finally:
        pipeline.stop()
        print(f"[T265:{arm_label}] Stopped")


def read_encoder(stop: threading.Event, state: SharedState,
                 encoder_port=ENCODER_PORT0, arm_label="arm0") -> None:
    """后台读取一路夹爪编码器角度，并写入共享状态。

    串口协议保持和原始 udp_sender.py 一致：
    - 启动时打开 ENCODER_PORT。
    - 发送 set_param 命令，把编码器设置为自动上报模式。
    - 自动上报间隔由 ENCODER_REPORT_INTERVAL_MS 决定，当前是 5 ms。
    - 后续循环解析 AA 55 开头的帧，只处理 cmd=0x01 的角度帧。
    - data 是 2 字节大端整数，单位是 0.1 度，所以除以 10 得到 angle_deg。

    stop 是主线程退出信号；退出时会关闭串口。
    """
    # timeout=0.05 让 read 不会永久阻塞，这样 Ctrl+C 后线程能及时退出。
    ser = serial.Serial(encoder_port, ENCODER_BAUDRATE, timeout=0.05)
    time.sleep(0.2)
    ser.reset_input_buffer()

    # 设置自动上报模式：AA 55 len cmd mode interval xor8。
    # len=2，因为 data 里有 mode 和 interval 两个字节。
    interval = max(1, min(255, int(ENCODER_REPORT_INTERVAL_MS)))
    body = SOF + bytes((2, CMD_SET_PARAM, MODE_AUTO_REPORT, interval))
    ser.write(body + bytes((xor8(body),)))
    print(
        f"[Encoder:{arm_label}] {encoder_port} open at {ENCODER_BAUDRATE} baud, "
        f"auto-report every {interval} ms"
    )

    buf = bytearray()
    try:
        while not stop.is_set():
            # 一次性读走当前串口缓冲区里已有的数据；如果没有数据，就读 1 字节
            # 等待 timeout，避免空转占满 CPU。
            waiting = ser.in_waiting
            chunk = ser.read(waiting if waiting > 0 else 1)
            if not chunk:
                time.sleep(0.001)
                continue

            buf.extend(chunk)
            while True:
                # 串口是字节流，没有消息边界。这里不断寻找帧头 AA 55；
                # 找不到时只保留可能成为下一帧开头的最后一个 AA。
                index = buf.find(SOF)
                if index < 0:
                    buf[:] = buf[-1:] if buf[-1:] == b"\xAA" else b""
                    break
                if index > 0:
                    del buf[:index]
                if len(buf) < 5:
                    break

                # 帧总长度 = 2 字节帧头 + 1 字节 len + 1 字节 cmd
                #          + len 字节 data + 1 字节 xor8。
                total = buf[2] + 5
                if len(buf) < total:
                    break

                raw = bytes(buf[:total])
                if xor8(raw[:-1]) != raw[-1]:
                    # 校验失败说明当前帧头可能是噪声或错位，丢掉一个字节后
                    # 继续重新同步。
                    del buf[0]
                    continue

                cmd = raw[3]
                data = raw[4:-1]
                del buf[:total]

                if cmd == CMD_ANGLE and len(data) == 2:
                    # 编码器角度单位是 0.1 度。例如原始整数 123 表示 12.3 度。
                    angle_deg = int.from_bytes(data, "big") / 10.0
                    with state.lock:
                        state.gripper = {
                            "sample_timestamp": time.time(),
                            "valid": 1,
                            "angle_deg": angle_deg,
                            "confidence": 1.0,
                        }
    finally:
        ser.close()
        print(f"[Encoder:{arm_label}] Stopped")


def _make_status(stage, ready, pose_valid, gripper_valid, sent_packets, send_errors):
    """生成写入共享内存 ring buffer 的 sender 状态。

    这份状态只用于主程序观察 sender 生命周期和传感器是否有效；
    真正的遥操 pose/gripper 数据由 _make_teleop_sample() 写入另一条
    共享内存 ring buffer。
    """
    return {
        "stage": np.array([stage], dtype=np.int32),
        "ready": np.array([ready], dtype=bool),
        "pose_valid": np.array([pose_valid], dtype=bool),
        "gripper_valid": np.array([gripper_valid], dtype=bool),
        "sent_packets": np.array([sent_packets], dtype=np.int64),
        "send_errors": np.array([send_errors], dtype=np.int64),
        "receive_timestamp": time.time(),
    }


def _make_teleop_sample(packet=None):
    """把原 lightumi.teleop_sender 包转换成共享内存可存储的定长数组。

    共享内存不存 Python dict/list 对象，只存 numpy 数组。因此这里保持
    原包里的数值语义不变，把最多 TELEOP_MAX_ARMS 路 arm 数据摊平成
    定长字段。接收端再恢复成原来的 ArmSample/PoseSample/GripperSample。
    """
    sample = {
        "sequence": np.array([0], dtype=np.int64),
        "timestamp_ns": np.array([0], dtype=np.int64),
        "elapsed_s": np.array([0.0], dtype=np.float64),
        "arm_count": np.array([0], dtype=np.int32),
        "arm_indices": np.full((TELEOP_MAX_ARMS,), -1, dtype=np.int32),
        "pose_valid": np.zeros((TELEOP_MAX_ARMS,), dtype=bool),
        "tracker_confidence": np.zeros((TELEOP_MAX_ARMS,), dtype=np.int32),
        "pose_sample_timestamp": np.zeros((TELEOP_MAX_ARMS,), dtype=np.float64),
        "position_m": np.zeros((TELEOP_MAX_ARMS, 3), dtype=np.float64),
        "quaternion_xyzw": np.tile(
            np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
            (TELEOP_MAX_ARMS, 1),
        ),
        "gripper_valid": np.zeros((TELEOP_MAX_ARMS,), dtype=bool),
        "gripper_angle_deg": np.zeros((TELEOP_MAX_ARMS,), dtype=np.float64),
        "gripper_confidence": np.zeros((TELEOP_MAX_ARMS,), dtype=np.float64),
        "gripper_sample_timestamp": np.zeros((TELEOP_MAX_ARMS,), dtype=np.float64),
        "receive_timestamp": time.time(),
    }
    if packet is None:
        return sample

    arms = packet.get("arms", [])[:TELEOP_MAX_ARMS]
    sample["sequence"][0] = int(packet.get("sequence", 0))
    sample["timestamp_ns"][0] = int(packet.get("timestamp_ns", 0))
    sample["elapsed_s"][0] = float(packet.get("elapsed_s", 0.0))
    sample["arm_count"][0] = len(arms)

    for out_idx, arm in enumerate(arms):
        pose = arm.get("pose") or {}
        gripper = arm.get("gripper") or {}
        sample["arm_indices"][out_idx] = int(arm.get("index", out_idx))
        sample["pose_valid"][out_idx] = bool(pose.get("valid", 0))
        sample["tracker_confidence"][out_idx] = int(
            pose.get("tracker_confidence", 0))
        sample["pose_sample_timestamp"][out_idx] = float(pose.get("sample_timestamp", 0.))
        sample["position_m"][out_idx] = np.array(
            pose.get("position_m", [0.0, 0.0, 0.0]),
            dtype=np.float64)
        sample["quaternion_xyzw"][out_idx] = np.array(
            pose.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0]),
            dtype=np.float64)
        sample["gripper_valid"][out_idx] = bool(gripper.get("valid", 0))
        sample["gripper_angle_deg"][out_idx] = float(
            gripper.get("angle_deg", 0.0))
        sample["gripper_confidence"][out_idx] = float(
            gripper.get("confidence", 0.0))
        sample["gripper_sample_timestamp"][out_idx] = float(gripper.get("sample_timestamp", 0.))

    return sample


def check_input_devices(
        t265_serial0=T265_SERIAL0,
        t265_serial1=T265_SERIAL1,
        encoder_port0=ENCODER_PORT0,
        encoder_port1=ENCODER_PORT1,
        verbose=True) -> bool:
    """检查所有已配置的 T265 和编码器是否能被系统枚举到。

    参数为空字符串或 None 表示该编号未配置，检查和读取都会跳过它。
    至少需要配置一台 T265 和一个编码器。
    """
    t265_serials = (t265_serial0, t265_serial1)
    encoder_ports = (encoder_port0, encoder_port1)
    configured_t265 = [str(value) for value in t265_serials if value]
    configured_encoders = [str(value) for value in encoder_ports if value]
    ok = bool(configured_t265) and bool(configured_encoders)

    try:
        import pyrealsense2 as rs

        ctx = rs.context()
        devices = list(ctx.query_devices())
        realsense_rows = []
        detected_serials = set()

        for dev in devices:
            def get_info(info):
                try:
                    if dev.supports(info):
                        return dev.get_info(info)
                except RuntimeError:
                    pass
                return ""

            name = get_info(rs.camera_info.name)
            serial_number = get_info(rs.camera_info.serial_number)
            product_line = get_info(rs.camera_info.product_line)
            realsense_rows.append((name, serial_number, product_line))
            detected_serials.add(serial_number)

        if verbose:
            print("[Device Check] RealSense devices:")
            if realsense_rows:
                for name, serial_number, product_line in realsense_rows:
                    print(
                        "  - "
                        f"name={name or 'unknown'} "
                        f"serial={serial_number or 'unknown'} "
                        f"product_line={product_line or 'unknown'}"
                    )
            else:
                print("  - none")
            for index, serial_number in enumerate(t265_serials):
                found = bool(serial_number) and str(serial_number) in detected_serials
                state_text = "DISABLED" if not serial_number else (
                    "OK" if found else "MISSING")
                print(
                    f"[Device Check] T265[{index}] serial "
                    f"{serial_number or '-'}: {state_text}"
                )

        ok = all(serial in detected_serials for serial in configured_t265) and ok
    except Exception as exc:
        ok = False
        if verbose:
            print(f"[Device Check] RealSense 枚举失败: {exc}")

    try:
        ports = list(list_ports.comports())
        detected_ports = {port.device for port in ports}

        if verbose:
            print("[Device Check] Serial ports:")
            if ports:
                for port in ports:
                    vid_pid = (
                        f"{port.vid:04X}:{port.pid:04X}"
                        if port.vid is not None and port.pid is not None
                        else "unknown"
                    )
                    serial_number = port.serial_number or "none"
                    print(
                        "  - "
                        f"device={port.device} "
                        f"vid_pid={vid_pid} "
                        f"serial={serial_number} "
                        f"desc={port.description}"
                    )
            else:
                print("  - none")
            for index, port_name in enumerate(encoder_ports):
                found = bool(port_name) and str(port_name) in detected_ports
                state_text = "DISABLED" if not port_name else (
                    "OK" if found else "MISSING")
                print(
                    f"[Device Check] Encoder[{index}] port "
                    f"{port_name or '-'}: {state_text}"
                )

        ok = all(port in detected_ports for port in configured_encoders) and ok
    except Exception as exc:
        ok = False
        if verbose:
            print(f"[Device Check] 串口枚举失败: {exc}")

    return bool(ok)


def run_sender_loop(stop, target_ip=TARGET_IP, target_port=TARGET_PORT,
                    delay_s=DELAY_S, status_callback=None,
                    teleop_callback=None, send_udp=True,
                    t265_serial0=T265_SERIAL0,
                    t265_serial1=T265_SERIAL1,
                    encoder_port0=ENCODER_PORT0,
                    encoder_port1=ENCODER_PORT1,
                    tcp_offset_enabled=TCP_OFFSET_ENABLED,
                    tcp_offset_m=TCP_OFFSET_M,
                    tcp_offset_quat=TCP_OFFSET_QUAT_XYZW) -> int:
    """启动所有已配置设备的读取线程，并发布编号稳定的遥操数据。

    ``t265_serial0/1`` 或 ``encoder_port0/1`` 为空时跳过对应设备。
    arm index 始终保持为 0/1，不会因为某一路未配置而重新编号。
    """
    def put_status(stage, ready=False, pose_valid=False, gripper_valid=False,
                   sent_packets=0, send_errors=0):
        if status_callback is not None:
            status_callback(_make_status(
                stage=stage,
                ready=ready,
                pose_valid=pose_valid,
                gripper_valid=gripper_valid,
                sent_packets=sent_packets,
                send_errors=send_errors,
            ))

    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        print("ERROR: pyrealsense2 is required. Use the t265/robodiff environment.")
        put_status(SENDER_STAGE_ERROR)
        return 1

    put_status(SENDER_STAGE_STARTING)
    print("=" * 60)
    print("Dual-arm Teleop Sender v3")
    if send_udp:
        print(f"  Transport: shared_memory + udp://{target_ip}:{target_port}")
    else:
        print("  Transport: shared_memory")
    print(f"  Send rate: {SEND_RATE_HZ} Hz")
    print(f"  Pose mode: {OUTPUT_FRAME}")
    print(f"  Sender smoothing: {'ON' if SMOOTHING_ENABLED else 'OFF'}")
    print(f"  Arm[0]: T265={t265_serial0 or '-'} Encoder={encoder_port0 or '-'}")
    print(f"  Arm[1]: T265={t265_serial1 or '-'} Encoder={encoder_port1 or '-'}")
    print("=" * 60)

    if delay_s > 0:
        # 延迟启动给操作者留时间拿稳 T265/夹爪，和原 sender_config.json
        # 里的 delay_s 行为一致。
        put_status(SENDER_STAGE_DELAY)
        print("请在倒计时内手动把 T265/夹爪摆到初始姿态。")
        remaining = delay_s
        while remaining > 0 and not stop.is_set():
            print(f"\rStarting in {remaining:.0f}s...", end="", flush=True)
            time.sleep(min(1.0, remaining))
            remaining -= 1.0
        print("\r" + " " * 30 + "\r", end="")
    if stop.is_set():
        put_status(SENDER_STAGE_STOPPED)
        return 0

    device_configs = [
        (0, t265_serial0, encoder_port0),
        (1, t265_serial1, encoder_port1),
    ]
    active_configs = [config for config in device_configs if config[1] or config[2]]
    if not active_configs:
        print("ERROR: 至少需要配置一路 T265 或编码器。")
        put_status(SENDER_STAGE_ERROR)
        return 1

    states = {index: SharedState() for index, _, _ in active_configs}
    sock = None
    if send_udp:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)

    threads = []
    for arm_index, t265_serial, encoder_port in active_configs:
        arm_label = f"arm{arm_index}"
        state = states[arm_index]
        if t265_serial:
            threads.append(threading.Thread(
                target=read_t265,
                args=(stop, state, t265_serial, arm_label),
                kwargs=dict(tcp_offset_enabled=tcp_offset_enabled,
                            tcp_offset_m=tcp_offset_m,
                            tcp_offset_quat=tcp_offset_quat),
                daemon=True,
            ))
        if encoder_port:
            threads.append(threading.Thread(
                target=read_encoder,
                args=(stop, state, encoder_port, arm_label),
                daemon=True,
            ))

    for thread in threads:
        thread.start()

    sequence = 0
    sent_packets = 0
    send_errors = 0
    started_at_ns = time.perf_counter_ns()
    interval_s = 1.0 / max(SEND_RATE_HZ, 1.0)
    last_print_at = time.monotonic()

    print("\nSending teleop samples. Press Ctrl+C to stop.\n")
    put_status(SENDER_STAGE_RUNNING, ready=False)
    try:
        while not stop.is_set():
            loop_start = time.monotonic()
            now_ns = time.perf_counter_ns()
            arms = []
            arm_values = {}
            for arm_index, _, _ in active_configs:
                state = states[arm_index]
                with state.lock:
                    pose = dict(state.pose)
                    gripper = dict(state.gripper)
                arm_values[arm_index] = (pose, gripper)
                arms.append({
                    "index": arm_index,
                    "pose": pose,
                    "gripper": gripper,
                })

            # 数据字段保持和原始 udp_sender.py 一致，后续控制逻辑无需改动。
            packet = {
                "type": "lightumi.teleop_sender",
                "version": 1,
                "sequence": sequence,
                "timestamp_ns": now_ns,
                "elapsed_s": (now_ns - started_at_ns) / 1_000_000_000.0,
                "arms": arms,
            }
            sequence += 1

            if teleop_callback is not None:
                teleop_callback(packet)

            try:
                # UDP 是无连接发送。这里不等待对端确认，保证发送端低延迟。
                if send_udp:
                    sock.sendto(
                        json.dumps(packet, ensure_ascii=False).encode("utf-8"),
                        (target_ip, target_port),
                    )
                    sent_packets += 1
                else:
                    sent_packets += 1
            except OSError as exc:
                send_errors += 1
                if send_errors <= 5 or send_errors % 100 == 0:
                    print(f"[send-error] {exc}")

            now = time.monotonic()
            if now - last_print_at >= 1.0:
                # parts = [f"sent={sent_packets}"]
                # for arm_index, _, _ in active_configs:
                #     pose, gripper = arm_values[arm_index]
                #     if pose["valid"]:
                #         px, py, pz = pose["position_m"]
                #         pose_text = f"pos=({px:.3f},{py:.3f},{pz:.3f})"
                #     else:
                #         pose_text = "no_pose"
                #     grip_text = (
                #         f"angle={gripper['angle_deg']:.1f}deg"
                #         if gripper["valid"] else "no_gripper"
                #     )
                #     parts.append(f"Arm[{arm_index}]:{pose_text},{grip_text}")
                # if send_errors:
                #     parts.append(f"err={send_errors}")
                # print("  ".join(parts))

                pose_ready = all(
                    bool(arm_values[index][0]["valid"])
                    for index, serial, _ in active_configs if serial
                )
                gripper_ready = all(
                    bool(arm_values[index][1]["valid"])
                    for index, _, port in active_configs if port
                )
                sensor_ready = pose_ready and gripper_ready
                put_status(
                    SENDER_STAGE_RUNNING,
                    ready=sensor_ready,
                    pose_valid=pose_ready,
                    gripper_valid=gripper_ready,
                    sent_packets=sent_packets,
                    send_errors=send_errors,
                )
                last_print_at = now

            elapsed = time.monotonic() - loop_start
            if elapsed < interval_s:
                # 简单限频到 SEND_RATE_HZ。这里保持原逻辑，不做额外调度封装。
                time.sleep(interval_s - elapsed)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # 通知读取线程退出，等待它们关闭硬件资源，然后按需关闭 UDP socket。
        stop.set()
        join_deadline = time.monotonic() + 3.0
        for thread in threads:
            thread.join(timeout=max(0.0, join_deadline - time.monotonic()))
        if sock is not None:
            sock.close()
        put_status(
            SENDER_STAGE_STOPPED,
            sent_packets=sent_packets,
            send_errors=send_errors,
        )
        print(f"Done. Sent {sent_packets} packets, {send_errors} errors.")

    return 0


class DualTeleopSenderProcess(mp.Process):
    """像 UMI Spacemouse 一样托管遥操输入采集的独立子进程。

    这个类负责生命周期和共享内存桥接：
    - start()/stop() 控制 sender 子进程；
    - SharedMemoryRingBuffer 保存最新状态；
    - 遥操位姿和夹爪数据也写入 SharedMemoryRingBuffer。

    send_udp=False 时，主遥操进程直接从共享内存读数据；send_udp=True
    主要用于兼容原来的 UDP 调试路径。
    """

    def __init__(self, shm_manager, target_ip=TARGET_IP, target_port=TARGET_PORT,
                 delay_s=DELAY_S, ready_timeout_s=None, get_max_k=30,
                 status_frequency=10, teleop_frequency=SEND_RATE_HZ,
                 send_udp=False,
                 t265_serial0=T265_SERIAL0,
                 t265_serial1=T265_SERIAL1,
                 encoder_port0=ENCODER_PORT0,
                 encoder_port1=ENCODER_PORT1,
                 tcp_offset_enabled=TCP_OFFSET_ENABLED,
                 tcp_offset_m=TCP_OFFSET_M,
                 tcp_offset_quat=TCP_OFFSET_QUAT_XYZW):
        super().__init__(name="DualTeleopSenderProcess")
        self.daemon = True
        self.target_ip = target_ip
        self.target_port = int(target_port)
        self.delay_s = float(delay_s)
        self.send_udp = bool(send_udp)
        self.t265_serial0 = t265_serial0
        self.t265_serial1 = t265_serial1
        self.encoder_port0 = encoder_port0
        self.encoder_port1 = encoder_port1
        self.tcp_offset_enabled = bool(tcp_offset_enabled)
        self.tcp_offset_m = tuple(tcp_offset_m)
        self.tcp_offset_quat = tuple(tcp_offset_quat)
        self.ready_timeout_s = (
            float(ready_timeout_s)
            if ready_timeout_s is not None
            else max(15.0, self.delay_s + 10.0)
        )
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()

        example = _make_status(
            stage=SENDER_STAGE_STARTING,
            ready=False,
            pose_valid=False,
            gripper_valid=False,
            sent_packets=0,
            send_errors=0,
        )
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=status_frequency,
        )
        self.teleop_ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=_make_teleop_sample(),
            get_max_k=get_max_k,
            get_time_budget=0.02,
            put_desired_frequency=teleop_frequency,
        )

    def get_status(self):
        return self.ring_buffer.get()

    def get_teleop(self):
        return self.teleop_ring_buffer.get()

    def start(self, wait=True, timeout=None):
        super().start()
        if wait:
            if not self.ready_event.wait(timeout):
                raise TimeoutError("遥操 sender 子进程初始化超时")
            if not self.is_alive():
                raise RuntimeError("遥操 sender 子进程在初始化阶段退出")
            status = self.get_status()
            if int(status["stage"][0]) == SENDER_STAGE_ERROR:
                raise RuntimeError(
                    "遥操 sender 初始化失败，请检查 T265、编码器和设备权限"
                )

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join(timeout=3.0)
            if self.is_alive():
                self.terminate()
                self.join(timeout=3.0)

    def __enter__(self):
        try:
            self.start(wait=True, timeout=self.ready_timeout_s)
        except BaseException:
            self.stop(wait=True)
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(wait=True)

    def run(self):
        def put_status(status):
            self.ring_buffer.put(status)
            if bool(status["ready"][0]) or int(status["stage"][0]) == SENDER_STAGE_ERROR:
                self.ready_event.set()

        def put_teleop(packet):
            self.teleop_ring_buffer.put(_make_teleop_sample(packet))

        try:
            code = run_sender_loop(
                stop=self.stop_event,
                target_ip=self.target_ip,
                target_port=self.target_port,
                delay_s=self.delay_s,
                status_callback=put_status,
                teleop_callback=put_teleop,
                send_udp=self.send_udp,
                t265_serial0=self.t265_serial0,
                t265_serial1=self.t265_serial1,
                encoder_port0=self.encoder_port0,
                encoder_port1=self.encoder_port1,
                tcp_offset_enabled=self.tcp_offset_enabled,
                tcp_offset_m=self.tcp_offset_m,
                tcp_offset_quat=self.tcp_offset_quat,
            )
            if code != 0:
                put_status(_make_status(
                    stage=SENDER_STAGE_ERROR,
                    ready=False,
                    pose_valid=False,
                    gripper_valid=False,
                    sent_packets=0,
                    send_errors=0,
                ))
        except BaseException as exc:
            print(f"[Teleop Sender Process] ERROR: {exc}", flush=True)
            put_status(_make_status(
                stage=SENDER_STAGE_ERROR,
                ready=False,
                pose_valid=False,
                gripper_valid=False,
                sent_packets=0,
                send_errors=0,
            ))
            raise


def main() -> int:
    """命令行入口：直接运行 sender。"""
    stop = threading.Event()
    return run_sender_loop(stop=stop)


if __name__ == "__main__":
    raise SystemExit(main())
