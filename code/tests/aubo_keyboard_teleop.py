#!/usr/bin/env python3
"""Real-time keyboard bring-up tool for the two AUBO arms.

Running without ``--enable-motion`` only queries and prints robot status.
Motion mode tracks key-down/key-up state and updates Cartesian targets at a
fixed frequency, matching the continuous action loop used by the UDP demo.
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import time
import tty

import click
import numpy as np
import scipy.spatial.transform as st

try:
    from pynput.keyboard import Key, Listener
except Exception as exc:  # Keep status-only mode and --help usable headlessly.
    Key = None
    Listener = None
    _PYNPUT_IMPORT_ERROR = exc
else:
    _PYNPUT_IMPORT_ERROR = None

from diffusion_policy.envs.realword.aubo_interpolation_controller import (
    query_robot_status,
)
from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv


class HeldKeyboard:
    """Track held keys and one-shot events with pynput press/release callbacks."""

    def __init__(self) -> None:
        if Listener is None:
            raise RuntimeError(
                "real-time keyboard control needs pynput and an active desktop "
                "display session"
            ) from _PYNPUT_IMPORT_ERROR
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard control requires an interactive terminal")
        self.fd = sys.stdin.fileno()
        self.previous = termios.tcgetattr(self.fd)
        self._held = set()
        self._events = []
        self._lock = threading.Lock()
        self._listener = Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )

    @staticmethod
    def _key_name(key):
        if key == Key.esc:
            return "esc"
        if key == Key.space:
            return "space"
        char = getattr(key, "char", None)
        return None if char is None else char.lower()

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            if name not in self._held:
                self._events.append(name)
            self._held.add(name)

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            self._held.discard(name)

    def held(self) -> set[str]:
        with self._lock:
            return set(self._held)

    def pop_events(self) -> list[str]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def drain_terminal(self) -> None:
        # pynput receives global events; discard the duplicate terminal bytes so
        # they are not executed by the shell after this program exits.
        while select.select([self.fd], [], [], 0)[0]:
            sys.stdin.read(1)

    def __enter__(self):
        tty.setcbreak(self.fd)
        self._listener.start()
        self._listener.wait()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._listener.stop()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.previous)


def _format_vector(values) -> str:
    return "[" + ", ".join(f"{float(value):+.5f}" for value in values) + "]"


def _print_status(label: str, status: dict) -> None:
    print(
        f"{label}: {status['robot_name']} @ {status['ip']}:{status['port']} "
        f"mode={status['robot_mode']} safety={status['safety_mode']}"
    )
    print(f"  joints(rad): {_format_vector(status['joint_positions'])}")
    print(f"  tcp xyz+rotvec: {_format_vector(status['tcp_pose_rotvec'])}")


def _query_both(left_ip, right_ip, port, username, password) -> None:
    print("只读状态检查：不会上电、进入伺服或发送运动命令。")
    left = query_robot_status(left_ip, port, username, password)
    _print_status("LEFT ", left)
    right = query_robot_status(right_ip, port, username, password)
    _print_status("RIGHT", right)


def _apply_delta(pose, translation, rotation):
    target = np.asarray(pose, dtype=np.float64).copy()
    target[:3] += translation
    if np.linalg.norm(rotation) > 0:
        target[3:] = (
            st.Rotation.from_rotvec(rotation)
            * st.Rotation.from_rotvec(target[3:])
        ).as_rotvec()
    return target


def _step_towards_home(current, home, translation_step, rotation_step_rad):
    """Take one bounded Cartesian step toward a saved startup pose."""
    current = np.asarray(current, dtype=np.float64)
    home = np.asarray(home, dtype=np.float64)
    target = current.copy()

    delta = home[:3] - current[:3]
    distance = float(np.linalg.norm(delta))
    position_arrived = distance <= 1e-5
    if position_arrived:
        target[:3] = home[:3]
    else:
        target[:3] += delta * (min(translation_step, distance) / distance)

    current_rotation = st.Rotation.from_rotvec(current[3:])
    home_rotation = st.Rotation.from_rotvec(home[3:])
    relative = home_rotation * current_rotation.inv()
    angle = float(relative.magnitude())
    rotation_arrived = angle <= 1e-4
    if rotation_arrived:
        target[3:] = home[3:]
    else:
        adjustment = st.Rotation.from_rotvec(
            relative.as_rotvec() * (min(rotation_step_rad, angle) / angle)
        )
        target[3:] = (adjustment * current_rotation).as_rotvec()

    return target, position_arrived and rotation_arrived


def _return_home(
    env: DualAuboRealEnv,
    targets: dict[str, np.ndarray],
    home: dict[str, np.ndarray],
    translation_speed: float,
    rotation_speed_rad: float,
    control_frequency: float,
    command_horizon: float,
) -> None:
    """Return both commanded poses to startup home at configured velocities."""
    print("双臂开始返回启动 home；按 Ctrl-C 或实体急停可中断。")
    period = 1.0 / control_frequency
    while True:
        cycle_started = time.monotonic()
        env.assert_healthy()
        arrived_all = True
        next_targets = {}
        for side in ("left", "right"):
            next_targets[side], arrived = _step_towards_home(
                targets[side],
                home[side],
                translation_speed * period,
                rotation_speed_rad * period,
            )
            arrived_all = arrived_all and arrived
        targets.update(next_targets)
        if arrived_all:
            print("双臂已回到启动 home。")
            return
        env.exec_actions(
            np.concatenate((targets["left"], targets["right"])),
            time.time() + command_horizon,
        )
        remaining = period - (time.monotonic() - cycle_started)
        if remaining > 0:
            time.sleep(remaining)


def _print_live_state(env: DualAuboRealEnv) -> None:
    state = env.get_state()
    for label in ("left", "right"):
        pose = state[label]["ActualTCPPose"]
        joints = state[label]["ActualQ"]
        print(
            f"{label:>5} tcp={_format_vector(pose)} "
            f"q={_format_vector(joints)}"
        )


def _run_keyboard(
    env: DualAuboRealEnv,
    translation_speed: float,
    rotation_speed_rad: float,
    control_frequency: float,
    command_horizon: float,
) -> None:
    state = env.get_state()
    for side in ("left", "right"):
        pose = np.asarray(state[side]["ActualTCPPose"], dtype=np.float64)
        joints = np.asarray(state[side]["ActualQ"], dtype=np.float64)
        if (
            pose.shape != (6,)
            or joints.shape != (6,)
            or not np.all(np.isfinite(pose))
            or not np.all(np.isfinite(joints))
            or (np.allclose(pose, 0.0) and np.allclose(joints, 0.0))
        ):
            raise RuntimeError(f"{side} received invalid initial RTDE state")
    home = {
        side: np.asarray(state[side]["ActualTCPPose"], dtype=np.float64).copy()
        for side in ("left", "right")
    }
    targets = {side: pose.copy() for side, pose in home.items()}
    selected = "both"
    last_motion_print = 0.0

    translation_directions = {
        "w": np.array([1.0, 0.0, 0.0]),
        "s": np.array([-1.0, 0.0, 0.0]),
        "a": np.array([0.0, 1.0, 0.0]),
        "d": np.array([0.0, -1.0, 0.0]),
        "r": np.array([0.0, 0.0, 1.0]),
        "f": np.array([0.0, 0.0, -1.0]),
    }
    rotation_directions = {
        "i": np.array([1.0, 0.0, 0.0]),
        "k": np.array([-1.0, 0.0, 0.0]),
        "j": np.array([0.0, 1.0, 0.0]),
        "l": np.array([0.0, -1.0, 0.0]),
        "u": np.array([0.0, 0.0, 1.0]),
        "o": np.array([0.0, 0.0, -1.0]),
    }

    print("\n================ AUBO 双臂键盘控制 ================")
    print("坐标系：每台 AUBO 各自的基坐标系")
    print("当前选择：双臂")
    print("")
    print("机械臂选择")
    print("  1          只控制左臂")
    print("  2          只控制右臂")
    print("  3          同时控制双臂")
    print("")
    print("平移控制")
    print("  W / S      沿基坐标 +X / -X")
    print("  A / D      沿基坐标 +Y / -Y")
    print("  R / F      沿基坐标 +Z / -Z（上升/下降取决于基坐标安装）")
    print("")
    print("旋转控制")
    print("  I / K      绕基坐标 +X / -X 旋转（Rx）")
    print("  J / L      绕基坐标 +Y / -Y 旋转（Ry）")
    print("  U / O      绕基坐标 +Z / -Z 旋转（Rz）")
    print("")
    print("状态与归位")
    print("  H          双臂分段返回程序启动时记录的 home")
    print("  P          打印双臂实时 TCP 位姿和关节角")
    print("  Space      取消未完成目标，以当前实测位姿重新保持")
    print("")
    print("退出与中断")
    print("  Q / Esc    正常退出伺服模式并断开连接，不下电")
    print("  Ctrl-C     中断程序；返回 home 过程中也可使用")
    print("")
    print(
        f"实时速度：平移 {translation_speed * 1000:.1f} mm/s，"
        f"旋转 {np.rad2deg(rotation_speed_rad):.1f} deg/s"
    )
    print(
        f"末端键盘目标 {control_frequency:.1f} Hz，"
        f"动作前瞻 {command_horizon:.3f} s"
    )
    print("按住运动键持续移动；松开后停止追加目标（不依赖键盘自动连发）。")
    print("未限制会话工作范围；程序仍保留伺服速度与关节增量保护。")
    print("启动实测位姿已记录为双臂 home。")
    print("====================================================")
    _print_live_state(env)

    period = 1.0 / control_frequency
    next_cycle = time.monotonic()
    with HeldKeyboard() as keyboard:
        while True:
            env.assert_healthy()
            keyboard.drain_terminal()
            for key in keyboard.pop_events():
                if key in ("q", "esc"):
                    print("退出键盘控制。")
                    return
                if key in ("1", "2", "3"):
                    selected = {"1": "left", "2": "right", "3": "both"}[key]
                    print(f"当前控制：{selected}")
                elif key == "p":
                    _print_live_state(env)
                elif key == "h":
                    _return_home(
                        env,
                        targets,
                        home,
                        translation_speed,
                        rotation_speed_rad,
                        control_frequency,
                        command_horizon,
                    )
                    next_cycle = time.monotonic()
                elif key == "space":
                    measured = env.get_state()
                    for side in ("left", "right"):
                        targets[side] = np.asarray(
                            measured[side]["ActualTCPPose"], dtype=np.float64
                        ).copy()
                    env.exec_actions(
                        np.concatenate((targets["left"], targets["right"])),
                        time.time() + command_horizon,
                    )
                    print("已用当前实测位姿重新保持。")

            held = keyboard.held()
            translation_direction = sum(
                (translation_directions[key] for key in held
                 if key in translation_directions),
                start=np.zeros(3),
            )
            rotation_direction = sum(
                (rotation_directions[key] for key in held
                 if key in rotation_directions),
                start=np.zeros(3),
            )
            translation_norm = float(np.linalg.norm(translation_direction))
            rotation_norm = float(np.linalg.norm(rotation_direction))
            if translation_norm > 1.0:
                translation_direction /= translation_norm
            if rotation_norm > 1.0:
                rotation_direction /= rotation_norm
            translation = translation_direction * translation_speed * period
            rotation = rotation_direction * rotation_speed_rad * period

            if translation_norm > 0.0 or rotation_norm > 0.0:
                selected_sides = (
                    ("left", "right") if selected == "both" else (selected,)
                )
                proposed = {
                    side: targets[side].copy() for side in ("left", "right")
                }
                for side in selected_sides:
                    proposed[side] = _apply_delta(
                        proposed[side], translation, rotation
                    )

                targets = proposed
                env.exec_actions(
                    np.concatenate((targets["left"], targets["right"])),
                    time.time() + command_horizon,
                )
                now = time.monotonic()
                if now - last_motion_print >= 0.5:
                    active = "+".join(sorted(
                        key.upper() for key in held
                        if key in translation_directions or key in rotation_directions
                    ))
                    print(
                        f"{selected}: held={active}  "
                        f"left={_format_vector(targets['left'])}  "
                        f"right={_format_vector(targets['right'])}"
                    )
                    last_motion_print = now

            next_cycle += period
            remaining = next_cycle - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif remaining < -period:
                next_cycle = time.monotonic()


@click.command(context_settings={"show_default": True})
@click.option("--left-robot-ip", default="192.168.3.200")
@click.option("--right-robot-ip", default="192.168.3.199")
@click.option("--robot-port", default=30004, type=int)
@click.option("--username", default="aubo")
@click.option("--password", default="123456", hide_input=True, show_default=False)
@click.option(
    "--enable-motion",
    is_flag=True,
    help="Explicitly enter servo mode and enable keyboard motion.",
)
@click.option(
    "--power-on",
    is_flag=True,
    help="Allow poweron/startup if an arm is not already Running.",
)
@click.option(
    "--translation-speed-mm-s",
    default=20.0,
    type=float,
    help="TCP translation speed while a movement key is held.",
)
@click.option(
    "--rotation-speed-deg-s",
    default=10.0,
    type=float,
    help="TCP rotation speed while a rotation key is held.",
)
@click.option(
    "--keyboard-frequency",
    default=10.0,
    type=float,
    help="Cartesian keyboard-target update frequency.",
)
@click.option(
    "--robot-frequency",
    default=200.0,
    type=float,
    help="Inverse-kinematics and servoJoint command frequency.",
)
@click.option(
    "--servo-t",
    default=0.02,
    type=float,
    help="servoJoint t parameter; official 5 ms example uses 0.02 s.",
)
@click.option(
    "--command-horizon",
    default=0.1,
    type=float,
    help="Look-ahead time for each continuously updated target.",
)
@click.option("--verbose", is_flag=True)
def main(
    left_robot_ip,
    right_robot_ip,
    robot_port,
    username,
    password,
    enable_motion,
    power_on,
    translation_speed_mm_s,
    rotation_speed_deg_s,
    keyboard_frequency,
    robot_frequency,
    servo_t,
    command_horizon,
    verbose,
):
    """Query or perform keyboard control of the dual AUBO arms."""
    if not enable_motion:
        print("请先关闭可能正在控制 AUBO 的其他 RPC 客户端。")
        try:
            _query_both(left_robot_ip, right_robot_ip, robot_port, username, password)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        print("\n未指定 --enable-motion，状态读取完成，未发送运动命令。")
        return

    translation_speed = translation_speed_mm_s / 1000.0
    rotation_speed = np.deg2rad(rotation_speed_deg_s)
    if translation_speed <= 0:
        raise click.UsageError("translation speed must be positive")
    if rotation_speed <= 0:
        raise click.UsageError("rotation speed must be positive")
    if keyboard_frequency <= 0:
        raise click.UsageError("keyboard frequency must be positive")
    if not 0 < robot_frequency <= 500:
        raise click.UsageError("robot frequency must be in (0, 500] Hz")
    if servo_t <= 0:
        raise click.UsageError("servo t must be positive")
    if command_horizon < 1.0 / keyboard_frequency:
        raise click.UsageError(
            "command horizon must be at least one keyboard control period"
        )

    print("即将连接并控制：")
    print(f"  LEFT  {left_robot_ip}:{robot_port}")
    print(f"  RIGHT {right_robot_ip}:{robot_port}")
    print(
        f"  实时平移速度 {translation_speed_mm_s:.1f} mm/s，"
        f"旋转速度 {rotation_speed_deg_s:.1f} deg/s"
    )
    print(
        f"  末端目标频率 {keyboard_frequency:.1f} Hz，"
        f"IK/servoJoint 频率 {robot_frequency:.1f} Hz"
    )
    print(
        f"  servoJoint t={servo_t:.3f} s，"
        f"动作前瞻 {command_horizon:.3f} s"
    )
    print(f"  自动上电/松刹车：{'允许' if power_on else '不允许'}")
    print("请先用示教器把双臂放到期望的 home；确认后会记录当前 TCP 位姿。")
    if not click.confirm(
        "确认双臂已在 home、急停可用，并进入双臂伺服控制？"
    ):
        print("已取消，未连接控制器。")
        return

    env = DualAuboRealEnv(
        left_robot_ip=left_robot_ip,
        right_robot_ip=right_robot_ip,
        left_robot_port=robot_port,
        right_robot_port=robot_port,
        username=username,
        password=password,
        robot_frequency=robot_frequency,
        servo_t=servo_t,
        max_pos_speed=max(translation_speed * 1.5, 1e-3),
        max_rot_speed=max(rotation_speed * 1.5, 1e-3),
        speed_fraction=0.2,
        max_joint_step=0.01,
        auto_power_on=power_on,
        verbose=verbose,
    )
    try:
        env.start(wait=True)
        print("双臂已进入伺服保持；尚未产生位移。")
        _run_keyboard(
            env,
            translation_speed,
            rotation_speed,
            keyboard_frequency,
            command_horizon,
        )
    finally:
        env.stop(wait=True)
        print("已退出两台机械臂的伺服模式；未执行下电。")


if __name__ == "__main__":
    main()
