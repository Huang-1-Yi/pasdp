"""AUBO Cartesian waypoint controller backed by ``pyaubo_sdk``.

The public pose convention is ``[x, y, z, rx, ry, rz]`` where translation is
in metres and rotation is a rotation vector in radians.  The AUBO RPC inverse
kinematics API expects RPY angles, so conversion is kept at the SDK boundary.

This controller intentionally uses the control path validated in the AUBO
reference checkout: Cartesian target -> inverse kinematics -> ``servoJoint``.
Robot state is received on the independent RTDE channel (port 30010) so RPC
state reads do not interrupt the servo stream.
"""

from __future__ import annotations

import enum
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from multiprocessing.managers import SharedMemoryManager
from typing import Any

import numpy as np
import scipy.spatial.transform as st

try:
    import pyaubo_sdk
except ImportError as exc:  # Keep --help/static imports usable before install.
    pyaubo_sdk = None
    _PYAUBO_IMPORT_ERROR = exc
else:
    _PYAUBO_IMPORT_ERROR = None

from diffusion_policy.common.pose_trajectory_interpolator import (
    PoseTrajectoryInterpolator,
)
from diffusion_policy.envs.realword.shared_memory.shared_memory_queue import (
    Empty,
    SharedMemoryQueue,
)
from diffusion_policy.envs.realword.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)


from diffusion_policy.envs.realword.aubo_homing import JointHome, joints as validate_joints
from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper

RTDE_PORT = 30010


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    HOME = 3
    HEARTBEAT = 4
    HOLD = 5


def require_pyaubo_sdk() -> Any:
    """Return the Python SDK or raise an installation-oriented error."""
    if pyaubo_sdk is None:
        raise RuntimeError(
            "pyaubo_sdk is not installed. Use the aubo_t265 launcher with "
            "Linux x86_64 Python 3.9 or install a matching AUBO Python SDK."
        ) from _PYAUBO_IMPORT_ERROR
    return pyaubo_sdk


def _enum_text(value: Any) -> str:
    return str(getattr(value, "name", value))


def rotvec_to_aubo(pose: np.ndarray) -> list[float]:
    """Convert xyz + rotation-vector pose to xyz + extrinsic xyz RPY."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"pose must have shape (6,), got {pose.shape}")
    rpy = st.Rotation.from_rotvec(pose[3:]).as_euler("xyz")
    return [*map(float, pose[:3]), *map(float, rpy)]


def aubo_to_rotvec(pose: Any) -> np.ndarray:
    """Convert AUBO xyz + extrinsic xyz RPY pose to xyz + rotation vector."""
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (6,):
        raise ValueError(f"AUBO pose must have shape (6,), got {pose.shape}")
    rotvec = st.Rotation.from_euler("xyz", pose[3:]).as_rotvec()
    return np.concatenate((pose[:3], rotvec))


def query_robot_status(
    robot_ip: str,
    robot_port: int = 30004,
    username: str = "aubo",
    password: str = "123456",
    request_timeout_ms: int = 1000,
) -> dict[str, Any]:
    """Login, read status once, then disconnect without enabling or moving."""
    sdk = require_pyaubo_sdk()
    rpc = sdk.RpcClient()
    rpc.setRequestTimeout(int(request_timeout_ms))
    try:
        rpc.connect(robot_ip, int(robot_port))
        if not rpc.hasConnected():
            raise RuntimeError(f"cannot connect to AUBO RPC {robot_ip}:{robot_port}")
        rpc.login(username, password)
        if not rpc.hasLogined():
            raise RuntimeError(f"cannot login to AUBO RPC {robot_ip}:{robot_port}")
        names = list(rpc.getRobotNames())
        if not names:
            raise RuntimeError(f"no robot reported by {robot_ip}:{robot_port}")
        robot = rpc.getRobotInterface(names[0])
        state = robot.getRobotState()
        tcp_pose = list(state.getTcpPose())
        return {
            "ip": robot_ip,
            "port": int(robot_port),
            "robot_name": names[0],
            "robot_mode": _enum_text(state.getRobotModeType()),
            "safety_mode": _enum_text(state.getSafetyModeType()),
            "joint_positions": np.asarray(
                state.getJointPositions(), dtype=np.float64
            ),
            "tcp_pose_aubo_rpy": np.asarray(tcp_pose, dtype=np.float64),
            "tcp_pose_rotvec": aubo_to_rotvec(tcp_pose),
        }
    finally:
        try:
            if rpc.hasLogined():
                rpc.logout()
        except Exception:
            pass
        try:
            rpc.disconnect()
        except Exception:
            pass


class AuboInterpolationController(mp.Process):
    """One AUBO arm controlled by interpolated Cartesian waypoints."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip: str,
        robot_port: int = 30004,
        username: str = "aubo",
        password: str = "123456",
        frequency: float = 100.0,
        state_frequency: float = 20.0,
        servo_t: float | None = None,
        lookahead_time: float = 0.1,
        gain: float = 200.0,
        servo_a: float = 0.2,
        servo_v: float = 0.3,
        max_joint_step: float = 0.01,
        max_pos_speed: float = 0.05,
        max_rot_speed: float = 0.3,
        speed_fraction: float = 0.2,
        auto_power_on: bool = False,
        launch_timeout: float = 20.0,
        soft_real_time: bool = False,
        verbose: bool = False,
        command_timeout: float | None = None,
        get_max_k: int = 128,
        gripper_config: dict | None = None,
        max_joint_speed: float | None = 0.5,
        max_following_error: float | None = 0.2,
        reject_joint_jumps: bool = False,
    ) -> None:
        if not 0 < frequency <= 500:
            raise ValueError("frequency must be in (0, 500]")
        if not 0 < state_frequency <= frequency:
            raise ValueError("state_frequency must be in (0, frequency]")
        if not 0 < speed_fraction <= 1:
            raise ValueError("speed_fraction must be in (0, 1]")
        if min(max_joint_step, max_pos_speed, max_rot_speed) <= 0:
            raise ValueError("motion limits must be positive")

        super().__init__(name=f"AuboController-{robot_ip}")
        self.robot_ip = robot_ip
        self.robot_port = int(robot_port)
        self.username = username
        self.password = password
        self.frequency = float(frequency)
        self.state_frequency = float(state_frequency)
        self.servo_t = float(servo_t if servo_t is not None else (2.0 / frequency))
        self.lookahead_time = float(lookahead_time)
        self.gain = float(gain)
        self.servo_a = float(servo_a)
        self.servo_v = float(servo_v)
        self.max_joint_step = float(max_joint_step)
        self.max_pos_speed = float(max_pos_speed)
        self.max_rot_speed = float(max_rot_speed)
        self.speed_fraction = float(speed_fraction)
        self.auto_power_on = bool(auto_power_on)
        self.launch_timeout = float(launch_timeout)
        self.soft_real_time = bool(soft_real_time)
        self.verbose = bool(verbose)
        self.command_timeout = command_timeout
        self.gripper_config = dict(gripper_config or {"type": "disabled"})
        self.gripper_config.setdefault('type', 'disabled')
        AuboDigitalGripper(None, self.gripper_config)
        self.max_joint_speed = None if max_joint_speed is None else float(max_joint_speed)
        self.max_following_error = None if max_following_error is None else float(max_following_error)
        self.reject_joint_jumps = bool(reject_joint_jumps)
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.max_joint_step,
                    self.max_pos_speed, self.max_rot_speed, self.lookahead_time,
                    self.gain, self.servo_a, self.servo_v, self.launch_timeout)):
            raise ValueError("Controller parameters must be finite and positive")
        for value in (self.max_joint_speed, self.max_following_error):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("Optional joint checks must be positive or None")
        if command_timeout is not None and (not np.isfinite(command_timeout) or command_timeout <= 0):
            raise ValueError("command_timeout must be positive")
        if not np.isfinite(self.servo_t) or self.servo_t < .005:
            raise ValueError("servo_t must be at least 5 ms")

        queue_example = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros(6, dtype=np.float64),
            "duration": 0.0,
            "target_time": 0.0,
            "gripper": -1.,
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=queue_example,
            buffer_size=256,
        )
        state_example = {
            "ActualTCPPose": np.zeros(6, dtype=np.float64),
            "ActualTCPSpeed": np.zeros(6, dtype=np.float64),
            "ActualQ": np.zeros(6, dtype=np.float64),
            "ActualQd": np.zeros(6, dtype=np.float64),
            "TargetTCPPose": np.zeros(6, dtype=np.float64),
            "robot_receive_timestamp": 0.0,
            "GripperState": -1.,
            "GripperEnabled": False,
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=state_example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=state_frequency,
        )
        self.home_state = mp.Value("i", 0)
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()
        self.startup_ok = mp.Value("b", False)

    def start(self, wait: bool = True) -> None:
        require_pyaubo_sdk()
        super().start()
        if wait:
            self.start_wait()

    def start_wait(self) -> None:
        if not self.ready_event.wait(self.launch_timeout):
            if not self.is_alive():
                raise RuntimeError(
                    f"AUBO controller {self.robot_ip} exited during startup "
                    f"(exitcode={self.exitcode})"
                )
            raise TimeoutError(f"AUBO controller {self.robot_ip} startup timed out")
        if not self.startup_ok.value:
            self.join(timeout=1.0)
            raise RuntimeError(
                f"AUBO controller {self.robot_ip} failed during startup; "
                "see child-process error above"
            )

    def stop(self, wait: bool = True) -> None:
        self.stop_event.set()
        if self.is_alive():
            from queue import Full
            try:
                self.input_queue.put({"cmd": Command.STOP.value})
            except Full:
                pass  # The independent stop event also preempts a saturated queue.
        if wait:
            self.stop_wait()

    def stop_wait(self, timeout: float = 5.0) -> None:
        if self.pid is None:
            return
        self.join(timeout=timeout)
        if self.is_alive():
            raise TimeoutError(f"AUBO controller {self.robot_ip} did not stop")

    @property
    def is_ready(self) -> bool:
        return bool(self.ready_event.is_set() and self.startup_ok.value and self.is_alive())

    def get_state(self, k: int | None = None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    def servoL(self, pose: Any, duration: float = 0.5) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f"pose must have shape (6,), got {pose.shape}")
        if not np.isfinite(duration) or duration < 1.0 / self.frequency:
            raise ValueError("duration is shorter than one controller period")
        if not self.is_ready:
            raise RuntimeError(f"AUBO controller {self.robot_ip} is not ready")
        self.input_queue.put(
            {"cmd": Command.SERVOL.value, "target_pose": pose, "duration": duration}
        )

    def schedule_waypoint(self, pose: Any, target_time: float, gripper=None) -> None:
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f"pose must have shape (6,), got {pose.shape}")
        if not self.is_ready:
            raise RuntimeError(f"AUBO controller {self.robot_ip} is not ready")
        if not np.isfinite(target_time):
            raise ValueError("target_time must be finite")
        if gripper is not None:
            if self.gripper_config["type"] == "disabled":
                raise RuntimeError("Gripper command supplied to disabled gripper")
            AuboDigitalGripper(None, self.gripper_config).validate_command(gripper)
        self.input_queue.put(
            {
                "cmd": Command.SCHEDULE_WAYPOINT.value,
                "target_pose": pose,
                "target_time": max(float(target_time), time.time() + 1e-3),
                "gripper": -1. if gripper is None else float(gripper),
            }
        )

    def return_home(self, target):
        target = validate_joints(target)
        if not self.is_ready:
            raise RuntimeError("Controller not ready for home")
        self.home_state.value = 1
        try:
            self.input_queue.put({"cmd": Command.HOME.value, "target_pose": target})
        except Exception:
            self.home_state.value = 0
            raise

    def hold(self):
        if not self.is_ready:
            raise RuntimeError("Controller not ready for hold")
        self.input_queue.put({"cmd": Command.HOLD.value})

    def heartbeat(self):
        if not self.is_ready:
            raise RuntimeError("Controller stopped during home")
        self.input_queue.put({"cmd": Command.HEARTBEAT.value})

    @staticmethod
    def _limit_joint_step(q_prev, q_target, max_step):
        q_prev = np.asarray(q_prev, dtype=np.float64)
        q_target = np.asarray(q_target, dtype=np.float64)
        if q_prev.shape != (6,) or q_target.shape != (6,) or not np.isfinite([q_prev, q_target]).all():
            raise ValueError("Invalid IK joint vector")
        delta = q_target - q_prev
        largest = float(np.max(np.abs(delta)))
        if largest > max_step:
            delta *= max_step / largest
        return q_prev + delta

    def _set_servo_mode(self, motion_control, enable: bool) -> None:
        target = 1 if enable else 0
        ret = None
        actual = None
        for attempt in range(3):
            ret = motion_control.setServoModeSelect(target)
            for _ in range(50):
                actual = motion_control.getServoModeSelect()
                if actual == target:
                    return
                time.sleep(0.01)
            if not enable:
                break
            # A previous planner request can leave the controller briefly busy.
            # Retry entering mode 1, but never unlock a protective stop here.
            time.sleep(0.1 * (attempt + 1))
        raise RuntimeError(
            f"setServoModeSelect({target}) failed on {self.robot_ip}, "
            f"ret={ret}, actual={actual}"
        )

    def _start_runtime(self, runtime_machine, sdk) -> None:
        """Start the planner and confirm it is Running before servo mode."""
        ret = runtime_machine.start()
        deadline = time.monotonic() + 2.0
        state = runtime_machine.getRuntimeState()
        while state != sdk.RuntimeState.Running and time.monotonic() < deadline:
            time.sleep(0.05)
            state = runtime_machine.getRuntimeState()
        if state != sdk.RuntimeState.Running:
            raise RuntimeError(
                f"runtime did not enter Running on {self.robot_ip}: "
                f"ret={ret}, state={_enum_text(state)}"
            )

    @staticmethod
    def _copy_rtde_state(rtde_lock, rtde_latest):
        with rtde_lock:
            raw_state = rtde_latest["state"]
            if raw_state is None:
                return None
            state = {key: value.copy() for key, value in raw_state.items()}
            received = rtde_latest["received"]
        if any(value.shape != (6,) for value in state.values()):
            return None
        if not all(np.all(np.isfinite(value)) for value in state.values()):
            return None
        if np.allclose(state["joint_angles"], 0.0) and np.allclose(
            state["eef_pose"], 0.0
        ):
            return None
        state["received"] = received
        return state

    def _warmup_servo(self, motion_control, q_hold, sdk) -> None:
        """Prove that an exact-current-joint hold is accepted before ready."""
        self._set_servo_mode(motion_control, True)
        time.sleep(0.02)
        last_ret = None
        for attempt in range(5):
            last_ret = self._servo_joint(motion_control, q_hold)
            if last_ret == 0:
                return
            if last_ret != -13:
                break
            # -13 is transient on some 0.26.0 controllers immediately after
            # entering servo mode.  Re-select mode 1 without an intervening
            # status RPC, then retry the same no-displacement hold command.
            select_ret = motion_control.setServoModeSelect(1)
            if self.verbose:
                print(
                    f"[AuboInterpolationController:{self.robot_ip}] "
                    f"startup hold ignored ({attempt + 1}/5); "
                    f"reselect servo ret={select_ret}",
                    flush=True,
                )
            time.sleep(0.05)
        description = sdk.returnValue2Str(int(last_ret))
        raise RuntimeError(
            f"initial servoJoint hold failed on {self.robot_ip}: "
            f"{last_ret} ({description})"
        )

    def _power_on(self, robot) -> None:
        sdk = require_pyaubo_sdk()
        state = robot.getRobotState()
        if state.getSafetyModeType() != sdk.SafetyModeType.Normal:
            raise RuntimeError("AUBO safety mode must be Normal before starting servo")
        if state.getRobotModeType() == sdk.RobotModeType.Running:
            return
        if not self.auto_power_on:
            raise RuntimeError(
                f"{self.robot_ip} is {_enum_text(state.getRobotModeType())}, not Running; "
                "start it on the pendant or pass auto_power_on=True explicitly"
            )

        manager = robot.getRobotManage()
        manager.poweron()
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            mode = state.getRobotModeType()
            if mode in (sdk.RobotModeType.Idle, sdk.RobotModeType.Running):
                break
            time.sleep(0.5)
        for _ in range(5):
            if manager.startup() == 0:
                break
            time.sleep(1.0)
        deadline = time.monotonic() + 30.0
        while state.getRobotModeType() != sdk.RobotModeType.Running:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"{self.robot_ip} power-on timed out in mode "
                    f"{_enum_text(state.getRobotModeType())}"
                )
            time.sleep(0.5)

    def _start_rtde(self):
        sdk = require_pyaubo_sdk()
        rtde = sdk.RtdeClient()
        rtde.connect(self.robot_ip, RTDE_PORT)
        if not rtde.hasConnected():
            raise RuntimeError(f"cannot connect to AUBO RTDE {self.robot_ip}:{RTDE_PORT}")
        rtde.login(self.username, self.password)
        if not rtde.hasLogined():
            raise RuntimeError(f"cannot login to AUBO RTDE {self.robot_ip}:{RTDE_PORT}")

        lock = threading.Lock()
        latest = {"state": None}

        def callback(parser):
            state = {
                "joint_angles": np.asarray(parser.popVectorDouble(), dtype=np.float64),
                "joint_speeds": np.asarray(parser.popVectorDouble(), dtype=np.float64),
                # R1_actual_TCP_pose uses the SDK's xyz + RPY convention.
                # Controller waypoints and the public state use rotation vectors.
                "eef_pose": aubo_to_rotvec(parser.popVectorDouble()),
                "eef_speed": np.asarray(parser.popVectorDouble(), dtype=np.float64),
            }
            with lock:
                latest["state"] = state
                latest["received"] = time.time()

        names = [
            "R1_actual_q",
            "R1_actual_qd",
            "R1_actual_TCP_pose",
            "R1_actual_TCP_speed",
        ]
        topic = rtde.setTopic(False, names, int(self.state_frequency), 0)
        if topic < 0:
            raise RuntimeError(f"RTDE setTopic failed on {self.robot_ip}: {topic}")
        rtde.subscribe(topic, callback)
        return rtde, topic, lock, latest

    def _servo_joint(self, motion_control, joints) -> int:
        # Queue pressure is handled on the next cycle; never busy-retry here.
        return int(motion_control.servoJoint(
            list(map(float, joints)), self.servo_a, self.servo_v,
            self.servo_t, self.lookahead_time, self.gain))

    def run(self) -> None:
        rpc = None
        rtde = None
        rtde_topic = None
        motion_control = None
        servo_enabled = False
        try:
            sdk = require_pyaubo_sdk()
            if self.soft_real_time:
                os.sched_setscheduler(0, os.SCHED_RR, os.sched_param(20))

            rpc = sdk.RpcClient()
            rpc.setRequestTimeout(1000)
            rpc.connect(self.robot_ip, self.robot_port)
            if not rpc.hasConnected():
                raise RuntimeError(
                    f"cannot connect to AUBO RPC {self.robot_ip}:{self.robot_port}"
                )
            rpc.login(self.username, self.password)
            if not rpc.hasLogined():
                raise RuntimeError(f"cannot login to AUBO RPC {self.robot_ip}")
            names = list(rpc.getRobotNames())
            if not names:
                raise RuntimeError(f"no robot reported by {self.robot_ip}")
            robot = rpc.getRobotInterface(names[0])
            motion_control = robot.getMotionControl()
            robot_algorithm = robot.getRobotAlgorithm()

            self._power_on(robot)
            # Required after stopMove, but never bypass Running/Normal checks.
            ret = motion_control.startMove()
            if ret not in (None, 0):
                raise RuntimeError(f"startMove rejected: {ret}")
            motion_control.setSpeedFraction(self.speed_fraction)
            gripper = AuboDigitalGripper(
                robot.getIoControl() if self.gripper_config["type"] != "disabled" else None,
                self.gripper_config)
            gripper.start()
            gripper_events = deque()

            rtde, rtde_topic, rtde_lock, rtde_latest = self._start_rtde()
            deadline = time.monotonic() + 3.0
            while True:
                rtde_state = self._copy_rtde_state(rtde_lock, rtde_latest)
                if rtde_state is not None:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"no valid RTDE state received from {self.robot_ip}"
                    )
                time.sleep(0.02)

            current_pose = rtde_state["eef_pose"].copy()
            q_seed = rtde_state["joint_angles"].copy()
            self._start_runtime(rpc.getRuntimeMachine(), sdk)
            # _warmup_servo enters mode 1.  Mark it here so cleanup also exits
            # servo mode when the startup hold itself is rejected.
            servo_enabled = True
            self._warmup_servo(motion_control, q_seed, sdk)
            dt = 1.0 / self.frequency
            now = time.monotonic()
            last_waypoint_time = now
            interpolator = PoseTrajectoryInterpolator(
                times=[now], poses=[current_pose]
            )
            target_pose = current_pose.copy()

            # Publish a real sample before reporting ready.  Otherwise a
            # caller can read the ring buffer's zero-valued schema example.
            self.ring_buffer.put(
                {
                    "ActualTCPPose": rtde_state["eef_pose"],
                    "ActualTCPSpeed": rtde_state["eef_speed"],
                    "ActualQ": rtde_state["joint_angles"],
                    "ActualQd": rtde_state["joint_speeds"],
                    "TargetTCPPose": target_pose,
                    "robot_receive_timestamp": rtde_state["received"],
                    "GripperState": gripper.state,
                    "GripperEnabled": gripper.enabled,
                }
            )
            self.startup_ok.value = True
            self.ready_event.set()

            home_plan = None
            joint_hold = False
            keep_running = True
            consecutive_ignored = 0
            consecutive_ik_failed = 0
            report_started = time.monotonic()
            report_cycles = 0
            last_command_at = time.monotonic()
            queue_full = overruns = 0
            ik_max = servo_max = 0.
            while keep_running:
                if self.stop_event.is_set():
                    break
                loop_start = time.perf_counter()
                now = time.monotonic()
                try:
                    commands = self.input_queue.get_all()
                    command_count = len(commands["cmd"])
                except Empty:
                    command_count = 0
                if command_count:
                    last_command_at = now
                    if np.any(commands["cmd"] == Command.STOP.value):
                        break
                    if np.any(commands["cmd"] == Command.HOLD.value):
                        if home_plan is not None and self.home_state.value == 1:
                            raise RuntimeError("Homing aborted; stopping servo")
                        held = self._copy_rtde_state(rtde_lock, rtde_latest)
                        if held is None or time.time() - held["received"] > .6:
                            raise RuntimeError("Stale state at hold")
                        # Keep the last accepted joint command to avoid a backwards
                        # jump into delayed feedback; discard all pending targets.
                        target_pose = interpolator(now)
                        interpolator = PoseTrajectoryInterpolator([now], [target_pose])
                        gripper_events.clear()
                        home_plan = None
                        joint_hold = True
                        self.home_state.value = 0
                        command_count = 0
                if self.command_timeout is not None and now - last_command_at > self.command_timeout:
                    raise RuntimeError("teleop command watchdog expired")
                for index in range(command_count):
                    # Coalesce consecutive streaming targets; retain scheduled waypoints.
                    if (index + 1 < command_count and
                        commands["cmd"][index] == Command.SERVOL.value and
                        commands["cmd"][index + 1] == Command.SERVOL.value):
                        continue
                    command = {key: value[index] for key, value in commands.items()}
                    command_type = int(command["cmd"])
                    if command_type == Command.STOP.value:
                        keep_running = False
                        break
                    if command_type == Command.HOME.value:
                        joint_hold = False
                        home_plan = JointHome(q_seed, command["target_pose"])
                        gripper_events.clear()
                        self.home_state.value = 1
                        continue
                    if command_type == Command.HEARTBEAT.value:
                        continue
                    joint_hold = False
                    if home_plan is not None:
                        if self.home_state.value == 1:
                            continue
                        state_now = self._copy_rtde_state(rtde_lock, rtde_latest)
                        if state_now is None or time.time()-state_now["received"] > .6:
                            raise RuntimeError("Stale state at home release")
                        interpolator = PoseTrajectoryInterpolator([now], [state_now["eef_pose"]])
                        home_plan = None
                        self.home_state.value = 0
                    if command_type == Command.SERVOL.value:
                        insert_time = now + dt + float(command["duration"])
                        interpolator = interpolator.drive_to_waypoint(
                            pose=command["target_pose"],
                            time=insert_time,
                            curr_time=now + dt,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = insert_time
                    elif command_type == Command.SCHEDULE_WAYPOINT.value:
                        monotonic_target = (
                            time.monotonic() - time.time() + float(command["target_time"])
                        )
                        interpolator = interpolator.schedule_waypoint(
                            pose=command["target_pose"],
                            time=monotonic_target,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=now + dt,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = monotonic_target
                        if float(command.get("gripper", -1.)) >= 0:
                            gripper_events.append((monotonic_target, float(command["gripper"])))
                    else:
                        raise RuntimeError(f"unknown controller command {command_type}")

                state = self._copy_rtde_state(rtde_lock, rtde_latest)
                if state is None or time.time() - state["received"] > 0.6:
                    raise RuntimeError("RTDE state is stale; stopping servo")
                if (self.max_following_error is not None and
                        np.max(np.abs(state["joint_angles"] - q_seed)) > self.max_following_error):
                    raise RuntimeError("Joint following error exceeded")
                if home_plan is not None:
                    target_pose = state["eef_pose"].copy()
                    if self.home_state.value == 1 and home_plan.check(state["joint_angles"], state["joint_speeds"], state["received"], q_seed):
                        self.home_state.value = 2
                    joints = home_plan.sample(dt)
                elif joint_hold:
                    joints = q_seed.copy()
                else:
                    target_pose = interpolator(now)
                    rpc_started = time.perf_counter()
                    ik_result = robot_algorithm.inverseKinematics(
                        list(map(float, q_seed)), rotvec_to_aubo(target_pose)
                    )
                    ik_max = max(ik_max, time.perf_counter() - rpc_started)
                    joints, ik_ret = np.asarray(ik_result[0]), int(ik_result[1])
                    if ik_ret != 0:
                        consecutive_ik_failed += 1
                        if consecutive_ik_failed > 15:  # ~0.3s of continuous IK failure
                            desc = sdk.returnValue2Str(ik_ret)
                            raise RuntimeError(
                                f"inverseKinematics failed on {self.robot_ip}: {ik_ret} ({desc})"
                            )
                        # Hold last valid joint configuration during transient singularities/boundaries
                        joints = q_seed
                    else:
                        consecutive_ik_failed = 0
                    limit = self.max_joint_step
                    if self.max_joint_speed is not None:
                        limit = min(limit, self.max_joint_speed * dt)
                    if self.reject_joint_jumps and np.max(np.abs(joints - q_seed)) > limit:
                        raise RuntimeError("IK joint jump exceeded configured limit")
                    joints = self._limit_joint_step(q_seed, joints, limit)
                while gripper_events and gripper_events[0][0] <= now:
                    gripper.command(gripper_events.popleft()[1])
                rpc_started = time.perf_counter()
                servo_ret = self._servo_joint(motion_control, joints)
                servo_max = max(servo_max, time.perf_counter() - rpc_started)
                if servo_ret == 0:
                    consecutive_ignored = 0
                    q_seed = joints
                    if home_plan is not None:
                        home_plan.accepted(dt)
                elif servo_ret == 2:
                    # Planning queue full: the robot controller buffer is saturated.
                    # This is normal transient flow-control; the robot is actively executing points.
                    consecutive_ignored = 0
                    queue_full += 1
                elif servo_ret == -13 and consecutive_ignored < 3:
                    consecutive_ignored += 1
                    # Re-select only; querying servo/safety state between
                    # streaming commands can itself interrupt AUBO servo mode.
                    select_ret = motion_control.setServoModeSelect(1)
                    if self.verbose:
                        print(
                            f"[AuboInterpolationController:{self.robot_ip}] "
                            f"servoJoint -13 ({consecutive_ignored}/3), "
                            f"reselect servo ret={select_ret}",
                            flush=True,
                        )
                    time.sleep(0.02)
                else:
                    description = sdk.returnValue2Str(servo_ret)
                    raise RuntimeError(
                        f"servoJoint failed on {self.robot_ip}: "
                        f"{servo_ret} ({description})"
                    )

                state = self._copy_rtde_state(rtde_lock, rtde_latest)
                if state is not None:
                    self.ring_buffer.put(
                        {
                            "ActualTCPPose": state["eef_pose"],
                            "ActualTCPSpeed": state["eef_speed"],
                            "ActualQ": state["joint_angles"],
                            "ActualQd": state["joint_speeds"],
                            "TargetTCPPose": target_pose,
                            "robot_receive_timestamp": state["received"],
                            "GripperState": gripper.state,
                            "GripperEnabled": gripper.enabled,
                        }
                    )

                sleep_time = dt - (time.perf_counter() - loop_start)
                if sleep_time <= 0:
                    overruns += 1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                report_cycles += 1
                report_now = time.monotonic()
                report_elapsed = report_now - report_started
                if self.verbose and report_elapsed >= 2.0:
                    print(
                        f"[AuboInterpolationController:{self.robot_ip}] "
                        f"servo_hz={report_cycles / report_elapsed:.1f} "
                        f"overruns={overruns} queue_full={queue_full} "
                        f"ik_max_ms={ik_max*1000:.1f} servo_max_ms={servo_max*1000:.1f}",
                        flush=True,
                    )
                    report_started = report_now
                    report_cycles = 0
                    queue_full = overruns = 0
                    ik_max = servo_max = 0.
        except Exception as exc:
            print(
                f"[AuboInterpolationController:{self.robot_ip}] ERROR: {exc}",
                flush=True,
            )
            raise
        finally:
            if not self.ready_event.is_set():
                self.ready_event.set()
            if motion_control is not None and servo_enabled:
                try:
                    self._set_servo_mode(motion_control, False)
                except Exception:
                    pass
            if rtde is not None:
                try:
                    if rtde_topic is not None:
                        rtde.removeTopic(False, rtde_topic)
                    rtde.logout()
                    rtde.disconnect()
                except Exception:
                    pass
            if rpc is not None:
                try:
                    if rpc.hasLogined():
                        rpc.logout()
                except Exception:
                    pass
                try:
                    rpc.disconnect()
                except Exception:
                    pass
