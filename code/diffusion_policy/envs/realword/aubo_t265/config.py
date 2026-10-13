"""Load one explicit dual-arm configuration, independent of the working directory."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .calibration import load_calibration, rotation_matrix

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config/dual.json"


def load_config(path=DEFAULT_CONFIG, validate_calibrations=True):
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    config = copy.deepcopy(data)
    arms = config.get("arms", [])
    if len(arms) != 2 or {arm.get("side") for arm in arms} != {"left", "right"}:
        raise ValueError("Exactly one left and one right arm are required")
    for field in ("ip", "serial", "index", "calibration"):
        values = [arm.get(field) for arm in arms]
        if any(value is None or value == "" for value in values) or len(set(values)) != 2:
            raise ValueError(f"Each arm needs a distinct {field}")
    if {arm["index"] for arm in arms} != {0, 1}:
        raise ValueError("Sender arm indices must be 0 and 1")
    for arm in arms:
        from diffusion_policy.envs.realword.aubo_gripper import AuboDigitalGripper
        arm["calibration"] = str((path.parent / arm["calibration"]).resolve())
        if validate_calibrations:
            load_calibration(arm["calibration"], arm["ip"], arm["serial"])
        for field in ("position_axis_signs", "rotation_axis_signs"):
            signs = arm.get(field, [1, 1, 1])
            if len(signs) != 3 or not np.isin(signs, [-1, 1]).all():
                raise ValueError(f"Invalid {field}")
            arm[field] = signs
        arm.setdefault("port", 30004)
        arm.setdefault("no_range_limit", False)
        if not isinstance(arm["no_range_limit"], bool):
            raise ValueError("no_range_limit must be true or false")
        if not isinstance(arm["port"], int) or not 0 < arm["port"] < 65536:
            raise ValueError("Invalid AUBO RPC port")
        arm.setdefault("gripper", {"type": "disabled"})
        AuboDigitalGripper(None, arm["gripper"])
        world = arm.get("world_from_base")
        if world is not None:
            matrix = np.asarray(world, dtype=float)
            if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
                raise ValueError("world_from_base must be a finite homogeneous transform")
            rotation_matrix(matrix[:3, :3])
    config["home_file"] = str((path.parent / config["home_file"]).resolve())
    motion = config["motion"]
    for field in ("teleop_frequency", "robot_frequency", "state_frequency", "record_frequency",
                  "position_scale", "max_translation", "max_rotation_deg", "max_speed",
                  "max_angular_speed_deg", "command_latency", "pose_timeout", "command_timeout"):
        if not np.isfinite(motion[field]) or motion[field] <= 0:
            raise ValueError(f"{field} must be finite and positive")
    if not 10 <= motion["teleop_frequency"] <= 100 or motion["robot_frequency"] > 200:
        raise ValueError("Invalid control frequencies")
    if motion["state_frequency"] > motion["robot_frequency"] or motion["record_frequency"] > motion["state_frequency"]:
        raise ValueError("Require record_frequency <= state_frequency <= robot_frequency")
    if not 0 <= motion.get("rotation_scale", 1.) <= 2:
        raise ValueError("rotation_scale must be 0..2")
    if not 0 < motion["max_rotation_deg"] < 180:
        raise ValueError("Rotation envelope must be below 180 degrees")
    if any(arm["no_range_limit"] for arm in arms) and motion.get("rotation_scale", 1.) not in (0., 1.):
        raise ValueError("Unbounded rotation requires rotation_scale 0 or 1")
    for field, default in (("smooth_alpha", .35), ("rotation_smooth_alpha", .7)):
        motion.setdefault(field, default)
        if not np.isfinite(motion[field]) or not 0 < motion[field] <= 1:
            raise ValueError(f"{field} must be in (0, 1]")
    motion.setdefault("deadzone", .0002)
    if not np.isfinite(motion["deadzone"]) or motion["deadzone"] < 0:
        raise ValueError("deadzone must be finite and nonnegative")
    controller = config.setdefault("controller", {})
    defaults = dict(max_pos_speed=motion["max_speed"], max_rot_speed_deg=motion["max_angular_speed_deg"],
                    servo_t=2. / motion["robot_frequency"], servo_a=.2, servo_v=.3,
                    lookahead_time=.1, gain=200., speed_fraction=.2, launch_timeout=20., max_joint_step=.01,
                    max_joint_speed=.5, max_following_error=.2, reject_joint_jumps=True)
    for field, default in defaults.items():
        controller.setdefault(field, default)
        value = controller[field]
        if field == "reject_joint_jumps":
            if not isinstance(value, bool):
                raise ValueError("reject_joint_jumps must be true or false")
        elif field in ("max_joint_speed", "max_following_error") and value is None:
            continue  # The original site controller has neither additional check.
        elif not np.isfinite(value) or value <= 0:
            raise ValueError(f"controller.{field} must be finite and positive")
    if controller["servo_t"] < .005 or controller["speed_fraction"] > 1:
        raise ValueError("Require servo_t >= 5 ms and speed_fraction <= 1")
    config.setdefault("workspace", {"enabled": False})
    encoders = [arm.get("encoder_port") for arm in arms if arm.get("encoder_port")]
    if len(encoders) != len(set(encoders)):
        raise ValueError("Each encoder needs a distinct serial port")
    if len({arm["gripper"]["type"] != "disabled" for arm in arms}) != 1:
        raise ValueError("Enable both end effectors or disable both")
    sender = config.get("sender", {})
    offset = np.asarray(sender.get("tcp_offset_m", [0, 0, 0]), dtype=float)
    quat = np.asarray(sender.get("tcp_offset_quat_xyzw", [0, 0, 0, 1]), dtype=float)
    if offset.shape != (3,) or quat.shape != (4,) or not np.isfinite(offset).all() or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1.):
        raise ValueError("Invalid sender TCP offset")
    cameras = config.setdefault("cameras", {"serial_numbers": []})
    serials = cameras.get("serial_numbers", [])
    if len(set(serials)) != len(serials) or set(serials) & {arm["serial"] for arm in arms}:
        raise ValueError("RGB camera serials must be unique and distinct from T265 devices")
    config["config_path"] = str(path)
    return config


def controller_kwargs(config):
    """Translate explicit controller configuration to environment/SDK units."""
    result = dict(config["controller"])
    result["max_rot_speed"] = float(np.radians(result.pop("max_rot_speed_deg")))
    return result


def recording_metadata(config):
    """Embed the actual calibration content, not only mutable file paths."""
    metadata = copy.deepcopy(config)
    metadata["pose_convention"] = "xyz_m+rotvec_rad_in_each_robot_base"
    metadata["calibrations"] = {
        arm["side"]: load_calibration(arm["calibration"], arm["ip"], arm["serial"])
        for arm in config["arms"]}
    return metadata
