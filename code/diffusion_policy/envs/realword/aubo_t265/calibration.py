"""Hardware-independent frame math and read-only AUBO installation checks."""
from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


def rotation_matrix(value):
    value = np.asarray(value, dtype=float)
    if (value.shape != (3, 3) or not np.isfinite(value).all()
            or not np.allclose(value.T @ value, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(value), 1, atol=1e-5)):
        raise ValueError("Expected a proper 3x3 rotation (det=+1)")
    return value


def bounded(vector, radius):
    return vector * min(1., radius / max(float(np.linalg.norm(vector)), 1e-12))


def unit(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-9:
        raise ValueError("Expected a finite nonzero direction")
    return value / np.linalg.norm(value)


def valid_gravity(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all() or not 8 <= np.linalg.norm(value) <= 11:
        raise ValueError("Invalid installation gravity")
    return value


def base_basis(gravity, forward):
    up = -unit(valid_gravity(gravity))
    forward = np.asarray(forward, dtype=float)
    if forward.shape != (3,) or not np.isfinite(forward).all():
        raise ValueError("Invalid forward direction")
    forward = unit(forward - up * np.dot(up, forward))
    return rotation_matrix(np.column_stack((forward, np.cross(up, forward), up)))


def t265_operator_basis(forward, left):
    up = np.array([0., 1., 0.])
    forward = unit(np.asarray(forward) - up * np.dot(forward, up))
    left = unit(np.asarray(left) - up * np.dot(left, up))
    separation = np.degrees(np.arccos(np.clip(forward @ left, -1., 1.)))
    if not 65 <= separation <= 115 or np.cross(forward, left) @ up < .9:
        raise ValueError("Forward/left gestures must form a right-handed horizontal pair")
    forward = unit(forward + unit(np.cross(left, up)))
    return rotation_matrix(np.column_stack((forward, np.cross(up, forward), up)))


def load_calibration(path, ip=None, serial=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if ip is not None and data.get("robot_ip") != ip:
        raise ValueError("Calibration robot IP mismatch")
    if serial is not None and data.get("t265_serial") != serial:
        raise ValueError("Calibration T265 serial mismatch")
    base = rotation_matrix(data["operator_basis_base"])
    sensor = rotation_matrix(data["operator_basis_t265"])
    mapping = rotation_matrix(data["rotation_base_t265"])
    if not np.allclose(mapping, base @ sensor.T, atol=1e-5):
        raise ValueError("Inconsistent calibration matrices")
    gravity = valid_gravity(data["gravity"])
    validate_installation(data, gravity)
    if "operator_rotation_basis_t265" in data:
        rotation_matrix(data["operator_rotation_basis_t265"])
    return data


def validate_installation(calibration, gravity, tolerance_deg=5.):
    up = -unit(valid_gravity(gravity))
    saved_up = rotation_matrix(calibration["operator_basis_base"])[:, 2]
    if saved_up @ up < np.cos(np.radians(tolerance_deg)):
        raise RuntimeError("Robot installation changed; recalibrate its base frame")


def read_installation(ip, port=30004, username="aubo", password="123456"):
    # No power-on, runtime, servo, or motion APIs are called here.
    from diffusion_policy.envs.realword.aubo_interpolation_controller import require_pyaubo_sdk
    rpc = require_pyaubo_sdk().RpcClient()
    rpc.setRequestTimeout(1000)
    try:
        rpc.connect(ip, int(port))
        if not rpc.hasConnected():
            raise RuntimeError(f"Cannot connect to AUBO RPC {ip}:{port}")
        rpc.login(username, password)
        if not rpc.hasLogined():
            raise RuntimeError("AUBO login failed")
        names = list(rpc.getRobotNames())
        if len(names) != 1:
            raise RuntimeError("Expected one robot per AUBO cabinet")
        config = rpc.getRobotInterface(names[0]).getRobotConfig()
        return valid_gravity(config.getGravity()), np.asarray(config.getMountingPose(), dtype=float)
    finally:
        try:
            if rpc.hasLogined():
                rpc.logout()
        finally:
            rpc.disconnect()


def _capture_translation(pipeline, label, duration=8.):
    time.sleep(1.)
    input(f"准备向{label}平移 T265（保持姿态），按回车开始：")
    anchor = last = None
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        frame = pipeline.wait_for_frames(1000).get_pose_frame()
        if not frame or frame.get_pose_data().tracker_confidence < 2:
            raise RuntimeError("Tracking lost during calibration")
        data = frame.get_pose_data()
        last = np.array([data.translation.x, data.translation.y, data.translation.z])
        if anchor is None:
            anchor = last.copy()
            print(f"向{label}移动 10–20 cm，然后保持。", flush=True)
    delta = last - anchor
    horizontal = delta - np.array([0., delta[1], 0.])
    if not .08 <= np.linalg.norm(horizontal) <= .6 or abs(delta[1]) > .5 * np.linalg.norm(horizontal):
        raise ValueError("Calibration gesture must be horizontal and 8–60 cm")
    return delta


def calibrate_arm(arm, forward_base=None):
    import pyrealsense2 as rs
    gravity, _ = read_installation(arm["ip"], arm.get("port", 30004))
    path = Path(arm["calibration"])
    saved = load_calibration(path, arm["ip"], arm["serial"]) if path.exists() else None
    if forward_base is not None:
        base = base_basis(gravity, forward_base)
    elif saved is not None:
        validate_installation(saved, gravity)
        base = rotation_matrix(saved["operator_basis_base"])
    else:
        raise ValueError("A new installation requires --forward-base X Y Z")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(arm["serial"])
    config.enable_stream(rs.stream.pose)
    pipeline.start(config)
    try:
        forward = _capture_translation(pipeline, "前")
        left = _capture_translation(pipeline, "左")
        sensor = t265_operator_basis(forward, left)
        data = dict(robot_ip=arm["ip"], t265_serial=arm["serial"],
                    operator_basis_base=base.tolist(), operator_basis_t265=sensor.tolist(),
                    rotation_base_t265=(base @ sensor.T).tolist(), gravity=gravity.tolist(),
                    forward_gesture=forward.tolist(), left_gesture=left.tolist())
        # Rotation gesture calibration refers to the previous tracking world.
        # A fresh translation calibration must not retain those stale axes.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return data
    finally:
        pipeline.stop()


def calibrate_live_arm(arm, get_sample, gravity):
    """Calibrate the sender's current tracking world without restarting T265."""
    path = Path(arm['calibration'])
    saved = load_calibration(path, arm['ip'], arm['serial'])
    validate_installation(saved, gravity)
    gestures = []
    for direction in ('forward', 'left'):
        time.sleep(1.)
        input(f"{arm['side']}: move T265 {direction} horizontally 10-30 cm after Enter, then hold for 8 seconds: ")
        anchor = get_sample().position.copy()
        deadline = time.monotonic() + 8.
        while time.monotonic() < deadline:
            last = get_sample().position.copy()
            time.sleep(.02)
        delta = last - anchor
        horizontal = np.linalg.norm(delta[[0, 2]])
        if not .08 <= horizontal <= .6 or abs(delta[1]) > .5 * horizontal:
            raise ValueError('Calibration gesture must be horizontal and 8-60 cm')
        gestures.append(delta)
    sensor = t265_operator_basis(*gestures)
    base = rotation_matrix(saved['operator_basis_base'])
    data = dict(robot_ip=arm['ip'], t265_serial=arm['serial'], gravity=np.asarray(gravity).tolist(),
                operator_basis_base=base.tolist(), operator_basis_t265=sensor.tolist(),
                rotation_base_t265=(base @ sensor.T).tolist(),
                forward_gesture=gestures[0].tolist(), left_gesture=gestures[1].tolist())
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    return data
