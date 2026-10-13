"""Fresh two-device samples from the framework's existing shared-memory sender."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class HandSample:
    position: np.ndarray
    orientation: Rotation
    sampled_at: float
    encoder_angle: float | None = None


def decode_samples(data, arms, timeout=.15, now=None, require_encoders=False):
    now = time.time() if now is None else now
    result = {}
    for arm in arms:
        slots = np.flatnonzero(data["arm_indices"] == arm["index"])
        if len(slots) != 1:
            raise RuntimeError(f"Missing input for {arm['side']}")
        index = int(slots[0])
        sampled_at = float(data["pose_sample_timestamp"][index])
        age = now - sampled_at
        if (not data["pose_valid"][index] or data["tracker_confidence"][index] < 2
                or not np.isfinite(sampled_at) or not -.05 <= age <= timeout):
            raise RuntimeError(f"{arm['side']} T265 input lost or stale (age={age:.3f}s)")
        position = np.asarray(data["position_m"][index], dtype=float)
        quat = np.asarray(data["quaternion_xyzw"][index], dtype=float)
        if position.shape != (3,) or quat.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < .5:
            raise RuntimeError("Invalid hand pose")
        encoder_angle = None
        if arm.get("encoder_port"):
            encoder_age = now - float(data["gripper_sample_timestamp"][index])
            angle = float(data["gripper_angle_deg"][index])
            if data["gripper_valid"][index] and -.05 <= encoder_age <= timeout and np.isfinite(angle):
                encoder_angle = angle
            elif require_encoders:
                raise RuntimeError(f"{arm['side']} encoder input lost or stale")
        result[arm["side"]] = HandSample(position.copy(), Rotation.from_quat(quat), sampled_at, encoder_angle)
    return result


class DualTeleopInput:
    def __init__(self, shm_manager, config):
        from diffusion_policy.envs.realword.udp_sender_dual_absolute_v3 import DualTeleopSenderProcess
        self.arms = config["arms"]
        self.timeout = config["motion"]["pose_timeout"]
        self.require_encoders = any(arm["gripper"]["type"] != "disabled" and arm.get("encoder_port") for arm in self.arms)
        devices = sorted(self.arms, key=lambda arm: arm["index"])
        settings = config.get("sender", {})
        self.sender = DualTeleopSenderProcess(
            shm_manager, delay_s=settings.get("delay_s", 3.),
            t265_serial0=devices[0]["serial"], t265_serial1=devices[1]["serial"],
            encoder_port0=devices[0].get("encoder_port"), encoder_port1=devices[1].get("encoder_port"),
            tcp_offset_enabled=settings.get("tcp_offset_enabled", False),
            tcp_offset_m=settings.get("tcp_offset_m", [0., 0., 0.]),
            tcp_offset_quat=settings.get("tcp_offset_quat_xyzw", [0., 0., 0., 1.]))

    def start(self):
        self.sender.start(timeout=20.)
        deadline = time.monotonic() + 10.
        while True:
            try:
                self.get()
                return
            except RuntimeError:
                if time.monotonic() > deadline or not self.sender.is_alive():
                    raise
                time.sleep(.02)

    def get(self):
        if not self.sender.is_alive():
            raise RuntimeError("T265 sender exited")
        return decode_samples(self.sender.get_teleop(), self.arms, self.timeout,
                              require_encoders=self.require_encoders)

    def stop(self):
        if self.sender.pid is not None:
            self.sender.stop()

    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.stop()
            raise

    def __exit__(self, *_):
        self.stop()
