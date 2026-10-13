"""One clutch/home state machine shared by preview and robot teleoperation."""
from __future__ import annotations

import time
import numpy as np

from .calibration import load_calibration
from .pose_mapper import QuaternionPoseMapper


def make_mappers(config, poses):
    motion = config["motion"]
    period = 1. / motion["teleop_frequency"]
    def time_constant(alpha):
        return 0. if alpha == 1. else -period / np.log1p(-alpha)
    result = {}
    for arm in config["arms"]:
        calibration = load_calibration(arm["calibration"], arm["ip"], arm["serial"])
        result[arm["side"]] = QuaternionPoseMapper(
            calibration["rotation_base_t265"], poses[arm["side"]],
            operator_basis_t265=calibration["operator_basis_t265"],
            rotation_basis_t265=calibration.get("operator_rotation_basis_t265"),
            position_axis_signs=arm["position_axis_signs"],
            rotation_axis_signs=arm["rotation_axis_signs"],
            scale=motion["position_scale"], rotation_scale=motion.get("rotation_scale", 1.),
            radius=None if arm.get("no_range_limit", False) else motion["max_translation"],
            angle=None if arm.get("no_range_limit", False) else np.radians(motion["max_rotation_deg"]),
            speed=motion["max_speed"], angular_speed=np.radians(motion["max_angular_speed_deg"]),
            tau=time_constant(motion.get("smooth_alpha", .35)),
            rotation_tau=time_constant(motion.get("rotation_smooth_alpha", .7)),
            deadzone=motion.get("deadzone", .0002))
    return result


class TeleopSession:
    def __init__(self, config, poses, workspace, env=None):
        self.config, self.workspace, self.env = config, workspace, env
        self.poses = {side: np.asarray(pose, dtype=float).copy() for side, pose in poses.items()}
        self.mappers = make_mappers(config, poses)
        self.mode = "paused"
        self.grippers = {"left": np.array([1.]), "right": np.array([1.])}
        self.encoder_anchors = {}
        self._encoder_toggled = {}
        if env and env.grippers_enabled:
            state = env.get_state()
            self.grippers = {side: env.decode_gripper(side, state[side]["GripperState"])[0].copy() for side in self.grippers}

    def pause(self):
        if self.env:
            self.env.hold()
            if self.env._recording:
                self.env.end_episode()
        self.mode = "paused"

    def resume(self, samples):
        if self.mode == "homing":
            raise RuntimeError("Wait for joint home to finish before resuming")
        poses = self.poses
        if self.env:
            state = self.env.get_state()
            poses = {side: value['ActualTCPPose'] for side, value in state.items()}
            if self.env.grippers_enabled:
                # Pending output targets cleared by HOLD must not reappear on resume.
                self.grippers = {side: self.env.decode_gripper(side, value['GripperState'])[0].copy()
                                 for side, value in state.items()}
        if self.env and self.env.grippers_enabled and any(not np.isin(v, [0., 1.]).all() for v in self.grippers.values()):
            raise RuntimeError("End effector outputs are in an unknown combination; set a known state before resuming")
        for side, mapper in self.mappers.items():
            mapper.anchor(samples[side].position, samples[side].orientation, poses[side])
        self.encoder_anchors = {side: sample.encoder_angle for side, sample in samples.items()}
        self._encoder_toggled = {side: False for side in samples}
        self.mode = "teleop"

    def toggle_output(self, side, pin):
        if not self.env or not self.env.grippers_enabled or self.mode != 'teleop':
            raise RuntimeError('Enable I/O and resume teleoperation before toggling outputs')
        settings = self.env.gripper_configs[side]
        if settings.get('mode') != 'independent' or pin not in settings['pins']:
            raise RuntimeError('This key requires an explicitly configured independent cabinet output')
        index = settings['pins'].index(pin)
        self.grippers[side][index] = 1. - self.grippers[side][index]

    def home(self, targets):
        if self.env is None:
            raise RuntimeError("Joint home is unavailable in read-only preview")
        if self.env._recording:
            self.env.end_episode()
        self.env.return_home(targets["left"], targets["right"])
        self.mode = "homing"

    def tick(self, samples, dt, target_time=None):
        if self.env:
            self.env.get_state()  # Detect either failed arm before sending further targets.
        if self.mode == "homing":
            self.env.heartbeat()
            if self.env.home_state == "holding":
                self.poses = {side: state["ActualTCPPose"].copy() for side, state in self.env.get_state().items()}
                self.mappers = make_mappers(self.config, self.poses)
                self.mode = "paused"
            return self.poses
        if self.mode != "teleop":
            if self.env:
                self.env.heartbeat()
            return self.poses
        next_poses = {side: mapper.step(samples[side].position, samples[side].orientation, dt)
                      for side, mapper in self.mappers.items()}
        for side, pose in next_poses.items():
            self.workspace.validate(side, pose)
        if self.env and self.env.grippers_enabled:
            threshold = self.config["motion"].get("gripper_toggle_deg", 30.)
            for side, sample in samples.items():
                if self.env.gripper_configs[side].get('mode') == 'independent':
                    continue
                anchor = self.encoder_anchors.get(side)
                if sample.encoder_angle is None or anchor is None:
                    continue
                delta = abs((sample.encoder_angle - anchor + 180.) % 360. - 180.)
                if delta > threshold and not self._encoder_toggled[side]:
                    self.grippers[side][0] = 1. - self.grippers[side][0]
                    self._encoder_toggled[side] = True
                elif delta < threshold * .5:
                    self._encoder_toggled[side] = False
            action = np.concatenate((next_poses["left"], self.grippers["left"],
                                     next_poses["right"], self.grippers["right"]))
        else:
            action = np.concatenate((next_poses["left"], next_poses["right"]))
        if self.env:
            self.env.exec_actions(action, time.time() + self.config["motion"]["command_latency"]
                                  if target_time is None else target_time)
        self.poses = next_poses
        return self.poses
