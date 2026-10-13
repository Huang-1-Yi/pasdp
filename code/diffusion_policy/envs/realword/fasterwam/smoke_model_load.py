#!/usr/bin/env python3
"""Load FasterWAM and run one offline forward without cameras or robots."""

from pathlib import Path

import cv2
import numpy as np

from .policy_adapter import FasterWAMPolicyAdapter


def _latest_preview(name: str) -> Path:
    root = Path("/home/hy/FasterWAM/real_runs/fold100")
    candidates = sorted(root.glob(f"*/camera_preflight/{name}_*.jpg"))
    if not candidates:
        raise FileNotFoundError(f"No saved preview found for {name} below {root}")
    return candidates[-1]


def _rgb_history(name: str) -> np.ndarray:
    path = _latest_preview(name)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Unable to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.stack((rgb, rgb), axis=0)


if __name__ == "__main__":
    adapter = FasterWAMPolicyAdapter()
    # Previously captured, read-only preflight data.  No robot or camera is
    # opened by this smoke test, and no command can be sent to the arms.
    left_pose = np.array(
        [0.41739, -0.03084, 0.78573, -2.92184, 1.14453, -0.00742],
        dtype=np.float32,
    )
    right_pose = np.array(
        [0.40905, -0.02388, 0.79903, -2.91573, 1.15689, -0.04913],
        dtype=np.float32,
    )
    observation = {
        "camera_0": _rgb_history("base"),
        "camera_1": _rgb_history("left_wrist"),
        "camera_2": _rgb_history("right_wrist"),
        "left_robot_eef_pose": np.stack((left_pose, left_pose)),
        "right_robot_eef_pose": np.stack((right_pose, right_pose)),
        "left_gripper_width": np.array([[0.08086], [0.08086]], dtype=np.float32),
        "right_gripper_width": np.array([[0.08083], [0.08083]], dtype=np.float32),
    }
    actions = adapter.predict_action(observation)
    print(
        "MODEL_LOAD_SMOKE_OK=true "
        f"text_embedding={adapter.policy.text_embedding_path} "
        f"actions_shape={actions.shape} actions_finite={np.isfinite(actions).all()} "
        f"left_gripper_values={np.unique(actions[:, 6]).tolist()} "
        f"right_gripper_values={np.unique(actions[:, 13]).tolist()}",
        flush=True,
    )
