"""FasterWAM model-side adapter for the existing dp_pro6000 real-robot loop."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FASTERWAM_ROOT = Path("/home/hy/FasterWAM")
DEFAULT_CHECKPOINT = (
    DEFAULT_FASTERWAM_ROOT
    / "checkpoints/fold100_action/step_010000_merged.pt"
)
DEFAULT_DATASET_STATS = (
    DEFAULT_FASTERWAM_ROOT / "checkpoints/fold100_action/dataset_stats.json"
)
DEFAULT_TEXT_EMBEDDING = (
    DEFAULT_FASTERWAM_ROOT
    / "checkpoints/fold100_action/text_embeddings/"
    "c5e39e32e704fe7a533eb4296adeffb7e910f86911713503866679c5afc646d6."
    "t5_len128.wan22ti2v5b.pt"
)


class FasterWAMPolicyAdapter:
    """Expose FasterWAM as ``observation -> native [H,14] actions``.

    The FasterWAM deployment policy performs state normalization and action
    denormalization internally.  Arm poses therefore pass through unchanged;
    only the two gripper widths are converted to dp_pro6000's binary Franka
    convention.
    """

    def __init__(
        self,
        *,
        fasterwam_root: str | Path = DEFAULT_FASTERWAM_ROOT,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        dataset_stats: str | Path = DEFAULT_DATASET_STATS,
        text_embedding: str | Path = DEFAULT_TEXT_EMBEDDING,
        instruction: str = "fold the T-shirt",
        device: str = "cuda:0",
        mixed_precision: str = "bf16",
        num_inference_steps: int = 10,
        action_horizon: int = 32,
        gripper_open_threshold: float = 0.04,
    ) -> None:
        self.fasterwam_root = Path(fasterwam_root).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_stats = Path(dataset_stats).expanduser().resolve()
        self.text_embedding = Path(text_embedding).expanduser().resolve()
        self.gripper_open_threshold = float(gripper_open_threshold)

        for required in (
            self.fasterwam_root,
            self.checkpoint,
            self.dataset_stats,
            self.text_embedding,
        ):
            if not required.exists():
                raise FileNotFoundError(required)
        if not 0.0 <= self.gripper_open_threshold <= 0.08:
            raise ValueError("gripper_open_threshold must be in [0.0, 0.08]")

        for path in (self.fasterwam_root, self.fasterwam_root / "src"):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)

        # DiffSynth resolves its base Wan/VAE weights through this variable.
        # Pin it here as well as in the shell launcher so the adapter remains
        # fully local/offline when imported from another entry point.
        os.environ.setdefault(
            "DIFFSYNTH_MODEL_BASE_PATH", str(self.fasterwam_root / "checkpoints")
        )
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")

        from experiments.robotwin.fasterwam_policy.deploy_policy_cached_text import (
            get_model,
        )

        self.policy = get_model(
            {
                "sim_cfg_name": "sim_robotwin.yaml",
                "ckpt_setting": str(self.checkpoint),
                "dataset_stats_path": str(self.dataset_stats),
                "text_embedding_path": str(self.text_embedding),
                "fixed_instruction": str(instruction),
                "create_text_embedding_if_missing": False,
                "device": str(device),
                "mixed_precision": str(mixed_precision),
                "num_inference_steps": int(num_inference_steps),
                "action_infer_mode": "one_pass_future_cache",
                "action_horizon": int(action_horizon),
            }
        )
        print(
            "FASTERWAM_MODEL_READY "
            f"checkpoint={self.checkpoint} "
            f"text_embedding={self.policy.text_embedding_path}",
            flush=True,
        )

    def predict_action(self, observation: dict[str, Any]) -> np.ndarray:
        actions = np.asarray(
            self.policy.predict_dp_observation(observation), dtype=np.float32
        )
        if actions.ndim != 2 or actions.shape[1] != 14 or actions.shape[0] == 0:
            raise ValueError(f"FasterWAM actions must be [H,14], got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("FasterWAM produced NaN or Inf actions")

        actions = actions.copy()
        for index in (6, 13):
            actions[:, index] = np.where(
                actions[:, index] > self.gripper_open_threshold, 0.08, 0.0
            )
        return actions

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
