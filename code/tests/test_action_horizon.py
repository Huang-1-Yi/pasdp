import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.common.action_horizon import (
    evaluate_action_horizon,
)


class _EvalDataset:
    def __init__(self):
        self.samples = [
            {
                "obs": {"x": torch.tensor([[2.0], [1.0]])},
                "action": torch.tensor([[1.0], [3.0]]),
            },
            {
                "obs": {"x": torch.tensor([[0.0], [4.0]])},
                "action": torch.tensor([[0.0], [2.0]]),
            },
        ]

    def get_validation_dataset(self):
        return self.samples

    def get_normalizer(self):
        return object()


class _IdentityActionPolicy(torch.nn.Module):
    horizon = 2

    def __init__(self):
        super().__init__()
        self.reset_count = 0
        self.normalizer = None

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def predict_action(self, obs):
        return {"action_pred": obs["x"]}

    def reset(self):
        self.reset_count += 1


class _Workspace:
    def __init__(self, target):
        self.cfg = OmegaConf.create({
            "_target_": target,
            "task": {"dataset": {}, "horizon": 2},
            "training": {"use_ema": False, "device": "cpu"},
            "val_dataloader": {
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
            },
        })
        self.model = _IdentityActionPolicy()
        self.ema_model = None


class ActionHorizonTest(unittest.TestCase):
    def _run(self, mode, target):
        workspace = _Workspace(target)
        dataset = _EvalDataset()
        with tempfile.TemporaryDirectory() as output_dir:
            with mock.patch(
                "diffusion_policy.workspace.common.action_horizon."
                "hydra.utils.instantiate",
                return_value=dataset,
            ):
                result = evaluate_action_horizon(
                    workspace,
                    checkpoint=None,
                    output_dir=output_dir,
                    device="cpu",
                    split="val",
                    mode=mode,
                )
            saved = np.load(Path(output_dir) / "per_pos_mse.npy")
        return workspace, result, saved

    def test_batched_path_keeps_per_position_mse(self):
        workspace, result, saved = self._run(
            "batched",
            "diffusion_policy.workspace.train_policy_workspace."
            "TrainPolicyWorkspace",
        )
        np.testing.assert_array_equal(saved, np.array([0.5, 4.0]))
        self.assertEqual(result["per_pos_mse"], [0.5, 4.0])
        self.assertEqual(result["count"], 2)
        self.assertEqual(workspace.model.reset_count, 0)

    def test_sequential_path_keeps_metrics_and_resets_each_unknown_sample(self):
        workspace, result, saved = self._run(
            "sequential",
            "diffusion_policy.workspace.train_padp_workspace."
            "TrainPADPWorkspace",
        )
        np.testing.assert_array_equal(saved, np.array([0.5, 4.0]))
        self.assertEqual(result["per_pos_mse"], [0.5, 4.0])
        self.assertEqual(result["per_pos_gt_abs"], [0.5, 2.5])
        self.assertEqual(result["per_pos_gt_power"], [0.5, 6.5])
        self.assertEqual(result["count"], 2)
        self.assertEqual(workspace.model.reset_count, 2)

    def test_auto_mode_uses_workspace_target_without_extra_policy_call(self):
        workspace, result, _ = self._run(
            "auto",
            "diffusion_policy.workspace.train_padp_workspace."
            "TrainPADPWorkspace",
        )
        self.assertIn("per_pos_nmse", result)
        self.assertEqual(workspace.model.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
