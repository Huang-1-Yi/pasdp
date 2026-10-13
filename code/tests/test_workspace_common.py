import random
import unittest

import numpy as np
import torch
from omegaconf import OmegaConf

from diffusion_policy.workspace.common import BaseTrainingWorkspace, run_rollout
from diffusion_policy.workspace.train_padp_unrolled_workspace import (
    TrainPADPUnrolledWorkspace,
)
from diffusion_policy.workspace.train_padp_workspace import (
    TrainPADPWorkspace,
)


class _PolicyWithOptimizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.optimizer = object()
        self.optimizer_kwargs = None

    def get_optimizer(self, **kwargs):
        self.optimizer_kwargs = kwargs
        return ("policy_optimizer", kwargs)


class WorkspaceCommonTest(unittest.TestCase):
    def _workspace_with_model(self):
        workspace = BaseTrainingWorkspace.__new__(BaseTrainingWorkspace)
        workspace.model = _PolicyWithOptimizer()
        return workspace

    def test_seed_everything_is_repeatable(self):
        BaseTrainingWorkspace._seed_everything(17)
        first = (
            random.random(),
            np.random.rand(),
            torch.rand(()).item(),
        )
        BaseTrainingWorkspace._seed_everything(17)
        second = (
            random.random(),
            np.random.rand(),
            torch.rand(()).item(),
        )
        self.assertEqual(first, second)

    def test_run_rollout_is_an_exact_single_call_forwarder(self):
        calls = []
        policy = object()
        result = {"test/mean_score": 0.75}

        class _Runner:
            def run(self, received_policy):
                calls.append(received_policy)
                return result

        returned = run_rollout(_Runner(), policy)
        self.assertIs(returned, result)
        self.assertEqual(calls, [policy])

    def test_optimizer_sources_preserve_existing_contracts(self):
        workspace = self._workspace_with_model()

        model_cfg = OmegaConf.create({
            "training": {"optimizer_source": "model"},
            "optimizer": {},
        })
        self.assertIs(
            workspace._create_optimizer(model_cfg),
            workspace.model.optimizer,
        )

        method_cfg = OmegaConf.create({
            "training": {"optimizer_source": "policy_method"},
            "optimizer": {"_target_": "unused", "lr": 0.25},
        })
        result = workspace._create_optimizer(method_cfg)
        self.assertEqual(result, ("policy_optimizer", {"lr": 0.25}))
        self.assertEqual(workspace.model.optimizer_kwargs, {"lr": 0.25})

        hydra_cfg = OmegaConf.create({
            "training": {"optimizer_source": "hydra"},
            "optimizer": {
                "_target_": "torch.optim.SGD",
                "lr": 0.125,
            },
        })
        optimizer = workspace._create_optimizer(hydra_cfg)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.125)

    def test_unrolled_reuses_padp_loop_and_only_overrides_hooks(self):
        self.assertTrue(
            issubclass(
                TrainPADPUnrolledWorkspace,
                TrainPADPWorkspace,
            )
        )
        self.assertNotIn(
            "run", TrainPADPUnrolledWorkspace.__dict__
        )
        self.assertNotIn(
            "run_eval_2",
            TrainPADPUnrolledWorkspace.__dict__,
        )

        standard = TrainPADPWorkspace.__new__(TrainPADPWorkspace)
        unrolled = TrainPADPUnrolledWorkspace.__new__(
            TrainPADPUnrolledWorkspace
        )
        disabled_cfg = OmegaConf.create({
            "training": {"enable_rollout": False, "num_epochs": 10}
        })
        self.assertTrue(standard._rollout_enabled(disabled_cfg))
        self.assertFalse(unrolled._rollout_enabled(disabled_cfg))

        calls = []

        class _CurriculumModel:
            def _set_self_rollout_prob(self, epoch, num_epochs):
                calls.append((epoch, num_epochs))

        unrolled.model = _CurriculumModel()
        unrolled.epoch = 3
        unrolled._on_epoch_start(disabled_cfg)
        self.assertEqual(calls, [(3, 10)])

    def test_padp_resolves_manifest_batch_plan_and_debug_cap(self):
        class _Dataset:
            def get_epoch_batch_counts(self, num_epochs):
                self.requested_epochs = num_epochs
                return np.asarray([7, 9, 8], dtype=np.int64)

        dataset = _Dataset()
        cfg = OmegaConf.create({
            'training': {
                'num_epochs': 3,
                'max_train_steps': None,
                'gradient_accumulate_every': 1,
            }
        })
        full, effective = TrainPADPWorkspace._resolve_epoch_batch_plan(
            dataset, object(), cfg)
        self.assertEqual(dataset.requested_epochs, 3)
        self.assertEqual(full, [7, 9, 8])
        self.assertEqual(effective, [7, 9, 8])

        cfg.training.max_train_steps = 8
        full, effective = TrainPADPWorkspace._resolve_epoch_batch_plan(
            dataset, object(), cfg)
        self.assertEqual(full, [7, 9, 8])
        self.assertEqual(effective, [7, 8, 8])


if __name__ == "__main__":
    unittest.main()
