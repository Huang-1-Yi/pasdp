import unittest

import torch
import torch.nn as nn

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.focal_policy import FocalPolicy
from diffusion_policy.policy.common.focal_objectives import (
    dct_over_horizon,
    orthonormal_dct_matrix,
)
from diffusion_policy.policy.common.focal_time_sampler import sample_two_timesteps


class _StateEncoder(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.projection = nn.Linear(state_dim, 8)

    def forward(self, obs_dict):
        return self.projection(obs_dict["state"])

    def output_shape(self):
        return (8,)


def _make_normalizer(state_dim, action_dim):
    normalizer = LinearNormalizer()
    normalizer.fit({
        "state": torch.randn(64, state_dim),
        "action": torch.randn(64, action_dim),
    })
    return normalizer


class FocalObjectiveTest(unittest.TestCase):
    def test_dct_matrix_is_orthonormal_and_preserves_energy(self):
        matrix = orthonormal_dct_matrix(8)
        torch.testing.assert_close(matrix @ matrix.T, torch.eye(8))
        trajectory = torch.randn(3, 8, 4)
        transformed = dct_over_horizon(trajectory, matrix)
        torch.testing.assert_close(
            trajectory.square().sum(dim=(1, 2)),
            transformed.square().sum(dim=(1, 2)),
        )

    def test_upstream_sampler_contract(self):
        cfg = {
            "tr_sampler": "v0",
            "P_mean_t": 0.6,
            "P_std_t": 1.6,
            "P_mean_r": 4.0,
            "P_std_r": 1.6,
            "ratio": 0.15,
        }
        generator = torch.Generator().manual_seed(7)
        t, r = sample_two_timesteps(
            cfg, 256, torch.device("cpu"), eps=1e-2, generator=generator)
        self.assertEqual(tuple(t.shape), (256,))
        self.assertEqual(tuple(r.shape), (256,))
        self.assertTrue(torch.all((t >= 1e-2) & (t <= 1.0)))
        self.assertTrue(torch.all((r >= 0.0) & (r <= 1.0)))


class FocalPolicyTest(unittest.TestCase):
    def _make_policy(self):
        state_dim = 6
        action_dim = 4
        policy = FocalPolicy(
            shape_meta={
                "obs": {"state": {"shape": [state_dim]}},
                "action": {"shape": [action_dim]},
            },
            horizon=8,
            n_action_steps=3,
            n_obs_steps=2,
            obs_encoder=_StateEncoder(state_dim),
            diffusion_step_embed_dim=32,
            down_dims=(32, 64, 128),
            kernel_size=3,
            n_groups=8,
            sample_cfg={
                "tr_sampler": "v0",
                "P_mean_t": 0.6,
                "P_std_t": 1.6,
                "P_mean_r": 4.0,
                "P_std_r": 1.6,
                "ratio": 0.15,
            },
        )
        policy.set_normalizer(_make_normalizer(state_dim, action_dim))
        return policy

    def test_loss_backward_and_prediction_shapes(self):
        policy = self._make_policy()
        batch = {
            "obs": {"state": torch.randn(2, 2, 6)},
            "action": torch.randn(2, 8, 4),
        }
        loss, info = policy.compute_loss(batch)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(info),
            {"space_loss", "freq_loss", "weighted_freq_loss", "total_loss"},
        )
        loss.backward()
        self.assertIsNotNone(policy.obs_encoder.projection.weight.grad)
        self.assertTrue(any(p.grad is not None for p in policy.model.parameters()))

        policy.eval()
        with torch.no_grad():
            result = policy.predict_action(batch["obs"])
        self.assertEqual(tuple(result["action"].shape), (2, 3, 4))
        self.assertEqual(tuple(result["action_pred"].shape), (2, 8, 4))


if __name__ == "__main__":
    unittest.main()
