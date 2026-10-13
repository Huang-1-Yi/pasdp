import unittest

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn

from diffusion_policy.policy.padp_ablation import (
    ColdStartEveryDecisionPolicy,
    PADPAblationPolicy,
)


class _DummyEncoder(nn.Module):
    def output_shape(self):
        return (1,)

    def forward(self, obs):
        value = next(iter(obs.values()))
        return torch.zeros(value.shape[0], 1, device=value.device)


def _make_policy(schedule="positionwise", weights="exponential"):
    return PADPAblationPolicy(
        shape_meta={
            "action": {"shape": [1]},
            "obs": {"state": {"shape": [1], "type": "low_dim"}},
        },
        noise_scheduler=DDIMScheduler(
            num_train_timesteps=6,
            beta_schedule="squaredcos_cap_v2",
        ),
        horizon=6,
        n_action_steps=1,
        n_obs_steps=1,
        obs_encoder=_DummyEncoder(),
        down_dims=(8, 16, 32),
        n_groups=8,
        ablation_noise_schedule_mode=schedule,
        ablation_loss_weight_mode=weights,
        fixed_permutation_seed=17,
    )


class PADPAblationTest(unittest.TestCase):
    def test_fixed_permutation_preserves_levels_but_changes_order(self):
        policy = _make_policy(schedule="fixed_permutation")
        original = policy.sqrt_one_minus_alpha_bar_h.flatten()
        ablated = policy.ablation_sqrt_one_minus_alpha_bar_h.flatten()
        self.assertTrue(torch.equal(original.sort().values, ablated.sort().values))
        self.assertFalse(torch.equal(original, ablated))

        samples = torch.zeros(2, 6, 1)
        noise = torch.ones_like(samples)
        output = policy._apply_position_noise(samples, noise)
        self.assertTrue(torch.equal(output[0, :, 0], ablated))

    def test_uniform_weights_preserve_exponential_weight_sum(self):
        baseline = _make_policy(weights="exponential")
        ablated = _make_policy(weights="uniform_sum_preserving")
        self.assertAlmostEqual(
            baseline.window_weights_h.sum().item(),
            ablated.window_weights_h.sum().item(),
            places=6,
        )
        expected = torch.full_like(
            ablated.window_weights_h,
            ablated.window_weights_h.mean(),
        )
        self.assertTrue(torch.equal(ablated.window_weights_h, expected))

    def test_cold_start_adapter_resets_before_every_decision(self):
        policy = _make_policy()
        calls = []

        def reset():
            calls.append("reset")

        def predict(obs):
            calls.append("predict")
            return {"action": torch.zeros(1, 1, 1)}

        policy.reset = reset
        policy.predict_action = predict
        adapter = ColdStartEveryDecisionPolicy(policy)
        adapter.predict_action({"state": torch.zeros(1, 1, 1)})
        adapter.predict_action({"state": torch.zeros(1, 1, 1)})
        self.assertEqual(calls, ["reset", "predict", "reset", "predict"])


if __name__ == "__main__":
    unittest.main()
