import unittest

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import nn

from diffusion_policy.policy.padp import PADPPolicy


class _DummyEncoder(nn.Module):
    def output_shape(self):
        return (1,)

    def forward(self, obs):
        value = next(iter(obs.values()))
        return torch.zeros(value.shape[0], 1, device=value.device)


class _DeterministicPADP(PADPPolicy):
    def normalize_obs(self, obs):
        return obs

    def _encode_normalized_obs(self, obs, n_steps=None):
        value = next(iter(obs.values()))
        return torch.zeros(
            value.shape[0] * n_steps, 1, device=value.device)

    def unnormalize_action(self, action):
        return action

    def _apply_position_noise(self, original_samples, noise):
        return original_samples


class _IndexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, sample, local_cond=None, global_cond=None):
        batch, horizon, action_dim = sample.shape
        indices = torch.arange(
            horizon, device=sample.device, dtype=sample.dtype)
        return indices.view(1, horizon, 1).expand(
            batch, horizon, action_dim)


class PADPChunkAlignmentTest(unittest.TestCase):
    def test_obs_offset_and_action_chunk_advance_together(self):
        policy = _DeterministicPADP(
            shape_meta={
                "action": {"shape": [1]},
                "obs": {"state": {"shape": [1], "type": "low_dim"}},
            },
            noise_scheduler=DDIMScheduler(
                num_train_timesteps=6,
                beta_schedule="squaredcos_cap_v2",
            ),
            horizon=6,
            n_action_steps=2,
            n_obs_steps=2,
            obs_encoder=_DummyEncoder(),
            down_dims=(8, 16, 32),
            n_groups=8,
        )
        policy.model = _IndexModel()
        policy._inference_buffer = torch.zeros(1, 6, 1)
        policy._inference_global_cond = torch.zeros(1, 2)

        result = policy.predict_action({"state": torch.zeros(1, 2, 1)})

        self.assertEqual(result["action"].flatten().tolist(), [1.0, 2.0])
        self.assertEqual(
            policy._inference_buffer[0, :4, 0].tolist(),
            [2.0, 3.0, 4.0, 5.0],
        )


if __name__ == "__main__":
    unittest.main()
