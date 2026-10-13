"""Isolated mechanism ablations for PADP.

The production PADP implementation stays in :mod:`diffusion_policy.policy.padp`.
This module only changes the two training mechanisms needed by the D ablation
matrix and provides an evaluation-only adapter for disabling buffer reuse.
"""

from typing import Dict

import torch

from diffusion_policy.policy.padp import PADPPolicy


class PADPAblationPolicy(PADPPolicy):
    """PADP with independently switchable schedule and loss-weight ablations."""

    VALID_SCHEDULES = {"positionwise", "fixed_permutation"}
    VALID_LOSS_WEIGHTS = {"exponential", "uniform_sum_preserving"}

    def __init__(
            self,
            *args,
            ablation_noise_schedule_mode: str = "positionwise",
            ablation_loss_weight_mode: str = "exponential",
            fixed_permutation_seed: int = 20260915,
            **kwargs):
        if ablation_noise_schedule_mode not in self.VALID_SCHEDULES:
            raise ValueError(
                "Unknown PADP ablation noise schedule: "
                f"{ablation_noise_schedule_mode!r}")
        if ablation_loss_weight_mode not in self.VALID_LOSS_WEIGHTS:
            raise ValueError(
                "Unknown PADP ablation loss weighting: "
                f"{ablation_loss_weight_mode!r}")

        # The parent builds the unchanged D0 tensors first. Ablations replace
        # only their isolated mechanism after construction.
        kwargs["noise_schedule_mode"] = "positionwise"
        kwargs["window_loss_weights"] = "exponential"
        super().__init__(*args, **kwargs)

        self.ablation_noise_schedule_mode = ablation_noise_schedule_mode
        self.ablation_loss_weight_mode = ablation_loss_weight_mode
        self.fixed_permutation_seed = int(fixed_permutation_seed)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.fixed_permutation_seed)
        permutation = torch.randperm(self.horizon, generator=generator)
        if self.horizon > 1 and torch.equal(
                permutation, torch.arange(self.horizon)):
            # Make the ablation non-trivial even for an unlucky tiny horizon.
            permutation = permutation.roll(1)
        self.register_buffer("ablation_position_permutation", permutation)
        self.register_buffer(
            "ablation_sqrt_alpha_bar_h",
            self.sqrt_alpha_bar_h[:, permutation, :].clone(),
        )
        self.register_buffer(
            "ablation_sqrt_one_minus_alpha_bar_h",
            self.sqrt_one_minus_alpha_bar_h[:, permutation, :].clone(),
        )

        baseline_weight_sum = self.window_weights_h.sum().detach().clone()
        self.register_buffer(
            "ablation_baseline_weight_sum", baseline_weight_sum)
        if self.ablation_loss_weight_mode == "uniform_sum_preserving":
            uniform_weight = baseline_weight_sum / float(self.horizon)
            self.window_weights_h.fill_(uniform_weight)

        if self.ablation_noise_schedule_mode == "fixed_permutation":
            # A newly appended far-future slot must use the coefficient assigned
            # to the final permuted position, rather than D0's hard-coded max.
            # Assign a new buffer tensor: the parent's tail buffer is a view into
            # its cosine table, so an in-place copy would corrupt the D1 table.
            self.sqrt_one_minus_alpha_bar_tail = (
                self.ablation_sqrt_one_minus_alpha_bar_h[:, -1:, :].clone())

        print(
            "PADP ablation: "
            f"schedule={self.ablation_noise_schedule_mode}, "
            f"loss_weights={self.ablation_loss_weight_mode}, "
            f"permutation_seed={self.fixed_permutation_seed}, "
            f"weight_sum={self.window_weights_h.sum().item():.8f}"
        )

    def _apply_position_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor) -> torch.Tensor:
        if self.ablation_noise_schedule_mode == "positionwise":
            return super()._apply_position_noise(original_samples, noise)
        return (
            self.ablation_sqrt_alpha_bar_h * original_samples
            + self.ablation_sqrt_one_minus_alpha_bar_h * noise
        )


class ColdStartEveryDecisionPolicy:
    """Evaluation-only D3 adapter that disables cross-decision buffer reuse."""

    def __init__(self, policy: PADPPolicy):
        if not isinstance(policy, PADPPolicy):
            raise TypeError(
                "D3 cold-start evaluation requires a PADPPolicy checkpoint, "
                f"got {type(policy)!r}")
        self.policy = policy
        self.cold_start_every_decision = True

    @property
    def device(self):
        return self.policy.device

    @property
    def dtype(self):
        return self.policy.dtype

    @property
    def horizon(self):
        return self.policy.horizon

    @property
    def n_obs_steps(self):
        return self.policy.n_obs_steps

    @property
    def n_action_steps(self):
        return self.policy.n_action_steps

    def reset(self):
        self.policy.reset()

    def predict_action(
            self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.policy.reset()
        return self.policy.predict_action(obs_dict)
