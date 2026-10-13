"""Modality- and environment-independent diffusion-policy base behavior."""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.common.base_policy import BasePolicy


class BaseDiffusionPolicy(BasePolicy, ABC):
    """Share normalizing and observation encoding across diffusion policies."""

    def _initialize_normalizer(self) -> None:
        """Register the normalizer at the child policy's original init position."""
        self.normalizer = LinearNormalizer()

    def set_normalizer(self, normalizer: LinearNormalizer) -> None:
        self.normalizer.load_state_dict(normalizer.state_dict())

    def normalize_obs(self, obs):
        return self.normalizer.normalize(obs)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer["action"].normalize(action)

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer["action"].unnormalize(action)

    def _encode_normalized_obs(
            self,
            nobs: Dict[str, torch.Tensor],
            n_steps: Optional[int] = None) -> torch.Tensor:
        """Flatten observation time and invoke the encoder without extra copies."""
        if n_steps is None:
            flat_obs = dict_apply(
                nobs,
                lambda x: x.reshape(-1, *x.shape[2:]),
            )
        else:
            flat_obs = dict_apply(
                nobs,
                lambda x: x[:, :n_steps, ...].reshape(-1, *x.shape[2:]),
            )
        return self.obs_encoder(flat_obs)

    def freeze_obs_encoder(self) -> None:
        self.obs_encoder.eval()
        self.obs_encoder.requires_grad_(False)

    @abstractmethod
    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def compute_loss(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError()
