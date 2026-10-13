from typing import Dict

import torch

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer


class BasePolicy(ModuleAttrMixin):
    """Common runtime contract for RGB, point-cloud, and multimodal policies."""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    def freeze_obs_encoder(self):
        """Freeze a conventional ``obs_encoder`` when the policy owns one."""
        if not hasattr(self, "obs_encoder"):
            raise AttributeError(
                f"{type(self).__name__} has no generic obs_encoder to freeze"
            )
        self.obs_encoder.eval()
        self.obs_encoder.requires_grad_(False)

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
