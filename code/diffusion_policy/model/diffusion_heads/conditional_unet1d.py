"""带 diffusion timestep 的 DP Conditional UNet 1D 公开类。"""
from typing import Union

import torch

from diffusion_policy.model.diffusion_heads.base_conditional_unet1d \
    import BaseConditionalUnet1D


class ConditionalUnet1D(BaseConditionalUnet1D):
    def __init__(self,
            input_dim,
            local_cond_dim=None,
            global_cond_dim=None,
            diffusion_step_embed_dim=256,
            down_dims=[256, 512, 1024],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            use_diffusion_step=True,
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            local_cond=None,
            global_cond=None,
            **kwargs):
        return self._forward(
            sample=sample,
            timestep=timestep,
            local_cond=local_cond,
            global_cond=global_cond,
            **kwargs,
        )
