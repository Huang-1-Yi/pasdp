"""不含 diffusion timestep 的 PADP Conditional UNet 1D 公开类。"""
import torch

from diffusion_policy.model.diffusion_heads.base_conditional_unet1d \
    import BaseConditionalUnet1D


class ConditionalUnet1DPadp(BaseConditionalUnet1D):
    def __init__(self,
            input_dim,
            local_cond_dim=None,
            global_cond_dim=None,
            down_dims=[256, 512, 1024],
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__(
            input_dim=input_dim,
            local_cond_dim=local_cond_dim,
            global_cond_dim=global_cond_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            use_diffusion_step=False,
        )

    def forward(self,
            sample: torch.Tensor,
            local_cond=None,
            global_cond=None,
            **kwargs):
        return self._forward(
            sample=sample,
            local_cond=local_cond,
            global_cond=global_cond,
            **kwargs,
        )


# 历史代码曾从本模块导入 ConditionalUnet1D，继续保留该别名。
ConditionalUnet1D = ConditionalUnet1DPadp
