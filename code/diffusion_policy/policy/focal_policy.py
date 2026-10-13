"""Observation-modality-independent Focal consistency-flow policy."""

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from diffusion_policy.model.diffusion_heads.conditional_unet1d import (
    ConditionalUnet1D,
)
from diffusion_policy.model.encoders.factory import build_obs_encoder
from diffusion_policy.policy.common.focal_objectives import (
    dct_over_horizon,
    orthonormal_dct_matrix,
)
from diffusion_policy.policy.common.focal_time_sampler import sample_two_timesteps
from diffusion_policy.policy.common.base_diffusion_policy import (
    BaseDiffusionPolicy,
)


class FocalPolicy(BaseDiffusionPolicy):
    """Consistency-flow action policy with full-horizon frequency supervision.

    The policy consumes only the common observation-encoder protocol.  It does
    not know whether the encoder input is RGB, point cloud, voxel, or a fused
    modality.  This permits an isolated Robomimic experiment using the existing
    RGB data/encoder/runner while changing only the policy objective.
    """

    def __init__(
            self,
            shape_meta: dict,
            horizon: int,
            n_action_steps: int,
            n_obs_steps: int,
            obs_encoder: Optional[nn.Module] = None,
            rgb_model: str = "robomimic",
            pointcloud_model: str = "pointnet_focal",
            voxel_model: str = "conv3d",
            rgb_encoder_kwargs: Optional[Dict] = None,
            pointcloud_encoder_kwargs: Optional[Dict] = None,
            voxel_encoder_kwargs: Optional[Dict] = None,
            crop_shape=(76, 76),
            obs_encoder_group_norm: bool = True,
            eval_fixed_crop: bool = True,
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 128,
            down_dims: Sequence[int] = (256, 512, 1024),
            kernel_size: int = 5,
            n_groups: int = 8,
            cond_predict_scale: bool = True,
            freq_weight: float = 1e-4,
            sample_cfg: Optional[Mapping] = None,
            eps: float = 1e-2,
            time_scale: float = 99.0):
        super().__init__()
        action_shape = tuple(int(x) for x in shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise NotImplementedError(
                f"FocalPolicy currently expects one action axis, got {action_shape}"
            )
        if not obs_as_global_cond:
            raise NotImplementedError(
                "The current Focal implementation supports only "
                "obs_as_global_cond=true"
            )
        horizon = int(horizon)
        n_action_steps = int(n_action_steps)
        n_obs_steps = int(n_obs_steps)
        if horizon <= 0 or n_action_steps <= 0 or n_obs_steps <= 0:
            raise ValueError("horizon, n_action_steps and n_obs_steps must be positive")
        if n_action_steps > horizon:
            raise ValueError("n_action_steps must not exceed horizon")

        if obs_encoder is None:
            rgb_kwargs = dict(rgb_encoder_kwargs or {})
            rgb_kwargs.setdefault("crop_shape", crop_shape)
            rgb_kwargs.setdefault("eval_fixed_crop", eval_fixed_crop)
            if rgb_model in {"robomimic", "robomimic_raw"}:
                rgb_kwargs.setdefault(
                    "obs_encoder_group_norm", obs_encoder_group_norm)
            elif rgb_model.startswith("timm"):
                rgb_kwargs.setdefault("use_group_norm", obs_encoder_group_norm)
            obs_encoder = build_obs_encoder(
                shape_meta=shape_meta,
                rgb_model=rgb_model,
                pointcloud_model=pointcloud_model,
                voxel_model=voxel_model,
                rgb_encoder_kwargs=rgb_kwargs,
                pointcloud_encoder_kwargs=pointcloud_encoder_kwargs,
                voxel_encoder_kwargs=voxel_encoder_kwargs,
            )

        output_shape = obs_encoder.output_shape()
        if isinstance(output_shape, int):
            obs_feature_dim = int(output_shape)
        elif len(output_shape) == 1:
            obs_feature_dim = int(output_shape[0])
        else:
            raise ValueError(
                f"obs_encoder.output_shape() must be one-dimensional, got "
                f"{output_shape}"
            )

        action_dim = action_shape[0]
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=obs_feature_dim * n_obs_steps,
            diffusion_step_embed_dim=int(diffusion_step_embed_dim),
            down_dims=tuple(int(x) for x in down_dims),
            kernel_size=int(kernel_size),
            n_groups=int(n_groups),
            cond_predict_scale=bool(cond_predict_scale),
        )
        self.obs_encoder = obs_encoder
        self._initialize_normalizer()
        self.register_buffer(
            "_dct_matrix",
            orthonormal_dct_matrix(horizon),
            persistent=False,
        )

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.obs_feature_dim = obs_feature_dim
        self.obs_as_global_cond = True
        self.freq_weight = float(freq_weight)
        self.sample_cfg = sample_cfg
        self.eps = float(eps)
        self.time_scale = float(time_scale)

        print(
            "Focal encoder selection: "
            f"requested_rgb={rgb_model}, "
            f"class={type(obs_encoder).__module__}.{type(obs_encoder).__name__}, "
            f"model_name={getattr(obs_encoder, 'model_name', None)}, "
            f"output_shape={output_shape}"
        )
        print(f"Focal flow params: {sum(p.numel() for p in self.model.parameters()):e}")
        print(f"Focal encoder params: {sum(p.numel() for p in self.obs_encoder.parameters()):e}")

    def _global_condition(
            self,
            normalized_obs: Dict[str, torch.Tensor],
            batch_size: int) -> torch.Tensor:
        features = self._encode_normalized_obs(
            normalized_obs, n_steps=self.n_obs_steps)
        expected = batch_size * self.n_obs_steps
        if features.shape[0] != expected:
            raise RuntimeError(
                f"encoder produced batch {features.shape[0]}, expected {expected}"
            )
        return features.reshape(batch_size, -1)

    def predict_action(
            self,
            obs_dict: Dict[str, torch.Tensor],
            generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
        if "past_action" in obs_dict:
            raise NotImplementedError("past_action conditioning is not implemented")
        normalized_obs = self.normalize_obs(obs_dict)
        example = next(iter(normalized_obs.values()))
        batch_size = int(example.shape[0])
        global_cond = self._global_condition(normalized_obs, batch_size)

        noise = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        time = torch.full(
            (batch_size,), self.eps, device=noise.device, dtype=noise.dtype)
        velocity = self.model(
            noise,
            time * self.time_scale,
            local_cond=None,
            global_cond=global_cond,
        )
        normalized_action = noise + velocity
        action_pred = self.unnormalize_action(normalized_action)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        normalized_obs = self.normalize_obs(batch["obs"])
        target = self.normalize_action(batch["action"])
        if target.ndim != 3:
            raise ValueError(
                f"action must have shape [B, T, D], got {tuple(target.shape)}"
            )
        batch_size, horizon, action_dim = target.shape
        if horizon != self.horizon or action_dim != self.action_dim:
            raise ValueError(
                f"action shape {tuple(target.shape)} does not match policy "
                f"[B, {self.horizon}, {self.action_dim}]"
            )
        global_cond = self._global_condition(normalized_obs, batch_size)

        noise = torch.randn_like(target)
        t, r = sample_two_timesteps(
            self.sample_cfg,
            num_samples=batch_size,
            device=target.device,
            eps=self.eps,
        )
        t_expand = t[:, None, None].to(dtype=target.dtype)
        r_expand = r[:, None, None].to(dtype=target.dtype)
        x_t = t_expand * target + (1.0 - t_expand) * noise
        x_r = r_expand * target + (1.0 - r_expand) * noise

        v_t = self.model(
            x_t,
            t * self.time_scale,
            local_cond=None,
            global_cond=global_cond,
        )
        with torch.no_grad():
            v_r = self.model(
                x_r,
                r * self.time_scale,
                local_cond=None,
                global_cond=global_cond,
            )
        v_r = torch.nan_to_num(v_r)
        f_t = x_t + (1.0 - t_expand) * v_t
        f_r = x_r + (1.0 - r_expand) * v_r

        short_t = f_t[:, :self.n_action_steps]
        short_r = f_r[:, :self.n_action_steps]
        loss_space = (short_t - short_r).square().flatten(1).mean(dim=1).mean()

        frequency_pred = dct_over_horizon(f_t, self._dct_matrix)
        frequency_target = dct_over_horizon(target, self._dct_matrix)
        loss_freq = (frequency_target - frequency_pred).square()
        loss_freq = loss_freq.sum(dim=(1, 2)).mean()
        weighted_freq_loss = self.freq_weight * loss_freq
        loss = loss_space + weighted_freq_loss
        loss_info = {
            "space_loss": loss_space.detach(),
            "freq_loss": loss_freq.detach(),
            "weighted_freq_loss": weighted_freq_loss.detach(),
            "total_loss": loss.detach(),
        }
        return loss, loss_info

    def reset(self) -> None:
        """Focal inference is stateless between rollout calls."""
        return None


__all__ = ["FocalPolicy"]
