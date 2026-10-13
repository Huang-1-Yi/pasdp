"""Shape-meta driven dense voxel observation encoder.

PADP's reference equivariant encoder is tied to escnn and volumes of size
16/58/64. This default branch keeps the same ``[C,D,H,W]`` contract while
using ordinary PyTorch Conv3d blocks plus adaptive pooling, so other positive
voxel sizes are valid and no optional dependency is imported.
"""

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.encoders.shape_meta import obs_keys_by_type


def _group_count(num_channels: int, channels_per_group: int = 16) -> int:
    groups = max(1, int(num_channels) // int(channels_per_group))
    while int(num_channels) % groups != 0:
        groups -= 1
    return groups


class Conv3dBackbone(nn.Module):
    """Resolution-independent dense voxel CNN."""

    def __init__(
            self,
            in_channels: int,
            feature_dim: int = 256,
            channels: Sequence[int] = (32, 64, 128),
            use_group_norm: bool = True,
            pool: str = "avg"):
        super().__init__()
        in_channels = int(in_channels)
        feature_dim = int(feature_dim)
        channels = tuple(int(x) for x in channels)
        if in_channels <= 0 or feature_dim <= 0 or not channels:
            raise ValueError("Voxel CNN dimensions must be positive")
        if min(channels) <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if pool not in {"avg", "max"}:
            raise ValueError(f"pool must be 'avg' or 'max', got {pool!r}")

        blocks = []
        current_channels = in_channels
        for output_channels in channels:
            blocks.append(nn.Conv3d(
                current_channels,
                output_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=not use_group_norm,
            ))
            if use_group_norm:
                blocks.append(nn.GroupNorm(
                    _group_count(output_channels), output_channels))
            blocks.append(nn.SiLU())
            current_channels = output_channels
        self.conv = nn.Sequential(*blocks)
        self.pool = (
            nn.AdaptiveAvgPool3d(1) if pool == "avg"
            else nn.AdaptiveMaxPool3d(1)
        )
        self.projection = nn.Linear(current_channels, feature_dim)
        self.in_channels = in_channels
        self.feature_dim = feature_dim

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        if voxels.ndim != 5 or voxels.shape[1] != self.in_channels:
            raise ValueError(
                f"Voxel CNN expected [B, {self.in_channels}, D, H, W], got "
                f"{tuple(voxels.shape)}"
            )
        features = self.pool(self.conv(voxels)).flatten(start_dim=1)
        return self.projection(features)


class VoxelObsEncoder(ModuleAttrMixin):
    """Encode dense voxel keys and concatenate optional low-dimensional state."""

    def __init__(
            self,
            shape_meta: dict,
            feature_dim: int = 256,
            channels: Sequence[int] = (32, 64, 128),
            use_group_norm: bool = True,
            pool: str = "avg",
            crop_shape: Optional[Sequence[int]] = None,
            eval_fixed_crop: bool = True,
            share_voxel_model: bool = False,
            frozen: bool = False,
            include_low_dim: bool = True,
            ignore_other_modalities: bool = True):
        super().__init__()
        grouped = obs_keys_by_type(shape_meta)
        if not grouped["voxel"]:
            raise ValueError("VoxelObsEncoder requires at least one voxel key")
        if not ignore_other_modalities:
            other = grouped["rgb"] + grouped["point_cloud"]
            if other:
                raise ValueError(
                    "VoxelObsEncoder received other spatial modalities: "
                    f"{other}. Use build_obs_encoder for multimodal fusion."
                )

        self.shape_meta = shape_meta
        self.model_name = "conv3d"
        self.feature_dim = int(feature_dim)
        self.eval_fixed_crop = bool(eval_fixed_crop)
        self.share_voxel_model = bool(share_voxel_model)
        self.frozen = bool(frozen)
        self.voxel_keys = grouped["voxel"]
        self.low_dim_keys = grouped["low_dim"] if include_low_dim else ()
        self.key_shape_map = {
            key: tuple(int(x) for x in attr["shape"])
            for key, attr in shape_meta["obs"].items()
            if not attr.get("ignore_by_policy", False)
        }

        key_channel_map = {}
        spatial_shapes = {}
        for key in self.voxel_keys:
            shape = self.key_shape_map[key]
            if len(shape) != 4:
                raise ValueError(
                    f"Voxel key {key!r} must have [C,D,H,W] shape, got {shape}"
                )
            key_channel_map[key] = shape[0]
            spatial_shapes[key] = shape[1:]
        self.key_channel_map = key_channel_map

        if crop_shape is None:
            self.crop_shape = None
        else:
            if len(crop_shape) != 3:
                raise ValueError(
                    f"crop_shape must contain (D,H,W), got {crop_shape}"
                )
            self.crop_shape = tuple(int(x) for x in crop_shape)
            if min(self.crop_shape) <= 0:
                raise ValueError(
                    f"crop_shape must be positive, got {self.crop_shape}"
                )
            for key, spatial_shape in spatial_shapes.items():
                if any(crop > size for crop, size in zip(
                        self.crop_shape, spatial_shape)):
                    raise ValueError(
                        f"crop_shape {self.crop_shape} exceeds voxel key "
                        f"{key!r} spatial shape {spatial_shape}"
                    )

        base_models = {
            input_channels: Conv3dBackbone(
                in_channels=input_channels,
                feature_dim=self.feature_dim,
                channels=channels,
                use_group_norm=use_group_norm,
                pool=pool,
            )
            for input_channels in sorted(set(key_channel_map.values()))
        }
        if self.share_voxel_model:
            if len(base_models) != 1:
                raise ValueError(
                    "Shared voxel model requires identical channel counts, "
                    f"got {key_channel_map}"
                )
            self.voxel_model = next(iter(base_models.values()))
            self.voxel_models = nn.ModuleDict()
        else:
            self.voxel_model = None
            self.voxel_models = nn.ModuleDict({
                key: copy.deepcopy(base_models[key_channel_map[key]])
                for key in self.voxel_keys
            })
        if self.frozen:
            for model in self._iter_models():
                model.requires_grad_(False)

        low_dim = sum(
            math.prod(self.key_shape_map[key]) for key in self.low_dim_keys
        )
        self._output_dim = len(self.voxel_keys) * self.feature_dim + low_dim

    def _iter_models(self):
        if self.share_voxel_model:
            yield self.voxel_model
        else:
            yield from self.voxel_models.values()

    def _model_for_key(self, key: str) -> nn.Module:
        return self.voxel_model if self.share_voxel_model else self.voxel_models[key]

    def _crop(self, voxels: torch.Tensor) -> torch.Tensor:
        if self.crop_shape is None:
            return voxels
        spatial_shape = voxels.shape[-3:]
        if self.training:
            offsets = tuple(
                torch.randint(
                    size - crop + 1, (), device=voxels.device).item()
                for size, crop in zip(spatial_shape, self.crop_shape)
            )
        elif self.eval_fixed_crop:
            offsets = tuple(
                (size - crop) // 2
                for size, crop in zip(spatial_shape, self.crop_shape)
            )
        else:
            return voxels
        d, h, w = offsets
        cd, ch, cw = self.crop_shape
        return voxels[..., d:d + cd, h:h + ch, w:w + cw]

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not obs_dict:
            raise ValueError("obs_dict must not be empty")
        batch_size = next(iter(obs_dict.values())).shape[0]
        features = []
        for key in self.voxel_keys:
            voxels = obs_dict[key]
            expected = self.key_shape_map[key]
            if voxels.ndim != 5 or tuple(voxels.shape[1:]) != expected:
                raise ValueError(
                    f"Voxel key {key!r} expected [B, {expected}], got "
                    f"{tuple(voxels.shape)}"
                )
            if not voxels.is_floating_point():
                raise TypeError(f"Voxel key {key!r} must be floating point")
            features.append(self._model_for_key(key)(self._crop(voxels)))
        for key in self.low_dim_keys:
            value = obs_dict[key]
            expected = self.key_shape_map[key]
            if value.shape[0] != batch_size or tuple(value.shape[1:]) != expected:
                raise ValueError(
                    f"Low-dimensional key {key!r} expected [B, {expected}], "
                    f"got {tuple(value.shape)}"
                )
            features.append(value.reshape(batch_size, -1))
        result = torch.cat(features, dim=-1)
        if result.shape != (batch_size, self._output_dim):
            raise RuntimeError(
                f"Unexpected encoder output {tuple(result.shape)}; expected "
                f"{(batch_size, self._output_dim)}"
            )
        return result

    def output_shape(self) -> Tuple[int, ...]:
        return (self._output_dim,)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            for model in self._iter_models():
                model.eval()
        return self


__all__ = ["Conv3dBackbone", "VoxelObsEncoder"]
