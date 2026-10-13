"""Shape-meta driven PointNet observation encoder.

The implementation follows PADP/DP3's useful core: a shared MLP is applied to
each point and a permutation-invariant reduction produces one feature vector.
It intentionally removes the reference file's task-name assumptions and does
not require PyTorch3D, Open3D, or termcolor.
"""

import copy
import math
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.encoders.shape_meta import obs_keys_by_type


class PointNetBackbone(nn.Module):
    """Point-wise MLP followed by max or mean pooling."""

    def __init__(
            self,
            in_channels: int,
            feature_dim: int = 256,
            hidden_dims: Sequence[int] = (64, 128, 256),
            use_layer_norm: bool = True,
            final_norm: str = "layernorm",
            pool: str = "max"):
        super().__init__()
        in_channels = int(in_channels)
        feature_dim = int(feature_dim)
        hidden_dims = tuple(int(x) for x in hidden_dims)
        if in_channels <= 0 or feature_dim <= 0 or not hidden_dims:
            raise ValueError("PointNet dimensions must be positive")
        if min(hidden_dims) <= 0:
            raise ValueError(f"hidden_dims must be positive, got {hidden_dims}")
        if pool not in {"max", "mean"}:
            raise ValueError(f"pool must be 'max' or 'mean', got {pool!r}")
        if final_norm not in {"none", "layernorm"}:
            raise ValueError(
                f"final_norm must be 'none' or 'layernorm', got {final_norm!r}"
            )

        layers = []
        current_dim = in_channels
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        self.point_mlp = nn.Sequential(*layers)
        projection = [nn.Linear(current_dim, feature_dim)]
        if final_norm == "layernorm":
            projection.append(nn.LayerNorm(feature_dim))
        self.projection = nn.Sequential(*projection)
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.pool = pool

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != self.in_channels:
            raise ValueError(
                f"PointNet expected [B, N, {self.in_channels}], got "
                f"{tuple(points.shape)}"
            )
        features = self.point_mlp(points)
        if self.pool == "max":
            features = features.amax(dim=1)
        else:
            features = features.mean(dim=1)
        return self.projection(features)


class PointCloudObsEncoder(ModuleAttrMixin):
    """Encode point-cloud keys and concatenate optional low-dimensional state.

    PADP shape metadata uses ``[N, C]`` (for example ``[1024, 6]``). Set
    ``point_layout='channels_first'`` for metadata and tensors shaped ``[C,N]``.
    ``input_channels`` can select the leading XYZ channels from XYZRGB input.
    """

    def __init__(
            self,
            shape_meta: dict,
            feature_dim: int = 256,
            hidden_dims: Sequence[int] = (64, 128, 256),
            input_channels: Optional[Union[int, Mapping[str, int]]] = None,
            point_layout: str = "points_last",
            use_layer_norm: bool = True,
            final_norm: str = "layernorm",
            pool: str = "max",
            share_pointcloud_model: bool = False,
            frozen: bool = False,
            include_low_dim: bool = True,
            ignore_other_modalities: bool = True):
        super().__init__()
        if point_layout not in {"points_last", "channels_first"}:
            raise ValueError(
                "point_layout must be 'points_last' or 'channels_first', got "
                f"{point_layout!r}"
            )
        grouped = obs_keys_by_type(shape_meta)
        if not grouped["point_cloud"]:
            raise ValueError(
                "PointCloudObsEncoder requires at least one point_cloud key"
            )
        if not ignore_other_modalities:
            other = grouped["rgb"] + grouped["voxel"]
            if other:
                raise ValueError(
                    "PointCloudObsEncoder received other spatial modalities: "
                    f"{other}. Use build_obs_encoder for multimodal fusion."
                )

        self.shape_meta = shape_meta
        self.model_name = "pointnet"
        self.feature_dim = int(feature_dim)
        self.point_layout = point_layout
        self.share_pointcloud_model = bool(share_pointcloud_model)
        self.frozen = bool(frozen)
        self.pointcloud_keys = grouped["point_cloud"]
        self.low_dim_keys = grouped["low_dim"] if include_low_dim else ()
        self.key_shape_map = {
            key: tuple(int(x) for x in attr["shape"])
            for key, attr in shape_meta["obs"].items()
            if not attr.get("ignore_by_policy", False)
        }

        key_channel_map = {}
        for key in self.pointcloud_keys:
            shape = self.key_shape_map[key]
            if len(shape) != 2:
                raise ValueError(
                    f"Point-cloud key {key!r} must have [N,C] or [C,N] "
                    f"shape, got {shape}"
                )
            available_channels = shape[-1] if point_layout == "points_last" else shape[0]
            if isinstance(input_channels, Mapping):
                selected_channels = int(input_channels.get(key, available_channels))
            elif input_channels is None:
                selected_channels = available_channels
            else:
                selected_channels = int(input_channels)
            if selected_channels <= 0 or selected_channels > available_channels:
                raise ValueError(
                    f"Point-cloud key {key!r} has {available_channels} channels; "
                    f"cannot select {selected_channels}"
                )
            key_channel_map[key] = selected_channels
        self.key_channel_map = key_channel_map

        base_models = {
            channels: PointNetBackbone(
                in_channels=channels,
                feature_dim=self.feature_dim,
                hidden_dims=hidden_dims,
                use_layer_norm=use_layer_norm,
                final_norm=final_norm,
                pool=pool,
            )
            for channels in sorted(set(key_channel_map.values()))
        }
        if self.share_pointcloud_model:
            if len(base_models) != 1:
                raise ValueError(
                    "Shared point-cloud model requires identical selected "
                    f"channel counts, got {key_channel_map}"
                )
            self.pointcloud_model = next(iter(base_models.values()))
            self.pointcloud_models = nn.ModuleDict()
        else:
            self.pointcloud_model = None
            self.pointcloud_models = nn.ModuleDict({
                key: copy.deepcopy(base_models[key_channel_map[key]])
                for key in self.pointcloud_keys
            })
        if self.frozen:
            for model in self._iter_models():
                model.requires_grad_(False)

        low_dim = sum(
            math.prod(self.key_shape_map[key]) for key in self.low_dim_keys
        )
        self._output_dim = len(self.pointcloud_keys) * self.feature_dim + low_dim

    def _iter_models(self):
        if self.share_pointcloud_model:
            yield self.pointcloud_model
        else:
            yield from self.pointcloud_models.values()

    def _model_for_key(self, key: str) -> nn.Module:
        if self.share_pointcloud_model:
            return self.pointcloud_model
        return self.pointcloud_models[key]

    def _prepare_points(self, key: str, points: torch.Tensor) -> torch.Tensor:
        expected = self.key_shape_map[key]
        if points.ndim != 3 or tuple(points.shape[1:]) != expected:
            raise ValueError(
                f"Point-cloud key {key!r} expected [B, {expected}], got "
                f"{tuple(points.shape)}"
            )
        if not points.is_floating_point():
            raise TypeError(f"Point-cloud key {key!r} must be floating point")
        if self.point_layout == "channels_first":
            points = points.transpose(1, 2)
        return points[..., :self.key_channel_map[key]]

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not obs_dict:
            raise ValueError("obs_dict must not be empty")
        batch_size = next(iter(obs_dict.values())).shape[0]
        features = []
        for key in self.pointcloud_keys:
            points = self._prepare_points(key, obs_dict[key])
            features.append(self._model_for_key(key)(points))
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


__all__ = ["PointCloudObsEncoder", "PointNetBackbone"]
