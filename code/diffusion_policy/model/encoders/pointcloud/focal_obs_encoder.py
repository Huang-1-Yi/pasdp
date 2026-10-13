"""FocalPolicy-compatible point-cloud observation encoder.

This module keeps the active FocalPolicy encoder architecture independent from
the generic :mod:`pointnet_obs_encoder` implementation.  The separation is
intentional: the upstream XYZ and XYZRGB branches do not share exactly the
same point-wise MLP topology, and FocalPolicy projects low-dimensional robot
state through a dedicated MLP before feature concatenation.

The implementation removes only repository-specific logging and configuration
objects.  Layer order and the active numerical operations match the upstream
``FocalPolicyEncoder`` path.
"""

from typing import Dict, Sequence, Tuple, Type

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def _build_state_mlp(
        input_dim: int,
        layer_dims: Sequence[int],
        activation: Type[nn.Module] = nn.ReLU) -> nn.Sequential:
    """Build the state MLP with the same final-layer semantics as upstream."""
    layer_dims = tuple(int(dim) for dim in layer_dims)
    if not layer_dims or min(layer_dims) <= 0:
        raise ValueError(
            f"state_mlp_size must contain positive dimensions, got {layer_dims}"
        )

    layers = []
    current_dim = int(input_dim)
    for hidden_dim in layer_dims[:-1]:
        layers.extend((nn.Linear(current_dim, hidden_dim), activation()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, layer_dims[-1]))
    return nn.Sequential(*layers)


class FocalPointNetXYZ(nn.Module):
    """The XYZ PointNet used by the active FocalPolicy experiments."""

    def __init__(
            self,
            out_channels: int = 64,
            use_layer_norm: bool = True,
            final_norm: str = "layernorm"):
        super().__init__()
        if final_norm not in {"none", "layernorm"}:
            raise ValueError(
                f"final_norm must be 'none' or 'layernorm', got {final_norm!r}"
            )

        block_channels = (64, 128, 256)
        layers = []
        in_channels = 3
        for out_dim in block_channels:
            layers.append(nn.Linear(in_channels, out_dim))
            layers.append(
                nn.LayerNorm(out_dim) if use_layer_norm else nn.Identity()
            )
            layers.append(nn.ReLU())
            in_channels = out_dim
        self.mlp = nn.Sequential(*layers)
        self.final_projection = _build_final_projection(
            block_channels[-1], int(out_channels), final_norm)
        self.in_channels = 3
        self.out_channels = int(out_channels)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        _validate_points(points, self.in_channels)
        features = self.mlp(points)
        features = torch.max(features, dim=1)[0]
        return self.final_projection(features)


class FocalPointNetXYZRGB(nn.Module):
    """The distinct four-layer XYZRGB PointNet from FocalPolicy."""

    def __init__(
            self,
            out_channels: int = 64,
            use_layer_norm: bool = True,
            final_norm: str = "layernorm"):
        super().__init__()
        if final_norm not in {"none", "layernorm"}:
            raise ValueError(
                f"final_norm must be 'none' or 'layernorm', got {final_norm!r}"
            )

        # Upstream deliberately has no norm or activation after the 512 layer.
        self.mlp = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(256, 512),
        )
        self.final_projection = _build_final_projection(
            512, int(out_channels), final_norm)
        self.in_channels = 6
        self.out_channels = int(out_channels)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        _validate_points(points, self.in_channels)
        features = self.mlp(points)
        features = torch.max(features, dim=1)[0]
        return self.final_projection(features)


def _build_final_projection(
        input_dim: int,
        output_dim: int,
        final_norm: str) -> nn.Module:
    if output_dim <= 0:
        raise ValueError(f"out_channels must be positive, got {output_dim}")
    if final_norm == "layernorm":
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
    return nn.Linear(input_dim, output_dim)


def _validate_points(points: torch.Tensor, channels: int) -> None:
    if points.ndim != 3 or points.shape[-1] != channels:
        raise ValueError(
            f"expected point cloud [B, N, {channels}], got {tuple(points.shape)}"
        )
    if not points.is_floating_point():
        raise TypeError("point cloud must be floating point")


class FocalPointCloudObsEncoder(ModuleAttrMixin):
    """Encode FocalPolicy point cloud and robot state into one feature vector.

    Defaults reproduce the active upstream configuration:

    * XYZ input (leading three channels from ``point_cloud``);
    * point feature dimension 64;
    * LayerNorm in the point MLP and final projection;
    * state MLP ``state_dim -> 64 -> 64``;
    * concatenated output dimension 128.

    ``include_low_dim=False`` is supported for the repository's multimodal
    factory contract and returns only the PointNet feature.  It is not the
    paper's full Focal observation encoder.
    """

    def __init__(
            self,
            shape_meta: dict,
            point_feature_dim: int = 64,
            state_mlp_size: Sequence[int] = (64, 64),
            use_pc_color: bool = False,
            use_layer_norm: bool = True,
            final_norm: str = "layernorm",
            point_cloud_key: str = "point_cloud",
            state_key: str = "agent_pos",
            imagination_key: str = "imagin_robot",
            include_low_dim: bool = True,
            frozen: bool = False,
            **kwargs):
        super().__init__()
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected Focal encoder arguments: {unknown}")
        if "obs" not in shape_meta:
            raise ValueError("shape_meta must contain an 'obs' mapping")

        obs_meta = shape_meta["obs"]
        if point_cloud_key not in obs_meta:
            raise KeyError(
                f"Focal encoder requires point-cloud key {point_cloud_key!r}"
            )
        point_shape = tuple(int(x) for x in obs_meta[point_cloud_key]["shape"])
        if len(point_shape) != 2:
            raise ValueError(
                f"{point_cloud_key!r} must have shape [N, C], got {point_shape}"
            )

        self.shape_meta = shape_meta
        self.model_name = "pointnet_focal"
        self.point_cloud_key = point_cloud_key
        self.state_key = state_key
        self.imagination_key = imagination_key
        self.point_shape = point_shape
        self.use_pc_color = bool(use_pc_color)
        self.include_low_dim = bool(include_low_dim)
        self.frozen = bool(frozen)
        self.selected_channels = 6 if self.use_pc_color else 3

        if point_shape[-1] < self.selected_channels:
            raise ValueError(
                f"{point_cloud_key!r} provides {point_shape[-1]} channels but "
                f"the selected Focal branch requires {self.selected_channels}"
            )

        pointnet_type = FocalPointNetXYZRGB if self.use_pc_color \
            else FocalPointNetXYZ
        self.extractor = pointnet_type(
            out_channels=int(point_feature_dim),
            use_layer_norm=bool(use_layer_norm),
            final_norm=final_norm,
        )

        self.use_imagined_robot = imagination_key in obs_meta
        if self.use_imagined_robot:
            imagination_shape = tuple(
                int(x) for x in obs_meta[imagination_key]["shape"])
            if len(imagination_shape) != 2:
                raise ValueError(
                    f"{imagination_key!r} must have shape [N, C], got "
                    f"{imagination_shape}"
                )
            if imagination_shape[-1] < self.selected_channels:
                raise ValueError(
                    f"{imagination_key!r} provides {imagination_shape[-1]} "
                    f"channels but {self.selected_channels} are required"
                )
            self.imagination_shape = imagination_shape
        else:
            self.imagination_shape = None

        self.point_feature_dim = int(point_feature_dim)
        self.state_feature_dim = 0
        if self.include_low_dim:
            if state_key not in obs_meta:
                raise KeyError(
                    f"full Focal encoder requires state key {state_key!r}"
                )
            state_shape = tuple(int(x) for x in obs_meta[state_key]["shape"])
            if len(state_shape) != 1:
                raise ValueError(
                    f"{state_key!r} must have one-dimensional shape, got "
                    f"{state_shape}"
                )
            self.state_shape = state_shape
            self.state_mlp = _build_state_mlp(
                input_dim=state_shape[0],
                layer_dims=state_mlp_size,
            )
            self.state_feature_dim = int(tuple(state_mlp_size)[-1])
        else:
            self.state_shape = None
            self.state_mlp = None

        self._output_dim = self.point_feature_dim + self.state_feature_dim
        if self.frozen:
            self.requires_grad_(False)

    def _prepare_point_tensor(
            self,
            obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        points = obs_dict[self.point_cloud_key]
        expected = self.point_shape
        if points.ndim != 3 or tuple(points.shape[1:]) != expected:
            raise ValueError(
                f"{self.point_cloud_key!r} expected [B, {expected}], got "
                f"{tuple(points.shape)}"
            )
        if not points.is_floating_point():
            raise TypeError(f"{self.point_cloud_key!r} must be floating point")
        points = points[..., :self.selected_channels]

        if self.use_imagined_robot:
            imagined = obs_dict[self.imagination_key]
            expected_imagined = self.imagination_shape
            if imagined.ndim != 3 or tuple(imagined.shape[1:]) != expected_imagined:
                raise ValueError(
                    f"{self.imagination_key!r} expected [B, "
                    f"{expected_imagined}], got {tuple(imagined.shape)}"
                )
            if not imagined.is_floating_point():
                raise TypeError(
                    f"{self.imagination_key!r} must be floating point"
                )
            imagined = imagined[..., :self.selected_channels]
            points = torch.cat((points, imagined), dim=1)
        return points

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        points = self._prepare_point_tensor(obs_dict)
        point_feature = self.extractor(points)
        features = [point_feature]

        if self.include_low_dim:
            state = obs_dict[self.state_key]
            if (state.ndim != 2
                    or tuple(state.shape[1:]) != self.state_shape
                    or state.shape[0] != points.shape[0]):
                raise ValueError(
                    f"{self.state_key!r} expected [B, {self.state_shape}], got "
                    f"{tuple(state.shape)}"
                )
            if not state.is_floating_point():
                raise TypeError(f"{self.state_key!r} must be floating point")
            features.append(self.state_mlp(state))

        result = torch.cat(features, dim=-1)
        expected_shape = (points.shape[0], self._output_dim)
        if tuple(result.shape) != expected_shape:
            raise RuntimeError(
                f"unexpected Focal encoder output {tuple(result.shape)}; "
                f"expected {expected_shape}"
            )
        return result

    def output_shape(self) -> Tuple[int, ...]:
        return (self._output_dim,)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            super().train(False)
        return self


__all__ = [
    "FocalPointCloudObsEncoder",
    "FocalPointNetXYZ",
    "FocalPointNetXYZRGB",
]
