"""Feature-level fusion for independently constructed modality encoders."""

import math
from typing import Dict, Mapping, Tuple

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.encoders.shape_meta import obs_keys_by_type


class MultiModalObsEncoder(ModuleAttrMixin):
    """Concatenate RGB, point-cloud, voxel, then low-dimensional features."""

    def __init__(self, shape_meta: dict, encoders: Mapping[str, nn.Module]):
        super().__init__()
        self.shape_meta = shape_meta
        self.encoders = nn.ModuleDict(dict(encoders))
        if not self.encoders:
            raise ValueError("MultiModalObsEncoder requires at least one branch")
        unsupported = set(self.encoders) - {"rgb", "point_cloud", "voxel"}
        if unsupported:
            raise ValueError(f"Unsupported encoder branches: {sorted(unsupported)}")

        grouped = obs_keys_by_type(shape_meta)
        self.low_dim_keys = grouped["low_dim"]
        self.key_shape_map = {
            key: tuple(int(x) for x in attr["shape"])
            for key, attr in shape_meta["obs"].items()
            if not attr.get("ignore_by_policy", False)
        }
        branch_dim = 0
        for name, encoder in self.encoders.items():
            output_shape = tuple(int(x) for x in encoder.output_shape())
            if len(output_shape) != 1:
                raise ValueError(
                    f"Encoder branch {name!r} must return a vector, got "
                    f"output_shape={output_shape}"
                )
            branch_dim += output_shape[0]
        low_dim = sum(
            math.prod(self.key_shape_map[key]) for key in self.low_dim_keys
        )
        self._output_dim = branch_dim + low_dim

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not obs_dict:
            raise ValueError("obs_dict must not be empty")
        batch_size = next(iter(obs_dict.values())).shape[0]
        features = [encoder(obs_dict) for encoder in self.encoders.values()]
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
                f"Unexpected multimodal output {tuple(result.shape)}; expected "
                f"{(batch_size, self._output_dim)}"
            )
        return result

    def output_shape(self) -> Tuple[int, ...]:
        return (self._output_dim,)

    def no_weight_decay(self):
        names = set()
        for branch_name, encoder in self.encoders.items():
            if hasattr(encoder, "no_weight_decay"):
                names.update(
                    f"encoders.{branch_name}.{name}"
                    for name in encoder.no_weight_decay()
                )
        return names


__all__ = ["MultiModalObsEncoder"]
