"""timm-backed RGB observation encoder.

This module is intentionally imported only through the RGB encoder factory so
the default Robomimic configuration does not require ``timm``.
"""

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF

from diffusion_policy.common.pytorch_util import replace_submodules
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class TimmObsEncoder(ModuleAttrMixin):
    """Encode flattened RGB/low-dimensional Robomimic observations.

    The active policies flatten time before invoking their observation
    encoder, so each value is shaped ``[B, ...]`` here (not ``[B, T, ...]``).
    Every RGB view produces one globally pooled feature vector; the vectors
    and flattened low-dimensional observations are concatenated to ``[B, D]``.
    """

    _VARIABLE_IMAGE_SIZE_PREFIXES = (
        "beit", "deit", "eva", "flexivit", "maxvit", "swin", "vit",
    )

    def __init__(
            self,
            shape_meta: dict,
            model_name: str = "resnet18.a1_in1k",
            pretrained: bool = False,
            frozen: bool = False,
            share_rgb_model: bool = False,
            imagenet_norm: Optional[bool] = None,
            input_range: str = "minus_one_one",
            crop_shape: Optional[Sequence[int]] = None,
            eval_fixed_crop: bool = True,
            model_image_size: Optional[Sequence[int]] = None,
            use_group_norm: bool = False,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.0):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_name = str(model_name)
        self.pretrained = bool(pretrained)
        self.frozen = bool(frozen)
        self.share_rgb_model = bool(share_rgb_model)
        self.imagenet_norm = (
            bool(pretrained) if imagenet_norm is None else bool(imagenet_norm)
        )
        self.input_range = str(input_range)
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(
                "input_range must be 'minus_one_one' or 'zero_one', got "
                f"{self.input_range!r}"
            )
        self.eval_fixed_crop = bool(eval_fixed_crop)

        rgb_keys = []
        low_dim_keys = []
        key_shape_map = {}
        image_shape: Optional[Tuple[int, int, int]] = None
        for key, attr in shape_meta["obs"].items():
            shape = tuple(int(x) for x in attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if obs_type == "rgb":
                if len(shape) != 3:
                    raise ValueError(f"RGB key {key!r} must have CHW shape, got {shape}")
                if image_shape is not None and image_shape != shape:
                    raise ValueError(
                        "TimmObsEncoder currently requires all RGB views to "
                        f"share one shape; got {image_shape} and {shape}"
                    )
                image_shape = shape
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")

        if not rgb_keys:
            raise ValueError("TimmObsEncoder requires at least one RGB observation")

        self.rgb_keys = tuple(sorted(rgb_keys))
        self.low_dim_keys = tuple(sorted(low_dim_keys))
        self.key_shape_map = key_shape_map
        assert image_shape is not None
        channels, image_height, image_width = image_shape
        if self.imagenet_norm and channels != 3:
            raise ValueError("ImageNet normalization requires three RGB channels")
        self.image_size = (image_height, image_width)
        if model_image_size is None:
            self.model_image_size = self.image_size
        else:
            if len(model_image_size) != 2:
                raise ValueError(
                    "model_image_size must be (H, W), got "
                    f"{model_image_size}"
                )
            self.model_image_size = tuple(int(x) for x in model_image_size)
            if min(self.model_image_size) <= 0:
                raise ValueError(
                    "model_image_size must be positive, got "
                    f"{self.model_image_size}"
                )

        if crop_shape is None:
            self.crop_shape = None
        else:
            if len(crop_shape) != 2:
                raise ValueError(f"crop_shape must be (H, W), got {crop_shape}")
            crop_height, crop_width = (int(x) for x in crop_shape)
            if crop_height > image_height or crop_width > image_width:
                raise ValueError(
                    f"crop_shape {(crop_height, crop_width)} exceeds image "
                    f"shape {(image_height, image_width)}"
                )
            self.crop_shape = (crop_height, crop_width)

        model_kwargs = {
            "pretrained": self.pretrained,
            "in_chans": channels,
            "num_classes": 0,
            "global_pool": "avg",
            "drop_rate": float(drop_rate),
            "drop_path_rate": float(drop_path_rate),
        }
        if self.model_name.startswith(self._VARIABLE_IMAGE_SIZE_PREFIXES):
            model_kwargs["img_size"] = self.model_image_size
        base_model = timm.create_model(self.model_name, **model_kwargs)
        self.feature_dim = int(base_model.num_features)
        data_config = timm.data.resolve_model_data_config(base_model)
        model_mean = tuple(float(x) for x in data_config["mean"])
        model_std = tuple(float(x) for x in data_config["std"])
        if len(model_mean) != channels or len(model_std) != channels:
            if self.imagenet_norm:
                raise ValueError(
                    f"Model normalization has {len(model_mean)} channels but "
                    f"input {image_shape} has {channels}"
                )
            model_mean = (0.0,) * channels
            model_std = (1.0,) * channels
        # Match the established UMI/PADP behavior: GroupNorm conversion is a
        # from-scratch training option.  Keep pretrained normalization intact.
        self.use_group_norm = bool(use_group_norm and not self.pretrained)
        if self.use_group_norm:
            base_model = replace_submodules(
                root_module=base_model,
                predicate=lambda module: isinstance(module, nn.BatchNorm2d),
                func=lambda module: nn.GroupNorm(
                    num_groups=self._group_count(module.num_features),
                    num_channels=module.num_features,
                ),
            )

        if self.share_rgb_model:
            self.rgb_model = base_model
            self.rgb_models = nn.ModuleDict()
        else:
            self.rgb_model = None
            self.rgb_models = nn.ModuleDict({
                key: copy.deepcopy(base_model) for key in self.rgb_keys
            })

        if self.frozen:
            self._set_rgb_requires_grad(False)

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor(model_mean).view(1, channels, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor(model_std).view(1, channels, 1, 1),
            persistent=False,
        )

        low_dim = sum(
            math.prod(self.key_shape_map[key]) for key in self.low_dim_keys
        )
        self._output_dim = len(self.rgb_keys) * self.feature_dim + low_dim

    @staticmethod
    def _group_count(num_channels: int) -> int:
        """Choose the historical channels-per-group~=16 layout safely."""
        groups = max(1, int(num_channels) // 16)
        while int(num_channels) % groups != 0:
            groups -= 1
        return groups

    def _iter_rgb_models(self):
        if self.share_rgb_model:
            yield self.rgb_model
        else:
            yield from self.rgb_models.values()

    def _set_rgb_requires_grad(self, requires_grad: bool) -> None:
        for model in self._iter_rgb_models():
            model.requires_grad_(requires_grad)

    def _model_for_key(self, key: str) -> nn.Module:
        if self.share_rgb_model:
            return self.rgb_model
        return self.rgb_models[key]

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        # Dataset normalizers in both active DP/PADP paths map RGB [0, 1] to
        # [-1, 1].  timm preprocessing and ImageNet statistics expect [0, 1].
        if self.input_range == "minus_one_one":
            image = image.add(1.0).mul(0.5)
        if self.crop_shape is not None:
            if self.training:
                top, left, height, width = (
                    torch.randint(
                        image.shape[-2] - self.crop_shape[0] + 1,
                        (), device=image.device,
                    ).item(),
                    torch.randint(
                        image.shape[-1] - self.crop_shape[1] + 1,
                        (), device=image.device,
                    ).item(),
                    self.crop_shape[0],
                    self.crop_shape[1],
                )
                image = TVF.crop(image, top, left, height, width)
            elif self.eval_fixed_crop:
                image = TVF.center_crop(image, self.crop_shape)
        if image.shape[-2:] != self.model_image_size:
            image = TVF.resize(
                image,
                list(self.model_image_size),
                antialias=True,
            )
        if self.imagenet_norm:
            image = (image - self._imagenet_mean) / self._imagenet_std
        return image

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not obs_dict:
            raise ValueError("obs_dict must not be empty")
        batch_size = next(iter(obs_dict.values())).shape[0]
        features = []

        for key in self.rgb_keys:
            image = obs_dict[key]
            expected = self.key_shape_map[key]
            if image.ndim != 4 or tuple(image.shape[1:]) != expected:
                raise ValueError(
                    f"RGB key {key!r} expected [B, {expected}], got "
                    f"{tuple(image.shape)}"
                )
            if not image.is_floating_point():
                raise TypeError(
                    f"RGB key {key!r} must be floating point"
                )
            image = self._prepare_image(image)
            feature = self._model_for_key(key)(image)
            if feature.ndim != 2:
                feature = feature.flatten(start_dim=1)
            features.append(feature)

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
                f"Unexpected encoder output {tuple(result.shape)}; "
                f"expected {(batch_size, self._output_dim)}"
            )
        return result

    def output_shape(self) -> Tuple[int, ...]:
        return (self._output_dim,)

    def no_weight_decay(self):
        names = set()
        for key in self.rgb_keys if not self.share_rgb_model else ("shared",):
            model = self._model_for_key(
                self.rgb_keys[0] if key == "shared" else key
            )
            if hasattr(model, "no_weight_decay"):
                prefix = "rgb_model" if key == "shared" else f"rgb_models.{key}"
                names.update(f"{prefix}.{name}" for name in model.no_weight_decay())
        return names

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            for model in self._iter_rgb_models():
                model.eval()
        return self


__all__ = ["TimmObsEncoder"]
