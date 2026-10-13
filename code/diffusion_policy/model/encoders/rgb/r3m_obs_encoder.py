"""R3M-backed multi-view observation encoder.

The optional ``r3m`` package is imported only when this provider is selected.
R3M's outer module expects uint8-like ``[0, 255]`` input and performs its own
preprocessing, while the active policies provide normalized ``[-1, 1]`` RGB.
This adapter therefore extracts the pretrained visual ``convnet`` and applies
the equivalent resize / ImageNet normalization explicitly.
"""

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class R3MObsEncoder(ModuleAttrMixin):
    """Encode flattened multi-camera observations with an R3M ResNet."""

    _FEATURE_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
            self,
            shape_meta: dict,
            model_name: str = "resnet18",
            frozen: bool = False,
            share_rgb_model: bool = True,
            input_range: str = "minus_one_one",
            crop_shape: Optional[Sequence[int]] = None,
            eval_fixed_crop: bool = True,
            resize_shape: Sequence[int] = (256, 256),
            model_crop_shape: Sequence[int] = (224, 224),
            model_override: Optional[nn.Module] = None):
        super().__init__()
        self.shape_meta = shape_meta
        self.model_name = str(model_name).lower()
        if self.model_name not in self._FEATURE_DIMS:
            choices = ", ".join(self._FEATURE_DIMS)
            raise ValueError(
                f"Unsupported R3M model {model_name!r}; choose: {choices}"
            )
        self.feature_dim = self._FEATURE_DIMS[self.model_name]
        self.frozen = bool(frozen)
        self.share_rgb_model = bool(share_rgb_model)
        self.input_range = str(input_range)
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(
                "input_range must be 'minus_one_one' or 'zero_one', got "
                f"{self.input_range!r}"
            )
        self.eval_fixed_crop = bool(eval_fixed_crop)
        self.resize_shape = self._pair("resize_shape", resize_shape)
        self.model_crop_shape = self._pair(
            "model_crop_shape", model_crop_shape)

        rgb_keys = []
        low_dim_keys = []
        key_shape_map = {}
        image_shape = None
        for key, attr in shape_meta["obs"].items():
            shape = tuple(int(x) for x in attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if obs_type == "rgb":
                if len(shape) != 3:
                    raise ValueError(
                        f"RGB key {key!r} must have CHW shape, got {shape}"
                    )
                if shape[0] != 3:
                    raise ValueError(
                        f"R3M requires three-channel RGB, got {shape}"
                    )
                if image_shape is not None and image_shape != shape:
                    raise ValueError(
                        "R3MObsEncoder requires all RGB views to share one "
                        f"shape; got {image_shape} and {shape}"
                    )
                image_shape = shape
                rgb_keys.append(key)
            elif obs_type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {obs_type}")
        if not rgb_keys:
            raise ValueError("R3MObsEncoder requires at least one RGB observation")

        self.rgb_keys = tuple(sorted(rgb_keys))
        self.low_dim_keys = tuple(sorted(low_dim_keys))
        self.key_shape_map = key_shape_map
        assert image_shape is not None
        image_height, image_width = image_shape[-2:]
        if crop_shape is None:
            self.crop_shape = None
        else:
            self.crop_shape = self._pair("crop_shape", crop_shape)
            if (self.crop_shape[0] > image_height
                    or self.crop_shape[1] > image_width):
                raise ValueError(
                    f"crop_shape {self.crop_shape} exceeds image shape "
                    f"{(image_height, image_width)}"
                )

        base_model = model_override
        if base_model is None:
            import r3m

            # The archived upstream loader chooses its initial device from
            # CUDA availability at import time. Force CPU construction; the
            # surrounding Policy later moves the complete module explicitly.
            r3m.device = "cpu"
            loaded = r3m.load_r3m(self.model_name)
            r3m_module = loaded.module if hasattr(loaded, "module") else loaded
            base_model = r3m_module.convnet.to("cpu")
        if not isinstance(base_model, nn.Module):
            raise TypeError("R3M visual model must be an nn.Module")

        if self.share_rgb_model:
            self.rgb_model = base_model
            self.rgb_models = nn.ModuleDict()
        else:
            self.rgb_model = None
            self.rgb_models = nn.ModuleDict({
                key: copy.deepcopy(base_model) for key in self.rgb_keys
            })

        if self.frozen:
            for model in self._iter_rgb_models():
                model.requires_grad_(False)

        self.register_buffer(
            "_imagenet_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

        low_dim = sum(
            math.prod(self.key_shape_map[key]) for key in self.low_dim_keys
        )
        self._output_dim = len(self.rgb_keys) * self.feature_dim + low_dim

    @staticmethod
    def _pair(name: str, value: Sequence[int]) -> Tuple[int, int]:
        if len(value) != 2:
            raise ValueError(f"{name} must contain (H, W), got {value}")
        result = tuple(int(x) for x in value)
        if min(result) <= 0:
            raise ValueError(f"{name} must be positive, got {result}")
        return result

    def _iter_rgb_models(self):
        if self.share_rgb_model:
            yield self.rgb_model
        else:
            yield from self.rgb_models.values()

    def _model_for_key(self, key: str) -> nn.Module:
        return self.rgb_model if self.share_rgb_model else self.rgb_models[key]

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.input_range == "minus_one_one":
            image = image.add(1.0).mul(0.5)
        if self.crop_shape is not None:
            if self.training:
                top = torch.randint(
                    image.shape[-2] - self.crop_shape[0] + 1,
                    (), device=image.device,
                ).item()
                left = torch.randint(
                    image.shape[-1] - self.crop_shape[1] + 1,
                    (), device=image.device,
                ).item()
                image = TVF.crop(
                    image, top, left,
                    self.crop_shape[0], self.crop_shape[1],
                )
            elif self.eval_fixed_crop:
                image = TVF.center_crop(image, self.crop_shape)
        image = TVF.resize(image, list(self.resize_shape), antialias=True)
        image = TVF.center_crop(image, self.model_crop_shape)
        return (image - self._imagenet_mean) / self._imagenet_std

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
            feature = self._model_for_key(key)(self._prepare_image(image))
            if feature.ndim != 2:
                feature = feature.flatten(start_dim=1)
            if feature.shape != (batch_size, self.feature_dim):
                raise RuntimeError(
                    f"R3M {self.model_name} returned {tuple(feature.shape)}; "
                    f"expected {(batch_size, self.feature_dim)}"
                )
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

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            for model in self._iter_rgb_models():
                model.eval()
        return self


__all__ = ["R3MObsEncoder"]
