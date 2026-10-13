"""Optional OpenAI CLIP / OpenCLIP multi-view observation encoder."""

import copy
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TVF

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class ClipObsEncoder(ModuleAttrMixin):
    """Use only CLIP's visual tower for flattened policy observations."""

    def __init__(
            self,
            shape_meta: dict,
            backend: str = "openai",
            model_name: str = "ViT-B/32",
            pretrained: Optional[str] = None,
            feature_dim: int = 512,
            frozen: bool = False,
            share_rgb_model: bool = True,
            normalize_features: bool = True,
            input_range: str = "minus_one_one",
            crop_shape: Optional[Sequence[int]] = None,
            eval_fixed_crop: bool = True,
            model_image_size: Sequence[int] = (224, 224),
            cache_dir: Optional[str] = None,
            model_override: Optional[nn.Module] = None):
        super().__init__()
        self.shape_meta = shape_meta
        self.backend = str(backend).lower()
        if self.backend not in {"openai", "open_clip"}:
            raise ValueError(
                f"backend must be 'openai' or 'open_clip', got {backend!r}"
            )
        self.model_name = str(model_name)
        self.pretrained = pretrained
        self.feature_dim = int(feature_dim)
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.frozen = bool(frozen)
        self.share_rgb_model = bool(share_rgb_model)
        self.normalize_features = bool(normalize_features)
        self.input_range = str(input_range)
        if self.input_range not in {"minus_one_one", "zero_one"}:
            raise ValueError(
                "input_range must be 'minus_one_one' or 'zero_one', got "
                f"{self.input_range!r}"
            )
        self.eval_fixed_crop = bool(eval_fixed_crop)
        self.model_image_size = self._pair(
            "model_image_size", model_image_size)

        rgb_keys = []
        low_dim_keys = []
        key_shape_map = {}
        image_shape = None
        for key, attr in shape_meta["obs"].items():
            shape = tuple(int(x) for x in attr["shape"])
            obs_type = attr.get("type", "low_dim")
            key_shape_map[key] = shape
            if obs_type == "rgb":
                if len(shape) != 3 or shape[0] != 3:
                    raise ValueError(
                        f"CLIP RGB key {key!r} must be 3-channel CHW, got "
                        f"{shape}"
                    )
                if image_shape is not None and image_shape != shape:
                    raise ValueError(
                        "ClipObsEncoder requires all RGB views to share one "
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
            raise ValueError("ClipObsEncoder requires at least one RGB observation")

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

        visual = model_override
        if visual is None:
            visual = self._load_visual_tower(cache_dir=cache_dir)
        if not isinstance(visual, nn.Module):
            raise TypeError("CLIP visual tower must be an nn.Module")

        if self.share_rgb_model:
            self.rgb_model = visual
            self.rgb_models = nn.ModuleDict()
        else:
            self.rgb_model = None
            self.rgb_models = nn.ModuleDict({
                key: copy.deepcopy(visual) for key in self.rgb_keys
            })
        if self.frozen:
            for model in self._iter_rgb_models():
                model.requires_grad_(False)

        # OpenAI CLIP statistics are also used by the selected OpenCLIP
        # ViT-B/32 LAION preset.
        self.register_buffer(
            "_clip_mean",
            torch.tensor(
                (0.48145466, 0.4578275, 0.40821073)
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(
                (0.26862954, 0.26130258, 0.27577711)
            ).view(1, 3, 1, 1),
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

    def _load_visual_tower(self, cache_dir: Optional[str]) -> nn.Module:
        if self.backend == "openai":
            import clip

            model, _ = clip.load(
                self.model_name,
                device="cpu",
                jit=False,
                download_root=cache_dir,
            )
            return model.visual.float()

        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device="cpu",
            cache_dir=cache_dir,
        )
        return model.visual

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
        if image.shape[-2:] != self.model_image_size:
            image = TVF.resize(
                image, list(self.model_image_size), antialias=True)
        return (image - self._clip_mean) / self._clip_std

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
            model = self._model_for_key(key)
            model_dtype = next(model.parameters()).dtype
            feature = model(self._prepare_image(image).to(dtype=model_dtype))
            if isinstance(feature, (tuple, list)):
                feature = feature[0]
            if feature.ndim != 2:
                feature = feature.flatten(start_dim=1)
            if feature.shape != (batch_size, self.feature_dim):
                raise RuntimeError(
                    f"CLIP visual tower returned {tuple(feature.shape)}; "
                    f"expected {(batch_size, self.feature_dim)}"
                )
            if self.normalize_features:
                feature = F.normalize(feature.float(), dim=-1)
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


def build_openai_clip_obs_encoder(shape_meta: dict, **kwargs):
    kwargs.setdefault("backend", "openai")
    return ClipObsEncoder(shape_meta=shape_meta, **kwargs)


def build_open_clip_obs_encoder(shape_meta: dict, **kwargs):
    kwargs.setdefault("backend", "open_clip")
    return ClipObsEncoder(shape_meta=shape_meta, **kwargs)


__all__ = [
    "ClipObsEncoder",
    "build_openai_clip_obs_encoder",
    "build_open_clip_obs_encoder",
]
