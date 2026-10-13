"""Lazy factory for RGB observation encoders.

Only the selected provider module is imported.  In particular, importing this
module does not require optional packages such as ``timm``.
"""

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Tuple


_RGB_ENCODERS: Dict[str, Tuple[str, str]] = {
    # PADP's historical wrapper layout: obs_encoder.encoder.*
    "robomimic": (
        "diffusion_policy.model.encoders.rgb.robomimic_obs_encoder",
        "RobomimicObsEncoder",
    ),
    # DP's historical raw layout: obs_encoder.*
    "robomimic_raw": (
        "diffusion_policy.model.encoders.rgb.robomimic_obs_encoder",
        "build_raw_robomimic_obs_encoder",
    ),
    "timm": (
        "diffusion_policy.model.encoders.rgb.timm_obs_encoder",
        "TimmObsEncoder",
    ),
    "r3m": (
        "diffusion_policy.model.encoders.rgb.r3m_obs_encoder",
        "R3MObsEncoder",
    ),
    "openai_clip": (
        "diffusion_policy.model.encoders.rgb.clip_obs_encoder",
        "build_openai_clip_obs_encoder",
    ),
    "open_clip": (
        "diffusion_policy.model.encoders.rgb.clip_obs_encoder",
        "build_open_clip_obs_encoder",
    ),
}

# CLASS-style string selectors backed by one provider implementation.  Keep
# experiment defaults here rather than in TimmObsEncoder so direct callers can
# still construct a neutral, from-scratch encoder.  imagenet_norm is omitted
# deliberately: TimmObsEncoder derives it from the final `pretrained` value.
_RGB_ENCODER_PRESETS: Dict[str, Tuple[str, Mapping[str, Any]]] = {
    "timm_resnet18": (
        "timm",
        {
            "model_name": "resnet18.a1_in1k",
            "pretrained": True,
            "share_rgb_model": True,
        },
    ),
    "timm_convnext_tiny": (
        "timm",
        {
            "model_name": "convnext_tiny",
            "pretrained": True,
            "share_rgb_model": True,
        },
    ),
    "timm_vit_tiny": (
        "timm",
        {
            "model_name": "vit_tiny_patch16_224",
            "pretrained": True,
            "share_rgb_model": True,
        },
    ),
    "timm_dino_vits16": (
        "timm",
        {
            "model_name": "vit_small_patch16_224.dino",
            "pretrained": True,
            "share_rgb_model": True,
        },
    ),
    "timm_dinov2_vits14": (
        "timm",
        {
            "model_name": "vit_small_patch14_dinov2.lvd142m",
            "pretrained": True,
            "share_rgb_model": True,
        },
    ),
    "timm_dinov3_vits16": (
        "timm",
        {
            "model_name": "vit_small_patch16_dinov3.lvd1689m",
            "pretrained": True,
            "share_rgb_model": True,
            # EVA PatchEmbed is strict: 84 is not divisible by patch16.
            "model_image_size": (80, 80),
        },
    ),
    "r3m_resnet18": (
        "r3m",
        {
            "model_name": "resnet18",
            "share_rgb_model": True,
        },
    ),
    "r3m_resnet34": (
        "r3m",
        {
            "model_name": "resnet34",
            "share_rgb_model": True,
        },
    ),
    "r3m_resnet50": (
        "r3m",
        {
            "model_name": "resnet50",
            "share_rgb_model": True,
        },
    ),
    "clip_vit_b32": (
        "openai_clip",
        {
            "model_name": "ViT-B/32",
            "feature_dim": 512,
            "share_rgb_model": True,
            "normalize_features": True,
        },
    ),
    "openclip_vit_b32": (
        "open_clip",
        {
            "model_name": "ViT-B-32",
            "pretrained": "laion2b_s34b_b79k",
            "feature_dim": 512,
            "share_rgb_model": True,
            "normalize_features": True,
        },
    ),
}

_INSTALL_HINTS = {
    "timm": (
        "python -m pip install --no-deps timm==1.0.28 safetensors==0.8.0"
    ),
    "r3m": (
        "python -m pip install --no-deps "
        "'git+https://github.com/facebookresearch/r3m.git@"
        "b2334e726887fa0206962d7984c69c5fb09cceab'"
    ),
    "openai_clip": (
        "python -m pip install --no-deps ftfy==6.3.1 wcwidth==0.8.2 "
        "regex==2024.11.6 "
        "'git+https://github.com/openai/CLIP.git@"
        "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'"
    ),
    "open_clip": (
        "python -m pip install --no-deps open-clip-torch==3.3.0 "
        "ftfy==6.3.1 wcwidth==0.8.2 regex==2024.11.6"
    ),
}


def available_rgb_models() -> Tuple[str, ...]:
    """Return registered model names without importing provider modules."""
    return tuple(_RGB_ENCODERS) + tuple(_RGB_ENCODER_PRESETS)


def build_rgb_obs_encoder(
        rgb_model: str,
        shape_meta: dict,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any):
    """Build one concrete encoder without adding a checkpoint wrapper layer.

    ``encoder_kwargs`` is convenient for Hydra configuration.  Additional
    keyword arguments are merged on top for direct Python callers.
    """
    requested_name = str(rgb_model).lower()
    if (requested_name not in _RGB_ENCODERS
            and requested_name not in _RGB_ENCODER_PRESETS):
        choices = ", ".join(available_rgb_models())
        raise ValueError(
            f"Unknown rgb_model={rgb_model!r}. Available models: {choices}"
        )

    provider_name = requested_name
    provider_kwargs: Dict[str, Any] = {}
    if requested_name in _RGB_ENCODER_PRESETS:
        provider_name, preset_kwargs = _RGB_ENCODER_PRESETS[requested_name]
        provider_kwargs.update(preset_kwargs)
    # Explicit config always wins over preset defaults.
    provider_kwargs.update(encoder_kwargs or {})
    provider_kwargs.update(kwargs)
    module_name, provider_object_name = _RGB_ENCODERS[provider_name]
    try:
        module = import_module(module_name)
        provider = getattr(module, provider_object_name)
        return provider(shape_meta=shape_meta, **provider_kwargs)
    except ModuleNotFoundError as exc:
        hint = _INSTALL_HINTS.get(provider_name)
        if hint is not None:
            raise ModuleNotFoundError(
                f"RGB encoder {requested_name!r} is optional and is not "
                "installed. "
                f"Install its pinned dependency group with: {hint}"
            ) from exc
        raise


__all__ = ["available_rgb_models", "build_rgb_obs_encoder"]
