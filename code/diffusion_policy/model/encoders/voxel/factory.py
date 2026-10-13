"""Lazy factory for voxel observation encoders."""

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Tuple


_VOXEL_ENCODERS: Dict[str, Tuple[str, str]] = {
    "conv3d": (
        "diffusion_policy.model.encoders.voxel.conv3d_obs_encoder",
        "VoxelObsEncoder",
    ),
}

_VOXEL_ENCODER_PRESETS: Dict[str, Tuple[str, Mapping[str, Any]]] = {
    "voxel_cnn": ("conv3d", {}),
    "voxel_cnn_small": (
        "conv3d",
        {"channels": (16, 32, 64), "feature_dim": 128},
    ),
}


def available_voxel_models() -> Tuple[str, ...]:
    return tuple(_VOXEL_ENCODERS) + tuple(_VOXEL_ENCODER_PRESETS)


def build_voxel_obs_encoder(
        voxel_model: str,
        shape_meta: dict,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any):
    requested_name = str(voxel_model).lower()
    if (requested_name not in _VOXEL_ENCODERS
            and requested_name not in _VOXEL_ENCODER_PRESETS):
        choices = ", ".join(available_voxel_models())
        raise ValueError(
            f"Unknown voxel_model={voxel_model!r}. Available models: {choices}"
        )

    provider_name = requested_name
    provider_kwargs: Dict[str, Any] = {}
    if requested_name in _VOXEL_ENCODER_PRESETS:
        provider_name, preset_kwargs = _VOXEL_ENCODER_PRESETS[requested_name]
        provider_kwargs.update(preset_kwargs)
    provider_kwargs.update(encoder_kwargs or {})
    provider_kwargs.update(kwargs)

    module_name, object_name = _VOXEL_ENCODERS[provider_name]
    provider = getattr(import_module(module_name), object_name)
    return provider(shape_meta=shape_meta, **provider_kwargs)


__all__ = ["available_voxel_models", "build_voxel_obs_encoder"]
