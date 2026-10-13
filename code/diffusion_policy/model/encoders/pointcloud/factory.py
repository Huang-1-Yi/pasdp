"""Lazy factory for point-cloud observation encoders."""

from importlib import import_module
from typing import Any, Dict, Mapping, Optional, Tuple


_POINTCLOUD_ENCODERS: Dict[str, Tuple[str, str]] = {
    "pointnet": (
        "diffusion_policy.model.encoders.pointcloud.pointnet_obs_encoder",
        "PointCloudObsEncoder",
    ),
    "pointnet_focal": (
        "diffusion_policy.model.encoders.pointcloud.focal_obs_encoder",
        "FocalPointCloudObsEncoder",
    ),
}

_POINTCLOUD_ENCODER_PRESETS: Dict[str, Tuple[str, Mapping[str, Any]]] = {
    "pointnet_xyz": ("pointnet", {"input_channels": 3}),
    "pointnet_xyzrgb": ("pointnet", {"input_channels": 6}),
    "pointnet_focal_xyz": ("pointnet_focal", {"use_pc_color": False}),
    "pointnet_focal_xyzrgb": ("pointnet_focal", {"use_pc_color": True}),
}


def available_pointcloud_models() -> Tuple[str, ...]:
    return tuple(_POINTCLOUD_ENCODERS) + tuple(_POINTCLOUD_ENCODER_PRESETS)


def build_pointcloud_obs_encoder(
        pointcloud_model: str,
        shape_meta: dict,
        encoder_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any):
    requested_name = str(pointcloud_model).lower()
    if (requested_name not in _POINTCLOUD_ENCODERS
            and requested_name not in _POINTCLOUD_ENCODER_PRESETS):
        choices = ", ".join(available_pointcloud_models())
        raise ValueError(
            f"Unknown pointcloud_model={pointcloud_model!r}. "
            f"Available models: {choices}"
        )

    provider_name = requested_name
    provider_kwargs: Dict[str, Any] = {}
    if requested_name in _POINTCLOUD_ENCODER_PRESETS:
        provider_name, preset_kwargs = _POINTCLOUD_ENCODER_PRESETS[requested_name]
        provider_kwargs.update(preset_kwargs)
    provider_kwargs.update(encoder_kwargs or {})
    provider_kwargs.update(kwargs)

    module_name, object_name = _POINTCLOUD_ENCODERS[provider_name]
    provider = getattr(import_module(module_name), object_name)
    return provider(shape_meta=shape_meta, **provider_kwargs)


__all__ = ["available_pointcloud_models", "build_pointcloud_obs_encoder"]
