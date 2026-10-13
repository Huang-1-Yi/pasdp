"""Observation encoder factory dispatched by ``shape_meta`` modalities."""

from typing import Any, Dict, Optional

from diffusion_policy.model.encoders.shape_meta import (
    obs_keys_by_type,
    select_shape_meta,
)


def build_obs_encoder(
        shape_meta: dict,
        rgb_model: str = "robomimic_raw",
        pointcloud_model: str = "pointnet",
        voxel_model: str = "conv3d",
        rgb_encoder_kwargs: Optional[Dict[str, Any]] = None,
        pointcloud_encoder_kwargs: Optional[Dict[str, Any]] = None,
        voxel_encoder_kwargs: Optional[Dict[str, Any]] = None):
    """Build one encoder per present spatial modality.

    A single-modality input is returned directly. In particular, RGB-only
    configs keep the exact existing module hierarchy and checkpoint keys.
    Mixed spatial inputs use feature-level concatenation and append low-dim
    state once, after all spatial branches.
    """
    grouped = obs_keys_by_type(shape_meta)
    active = [
        modality for modality in ("rgb", "point_cloud", "voxel")
        if grouped[modality]
    ]
    if not active:
        raise ValueError(
            "shape_meta has no rgb, point_cloud, or voxel observation"
        )

    builders = {}
    if "rgb" in active:
        from diffusion_policy.model.encoders.rgb.factory import (
            build_rgb_obs_encoder,
        )
        builders["rgb"] = lambda meta, include_low_dim: build_rgb_obs_encoder(
            rgb_model=rgb_model,
            shape_meta=meta,
            encoder_kwargs=rgb_encoder_kwargs,
        )
    if "point_cloud" in active:
        from diffusion_policy.model.encoders.pointcloud.factory import (
            build_pointcloud_obs_encoder,
        )
        builders["point_cloud"] = (
            lambda meta, include_low_dim: build_pointcloud_obs_encoder(
                pointcloud_model=pointcloud_model,
                shape_meta=meta,
                encoder_kwargs={
                    **(pointcloud_encoder_kwargs or {}),
                    "include_low_dim": include_low_dim,
                },
            )
        )
    if "voxel" in active:
        from diffusion_policy.model.encoders.voxel.factory import (
            build_voxel_obs_encoder,
        )
        builders["voxel"] = lambda meta, include_low_dim: build_voxel_obs_encoder(
            voxel_model=voxel_model,
            shape_meta=meta,
            encoder_kwargs={
                **(voxel_encoder_kwargs or {}),
                "include_low_dim": include_low_dim,
            },
        )

    if len(active) == 1:
        modality = active[0]
        return builders[modality](shape_meta, True)

    from diffusion_policy.model.encoders.multimodal_obs_encoder import (
        MultiModalObsEncoder,
    )
    branches = {}
    for modality in active:
        branch_meta = select_shape_meta(
            shape_meta, (modality,), include_low_dim=False)
        branches[modality] = builders[modality](branch_meta, False)
    return MultiModalObsEncoder(shape_meta=shape_meta, encoders=branches)


__all__ = ["build_obs_encoder"]
