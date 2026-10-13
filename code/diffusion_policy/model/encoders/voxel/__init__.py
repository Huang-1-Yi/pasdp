"""Voxel observation encoders."""

from diffusion_policy.model.encoders.voxel.factory import (
    available_voxel_models,
    build_voxel_obs_encoder,
)

__all__ = [
    "Conv3dBackbone",
    "VoxelObsEncoder",
    "available_voxel_models",
    "build_voxel_obs_encoder",
]


def __getattr__(name):
    if name in {"Conv3dBackbone", "VoxelObsEncoder"}:
        from diffusion_policy.model.encoders.voxel.conv3d_obs_encoder import (
            Conv3dBackbone,
            VoxelObsEncoder,
        )
        return {
            "Conv3dBackbone": Conv3dBackbone,
            "VoxelObsEncoder": VoxelObsEncoder,
        }[name]
    raise AttributeError(name)
