"""Point-cloud observation encoders."""

from diffusion_policy.model.encoders.pointcloud.factory import (
    available_pointcloud_models,
    build_pointcloud_obs_encoder,
)

__all__ = [
    "FocalPointCloudObsEncoder",
    "FocalPointNetXYZ",
    "FocalPointNetXYZRGB",
    "DP3Encoder",
    "PointCloudObsEncoder",
    "PointNetBackbone",
    "available_pointcloud_models",
    "build_pointcloud_obs_encoder",
]


def __getattr__(name):
    if name == "DP3Encoder":
        from diffusion_policy.model.encoders.pointcloud.dp3_encoder import (
            DP3Encoder,
        )
        return DP3Encoder
    if name in {
            "FocalPointCloudObsEncoder",
            "FocalPointNetXYZ",
            "FocalPointNetXYZRGB"}:
        from diffusion_policy.model.encoders.pointcloud.focal_obs_encoder import (
            FocalPointCloudObsEncoder,
            FocalPointNetXYZ,
            FocalPointNetXYZRGB,
        )
        return {
            "FocalPointCloudObsEncoder": FocalPointCloudObsEncoder,
            "FocalPointNetXYZ": FocalPointNetXYZ,
            "FocalPointNetXYZRGB": FocalPointNetXYZRGB,
        }[name]
    if name in {"PointCloudObsEncoder", "PointNetBackbone"}:
        from diffusion_policy.model.encoders.pointcloud.pointnet_obs_encoder import (
            PointCloudObsEncoder,
            PointNetBackbone,
        )
        return {
            "PointCloudObsEncoder": PointCloudObsEncoder,
            "PointNetBackbone": PointNetBackbone,
        }[name]
    raise AttributeError(name)
