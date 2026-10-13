"""RGB 图像与配套低维 observation 编码组件。"""

from diffusion_policy.model.encoders.rgb.factory import (
    available_rgb_models,
    build_rgb_obs_encoder,
)

__all__ = [
    "RobomimicObsEncoder",
    "available_rgb_models",
    "build_rgb_obs_encoder",
]


def __getattr__(name):
    """Preserve the package-level legacy export without eager imports."""
    if name == "RobomimicObsEncoder":
        from diffusion_policy.model.encoders.rgb.robomimic_obs_encoder import (
            RobomimicObsEncoder,
        )
        return RobomimicObsEncoder
    raise AttributeError(name)
