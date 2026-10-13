"""Time-pair samplers used by the published FocalPolicy implementation."""

from typing import Mapping, Optional, Tuple

import torch


def _logit_normal_sample(
        mean: float,
        std: float,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator] = None) -> torch.Tensor:
    normal = torch.randn(
        num_samples, device=device, generator=generator)
    return torch.sigmoid(normal * float(std) + float(mean)).clamp_(0.0, 1.0)


def _joint_v0(
        cfg: Mapping,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
    t = _logit_normal_sample(
        cfg["P_mean_t"], cfg["P_std_t"], num_samples, device, generator)
    r = _logit_normal_sample(
        cfg["P_mean_r"], cfg["P_std_r"], num_samples, device, generator)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    same_time = torch.rand(
        num_samples, device=device, generator=generator) < 1.0 - float(cfg["ratio"])
    r = torch.where(same_time, t, r)
    return t, r


def _joint_v1(
        cfg: Mapping,
        num_samples: int,
        device: torch.device,
        generator: Optional[torch.Generator]) -> Tuple[torch.Tensor, torch.Tensor]:
    t = _logit_normal_sample(
        cfg["P_mean_t"], cfg["P_std_t"], num_samples, device, generator)
    r = _logit_normal_sample(
        cfg["P_mean_r"], cfg["P_std_r"], num_samples, device, generator)
    same_time = torch.rand(
        num_samples, device=device, generator=generator) < 1.0 - float(cfg["ratio"])
    r = torch.where(same_time, t, r)
    return t, torch.minimum(t, r)


def sample_two_timesteps(
        cfg: Mapping,
        num_samples: int,
        device: torch.device,
        eps: float = 1e-5,
        generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Match the upstream sampler, including its independently sampled ``t``.

    Upstream obtains a joint ``(t, r)`` pair, discards that pair's ``t``, and
    returns a separate uniform ``t`` with the joint sampler's ``r``.  It is
    unusual, but is kept here as the reproduction default.  A corrected
    sampler, if desired, should be added under a different selector.
    """
    if cfg is None:
        raise ValueError("sample_cfg is required")
    sampler = str(cfg["tr_sampler"])
    t = torch.rand(
        num_samples, device=device, generator=generator)
    t = t * (1.0 - float(eps)) + float(eps)
    if sampler == "v0":
        _, r = _joint_v0(cfg, num_samples, device, generator)
    elif sampler == "v1":
        _, r = _joint_v1(cfg, num_samples, device, generator)
    else:
        raise ValueError(f"Unknown joint time sampler: {sampler!r}")
    return t, r


__all__ = ["sample_two_timesteps"]
