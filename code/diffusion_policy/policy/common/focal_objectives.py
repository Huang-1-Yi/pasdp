"""Differentiable trajectory objectives used by FocalPolicy."""

import math

import torch


def orthonormal_dct_matrix(
        length: int,
        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix used along the action horizon."""
    length = int(length)
    if length <= 0:
        raise ValueError(f"DCT length must be positive, got {length}")
    sample_index = torch.arange(length, dtype=torch.float64)
    frequency_index = torch.arange(length, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(
        math.pi / length * (sample_index + 0.5) * frequency_index)
    matrix[0] *= math.sqrt(1.0 / length)
    if length > 1:
        matrix[1:] *= math.sqrt(2.0 / length)
    return matrix.to(dtype=dtype)


def dct_over_horizon(
        trajectory: torch.Tensor,
        matrix: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II to ``[B, T, D]`` along ``T``."""
    if trajectory.ndim != 3:
        raise ValueError(
            f"trajectory must have shape [B, T, D], got {tuple(trajectory.shape)}"
        )
    if matrix.shape != (trajectory.shape[1], trajectory.shape[1]):
        raise ValueError(
            f"DCT matrix {tuple(matrix.shape)} does not match horizon "
            f"{trajectory.shape[1]}"
        )
    matrix = matrix.to(device=trajectory.device, dtype=trajectory.dtype)
    return torch.einsum("kt,btd->bkd", matrix, trajectory)


__all__ = ["dct_over_horizon", "orthonormal_dct_matrix"]
