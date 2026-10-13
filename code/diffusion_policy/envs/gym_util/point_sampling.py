"""Point-cloud sampling backends for legacy MetaWorld and Adroit rollout.

The project's vendored PyTorch3D wheel intentionally has no C++/CUDA
extension, so importing ``pytorch3d.ops`` is not a valid default.  Open3D is
used by the legacy RGB-D reconstruction path already and is therefore the
default FPS backend.  Optional backends are imported only when selected.
"""

from typing import Optional

import numpy as np


def point_cloud_sampling(
        point_cloud: np.ndarray,
        num_points: int,
        method: str = "fps",
        fps_backend: str = "open3d",
        device: Optional[str] = None,
        rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Return exactly ``num_points`` rows while preserving extra channels."""
    points = np.asarray(point_cloud)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(
            f"point_cloud must have shape [N,C>=3], got {points.shape}")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError(f"point_cloud must be floating point, got {points.dtype}")
    num_points = int(num_points)
    if num_points <= 0:
        raise ValueError(f"num_points must be positive, got {num_points}")
    if not np.isfinite(points).all():
        raise ValueError("point_cloud contains NaN or Inf")

    n_points = int(points.shape[0])
    if n_points == 0:
        return np.zeros((num_points, points.shape[1]), dtype=points.dtype)
    if n_points < num_points:
        padding = np.zeros(
            (num_points - n_points, points.shape[1]), dtype=points.dtype)
        return np.concatenate((points, padding), axis=0)
    if n_points == num_points:
        return np.ascontiguousarray(points)

    if method == "uniform":
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.choice(n_points, num_points, replace=False)
    elif method == "fps":
        indices = _farthest_point_indices(
            points[:, :3], num_points, backend=fps_backend, device=device)
    else:
        raise ValueError(
            f"method must be 'uniform' or 'fps', got {method!r}")
    return np.ascontiguousarray(points[np.asarray(indices, dtype=np.int64)])


def _farthest_point_indices(
        xyz: np.ndarray,
        num_points: int,
        backend: str,
        device: Optional[str]) -> np.ndarray:
    if backend == "open3d":
        return _open3d_farthest_point_indices(xyz, num_points)
    if backend == "torch":
        return _torch_farthest_point_indices(xyz, num_points, device=device)
    if backend == "pytorch3d":
        return _pytorch3d_farthest_point_indices(xyz, num_points, device=device)
    raise ValueError(
        "fps_backend must be 'open3d', 'torch', or 'pytorch3d', "
        f"got {backend!r}")


def _open3d_farthest_point_indices(
        xyz: np.ndarray,
        num_points: int) -> np.ndarray:
    try:
        import open3d as o3d
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "fps_backend='open3d' requires Open3D; install the optional "
            "Focal legacy environment group") from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.asarray(xyz, dtype=np.float64))
    sampled = cloud.farthest_point_down_sample(int(num_points))
    sampled_xyz = np.asarray(sampled.points)
    if sampled_xyz.shape != (num_points, 3):
        raise RuntimeError(
            "Open3D FPS returned unexpected shape "
            f"{sampled_xyz.shape}, expected {(num_points, 3)}")
    # Open3D does not expose source indices. Exact nearest-neighbour lookup
    # restores any non-XYZ channels from the original array.
    _, indices = cKDTree(np.asarray(xyz)).query(sampled_xyz, k=1)
    return np.asarray(indices, dtype=np.int64)


def _torch_farthest_point_indices(
        xyz: np.ndarray,
        num_points: int,
        device: Optional[str]) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(xyz, dtype=torch.float32, device=device or "cpu")
    n_points = tensor.shape[0]
    indices = torch.empty(num_points, dtype=torch.long, device=tensor.device)
    min_distances = torch.full(
        (n_points,), torch.inf, dtype=tensor.dtype, device=tensor.device)
    farthest = torch.tensor(0, dtype=torch.long, device=tensor.device)
    for idx in range(num_points):
        indices[idx] = farthest
        centroid = tensor[farthest]
        distances = torch.sum((tensor - centroid) ** 2, dim=-1)
        min_distances = torch.minimum(min_distances, distances)
        farthest = torch.argmax(min_distances)
    return indices.cpu().numpy()


def _pytorch3d_farthest_point_indices(
        xyz: np.ndarray,
        num_points: int,
        device: Optional[str]) -> np.ndarray:
    try:
        import torch
        from pytorch3d.ops import sample_farthest_points
    except (ImportError, RuntimeError) as exc:
        raise ImportError(
            "fps_backend='pytorch3d' requires a PyTorch3D build with its "
            "compiled _C extension; the default no-extension wheel is not "
            "sufficient") from exc
    tensor = torch.as_tensor(
        xyz, dtype=torch.float32, device=device or "cuda:0").unsqueeze(0)
    _, indices = sample_farthest_points(
        tensor, K=int(num_points), random_start_point=False)
    return indices.squeeze(0).cpu().numpy()


__all__ = ["point_cloud_sampling"]
