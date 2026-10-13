"""Utilities for validating and selecting observation modalities."""

from copy import deepcopy
from typing import Dict, Iterable, Mapping, Tuple


_OBS_TYPE_ALIASES = {
    "rgb": "rgb",
    "image": "rgb",
    "low_dim": "low_dim",
    "lowdim": "low_dim",
    "point_cloud": "point_cloud",
    "pointcloud": "point_cloud",
    "pc": "point_cloud",
    "voxel": "voxel",
    "voxels": "voxel",
}


def canonical_obs_type(value: object) -> str:
    """Return the canonical shape-meta observation type.

    Missing ``type`` fields are handled by :func:`obs_type_for_attr` and keep
    the historical Diffusion Policy convention of meaning ``low_dim``.
    """
    key = str(value).strip().lower()
    try:
        return _OBS_TYPE_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(set(_OBS_TYPE_ALIASES.values())))
        raise ValueError(
            f"Unsupported observation type {value!r}; supported: {choices}"
        ) from exc


def obs_type_for_attr(attr: Mapping) -> str:
    return canonical_obs_type(attr.get("type", "low_dim"))


def obs_keys_by_type(shape_meta: Mapping) -> Dict[str, Tuple[str, ...]]:
    """Validate ``shape_meta`` and return sorted, non-ignored keys by type."""
    if "obs" not in shape_meta or not isinstance(shape_meta["obs"], Mapping):
        raise ValueError("shape_meta must contain an 'obs' mapping")

    grouped: Dict[str, list] = {
        "rgb": [],
        "point_cloud": [],
        "voxel": [],
        "low_dim": [],
    }
    for key, attr in shape_meta["obs"].items():
        if not isinstance(attr, Mapping) or "shape" not in attr:
            raise ValueError(f"Observation {key!r} must define a shape")
        shape = tuple(int(x) for x in attr["shape"])
        if not shape or min(shape) <= 0:
            raise ValueError(
                f"Observation {key!r} must have a non-empty positive shape, "
                f"got {shape}"
            )
        if attr.get("ignore_by_policy", False):
            continue
        grouped[obs_type_for_attr(attr)].append(str(key))
    return {key: tuple(sorted(value)) for key, value in grouped.items()}


def select_shape_meta(
        shape_meta: Mapping,
        obs_types: Iterable[str],
        include_low_dim: bool = False) -> dict:
    """Copy ``shape_meta`` while retaining only the requested modalities."""
    selected_types = {canonical_obs_type(value) for value in obs_types}
    if include_low_dim:
        selected_types.add("low_dim")

    result = {
        key: deepcopy(value)
        for key, value in shape_meta.items()
        if key != "obs"
    }
    result["obs"] = {
        key: deepcopy(attr)
        for key, attr in shape_meta["obs"].items()
        if (not attr.get("ignore_by_policy", False)
            and obs_type_for_attr(attr) in selected_types)
    }
    return result


__all__ = [
    "canonical_obs_type",
    "obs_keys_by_type",
    "obs_type_for_attr",
    "select_shape_meta",
]
