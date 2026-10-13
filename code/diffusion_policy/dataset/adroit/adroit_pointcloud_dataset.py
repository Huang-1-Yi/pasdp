"""Point-cloud dataset adapter for the FocalPolicy Adroit Zarr files."""

import copy
from typing import Dict, Optional

import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


_TASK_DIMS = {
    "door": (24, 28),
    "hammer": (24, 26),
    "pen": (24, 24),
}


class AdroitPointCloudDataset(BaseImageDataset):
    """Read state, action and XYZ point clouds from an Adroit Zarr dataset.

    The migrated FocalPolicy datasets store 512 XYZRGB points per frame.  The
    first three channels are selected by default because the online Adroit
    wrapper and ``pointnet_focal_xyz`` encoder both expose XYZ only.
    """

    def __init__(
            self,
            zarr_path: str,
            horizon: int = 1,
            n_obs_steps: Optional[int] = None,
            pad_before: int = 0,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.0,
            max_train_episodes: Optional[int] = None,
            task_name: Optional[str] = None,
            num_points: Optional[int] = 512,
            point_cloud_channels: int = 3,
            expected_state_dim: Optional[int] = None,
            expected_action_dim: Optional[int] = None):
        super().__init__()
        if point_cloud_channels not in (3, 6):
            raise ValueError(
                "point_cloud_channels must be 3 (XYZ) or 6 (XYZRGB), got "
                f"{point_cloud_channels}"
            )
        if task_name is not None and task_name not in _TASK_DIMS:
            raise ValueError(
                f"unsupported Adroit task_name={task_name!r}; "
                f"expected one of {tuple(_TASK_DIMS)}"
            )

        replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=("state", "action", "point_cloud"))
        source_points = int(replay_buffer["point_cloud"].shape[-2])
        source_channels = int(replay_buffer["point_cloud"].shape[-1])
        if num_points is not None and source_points != int(num_points):
            raise ValueError(
                f"source point cloud has {source_points} points per frame, "
                f"but num_points={num_points}; preprocess explicitly instead "
                "of silently changing the training distribution"
            )
        if source_channels < point_cloud_channels:
            raise ValueError(
                f"source point cloud has {source_channels} channels, but "
                f"{point_cloud_channels} were requested"
            )

        if task_name is not None:
            task_state_dim, task_action_dim = _TASK_DIMS[task_name]
            if expected_state_dim is not None and expected_state_dim != task_state_dim:
                raise ValueError(
                    f"task_name={task_name!r} requires expected_state_dim="
                    f"{task_state_dim}, got {expected_state_dim}"
                )
            if expected_action_dim is not None and expected_action_dim != task_action_dim:
                raise ValueError(
                    f"task_name={task_name!r} requires expected_action_dim="
                    f"{task_action_dim}, got {expected_action_dim}"
                )
            expected_state_dim = task_state_dim
            expected_action_dim = task_action_dim

        state_dim = int(replay_buffer["state"].shape[-1])
        action_dim = int(replay_buffer["action"].shape[-1])
        if task_name is not None and (
                state_dim, action_dim) != (task_state_dim, task_action_dim):
            raise ValueError(
                f"Adroit {task_name} expects state/action dimensions "
                f"{task_state_dim}/{task_action_dim}, but dataset contains "
                f"{state_dim}/{action_dim}"
            )
        if expected_state_dim is not None and state_dim != int(expected_state_dim):
            raise ValueError(
                f"expected_state_dim={expected_state_dim}, but dataset contains "
                f"state_dim={state_dim}"
            )
        if expected_action_dim is not None and action_dim != int(expected_action_dim):
            raise ValueError(
                f"expected_action_dim={expected_action_dim}, but dataset contains "
                f"action_dim={action_dim}"
            )

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = downsample_mask(
            mask=~val_mask,
            max_n=max_train_episodes,
            seed=seed,
        )
        key_first_k = None
        if n_obs_steps is not None:
            key_first_k = {
                "state": int(n_obs_steps),
                "point_cloud": int(n_obs_steps),
            }

        self.replay_buffer = replay_buffer
        self.sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )
        self.train_mask = train_mask
        self.horizon = int(horizon)
        self.n_obs_steps = n_obs_steps
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        self.task_name = task_name
        self.point_cloud_channels = int(point_cloud_channels)
        self.num_points = source_points

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        key_first_k = None
        if self.n_obs_steps is not None:
            key_first_k = {
                "state": int(self.n_obs_steps),
                "point_cloud": int(self.n_obs_steps),
            }
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=key_first_k,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs):
        data = {
            "action": self.replay_buffer["action"],
            "agent_pos": self.replay_buffer["state"],
            "point_cloud": self.replay_buffer["point_cloud"][
                ..., :self.point_cloud_channels
            ],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer["action"]))

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        obs_slice = slice(self.n_obs_steps)
        data = {
            "obs": {
                "point_cloud": sample["point_cloud"][
                    obs_slice, ..., :self.point_cloud_channels
                ].astype(np.float32),
                "agent_pos": sample["state"][obs_slice].astype(np.float32),
            },
            "action": sample["action"].astype(np.float32),
        }
        return dict_apply(data, torch.from_numpy)


__all__ = ["AdroitPointCloudDataset"]
