"""RTV8 PADP dataset adapter for MetaWorld point-cloud Zarr files.

Mirrors ``replay_image_dataset_padp.py`` but for point-cloud + agent-pos data.
The conversion flattens episode-structured Zarr arrays into padded flat arrays
with per-epoch ``[batch_size, max_cols, 2]`` mapping tables, enabling the same
deterministic batch organisation used by the RGB PADP dataset.
"""

from typing import Dict, Optional

import os
import time

import numpy as np
import zarr
from tqdm import tqdm

import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.rtv8 import (
    RTV8CacheMessages,
    RTV8EpochStateMixin,
    load_or_build_rtv8_replay_buffer,
    resolve_mapping_index,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer


class MetaWorldPointCloudDatasetPadp(RTV8EpochStateMixin, BaseImageDataset):
    """RTV8 mapping-table dataset for MetaWorld point-cloud expert demonstrations.

    See ``RobomimicReplayImageDataset`` for the mapping-table protocol; the
    point-cloud variant replaces RGB with float32 XYZ and stores
    ``agent_pos`` instead of generic ``low_dim`` keys.
    """

    def __init__(
            self,
            zarr_path: str,
            horizon: int = 1,
            n_obs_steps: Optional[int] = None,
            n_action_steps: Optional[int] = None,
            seed: int = 42,
            val_ratio: float = 0.0,
            n_demo: int = 10,
            num_points: Optional[int] = 1024,
            point_cloud_channels: int = 3,
            batch_size: int = 1,
            mapping_dir: Optional[str] = None,
            num_epochs: Optional[int] = None,
            use_cache: bool = True,
    ):
        start_time = time.time()
        print(f"RTV8 pointcloud dataset init start: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 1
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else self.horizon
        self.n_demo = int(n_demo)
        self.batch_size = int(batch_size) if batch_size is not None else 1
        if self.batch_size <= 0:
            self.batch_size = 1
        self.num_points = int(num_points) if num_points is not None else 1024
        self.point_cloud_channels = int(point_cloud_channels)
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

        cache_zarr_path = (
            zarr_path + f".n{n_demo}.h{self.horizon}.obs{self.n_obs_steps}"
            f".act{self.n_action_steps}.bs{self.batch_size}.v8_pc.zarr"
        )
        def convert(store):
            return _convert_metaworld_to_replay_v8(
                store=store,
                zarr_path=zarr_path,
                horizon=self.horizon,
                n_demo=n_demo,
                num_points=self.num_points,
                point_cloud_channels=self.point_cloud_channels,
            )

        replay_buffer = load_or_build_rtv8_replay_buffer(
            cache_zarr_path=cache_zarr_path,
            use_cache=use_cache,
            converter=convert,
            messages=RTV8CacheMessages(
                acquiring="Acquiring lock on pointcloud RTV8 cache.",
                creating="Pointcloud RTV8 cache not found. Creating ...",
                created="Pointcloud RTV8 cache created.",
                loading="Loading pointcloud RTV8 ReplayBuffer from disk.",
            ),
        )

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask

        self.replay_buffer = replay_buffer
        self.train_mask = train_mask
        self.seed = seed

        # episode_map: [n_episodes, 6] = [ep_id, real_len, real_len_new,
        #                                 buf_start(1b), buf_end(1b), window_num]
        self.episode_map = np.asarray(
            self.replay_buffer.meta["episode_map"][:], dtype=np.int64)
        assert self.episode_map.shape[0] == self.replay_buffer.n_episodes

        self.train_episode_ids = np.nonzero(self.train_mask)[0].astype(np.int64)
        if self.train_episode_ids.size == 0:
            raise RuntimeError("No training episodes available!")

        # mapping directory
        if mapping_dir is None:
            mapping_dir = (
                os.path.splitext(zarr_path)[0]
                + f".rtdp_v8_pc_maps.n{self.n_demo}.h{self.horizon}"
                f".obs{self.n_obs_steps}.act{self.n_action_steps}"
                f".b{self.batch_size}"
            )
        os.makedirs(mapping_dir, exist_ok=True)
        self.mapping_dir = mapping_dir

        if self.num_epochs is not None and self.num_epochs > 0:
            for epk in range(self.num_epochs):
                path = self._mapping_file_path(epk)
                if not os.path.exists(path):
                    self._build_and_save_epoch_mapping(
                        epoch_index=epk, episode_ids=self.train_episode_ids)
                    print(f"Built mapping for epoch {epk}.")

        self._epoch_index = 0
        self._load_or_build_epoch_mapping(
            self._epoch_index, self.train_episode_ids)
        self._write_current_epoch_file(self._epoch_index)
        try:
            self._cached_epoch_mtime = os.stat(
                self._current_epoch_file_path()).st_mtime
        except Exception:
            self._cached_epoch_mtime = 0.0
        self._cached_epoch_value = int(self._epoch_index)
        print(f"RTV8 pointcloud dataset init done: {time.time() - start_time:.1f}s")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        data = {
            "action": np.asarray(self.replay_buffer["action"]),
            "agent_pos": np.asarray(self.replay_buffer["state"]),
            "point_cloud": np.asarray(self.replay_buffer["point_cloud"])[
                ..., :self.point_cloud_channels
            ],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.replay_buffer["action"]))

    # ------------------------------------------------------------------
    # DataLoader interface
    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._sync_epoch_mapping_from_file()
        _, _, abs_start = resolve_mapping_index(
            self._global_pairs,
            self.episode_map,
            self.batch_size,
            self._max_cols,
            idx,
        )

        S = self.n_obs_steps
        h = self.horizon

        obs_dict = {}
        pc_slice = self.replay_buffer["point_cloud"][
            abs_start: abs_start + S, :, :self.point_cloud_channels
        ].astype(np.float32)
        obs_dict["point_cloud"] = pc_slice

        state_slice = self.replay_buffer["state"][
            abs_start: abs_start + S].astype(np.float32)
        obs_dict["agent_pos"] = state_slice

        actions = self.replay_buffer["action"][
            abs_start: abs_start + h].astype(np.float32)
        pos_info = self.replay_buffer["pos_info"][
            abs_start: abs_start + h].astype(np.int64)

        out = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(actions),
            "window_info": torch.from_numpy(pos_info),
        }
        return out

    # ------------------------------------------------------------------
    # Internal: epoch mapping table
    # ------------------------------------------------------------------
# ------------------------------------------------------------------
# Conversion: physical padding + pos_info + episode_map
# ------------------------------------------------------------------
def _convert_metaworld_to_replay_v8(
        store,
        zarr_path: str,
        horizon: int = 1,
        n_demo: int = 10,
        num_points: int = 1024,
        point_cloud_channels: int = 3,
):
    start_time = time.time()
    print(f"Pointcloud RTV8 conversion start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    h = int(horizon)
    pad_b = max(h - 1, 0)
    pad_a = max(h - 1, 0)

    src = ReplayBuffer.copy_from_path(
        zarr_path, keys=("state", "action", "point_cloud"))

    n_eps = src.n_episodes
    if n_demo > n_eps:
        n_demo = n_eps

    episode_ends = np.asarray(src.meta["episode_ends"])
    real_lens = np.diff(np.concatenate([[0], episode_ends])).astype(np.int64)
    real_lens = real_lens[:n_demo]
    real_len_news = real_lens + 2 * (h - 1)
    total_padded = int(real_len_news.sum())

    root = zarr.group(store)
    data_group = root.require_group("data", overwrite=True)
    meta_group = root.require_group("meta", overwrite=True)

    meta_group.array(
        "episode_ends", np.cumsum(real_len_news), dtype=np.int64,
        compressor=None, overwrite=True)

    episode_map = np.zeros((n_demo, 6), dtype=np.int64)
    episode_map_arr = meta_group.array(
        name="episode_map",
        data=np.zeros((n_demo, 6), dtype=np.int64),
        shape=(n_demo, 6),
        chunks=(n_demo, 6),
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    state_dim = int(src["state"].shape[-1])
    action_dim = int(src["action"].shape[-1])
    pc_dim = int(src["point_cloud"].shape[-1])
    use_channels = min(pc_dim, point_cloud_channels)

    state_arr = data_group.array(
        name="state",
        data=np.empty((total_padded, state_dim), dtype=np.float32),
        shape=(total_padded, state_dim),
        chunks=(min(total_padded, max(1, 1024)), state_dim),
        compressor=None, dtype=np.float32, overwrite=True)

    action_arr = data_group.array(
        name="action",
        data=np.empty((total_padded, action_dim), dtype=np.float32),
        shape=(total_padded, action_dim),
        chunks=(min(total_padded, max(1, 1024)), action_dim),
        compressor=None, dtype=np.float32, overwrite=True)

    point_cloud_arr = data_group.array(
        name="point_cloud",
        data=np.empty(
            (total_padded, num_points, use_channels), dtype=np.float32),
        shape=(total_padded, num_points, use_channels),
        chunks=(min(total_padded, max(1, 64)), num_points, use_channels),
        compressor=None, dtype=np.float32, overwrite=True)

    pos_info_arr = data_group.array(
        name="pos_info",
        data=np.empty((total_padded, 5), dtype=np.int64),
        shape=(total_padded, 5),
        chunks=(min(total_padded, max(1, 1024)), 5),
        compressor=None, dtype=np.int64, overwrite=True)

    write_ptr = 0
    for e in tqdm(range(n_demo), desc="RTV8 pointcloud: materialize episodes"):
        start = int(np.concatenate([[0], episode_ends])[e])
        end = int(episode_ends[e])
        real_len = int(end - start)
        real_len_new = real_len + 2 * (h - 1)
        win_num = real_len + h - 1

        ofs = write_ptr
        ofs_end = ofs + real_len_new

        act = src["action"][start:end].astype(np.float32)
        st = src["state"][start:end].astype(np.float32)
        pc = src["point_cloud"][start:end, :num_points, :use_channels].astype(
            np.float32)

        # action
        if pad_b > 0:
            action_arr[ofs: ofs + pad_b] = act[0:1].repeat(pad_b, axis=0)
        action_arr[ofs + pad_b: ofs + pad_b + real_len] = act
        if pad_a > 0:
            action_arr[ofs + pad_b + real_len: ofs_end] = act[-1:].repeat(
                pad_a, axis=0)

        # state
        if pad_b > 0:
            state_arr[ofs: ofs + pad_b] = st[0:1].repeat(pad_b, axis=0)
        state_arr[ofs + pad_b: ofs + pad_b + real_len] = st
        if pad_a > 0:
            state_arr[ofs + pad_b + real_len: ofs_end] = st[-1:].repeat(
                pad_a, axis=0)

        # point_cloud
        if pad_b > 0:
            point_cloud_arr[ofs: ofs + pad_b] = pc[0:1].repeat(pad_b, axis=0)
        point_cloud_arr[ofs + pad_b: ofs + pad_b + real_len] = pc
        if pad_a > 0:
            point_cloud_arr[ofs + pad_b + real_len: ofs_end] = pc[-1:].repeat(
                pad_a, axis=0)

        # pos_info
        pos = np.zeros((real_len_new, 5), dtype=np.int64)
        if pad_b > 0:
            pos[:pad_b, 0] = 1
        pos[:, 2] = np.arange(real_len_new, dtype=np.int64)
        pos[:, 3] = e
        buf_start_1b = ofs + 1
        pos[:, 4] = buf_start_1b + pos[:, 2]
        if real_len > 0:
            pos[pad_b: pad_b + real_len, 1] = np.arange(real_len, dtype=np.int64)
            if pad_b > 0:
                pos[:pad_b, 1] = 0
            if pad_a > 0:
                pos[pad_b + real_len:, 1] = real_len - 1
        pos_info_arr[ofs: ofs_end] = pos

        episode_map[e, 0] = e
        episode_map[e, 1] = real_len
        episode_map[e, 2] = real_len_new
        episode_map[e, 3] = buf_start_1b
        episode_map[e, 4] = ofs_end
        episode_map[e, 5] = win_num
        episode_map_arr[e] = episode_map[e]

        write_ptr = ofs_end

    episode_map_arr[:, :] = episode_map

    assert write_ptr == total_padded

    if hasattr(store, "flush"):
        store.flush()

    print(f"Pointcloud RTV8 conversion done: {time.time() - start_time:.1f}s")
    return ReplayBuffer(root)


__all__ = ["MetaWorldPointCloudDatasetPadp"]
