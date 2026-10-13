"""Shared RTV8 epoch-file state without sample decoding policy."""

from typing import Optional

import copy
import os
import time

import numpy as np
from threadpoolctl import threadpool_limits

from diffusion_policy.dataset.rtv8.mapping_table import (
    build_balanced_epoch_pairs,
)


class RTV8EpochStateMixin:
    """Mechanically shared epoch state used by all RTV8 suite adapters.

    Subclasses continue to implement ``_load_or_build_epoch_mapping`` so that
    their existing cache validation rules remain unchanged.
    """

    def set_epoch(self, epoch_index: int):
        self._epoch_index = int(epoch_index)
        self._load_or_build_epoch_mapping(
            self._epoch_index, self.train_episode_ids)
        self._write_current_epoch_file(self._epoch_index)
        try:
            self._cached_epoch_mtime = os.stat(
                self._current_epoch_file_path()).st_mtime
        except Exception:
            self._cached_epoch_mtime = 0.0
        self._cached_epoch_value = int(self._epoch_index)

    @staticmethod
    def worker_init_fn(worker_id):
        threadpool_limits(1)

    def __len__(self):
        return int(self.batch_size * self._max_cols)

    def get_validation_dataset(self):
        """Create the legacy shallow validation view used by point clouds."""

        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_episode_ids = np.nonzero(
            val_set.train_mask)[0].astype(np.int64)
        if val_set.train_episode_ids.size == 0:
            return val_set
        val_set.mapping_dir = self.mapping_dir + ".val"
        os.makedirs(val_set.mapping_dir, exist_ok=True)
        val_set._load_or_build_epoch_mapping(
            0, val_set.train_episode_ids)
        return val_set

    def _mapping_file_path(self, epoch_index: int) -> str:
        return os.path.join(
            self.mapping_dir, f"mapping_epoch_{epoch_index:05d}.npz")

    def _current_epoch_file_path(self) -> str:
        return os.path.join(self.mapping_dir, "current_epoch.txt")

    def _load_or_build_epoch_mapping(
            self, epoch_index: int, episode_ids: np.ndarray):
        """Load the legacy point-cloud archive without RGB fingerprinting."""

        path = self._mapping_file_path(epoch_index)
        if not os.path.exists(path):
            print(f"Mapping for epoch {epoch_index} not found. Building...")
            self._build_and_save_epoch_mapping(epoch_index, episode_ids)

        data = np.load(path)
        pairs = data["pairs"]
        B_file = int(data["batch_size"])
        max_cols = int(data["max_cols"])

        if B_file != self.batch_size:
            raise RuntimeError(
                f"Mapping batch_size={B_file} != dataset batch_size={self.batch_size}")

        self._global_pairs = pairs.astype(np.int32)
        self._max_cols = max_cols

    def _build_and_save_epoch_mapping(
            self, epoch_index: int, episode_ids: np.ndarray):
        """Write the legacy point-cloud mapping archive."""

        start_time = time.time()
        print(f"Building epoch mapping {epoch_index} ...")
        pairs, max_cols = build_balanced_epoch_pairs(
            episode_map=self.episode_map,
            episode_ids=episode_ids,
            batch_size=self.batch_size,
            seed=self.seed,
            epoch_index=epoch_index,
        )
        path = self._mapping_file_path(epoch_index)
        np.savez_compressed(
            path,
            pairs=pairs,
            batch_size=self.batch_size,
            max_cols=max_cols,
        )
        print(
            f"Epoch mapping {epoch_index} built in "
            f"{time.time() - start_time:.1f}s")

    def _write_current_epoch_file(self, epoch_index: int):
        try:
            with open(self._current_epoch_file_path(), "w") as f:
                f.write(str(int(epoch_index)))
        except Exception:
            pass

    def _read_current_epoch_fast(self) -> Optional[int]:
        path = self._current_epoch_file_path()
        try:
            mtime = os.stat(path).st_mtime
        except Exception:
            return None
        if mtime == getattr(self, "_cached_epoch_mtime", None):
            return getattr(self, "_cached_epoch_value", None)
        try:
            with open(path, "r") as f:
                txt = f.read().strip()
            if txt:
                val = int(txt)
                self._cached_epoch_mtime = mtime
                self._cached_epoch_value = val
                return val
        except Exception:
            return None
        return None

    def _sync_epoch_mapping_from_file(self) -> None:
        new_epoch = self._read_current_epoch_fast()
        if (new_epoch is not None) and (
                new_epoch != getattr(self, "_epoch_index", None)):
            self._epoch_index = new_epoch
            self._load_or_build_epoch_mapping(
                self._epoch_index, self.train_episode_ids)


__all__ = ["RTV8EpochStateMixin"]
