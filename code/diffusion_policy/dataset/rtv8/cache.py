"""Shared RTV8 ReplayBuffer cache lifecycle."""

from dataclasses import dataclass
from typing import Callable

import os
import shutil

from filelock import FileLock
import zarr

from diffusion_policy.common.replay_buffer import ReplayBuffer


@dataclass(frozen=True)
class RTV8CacheMessages:
    acquiring: str
    creating: str
    created: str
    loading: str
    loaded: str = "Loaded!"


def load_or_build_rtv8_replay_buffer(
        cache_zarr_path: str,
        use_cache: bool,
        converter: Callable,
        messages: RTV8CacheMessages,
):
    """Run the legacy lock/build/load/memory-store sequence exactly once."""

    if not use_cache:
        return converter(zarr.MemoryStore())

    cache_lock_path = cache_zarr_path + ".rtv8.lock"
    print(messages.acquiring)
    with FileLock(cache_lock_path):
        if not os.path.exists(cache_zarr_path):
            print(messages.creating)
            try:
                store = zarr.DirectoryStore(cache_zarr_path)
                replay_buffer = converter(store)
                print(messages.created)
                return replay_buffer
            except Exception:
                if os.path.exists(cache_zarr_path):
                    shutil.rmtree(cache_zarr_path)
                raise

        print(messages.loading)
        store = zarr.DirectoryStore(cache_zarr_path)
        replay_buffer = ReplayBuffer(zarr.group(store))
        print(messages.loaded)
        return replay_buffer


__all__ = ["RTV8CacheMessages", "load_or_build_rtv8_replay_buffer"]
