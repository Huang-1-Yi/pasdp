"""Real-image dataset adapted from task_padp/0518's RealPushTImageDataset."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile

from filelock import FileLock
import numpy as np
from omegaconf import OmegaConf
import torch
import zarr

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.envs.realword.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer

# Also register when opening an existing cache in a fresh process or worker.
register_codecs(verbose=False)


class RealImageDataset(BaseImageDataset):
    def __init__(self, shape_meta, dataset_path, horizon=1, pad_before=0,
                 pad_after=0, n_obs_steps=None, n_latency_steps=0,
                 use_cache=True, cache_dir=None, seed=42, val_ratio=0.0,
                 max_train_episodes=None, max_episodes=None, delta_action=False,
                 n_decoding_threads=2, n_encoding_threads=4):
        if delta_action:
            raise ValueError("This dataset preserves recorded absolute actions; delta_action must be false")
        self.shape_meta = (OmegaConf.to_container(shape_meta, resolve=True)
                           if OmegaConf.is_config(shape_meta) else copy.deepcopy(shape_meta))
        action_shape = tuple(self.shape_meta["action"]["shape"])
        assert action_shape in [(2,), (6,), (7,), (15,)]
        self.rgb_keys = [k for k, v in self.shape_meta["obs"].items() if v.get("type", "low_dim") == "rgb"]
        self.lowdim_keys = [k for k, v in self.shape_meta["obs"].items() if v.get("type", "low_dim") == "low_dim"]
        self.horizon = int(horizon)
        self.n_obs_steps = n_obs_steps
        self.n_latency_steps = int(n_latency_steps)
        self.pad_before, self.pad_after = int(pad_before), int(pad_after)
        if self.horizon < 1 or not 0 <= float(val_ratio) < 1:
            raise ValueError("horizon must be positive and val_ratio must be in [0, 1)")
        source = Path(dataset_path).expanduser().resolve()
        root = zarr.open_group(str(source / "replay_buffer.zarr"), mode="r")
        resolutions = {k: tuple(self.shape_meta["obs"][k]["shape"][:0:-1]) for k in self.rgb_keys}
        shapes = {k: tuple(self.shape_meta["obs"][k]["shape"]) for k in self.lowdim_keys}
        shapes["action"] = action_shape
        conversion = dict(dataset_path=str(source), out_resolutions=resolutions,
                          lowdim_keys=self.lowdim_keys + ["action"], image_keys=self.rgb_keys,
                          lowdim_shapes=shapes, max_episodes=max_episodes,
                          n_decoding_threads=n_decoding_threads, n_encoding_threads=n_encoding_threads)
        if use_cache:
            # Cache identity includes source values and videos, not just shape_meta.
            digest = hashlib.sha256(json.dumps(dict(version=1, path=str(source),
                                      shape_meta=self.shape_meta, max_episodes=max_episodes), sort_keys=True).encode())
            ends = np.asarray(root["meta/episode_ends"][:])
            if max_episodes is not None:
                ends = ends[:int(max_episodes)]
            if not len(ends):
                raise ValueError("No episodes selected")
            digest.update(ends.tobytes())
            for key in self.lowdim_keys + ["action", "timestamp"]:
                values = np.asarray(root["data"][key][:int(ends[-1])])
                digest.update(f"{key}:{values.shape}:{values.dtype}".encode())
                digest.update(values.tobytes())
            for episode in range(len(ends)):
                for key in self.rgb_keys:
                    video = source / "videos" / str(episode) / f"{int(key.rsplit('_', 1)[1])}.mp4"
                    stat = video.stat()
                    digest.update(f"{video}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else source / "image_cache"
            if cache_root == source / "replay_buffer.zarr" or (source / "replay_buffer.zarr") in cache_root.parents:
                raise ValueError("Image cache must not overwrite source Zarr")
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / (digest.hexdigest() + ".zarr")
            with FileLock(str(cache_path) + ".lock"):
                if not cache_path.exists():
                    # Only publish a fully converted cache. Failed builds stay identifiable.
                    build = Path(tempfile.mkdtemp(prefix="building-", dir=cache_root))
                    replay = real_data_to_replay_buffer(out_store=zarr.DirectoryStore(str(build)), **conversion)
                    replay.root.attrs["complete"] = True
                    build.rename(cache_path)
                cached = zarr.open_group(str(cache_path), mode="r")
                if not cached.attrs.get("complete", False):
                    raise RuntimeError(f"Incomplete image cache: {cache_path}")
                self.replay_buffer = ReplayBuffer(cached)
            self.cache_path = str(cache_path)
        else:
            self.replay_buffer = real_data_to_replay_buffer(**conversion)
            self.cache_path = None
        self.val_mask = get_val_mask(self.replay_buffer.n_episodes, float(val_ratio), seed)
        self.train_mask = downsample_mask(~self.val_mask, max_train_episodes, seed)
        if not self.train_mask.any():
            raise ValueError("No training episodes selected")
        self.sampler = self._sampler(self.train_mask)
        if not len(self.sampler):
            raise ValueError("No training windows; adjust horizon or padding")

    def _sampler(self, mask):
        first_k = {} if self.n_obs_steps is None else {
            k: int(self.n_obs_steps) for k in self.rgb_keys + self.lowdim_keys}
        return SequenceSampler(self.replay_buffer, self.horizon + self.n_latency_steps,
                               self.pad_before, self.pad_after, episode_mask=mask,
                               key_first_k=first_k)

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._sampler(self.val_mask)
        return result

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        ends = self.replay_buffer.episode_ends[:]
        starts = np.r_[0, ends[:-1]]
        for key in ["action"] + self.lowdim_keys:
            # Fit only training episodes; validation data does not affect scaling.
            values = np.concatenate([self.replay_buffer[key][int(a):int(b)]
                                     for a, b, take in zip(starts, ends, self.train_mask) if take])
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(values, **kwargs)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self.replay_buffer["action"][:])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        data = self.sampler.sample_sequence(index)
        selection = slice(self.n_obs_steps)
        obs = {key: np.moveaxis(data[key][selection], -1, 1).astype(np.float32) / 255.
               for key in self.rgb_keys}
        obs.update({key: data[key][selection].astype(np.float32) for key in self.lowdim_keys})
        action = data["action"][self.n_latency_steps:].astype(np.float32)
        return {"obs": dict_apply(obs, torch.from_numpy), "action": torch.from_numpy(action)}
