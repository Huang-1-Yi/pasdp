"""PADP 展开训练 dataset: 返回 K 个连续观测 + (H+K-1) 帧连续动作。
继承 RobomimicReplayImageDataset (RTV8 padded zarr), 只改 __getitem__。
"""
from typing import Dict, List, Optional
import numpy as np
import torch

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.rtv8 import resolve_mapping_index
from diffusion_policy.dataset.robomimic.replay_image_dataset_padp import (
    RobomimicReplayImageDataset,
)


class RobomimicReplayImageDatasetUnrolled(RobomimicReplayImageDataset):
    """与 RobomimicReplayImageDataset 完全相同的 mapping/zarr 后端,
    仅 __getitem__ 多取 (K-1) 个 obs 和 (K-1) 帧 action, 支持 K 步展开训练。"""

    def __init__(self, *args, rollout_steps: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.K = int(rollout_steps)
        assert self.K >= 1, f"rollout_steps must be >= 1, got {self.K}"
        if self.K > 1:
            print(f"[UnrolledDataset] K={self.K}: each sample returns "
                  f"{self.K} obs frames + {self.horizon + self.K - 1} action steps")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        e, _, abs_start = resolve_mapping_index(
            self._global_pairs,
            self.episode_map,
            self.batch_size,
            self._max_cols,
            idx,
        )
        info = self.episode_map[e]

        h = self.horizon

        # ---- 1. obs: 取 K 帧 (每帧 n_obs_steps=1) ----
        obs_dict = {}
        K = self.K
        n_pad_end = int(info[2])  # real_len_new = real_len + 2*(h-1)
        buf_end_0b = buf_start_1b - 1 + n_pad_end - 1  # zero-based last valid index

        if self.uint8_to_gpu:
            for key in self.rgb_keys:
                # 安全截断: 不超出 zarr padding 范围
                end = min(abs_start + K, buf_end_0b + 1)
                seq = self.replay_buffer[key][abs_start:end]  # (len, H, W, C) uint8
                if end - abs_start < K:
                    pad_len = K - (end - abs_start)
                    last_frame = seq[-1:].repeat(pad_len, axis=0)
                    seq = np.concatenate([seq, last_frame], axis=0)
                seq = np.moveaxis(seq, -1, 1)  # -> (K, C, H, W) uint8
                obs_dict[key] = seq
        else:
            for key in self.rgb_keys:
                end = min(abs_start + K, buf_end_0b + 1)
                seq = self.replay_buffer[key][abs_start:end]
                if end - abs_start < K:
                    pad_len = K - (end - abs_start)
                    last_frame = seq[-1:].repeat(pad_len, axis=0)
                    seq = np.concatenate([seq, last_frame], axis=0)
                seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0
                obs_dict[key] = seq

        for key in self.lowdim_keys:
            end = min(abs_start + K, buf_end_0b + 1)
            seq = self.replay_buffer[key][abs_start:end].astype(np.float32)
            if end - abs_start < K:
                pad_len = K - (end - abs_start)
                last_frame = seq[-1:].repeat(pad_len, axis=0)
                seq = np.concatenate([seq, last_frame], axis=0)
            obs_dict[key] = seq

        # ---- 2. action: 取 H+K-1 帧 ----
        act_len = h + K - 1
        act_end = min(abs_start + act_len, buf_end_0b + 1)
        actions = self.replay_buffer['action'][abs_start:act_end].astype(np.float32)
        if act_end - abs_start < act_len:
            pad_a_len = act_len - (act_end - abs_start)
            last_act = actions[-1:].repeat(pad_a_len, axis=0)
            actions = np.concatenate([actions, last_act], axis=0)

        # ---- 3. pos_info: 只保留 cycle 0 的 H 帧 ----
        pos_end = min(abs_start + h, buf_end_0b + 1)
        pos_info = self.replay_buffer['pos_info'][abs_start:pos_end].astype(np.int64)
        if pos_end - abs_start < h:
            pad_p_len = h - (pos_end - abs_start)
            last_pos = pos_info[-1:].repeat(pad_p_len, axis=0)
            pos_info = np.concatenate([pos_info, last_pos], axis=0)

        out = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions),
            'window_info': torch.from_numpy(pos_info),
        }
        return out
