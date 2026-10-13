"""PADP 流式缓冲区展开训练 (方向 A: Streaming-Unrolled Training).

继承 SlidingWindowDiffusionPolicy, 在训练时展开 K 个控制周期,
让模型处理自身前一周期预测作为输入, 对齐训练/推理输入分布。
"""
from typing import Dict, Optional
import torch
import torch.nn.functional as F

from diffusion_policy.policy.padp import SlidingWindowDiffusionPolicy


class SlidingWindowDiffusionPolicyUnrolled(SlidingWindowDiffusionPolicy):
    def __init__(self, *args,
                 rollout_steps: int = 2,
                 self_rollout_warmup_fraction: float = 0.4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_steps = rollout_steps
        self.warmup_frac = self_rollout_warmup_fraction
        self._cur_self_prob = 0.0

    def _set_self_rollout_prob(self, epoch: int, num_epochs: int):
        warmup = max(1, int(self.warmup_frac * num_epochs))
        self._cur_self_prob = min(1.0, epoch / warmup)

    @torch.no_grad()
    def _current_self_prob(self) -> float:
        return self._cur_self_prob

    def compute_loss(self, batch: Dict[str, torch.Tensor]):
        assert 'obs' in batch
        assert 'action' in batch

        nobs = self.normalize_obs(batch['obs'])
        nactions = self.normalize_action(batch['action'])
        B, full_len, D = nactions.shape
        K = self.rollout_steps
        H = self.horizon

        assert full_len >= H + K - 1, \
            f"action length {full_len} too short for H={H} K={K} (need {H+K-1})"

        total_loss_b = torch.zeros(B, device=nactions.device, dtype=nactions.dtype)
        pred_prev = None
        preds = []  # 保存每 cycle 预测 (方向 B 需要)

        for i in range(K):
            target_i = nactions[:, i:i + H]

            # ---- obs encoding per cycle (n_obs_steps=1) ----
            nobs_i: Dict[str, torch.Tensor] = {}
            for k, v in nobs.items():
                if v.ndim >= 4:  # rgb: (B, K, C, H, W)
                    nobs_i[k] = v[:, i:i + 1]
                else:  # lowdim: (B, K, D)
                    nobs_i[k] = v[:, i:i + 1]

            feats_i = self._encode_normalized_obs(nobs_i, n_steps=self.n_obs_steps)
            glob_i = feats_i.reshape(B, -1)

            # ---- buffer ----
            if i == 0:
                noise = torch.randn(target_i.shape, device=target_i.device, dtype=target_i.dtype)
                buffer = self._apply_position_noise(target_i, noise)
            else:
                use_student = (
                    self.training and
                    torch.rand(1, device=target_i.device).item() < self._cur_self_prob
                )
                if use_student:
                    x0 = torch.cat([
                        pred_prev[:, 1:].detach(),
                        torch.randn(B, 1, D, device=target_i.device, dtype=target_i.dtype),
                    ], dim=1)
                else:
                    x0 = target_i
                noise = torch.randn(target_i.shape, device=target_i.device, dtype=target_i.dtype)
                buffer = self._apply_position_noise(x0, noise)

            # ---- forward + loss ----
            pred_i = self.model(buffer, local_cond=None, global_cond=glob_i)
            pred_prev = pred_i
            preds.append(pred_i)

            mse = F.mse_loss(pred_i, target_i, reduction='none')
            w_mse = (mse * self.window_weights_h).sum(dim=1)
            loss_cycle_b = w_mse.mean(dim=-1)
            total_loss_b = total_loss_b + loss_cycle_b

        # 保存 preds 供方向 B 使用
        if K >= 2:
            self._debug_preds = preds
            self._debug_targets = [nactions[:, i:i + H] for i in range(K)]

        return total_loss_b / K
