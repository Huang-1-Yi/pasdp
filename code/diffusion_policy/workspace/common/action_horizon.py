"""Offline action-horizon diagnostics shared across policy workspaces.

The two private implementations are mechanically extracted from the former
DP-style and PADP workspace methods. Their arithmetic, sample traversal,
normalization, reset behavior, and output fields remain unchanged.
"""

import json
import os
import time

import dill
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply


def _evaluate_batched(
    self,
    checkpoint: str,
    output_dir: str,
    device: str = None,
    split: str = "test",
    batch_size: int = None,
    max_batches: int = None,
):
    """
        Iterate the validation or test dataset and write per-position MSE.
        """
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
        try:
            self.load_payload(payload)
        except Exception as e:
            print(f"Warning: load_payload failed: {e}")

    cfg = self.cfg
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if (split == "test") and hasattr(dataset, "get_test_dataset"):
        eval_dataset = dataset.get_test_dataset()
    else:
        eval_dataset = dataset.get_validation_dataset()

    val_loader_cfg = dict(cfg.val_dataloader)
    if batch_size is not None:
        val_loader_cfg["batch_size"] = int(batch_size)
    eval_dataloader = DataLoader(eval_dataset, **val_loader_cfg)

    normalizer = dataset.get_normalizer()
    try:
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
    except Exception:
        pass

    policy = self.ema_model if cfg.training.use_ema and self.ema_model is not None else self.model

    if device is None:
        device = cfg.training.device if "training" in cfg and "device" in cfg.training else "cpu"
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    H = int(getattr(policy, "horizon", getattr(self, "horizon", None) or 0))
    if H == 0:
        H = int(cfg.task.horizon) if "task" in cfg and "horizon" in cfg.task else None
    assert H is not None and H > 0, f"Cannot determine horizon H (got {H})"

    sum_sqerr = None
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_dataloader):
            if (max_batches is not None) and (batch_idx >= int(max_batches)):
                break

            if "window_info" in batch:
                batch.pop("window_info")

            if "obs" not in batch or "action" not in batch:
                continue

            obs_dict = batch["obs"]
            gt_actions = batch["action"]
            if gt_actions.shape[1] < H:
                continue

            obs_t = dict_apply(obs_dict, lambda x: x.to(device, non_blocking=True))
            out = policy.predict_action(obs_t)
            if "action_pred" not in out:
                continue

            pred_full = out["action_pred"].detach().cpu().numpy()
            gt_np = (
                gt_actions.numpy()
                if hasattr(gt_actions, "numpy")
                else np.array(gt_actions)
            )[:, :H, :]

            per_pos_sqerr = ((pred_full - gt_np) ** 2).mean(axis=-1)

            if sum_sqerr is None:
                sum_sqerr = np.zeros(H, dtype=np.float64)
            sum_sqerr += per_pos_sqerr.sum(axis=0)
            total_count += per_pos_sqerr.shape[0]

    if total_count == 0:
        print("No samples processed in eval_2")
        return {"per_pos_mse": None, "count": 0}

    mean_mse_per_pos = (sum_sqerr / float(total_count)).astype(float)
    out_np_path = os.path.join(output_dir, "per_pos_mse.npy")
    out_json_path = os.path.join(output_dir, "per_pos_mse.json")
    np.save(out_np_path, mean_mse_per_pos)
    json.dump(
        {
            "per_pos_mse": mean_mse_per_pos.tolist(),
            "horizon": int(H),
            "count": int(total_count),
        },
        open(out_json_path, "w"),
        indent=2,
    )

    print(f"Wrote per-position MSE to {out_json_path} (n={int(total_count)})")
    return {"per_pos_mse": mean_mse_per_pos.tolist(), "count": int(total_count)}


def _evaluate_sequential(self, checkpoint: str, output_dir: str, device: str = None,
               split: str = 'test', batch_size: int = None, max_batches: int = None):
    """
        Eval mode 2: iterate dataset (test/val) and compute per-position MSE across whole dataset.
        Writes per_pos_mse.npy and per_pos_mse.json into output_dir.
        Parameters:
            checkpoint: path to checkpoint file (will be loaded if provided)
            output_dir: directory to save results
            device: device string like 'cuda:0' or 'cpu'
            split: 'test' or 'val'
            batch_size: override val_dataloader batch size if provided
            max_batches: if set, only process this many batches (useful for smoke-test)
        """
    # prepare output dir
    os.makedirs(output_dir, exist_ok=True)

    # load checkpoint payload if provided
    if checkpoint is not None:
        print(f"Loading checkpoint from {checkpoint}")
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        try:
            # try to load model weights etc into this workspace
            self.load_payload(payload)
        except Exception as e:
            print(f"Warning: load_payload failed: {e}")

    cfg = self.cfg
    # 【v4改动·B Step3】eval_2 沿用 checkpoint 里保存的 cfg.training.uint8_to_gpu，
    # 让训练时存的是什么开关，评估就走对应路径。旧 checkpoint 没此字段则 False。
    _uint8_to_gpu_eval = bool(cfg.training.get('uint8_to_gpu', False)) if hasattr(cfg.training, 'get') else False

    # dataset
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert hasattr(dataset, 'get_validation_dataset')
    if (split == 'test') and hasattr(dataset, 'get_test_dataset'):
        eval_dataset = dataset.get_test_dataset()
    else:
        eval_dataset = dataset.get_validation_dataset()

    # dataloader config
    val_loader_cfg = dict(cfg.val_dataloader)
    if batch_size is not None:
        val_loader_cfg['batch_size'] = int(batch_size)
    eval_dataloader = DataLoader(eval_dataset, **val_loader_cfg)

    # set normalizer to model(s)
    normalizer = dataset.get_normalizer()
    try:
        self.model.set_normalizer(normalizer)
        if self.ema_model is not None:
            self.ema_model.set_normalizer(normalizer)
    except Exception:
        pass

    # pick policy (use ema if configured)
    policy = self.model
    if cfg.training.use_ema and (self.ema_model is not None):
        policy = self.ema_model

    # device
    if device is None:
        device = cfg.training.device if 'training' in cfg and 'device' in cfg.training else 'cpu'
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    H = int(getattr(policy, 'horizon', getattr(self, 'horizon', None) or 0))
    if H == 0:
        # try cfg
        H = int(cfg.task.horizon) if 'task' in cfg and 'horizon' in cfg.task else None
    assert H is not None and H > 0, f"Cannot determine horizon H (got {H})"

    import numpy as _np
    from diffusion_policy.common.pytorch_util import dict_apply as _dict_apply

    sum_sqerr = _np.zeros(H, dtype=_np.float64)
    sum_gt_abs = _np.zeros(H, dtype=_np.float64)
    sum_gt_l2 = _np.zeros(H, dtype=_np.float64)
    sum_gt_power = _np.zeros(H, dtype=_np.float64)
    total_count = 0
    skipped_batches = 0
    eval_start_time = time.time()
    progress_every = 500
    total_samples_hint = len(eval_dataset) if hasattr(eval_dataset, '__len__') else None

    # track initialization (warmup) counts and timing
    init_count = 0
    total_init_time = 0.0
    reset_reason_counts = {
        'first_sample': 0,
        'episode_change': 0,
        'non_contiguous_step': 0,
        'explicit_step_zero': 0,
        'metadata_missing_fallback': 0,
    }
    prev_episode_id = None
    prev_local_pos = None

    # PADP eval_2 专用：优先直接按 episode 顺序从 replay_buffer 读取，
    # 不依赖数据集内部已存 mapping 的全局顺序。
    use_direct_episode_loading = all([
        hasattr(eval_dataset, 'replay_buffer'),
        hasattr(eval_dataset, 'episode_map'),
        hasattr(eval_dataset, 'rgb_keys'),
        hasattr(eval_dataset, 'lowdim_keys'),
        hasattr(eval_dataset, 'n_obs_steps'),
    ])
    direct_episode_ids = _np.array([], dtype=_np.int64)
    if use_direct_episode_loading:
        if hasattr(eval_dataset, 'train_episode_ids') and (eval_dataset.train_episode_ids is not None):
            direct_episode_ids = _np.array(eval_dataset.train_episode_ids, dtype=_np.int64)
        elif hasattr(eval_dataset, 'train_mask'):
            direct_episode_ids = _np.nonzero(eval_dataset.train_mask)[0].astype(_np.int64)
        else:
            # fallback: use all episodes
            direct_episode_ids = _np.arange(eval_dataset.replay_buffer.n_episodes, dtype=_np.int64)

        if direct_episode_ids.size == 0:
            use_direct_episode_loading = False
        else:
            direct_episode_ids = _np.sort(direct_episode_ids)
            try:
                total_samples_hint = int(sum(int(eval_dataset.episode_map[int(e), 5]) for e in direct_episode_ids))
            except Exception:
                pass

    with torch.no_grad():
        if use_direct_episode_loading:
            stop_eval = False
            S = int(getattr(eval_dataset, 'n_obs_steps', 1))
            rb = eval_dataset.replay_buffer
            ep_map = _np.asarray(eval_dataset.episode_map)
            for ep_idx, ep_id in enumerate(direct_episode_ids):
                if stop_eval:
                    break

                # 每个 episode 开始时重置一次推理状态
                t0 = time.time()
                try:
                    policy.reset()
                except Exception:
                    pass
                t1 = time.time()
                init_count += 1
                total_init_time += (t1 - t0)
                if total_count == 0:
                    reset_reason_counts['first_sample'] += 1
                else:
                    reset_reason_counts['episode_change'] += 1

                info = ep_map[int(ep_id)]
                buf_start_1b = int(info[3])
                win_num = int(info[5])

                prev_episode_id = int(ep_id)
                prev_local_pos = None

                for local_pos in range(win_num):
                    if (max_batches is not None) and (total_count >= int(max_batches)):
                        stop_eval = True
                        break

                    abs_start = (buf_start_1b - 1) + local_pos

                    # build one sample directly from replay buffer
                    obs_np = {}
                    # 【v4改动·B Step3】run_eval_2 走 direct replay_buffer 路径，
                    # 不经 Dataset，故这里独立按 cfg.training.uint8_to_gpu 分支：
                    #   False（默认）：CPU 内 uint8 -> float32 /255（原行为，与改前完全一致）。
                    #   True：仅 moveaxis 出 uint8 CHW，搬到 GPU 后再做 float().div_(255.)。
                    #   low_dim/gt_window 不受开关影响，始终 CPU float32。
                    if _uint8_to_gpu_eval:
                        for key in eval_dataset.rgb_keys:
                            seq = rb[key][abs_start: abs_start + S]  # (S, H, W, C) uint8
                            seq = _np.moveaxis(seq, -1, 1)  # -> (S, C, H, W) uint8
                            obs_np[key] = seq[None, ...]
                    else:
                        for key in eval_dataset.rgb_keys:
                            seq = rb[key][abs_start: abs_start + S]
                            seq = _np.moveaxis(seq, -1, 1).astype(_np.float32) / 255.0
                            obs_np[key] = seq[None, ...]
                    for key in eval_dataset.lowdim_keys:
                        seq = rb[key][abs_start: abs_start + S].astype(_np.float32)
                        obs_np[key] = seq[None, ...]

                    gt_window = rb['action'][abs_start: abs_start + H].astype(_np.float32)
                    if gt_window.shape[0] < H:
                        skipped_batches += 1
                        continue

                    sample_obs = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in obs_np.items()}

                    # 【v4改动·B Step3】uint8_to_gpu 开关打开时，rgb obs 在 GPU 上做归一化，
                    # 与训练路径数学等价（Step1 实测 max|diff|=0）。
                    if _uint8_to_gpu_eval:
                        for _k in list(sample_obs.keys()):
                            _v = sample_obs[_k]
                            if _v.dtype == torch.uint8:
                                sample_obs[_k] = _v.float().div_(255.0)

                    out = policy.predict_action(sample_obs)
                    if 'action_pred' not in out:
                        skipped_batches += 1
                        continue

                    pred_full = out['action_pred'].detach().cpu().numpy()      # [1,H,D]
                    gt_sample = gt_window[None, :H, :]                          # [1,H,D]

                    per_pos_sqerr = ((pred_full - gt_sample) ** 2).mean(axis=-1)
                    per_pos_gt_abs = _np.abs(gt_sample).mean(axis=-1)
                    per_pos_gt_l2 = _np.linalg.norm(gt_sample, axis=-1)
                    per_pos_gt_power = (gt_sample ** 2).mean(axis=-1)

                    sum_sqerr += per_pos_sqerr.sum(axis=0)
                    sum_gt_abs += per_pos_gt_abs.sum(axis=0)
                    sum_gt_l2 += per_pos_gt_l2.sum(axis=0)
                    sum_gt_power += per_pos_gt_power.sum(axis=0)
                    total_count += 1
                    prev_local_pos = local_pos

                    if (total_count % progress_every) == 0:
                        elapsed = max(time.time() - eval_start_time, 1e-9)
                        sps = total_count / elapsed
                        if total_samples_hint is not None and total_samples_hint > 0:
                            remain = max(total_samples_hint - total_count, 0)
                            eta_sec = remain / max(sps, 1e-9)
                            print(f"[eval_2] processed={total_count}/{total_samples_hint}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s, eta={eta_sec:.1f}s")
                        else:
                            print(f"[eval_2] processed={total_count}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s")
        else:
            for batch_idx, batch in enumerate(eval_dataloader):
                if (max_batches is not None) and (batch_idx >= int(max_batches)):
                    break

                # handle window_info -> extract episode_id/local_pos for natural sequential stepping
                episode_id_arr = None
                local_pos_arr = None
                if 'window_info' in batch:
                    window_info = batch.pop('window_info')
                    try:
                        wi_np = (window_info.numpy() if hasattr(window_info, 'numpy') else _np.array(window_info))
                        # supported formats:
                        # 1) SequenceSamplerRT window_info: [*, *, 5], col0=episode_id, col4=local_wid
                        # 2) PADP pos_info:            [*, *, 5], col3=episode_id, col2=episode_new_pos_id
                        if wi_np.ndim == 3 and wi_np.shape[-1] >= 5:
                            # detect PADP format by warmup flag at col0 (mostly 0/1) and 1-based buffer id at col4
                            col0 = wi_np[:, 0, 0]
                            col4 = wi_np[:, 0, 4]
                            is_padp_like = _np.all(_np.isin(col0, [0, 1])) and _np.all(col4 >= 1)
                            if is_padp_like:
                                episode_id_arr = wi_np[:, 0, 3].astype(_np.int64)
                                local_pos_arr = wi_np[:, 0, 2].astype(_np.int64)
                            else:
                                episode_id_arr = wi_np[:, 0, 0].astype(_np.int64)
                                local_pos_arr = wi_np[:, 0, 4].astype(_np.int64)
                        elif wi_np.ndim == 2 and wi_np.shape[-1] >= 5:
                            col0 = wi_np[:, 0]
                            col4 = wi_np[:, 4]
                            is_padp_like = _np.all(_np.isin(col0, [0, 1])) and _np.all(col4 >= 1)
                            if is_padp_like:
                                episode_id_arr = wi_np[:, 3].astype(_np.int64)
                                local_pos_arr = wi_np[:, 2].astype(_np.int64)
                            else:
                                episode_id_arr = wi_np[:, 0].astype(_np.int64)
                                local_pos_arr = wi_np[:, 4].astype(_np.int64)
                    except Exception:
                        episode_id_arr = None
                        local_pos_arr = None

                # prepare obs; keep gt on cpu to avoid extra transfers
                if 'obs' not in batch or 'action' not in batch:
                    print('Batch missing obs or action keys, skipping')
                    skipped_batches += 1
                    continue

                obs_dict = batch['obs']
                gt_actions = batch['action']

                # ensure gt has enough horizon
                if gt_actions.shape[1] < H:
                    print(f"Skipping batch {batch_idx} because gt length {gt_actions.shape[1]} < H {H}")
                    skipped_batches += 1
                    continue

                # move obs to device
                obs_t = _dict_apply(obs_dict, lambda x: x.to(device, non_blocking=True))

                # keep gt on cpu for numpy ops
                gt_np = (gt_actions.numpy() if hasattr(gt_actions, 'numpy') else _np.array(gt_actions))[:, :H, :]
                B = gt_np.shape[0]

                # 【v4改动·B Step3-fix】DataLoader fallback 分支（val_ratio=0 时走这里）
                # 也必须同步做 uint8_to_gpu 的 GPU 归一化，否则 uint8 obs 会直接喂给
                # normalizer(2x-1) 而数值错位。与 direct 分支、训练循环分支保持一致。
                if _uint8_to_gpu_eval:
                    for _k in list(obs_t.keys()):
                        _v = obs_t[_k]
                        if _v.dtype == torch.uint8:
                            obs_t[_k] = _v.float().div_(255.0)

                # iterate samples individually to respect per-episode initialization
                for i in range(B):
                    # determine whether to reset for natural episode stepping
                    cur_episode_id = None
                    cur_local_pos = None
                    if episode_id_arr is not None:
                        try:
                            cur_episode_id = int(episode_id_arr[i])
                        except Exception:
                            cur_episode_id = None
                    if local_pos_arr is not None:
                        try:
                            cur_local_pos = int(local_pos_arr[i])
                        except Exception:
                            cur_local_pos = None

                    need_reset = False
                    reset_reason = None
                    if total_count == 0:
                        need_reset = True
                        reset_reason = 'first_sample'
                    elif cur_episode_id is not None:
                        # episode boundary reset
                        if (prev_episode_id is None) or (cur_episode_id != prev_episode_id):
                            need_reset = True
                            reset_reason = 'episode_change'
                        # same episode but non-contiguous local step
                        elif (cur_local_pos is not None) and (prev_local_pos is not None):
                            if cur_local_pos == 0:
                                need_reset = True
                                reset_reason = 'explicit_step_zero'
                            elif cur_local_pos != (prev_local_pos + 1):
                                need_reset = True
                                reset_reason = 'non_contiguous_step'
                    elif cur_local_pos is not None:
                        # fallback when only local step is available
                        if cur_local_pos == 0:
                            need_reset = True
                            reset_reason = 'explicit_step_zero'
                        elif (prev_local_pos is not None) and (cur_local_pos != (prev_local_pos + 1)):
                            need_reset = True
                            reset_reason = 'non_contiguous_step'
                    else:
                        # no metadata: conservative fallback
                        need_reset = True
                        reset_reason = 'metadata_missing_fallback'

                    if need_reset:
                        t0 = time.time()
                        try:
                            policy.reset()
                        except Exception:
                            # best-effort: if policy has no reset, ignore
                            pass
                        t1 = time.time()
                        init_count += 1
                        total_init_time += (t1 - t0)
                        if reset_reason in reset_reason_counts:
                            reset_reason_counts[reset_reason] += 1

                    prev_episode_id = cur_episode_id if cur_episode_id is not None else prev_episode_id
                    prev_local_pos = cur_local_pos if cur_local_pos is not None else prev_local_pos

                    # build per-sample obs (keep batch dim = 1)
                    sample_obs = {}
                    for k, v in obs_t.items():
                        try:
                            sample_obs[k] = v[i:i+1]
                        except Exception:
                            sample_obs[k] = v[i:i+1]

                    out = policy.predict_action(sample_obs)
                    if 'action_pred' not in out:
                        print('policy.predict_action did not return action_pred for sample, skipping')
                        skipped_batches += 1
                        continue

                    pred_full = out['action_pred'].detach().cpu().numpy()  # [1,H,D]
                    gt_sample = gt_np[i:i+1, :H, :]

                    # compute per-sample per-position squared error (mean over action dim)
                    per_pos_sqerr = ((pred_full - gt_sample) ** 2).mean(axis=-1)  # [1,H]
                    # 统计真实动作每个位置的幅度（便于分析 MSE 是否受目标尺度影响）
                    per_pos_gt_abs = _np.abs(gt_sample).mean(axis=-1)             # [1,H]
                    per_pos_gt_l2 = _np.linalg.norm(gt_sample, axis=-1)           # [1,H]
                    # 归一化分母：每个位置真实动作的均方能量 E[a^2]
                    per_pos_gt_power = (gt_sample ** 2).mean(axis=-1)             # [1,H]

                    # accumulate
                    sum_sqerr += per_pos_sqerr.sum(axis=0)
                    sum_gt_abs += per_pos_gt_abs.sum(axis=0)
                    sum_gt_l2 += per_pos_gt_l2.sum(axis=0)
                    sum_gt_power += per_pos_gt_power.sum(axis=0)
                    total_count += 1

                    if (total_count % progress_every) == 0:
                        elapsed = max(time.time() - eval_start_time, 1e-9)
                        sps = total_count / elapsed
                        if total_samples_hint is not None and total_samples_hint > 0:
                            remain = max(total_samples_hint - total_count, 0)
                            eta_sec = remain / max(sps, 1e-9)
                            print(f"[eval_2] processed={total_count}/{total_samples_hint}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s, eta={eta_sec:.1f}s")
                        else:
                            print(f"[eval_2] processed={total_count}, skipped_batches={skipped_batches}, speed={sps:.2f} samples/s")

    if total_count == 0:
        print('No samples were processed. Exiting without writing results.')
        return {'per_pos_mse': None, 'count': 0}

    mean_mse_per_pos = (sum_sqerr / float(total_count)).astype(float)
    mean_gt_abs_per_pos = (sum_gt_abs / float(total_count)).astype(float)
    mean_gt_l2_per_pos = (sum_gt_l2 / float(total_count)).astype(float)
    mean_gt_power_per_pos = (sum_gt_power / float(total_count)).astype(float)
    nmse_eps = 1e-12
    per_pos_nmse = (mean_mse_per_pos / (mean_gt_power_per_pos + nmse_eps)).astype(float)

    # save results
    out_np_path = os.path.join(output_dir, 'per_pos_mse.npy')
    out_json_path = os.path.join(output_dir, 'per_pos_mse.json')
    _np.save(out_np_path, mean_mse_per_pos)
    # 保持原有格式，但添加位置索引信息
    result_dict = {
        'per_pos_mse': mean_mse_per_pos.tolist(),  # 保持原字段
        'per_pos_gt_abs': mean_gt_abs_per_pos.tolist(),
        'per_pos_gt_l2': mean_gt_l2_per_pos.tolist(),
        'per_pos_gt_power': mean_gt_power_per_pos.tolist(),
        'per_pos_nmse': per_pos_nmse.tolist(),
        'nmse_eps': nmse_eps,
        'count': int(total_count),
        'horizon': int(H),
        'time_steps': list(range(int(H)))  # 新增位置索引
    }

    json.dump(result_dict, open(out_json_path, 'w'), indent=2)

    # write eval log (initialization counts and timing)
    eval_log = {
        'init_count': int(init_count),
        'total_init_time': float(total_init_time),
        'avg_init_time': float(total_init_time / init_count) if init_count > 0 else 0.0,
        'reset_reason_counts': reset_reason_counts,
        'processed_samples': int(total_count),
        'skipped_batches': int(skipped_batches),
        'progress_every': int(progress_every),
        'eval_elapsed_sec': float(time.time() - eval_start_time),
        'eval_samples_per_sec': float(total_count / max(time.time() - eval_start_time, 1e-9)),
        'total_samples_hint': int(total_samples_hint) if total_samples_hint is not None else None
    }
    eval_log_path = os.path.join(output_dir, 'eval_2_log.json')
    json.dump(eval_log, open(eval_log_path, 'w'), indent=2)

    print(f"Wrote per-position MSE to {out_json_path} (n={int(total_count)})")
    return {
        'per_pos_mse': mean_mse_per_pos.tolist(),
        'per_pos_gt_abs': mean_gt_abs_per_pos.tolist(),
        'per_pos_gt_l2': mean_gt_l2_per_pos.tolist(),
        'per_pos_gt_power': mean_gt_power_per_pos.tolist(),
        'per_pos_nmse': per_pos_nmse.tolist(),
        'count': int(total_count)
    }

def evaluate_action_horizon(
    workspace,
    checkpoint: str = None,
    output_dir: str = ".",
    device: str = None,
    split: str = "test",
    batch_size: int = None,
    max_batches: int = None,
    mode: str = "auto",
):
    """Evaluate action errors without changing either legacy calculation path."""

    if mode == "auto":
        target = str(workspace.cfg.get("_target_", "")).lower()
        mode = "sequential" if "padp" in target else "batched"

    if mode == "batched":
        return _evaluate_batched(
            workspace,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=device,
            split=split,
            batch_size=batch_size,
            max_batches=max_batches,
        )
    if mode == "sequential":
        return _evaluate_sequential(
            workspace,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=device,
            split=split,
            batch_size=batch_size,
            max_batches=max_batches,
        )
    raise ValueError(
        "mode must be one of auto, batched, sequential; "
        f"got {mode!r}"
    )


__all__ = ["evaluate_action_horizon"]
