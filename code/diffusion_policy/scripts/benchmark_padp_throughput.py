"""对比不同 Conda 环境中的 PADP 单卡训练吞吐，不执行 rollout 或保存 checkpoint。"""

import argparse
import copy
import json
import multiprocessing as mp
import pathlib
import random
import time

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def register_resolvers():
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "get_max_steps",
        lambda task_name: {
            "coffee_d0": 400,
            "stack_d1": 400,
        }.get(task_name, 400),
        replace=True,
    )
    OmegaConf.register_new_resolver("get_ws_x_center", lambda _: 0.0, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda _: 0.0, replace=True)


def train_step(model, optimizer, ema, batch_cpu, device):
    batch = {
        key: value
        for key, value in batch_cpu.items()
        if key != "window_info"
    }
    batch = dict_apply(batch, lambda tensor: tensor.to(device, non_blocking=True))
    raw_loss = model.compute_loss(batch)
    loss = raw_loss.mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ema.step(model)


def benchmark_fixed_batch(model, optimizer, ema, batch_cpu, device, warmup_steps, steps):
    for _ in range(warmup_steps):
        train_step(model, optimizer, ema, batch_cpu, device)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, optimizer, ema, batch_cpu, device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_dataloader(model, optimizer, ema, loader, device, warmup_steps, steps):
    iterator = iter(loader)
    for _ in range(warmup_steps):
        train_step(model, optimizer, ema, next(iterator), device)
    torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(steps):
        train_step(model, optimizer, ema, next(iterator), device)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return elapsed


def main():
    args = parse_args()
    mp.set_start_method("spawn", force=True)
    register_resolvers()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    root = pathlib.Path(__file__).resolve().parents[2]
    config_dir = root / "diffusion_policy" / "config"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="robomimic_padp_position_wise_v3",
            overrides=[
                "task_name=coffee_d0",
                "n_demo=200",
                "n_obs_steps=1",
                "horizon=40",
                "window_exp_gamma=0.25",
                "policy.noise_schedule_mode=positionwise",
                f"dataloader.batch_size={args.batch_size}",
                f"dataloader.num_workers={args.num_workers}",
                "policy.pred_type=sample",
                "policy.noise_scheduler.prediction_type=sample",
                "training.num_epochs=1",
            ],
        )
    OmegaConf.resolve(cfg)

    model = hydra.utils.instantiate(cfg.policy)
    ema_model = copy.deepcopy(model)
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    normalizer = dataset.get_normalizer()
    model.set_normalizer(normalizer)
    ema_model.set_normalizer(normalizer)

    loader_cfg = dict(cfg.dataloader)
    # 与真实 workspace 保持一致：v4 将线程池限制从每次 __getitem__
    # 移到了 DataLoader worker 初始化阶段。若基准遗漏此项，12 个 worker
    # 会各自启动多线程，造成严重 CPU 过度订阅并得到无效的低吞吐结果。
    if loader_cfg.get("num_workers", 0) > 0:
        loader_cfg["worker_init_fn"] = getattr(dataset, "worker_init_fn", None)
    loader = DataLoader(dataset, **loader_cfg)

    device = torch.device(cfg.training.device)
    model.to(device)
    ema_model.to(device)
    from diffusion_policy.model.common.ema_model import EMAModel

    ema: EMAModel = hydra.utils.instantiate(cfg.ema, model=ema_model)
    torch.cuda.reset_peak_memory_stats(device)

    first_batch = next(iter(loader))
    fixed_elapsed = benchmark_fixed_batch(
        model=model,
        optimizer=optimizer,
        ema=ema,
        batch_cpu=first_batch,
        device=device,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
    )
    full_elapsed = benchmark_dataloader(
        model=model,
        optimizer=optimizer,
        ema=ema,
        loader=loader,
        device=device,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
    )

    result = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "steps": args.steps,
        "fixed_batch_steps_per_second": args.steps / fixed_elapsed,
        "dataloader_steps_per_second": args.steps / full_elapsed,
        "fixed_batch_seconds": fixed_elapsed,
        "dataloader_seconds": full_elapsed,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
    }
    print("BENCH_RESULT=" + json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
