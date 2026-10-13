#!/usr/bin/env python3
"""
Run the repo's original eval flow (env_runner.run(policy)) but additionally
record the pure model forward time (inside policy.model.forward) vs the
total policy time recorded by the runner. Writes timing metrics to
output_dir/timing_log.json.

Usage:
  python3 scripts/benchmark_eval_actual_timing.py \
      --checkpoint /path/to/ckpt.ckpt \
      --output_dir outputs/bench_myrun \
      --device cuda:0

This keeps the original evaluation behaviour (videos, metrics, etc.) but
monkeypatches the model forward to accumulate model-only time so we can
report eval_overhead = total_policy_time - model_forward_time.
"""
import argparse
import os
import json
import time
import pathlib
import dill
import torch
import hydra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--n_test', type=int, default=None,
                        help='If set, override cfg.task.env_runner.n_test')
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load checkpoint like eval_sim_rollout.py
    payload = torch.load(open(args.checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=str(out_dir))
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if getattr(cfg.training, 'use_ema', False):
        # if training indicated to use ema, allow override
        policy = getattr(workspace, 'ema_model', policy)

    # optionally override runner n_test from CLI
    if args.n_test is not None:
        try:
            cfg.task.env_runner.n_test = int(args.n_test)
        except Exception:
            pass

    # align some defaults used in eval_sim_rollout.py
    cfg.task.env_runner.n_envs = getattr(cfg.task.env_runner, 'n_envs', 32)
    cfg.task.env_runner.n_train = getattr(cfg.task.env_runner, 'n_train', 32)
    cfg.task.env_runner.n_train_vis = getattr(cfg.task.env_runner, 'n_train_vis', 6)
    if args.n_test is None:
        cfg.task.env_runner.n_test = getattr(cfg.task.env_runner, 'n_test', 200)
    cfg.task.env_runner.n_test_vis = getattr(cfg.task.env_runner, 'n_test_vis', 6)

    device = torch.device(args.device)
    policy.to(device)
    policy.eval()

    # instrumentation counters
    total_model_forward_time = 0.0
    model_forward_calls = 0

    # Helper to wrap model.forward if present
    model = getattr(policy, 'model', None)
    orig_forward = None
    wrapped = False
    if model is not None:
        # try to wrap forward method
        if hasattr(model, 'forward') and callable(getattr(model, 'forward')):
            orig_forward = model.forward

            def make_wrapped(orig):
                def wrapped_forward(*args, **kwargs):
                    nonlocal total_model_forward_time, model_forward_calls
                    # sync for accurate timing on CUDA
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    out = orig(*args, **kwargs)
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    total_model_forward_time += (t1 - t0)
                    model_forward_calls += 1
                    return out
                return wrapped_forward

            model.forward = make_wrapped(orig_forward)
            wrapped = True
        else:
            # best-effort: many models are nn.Module and use forward; if not,
            # we can try to wrap __call__ but that's less safe. Skip if not found.
            wrapped = False

    # instantiate env_runner and run (keeps original eval flow)
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=str(out_dir))

    print(f"Starting eval with checkpoint={args.checkpoint} on device={device}")
    runner_log = env_runner.run(policy)

    # collect runner timing that env_runner reports (policy/avg_inference_time_ms)
    reported_avg_ms = runner_log.get('policy/avg_inference_time_ms', None)

    # compute model forward avg ms
    avg_model_ms = None
    avg_overhead_ms = None
    if model_forward_calls > 0:
        avg_model_ms = (total_model_forward_time / model_forward_calls) * 1000.0
        if reported_avg_ms is not None:
            avg_overhead_ms = float(reported_avg_ms) - avg_model_ms

    # extend runner_log with our timing
    runner_log['policy/avg_model_forward_time_ms'] = avg_model_ms
    runner_log['policy/model_forward_calls'] = model_forward_calls
    runner_log['policy/avg_eval_overhead_ms'] = avg_overhead_ms

    # save timing json
    timing_path = out_dir.joinpath('timing_log.json')
    with open(timing_path, 'w') as f:
        json.dump(runner_log, f, indent=2, sort_keys=True)

    print(f"Wrote timing log to: {timing_path}")
    if wrapped:
        print(f"Model.forward was wrapped; model_forward_calls={model_forward_calls}")
    else:
        print("Model.forward could not be wrapped (model.forward missing). Only runner-level timing will be available.")


if __name__ == '__main__':
    main()
