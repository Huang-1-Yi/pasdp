"""Run one formal PADP D-ablation rollout without changing the base evaluator."""

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import click
import dill
import hydra
import torch
import wandb

from diffusion_policy.policy.padp import PADPPolicy
from diffusion_policy.policy.padp_ablation import ColdStartEveryDecisionPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.common import run_rollout


@click.command()
@click.option("--checkpoint", required=True, type=click.Path(exists=True))
@click.option("--output-dir", required=True, type=click.Path())
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--n-envs", default=25, type=click.IntRange(min=1))
@click.option("--n-test", default=100, type=click.IntRange(min=1))
@click.option("--n-test-vis", default=2, type=click.IntRange(min=0))
@click.option("--test-start-seed", default=20000, type=int)
@click.option("--cold-start-every-decision", is_flag=True)
def main(
        checkpoint, output_dir, device, n_envs, n_test, n_test_vis,
        test_start_seed, cold_start_every_decision):
    output = Path(output_dir)
    if output.exists():
        raise click.ClickException(f"Refusing to overwrite: {output}")
    output.mkdir(parents=True)

    with open(checkpoint, "rb") as stream:
        payload = torch.load(stream, pickle_module=dill)
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg, output_dir=str(output))
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    if not isinstance(policy, PADPPolicy):
        raise click.ClickException(
            f"D ablation evaluator requires PADP, got {type(policy)!r}")

    cfg.task.env_runner.n_envs = n_envs
    cfg.task.env_runner.n_train = 0
    cfg.task.env_runner.n_train_vis = 0
    cfg.task.env_runner.n_test = n_test
    cfg.task.env_runner.n_test_vis = n_test_vis
    cfg.task.env_runner.test_start_seed = test_start_seed

    policy.to(torch.device(device))
    policy.eval()
    rollout_policy = (
        ColdStartEveryDecisionPolicy(policy)
        if cold_start_every_decision else policy
    )
    warmup_calls = int(policy.horizon + policy.n_obs_steps - 1)

    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner, output_dir=str(output))
    try:
        runner_log = run_rollout(env_runner, rollout_policy)
    finally:
        env_runner.close()

    record = {}
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            record[key] = value._path
        else:
            record[key] = value
    record.update({
        "evaluation/n_action_steps": int(cfg.task.env_runner.n_action_steps),
        "evaluation/n_envs": int(n_envs),
        "evaluation/n_train": 0,
        "evaluation/n_test": int(n_test),
        "evaluation/n_test_vis": int(n_test_vis),
        "evaluation/test_start_seed": int(test_start_seed),
        "evaluation/training_seed": int(cfg.training.seed),
        "evaluation/training_n_demo": int(cfg.n_demo),
        "evaluation/buffer_reuse": not cold_start_every_decision,
        "evaluation/warmup_model_calls": warmup_calls,
        "evaluation/cold_start_model_calls": warmup_calls + 1,
        "evaluation/steady_state_model_calls": (
            warmup_calls + 1 if cold_start_every_decision else 1
        ),
    })
    temporary = output / "eval_log.json.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output / "eval_log.json")


if __name__ == "__main__":
    main()
