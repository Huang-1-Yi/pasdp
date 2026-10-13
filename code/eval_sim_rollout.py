"""
Usage:
python eval_sim_rollout.py --checkpoint /path/to/latest.ckpt -o /path/to/eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.common import run_rollout

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-envs', default=32, type=int, show_default=True)
@click.option('--n-train', default=32, type=int, show_default=True)
@click.option('--n-train-vis', default=6, type=int, show_default=True)
@click.option('--n-test', default=200, type=int, show_default=True)
@click.option('--n-test-vis', default=6, type=int, show_default=True)
@click.option(
    '--test-start-seed',
    default=None,
    type=int,
    help='Override the first test environment seed.',
)
@click.option(
    '--num-inference-steps',
    default=None,
    type=click.IntRange(min=1),
    help=(
        'Override policy.num_inference_steps at evaluation time. '
        'Used by the DP inference-budget experiment; does not retrain.'
    ),
)
def main(
        checkpoint, output_dir, device, n_envs, n_train, n_train_vis,
        n_test, n_test_vis, test_start_seed, num_inference_steps):
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    cfg.task.env_runner.n_envs = n_envs
    cfg.task.env_runner.n_train = n_train
    cfg.task.env_runner.n_train_vis = n_train_vis
    cfg.task.env_runner.n_test = n_test
    cfg.task.env_runner.n_test_vis = n_test_vis
    if test_start_seed is not None:
        cfg.task.env_runner.test_start_seed = int(test_start_seed)
    print("=== Evaluation policy parameters ===")
    if hasattr(policy, 'num_inference_steps'):
        print(f"policy.num_inference_steps: {policy.num_inference_steps}")
    elif hasattr(policy, 'noise_scheduler') and hasattr(policy.noise_scheduler, 'config'):
        print(f"noise_scheduler.config.num_inference_steps: {policy.noise_scheduler.config.get('num_inference_steps', 'Not found')}")

    if num_inference_steps is not None:
        if not hasattr(policy, 'num_inference_steps'):
            raise click.ClickException(
                'Loaded policy does not expose num_inference_steps.'
            )
        train_steps = None
        if (hasattr(policy, 'noise_scheduler')
                and hasattr(policy.noise_scheduler, 'config')):
            train_steps = policy.noise_scheduler.config.get(
                'num_train_timesteps', None)
        if (train_steps is not None
                and int(num_inference_steps) > int(train_steps)):
            raise click.ClickException(
                f'num_inference_steps={num_inference_steps} exceeds '
                f'num_train_timesteps={train_steps}.'
            )
        original_steps = int(policy.num_inference_steps)
        policy.num_inference_steps = int(num_inference_steps)
        if hasattr(cfg, 'policy') and hasattr(cfg.policy, 'num_inference_steps'):
            cfg.policy.num_inference_steps = int(num_inference_steps)
        print(
            'evaluation override: '
            f'num_inference_steps {original_steps} -> '
            f'{policy.num_inference_steps}'
        )

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    try:
        runner_log = run_rollout(env_runner, policy)
    finally:
        env_runner.close()
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    if hasattr(policy, 'num_inference_steps'):
        json_log['evaluation/num_inference_steps'] = int(
            policy.num_inference_steps)
    json_log['evaluation/n_action_steps'] = int(
        cfg.task.env_runner.n_action_steps)
    json_log['evaluation/training_seed'] = int(cfg.training.seed)
    json_log['evaluation/training_n_demo'] = int(cfg.n_demo)
    json_log['evaluation/test_start_seed'] = int(
        cfg.task.env_runner.test_start_seed)
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
