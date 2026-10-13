import os
import pathlib
import click
import torch
import dill
import json

from diffusion_policy.workspace.common.action_horizon import (
    evaluate_action_horizon,
)


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('--split', default='test')
@click.option('--batch_size', default=None, type=int)
@click.option('--max_batches', default=None, type=int)
@click.option('--overwrite', is_flag=True, default=False)
@click.option(
    '--mode',
    type=click.Choice(['auto', 'batched', 'sequential']),
    default='auto',
    show_default=True,
)
def main(
        checkpoint, output_dir, device, split, batch_size, max_batches,
        overwrite, mode):
    if os.path.exists(output_dir) and not overwrite:
        click.confirm(
            f"Output path {output_dir} already exists! Overwrite?",
            abort=True,
        )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint payload
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload.get('cfg', None)
    if cfg is None:
        raise RuntimeError('No cfg found in checkpoint payload')

    # Instantiate the workspace class recorded by the checkpoint.
    import hydra
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)

    # load payload into workspace
    workspace.load_payload(payload)

    # resolve and print runtime device info
    requested_device = str(device)
    cuda_available = torch.cuda.is_available()
    if requested_device.startswith('cuda') and (not cuda_available):
        print(f"[eval_action_horizon] Requested device={requested_device}, but CUDA is not available. Falling back to cpu.")
        device = 'cpu'
    else:
        device = requested_device

    print(f"[eval_action_horizon] Requested device: {requested_device}")
    print(f"[eval_action_horizon] CUDA available: {cuda_available}")
    print(f"[eval_action_horizon] Actual device: {device}")

    print('Starting action-horizon evaluation...')
    # The payload has already been loaded above. Passing checkpoint=None keeps
    # the legacy calculation intact while avoiding a second checkpoint load.
    result = evaluate_action_horizon(
        workspace,
        checkpoint=None,
        output_dir=output_dir,
        device=device,
        split=split,
        batch_size=batch_size,
        max_batches=max_batches,
        mode=mode,
    )
    out_path = os.path.join(output_dir, 'action_horizon_summary.json')
    json.dump(result, open(out_path, 'w'), indent=2)
    print(f'Wrote eval summary to {out_path}')


if __name__ == '__main__':
    main()
