"""Sequential one-batch real-data training and two-decision inference checks.

Run from the project root. These are wiring checks, not performance evaluations.
They use two recorded episodes and do not access robot hardware.
"""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo


METHODS = {
    "padp": [],
    "dp": ["workspace=policy", "policy=dp_ddim", "encoder=rgb/robomimic_raw", "policy.num_inference_steps=10"],
    "sdp": ["workspace=policy", "policy=streaming_diffusion", "encoder=rgb/robomimic_raw"],
    "sdp_optimized": ["workspace=policy", "policy=streaming_diffusion_optimized", "encoder=rgb/robomimic_raw"],
    "flow": ["workspace=policy", "policy=flow_policy", "encoder=rgb/robomimic_raw"],
    "focal": ["workspace=policy", "policy=focal_flow"],
    "transformer": ["workspace=policy", "policy=diffusion_transformer", "encoder=rgb/robomimic_raw", "~capacity"],
    "act": ["workspace=policy", "policy=act", "~encoder", "~capacity", "policy.max_timesteps=1000", "policy.n_envs=1", "training.optimizer_source=model", "training.use_ema=false"],
    "bcrnn": ["workspace=bcrnn", "policy=robomimic_bcrnn", "~encoder", "~capacity", "temporal=bcrnn_h10_obs1_action1", "training.use_ema=false"],
    "padp_unrolled": ["workspace=padp_unrolled", "policy=padp_unrolled", "task.dataset.horizon=41", "task.dataset.n_obs_steps=2"],
}


def training_overrides(method, output):
    return ["experiment=padp_real_aubo_1013", *METHODS[method],
            f"name={method}_real_smoke", f"output_group={method}_real_smoke",
            "task.dataset.max_episodes=2", "dataloader.batch_size=1",
            "dataloader.num_workers=0", "dataloader.persistent_workers=false",
            "training.num_epochs=1", "training.max_train_steps=1",
            "training.lr_warmup_steps=0", "training.use_ema=false",
            "logging.mode=disabled", "checkpoint.save_last_ckpt=false",
            "checkpoint.save_last_snapshot=false", "checkpoint.topk.k=0",
            f"hydra.run.dir={output}"]


def probe(method, project):
    # Fresh, untrained weights isolate inference API compatibility from fitting.
    # The actual training/optimizer path is checked separately through train.py.
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(project))
    from diffusion_policy.common.pytorch_util import dict_apply
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(config_dir=str(project / "diffusion_policy/config"), version_base=None):
        cfg = compose(config_name="train", overrides=training_overrides(method, "unused"))
    OmegaConf.resolve(cfg)
    dataset = instantiate(cfg.task.dataset)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    policy = instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to("cuda:0")
    policy.eval()
    obs = dict_apply(batch["obs"], lambda x: x.to("cuda:0"))
    policy.reset()
    shapes = []
    with torch.no_grad():
        for decision in range(2):
            result = policy.predict_action(obs)
            action = result["action"]
            assert action.ndim == 3 and action.shape[0] == 1 and action.shape[-1] == 15, action.shape
            assert torch.isfinite(action).all(), "Nonfinite action"
            shapes.append(list(action.shape))
    print("INFERENCE_OK " + json.dumps({"method": method, "two_decision_shapes": shapes}), flush=True)


def run_logged(command, path, timeout):
    with path.open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return result.returncode
        except subprocess.TimeoutExpired:
            return "timeout"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="1")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--probe", choices=list(METHODS))
    parser.add_argument("--methods", nargs="+", choices=list(METHODS), default=list(METHODS))
    parser.add_argument("--run-id", default=datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    os.chdir(project)
    if args.probe:
        probe(args.probe, project)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS="4",
                      PYTHONDONTWRITEBYTECODE="1", HYDRA_FULL_ERROR="1")
    root = project / "data/real_smoke_policies" / args.run_id
    root.mkdir(parents=True, exist_ok=False)
    results = []
    print(f"SMOKE_ROOT {root}", flush=True)
    for method in args.methods:
        output = root / method
        output.mkdir()
        started = time.monotonic()
        command = [sys.executable, "train.py", "--config-name=train", *training_overrides(method, output)]
        print(f"START {method}", flush=True)
        status = run_logged(command, output / "console.log", args.timeout)
        record = {"method": method, "training_exit": status, "command": command,
                  "output_dir": str(output)}
        rows = []
        for filename in ["train_logs.json.txt", "logs.json.txt"]:
            path = output / filename
            if path.exists():
                rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        losses = [r["train_loss"] for r in rows if "train_loss" in r]
        record["train_loss"] = losses[-1] if losses else None
        record["training_ok"] = status == 0 and bool(losses) and all(math.isfinite(float(v)) for v in losses)
        if record["training_ok"]:
            probe_command = [sys.executable, str(Path(__file__).resolve()), "--probe", method]
            record["inference_exit"] = run_logged(probe_command, output / "inference.log", args.timeout)
            record["inference_ok"] = record["inference_exit"] == 0
        else:
            record["inference_ok"] = None
        record["seconds"] = round(time.monotonic() - started, 2)
        results.append(record)
        (root / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print("RESULT " + json.dumps({k: v for k, v in record.items() if k != "command"}), flush=True)
    results.append({"method": "dp3", "training_ok": False, "status": "blocked",
                    "reason": "1013 recordings contain RGB and low-dimensional state, but no point_cloud input required by DP3."})
    (root / "summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"DONE {root / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
