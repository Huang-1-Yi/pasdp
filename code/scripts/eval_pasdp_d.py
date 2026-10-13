"""Sequential formal evaluation for the isolated PASDP D ablations."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "data/experiments/train/D_ablation"
EVAL_ROOT = ROOT / "data/experiments/eval/D_ablation"
PROTOCOL = "seed42/n_test100_seed20000"
TASKS = [
    "coffee_d0", "threading_d0", "pick_place_d0",
    "coffee_preparation_d0", "hammer_cleanup_d0", "square_d0",
    "stack_d0", "stack_three_d0", "nut_assembly_d0",
    "three_piece_assembly_d0",
]
TRAINED_CONDITIONS = {
    "d1": ("fixed_permutation", "exponential"),
    "d2": ("positionwise", "uniform_sum_preserving"),
    "d12": ("fixed_permutation", "uniform_sum_preserving"),
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_a(task):
    record = (
        ROOT / "data/experiments/eval/A/pasdp" / task / PROTOCOL
        / "default/eval_log.json"
    )
    data = read_json(record)
    expected = {
        "evaluation/stage": "A",
        "evaluation/algorithm": "pasdp",
        "evaluation/task_name": task,
        "evaluation/training_seed": 42,
        "evaluation/training_n_demo": 200,
        "evaluation/n_test": 100,
        "evaluation/n_envs": 25,
        "evaluation/test_start_seed": 20000,
    }
    for key, value in expected.items():
        if data.get(key) != value:
            raise RuntimeError(f"{record}: {key} mismatch")
    checkpoint = Path(data["evaluation/checkpoint_path"])
    digest = sha256(checkpoint)
    if digest != data["evaluation/checkpoint_sha256"]:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {checkpoint}")
    return record, data, checkpoint, digest


def selected_checkpoint(task, condition):
    root = TRAIN_ROOT / condition / task / "seed42"
    latest = sorted(root.glob("**/checkpoints/latest.ckpt"),
                    key=lambda path: path.stat().st_mtime, reverse=True)
    if not latest:
        raise RuntimeError(f"Missing completed training run: {root}")
    checkpoint_dir = latest[0].parent
    candidates = []
    pattern = re.compile(r"epoch(\d+)_score([-+0-9.eE]+)\.ckpt$")
    for path in checkpoint_dir.glob("epoch*_score*.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((float(match.group(2)), int(match.group(1)), path))
    if not candidates:
        raise RuntimeError(f"No top-k checkpoint: {checkpoint_dir}")
    checkpoint = max(candidates, key=lambda item: (item[0], item[1]))[2]

    run_dir = checkpoint_dir.parent
    cfg_path = run_dir / ".hydra/config.yaml"
    log_path = run_dir / "train_logs.json.txt"
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text())
    expected_schedule, expected_weights = TRAINED_CONDITIONS[condition]
    expected = {
        "task_name": task,
        "n_demo": 200,
        "seed": 42,
        "n_envs": 20,
        "n_test": 30,
        "test_start_seed": 100000,
        "target": "diffusion_policy.policy.padp_ablation.PADPAblationPolicy",
        "schedule": expected_schedule,
        "weights": expected_weights,
        "permutation_seed": 20260915,
    }
    actual = {
        "task_name": cfg["task_name"],
        "n_demo": int(cfg["n_demo"]),
        "seed": int(cfg["training"]["seed"]),
        "n_envs": int(cfg["task"]["env_runner"]["n_envs"]),
        "n_test": int(cfg["task"]["env_runner"]["n_test"]),
        "test_start_seed": int(cfg["task"]["env_runner"]["test_start_seed"]),
        "target": cfg["policy"]["_target_"],
        "schedule": cfg["policy"]["ablation_noise_schedule_mode"],
        "weights": cfg["policy"]["ablation_loss_weight_mode"],
        "permutation_seed": int(cfg["policy"]["fixed_permutation_seed"]),
    }
    if actual != expected:
        raise RuntimeError(
            f"Training protocol mismatch: {cfg_path}\n"
            f"actual={actual}\nexpected={expected}")
    last = json.loads([line for line in log_path.read_text().splitlines()
                       if line.strip()][-1])
    if int(last.get("epoch", -1)) != 250:
        raise RuntimeError(f"Training incomplete: {log_path}")
    return checkpoint, sha256(checkpoint)


def validate_result(path, metadata):
    data = read_json(path)
    for key, value in metadata.items():
        if data.get(key) != value:
            raise RuntimeError(
                f"{path}: {key}={data.get(key)!r}, expected {value!r}")
    if "test/success_rate" not in data:
        raise RuntimeError(f"Missing success rate: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+",
                        choices=["d0", "d1", "d2", "d12", "d3"],
                        default=["d0", "d1", "d2", "d12", "d3"])
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    conditions = list(dict.fromkeys(args.conditions))
    tasks = list(dict.fromkeys(args.tasks))
    jobs = []
    reuse = []
    failures = []
    for task in tasks:
        try:
            a_record, a_data, a_checkpoint, a_sha = source_a(task)
        except Exception as error:
            failures.append(str(error))
            continue
        for condition in conditions:
            if condition == "d0":
                reuse.append((task, a_record, a_data, a_checkpoint, a_sha))
                print(f"[READY] {task} D0 reuses formal A-PASDP", flush=True)
                continue
            try:
                if condition == "d3":
                    checkpoint, digest = a_checkpoint, a_sha
                else:
                    checkpoint, digest = selected_checkpoint(task, condition)
            except Exception as error:
                failures.append(f"{task}/{condition}: {error}")
                continue
            output = EVAL_ROOT / condition / task / PROTOCOL / "default"
            metadata = {
                "evaluation/stage": "D_ablation",
                "evaluation/condition": condition,
                "evaluation/task_name": task,
                "evaluation/training_seed": 42,
                "evaluation/training_n_demo": 200,
                "evaluation/checkpoint_path": str(checkpoint.resolve()),
                "evaluation/checkpoint_sha256": digest,
                "evaluation/checkpoint_rule": (
                    "reuse_A_PASDP" if condition == "d3"
                    else "max_topk_score_then_latest_epoch"),
                "evaluation/n_envs": 25,
                "evaluation/n_train": 0,
                "evaluation/n_test": 100,
                "evaluation/n_test_vis": 2,
                "evaluation/test_start_seed": 20000,
                "evaluation/buffer_reuse": condition != "d3",
            }
            result = output / "eval_log.json"
            if result.exists():
                validate_result(result, metadata)
                print(f"[SKIP] {task} {condition}: verified", flush=True)
                continue
            if output.exists():
                failures.append(
                    f"Incomplete output; inspect/move before retry: {output}")
                continue
            command = [
                sys.executable, "-u", str(ROOT / "scripts/eval_pasdp_d_one.py"),
                "--checkpoint", str(checkpoint), "--output-dir", str(output),
                "--device", "cuda:0", "--n-envs", "25", "--n-test", "100",
                "--n-test-vis", "2", "--test-start-seed", "20000",
            ]
            if condition == "d3":
                command.append("--cold-start-every-decision")
            jobs.append((task, condition, output, metadata, command))
            print(f"[READY] {task} {condition}", flush=True)
            if args.dry_run:
                print(shlex.join(command))

    if failures:
        raise RuntimeError("D preflight failed:\n- " + "\n- ".join(failures))
    print(f"Pending rollouts: {len(jobs)}; D0 reuse records: {len(reuse)}")
    if args.check or args.dry_run:
        return

    import fcntl
    lock_path = EVAL_ROOT / "queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for task, a_record, a_data, checkpoint, digest in reuse:
            output = EVAL_ROOT / "d0" / task / PROTOCOL
            output.mkdir(parents=True, exist_ok=True)
            manifest = output / "reuse_manifest.json"
            expected = {
                "evaluation/stage": "D_ablation",
                "evaluation/condition": "d0",
                "evaluation/task_name": task,
                "evaluation/source_record": str(a_record.resolve()),
                "evaluation/checkpoint_path": str(checkpoint.resolve()),
                "evaluation/checkpoint_sha256": digest,
                "test/success_rate": a_data["test/success_rate"],
            }
            if manifest.exists() and read_json(manifest) != expected:
                raise RuntimeError(f"D0 reuse manifest mismatch: {manifest}")
            if not manifest.exists():
                manifest.write_text(json.dumps(expected, indent=2) + "\n")
        for task, condition, output, metadata, command in jobs:
            log = (ROOT / "data/experiments/logs/eval/D_ablation" /
                   condition / task / PROTOCOL / "console.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"[START] {task} {condition}; log={log}", flush=True)
            with log.open("a") as console:
                subprocess.run(command, cwd=ROOT, stdout=console,
                               stderr=subprocess.STDOUT, check=True)
            result = output / "eval_log.json"
            data = read_json(result)
            data.update(metadata)
            temporary = result.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, result)
            validate_result(result, metadata)
            print(f"[DONE] {task} {condition}: {data['test/success_rate']}")

        summary = (ROOT / "data/experiments/eval/summary/D_ablation" /
                   PROTOCOL)
        summary.mkdir(parents=True, exist_ok=True)
        rows = []
        for task in TASKS:
            for condition in ["d0", "d1", "d2", "d12", "d3"]:
                if condition == "d0":
                    record = EVAL_ROOT / condition / task / PROTOCOL / "reuse_manifest.json"
                else:
                    record = EVAL_ROOT / condition / task / PROTOCOL / "default/eval_log.json"
                if record.exists():
                    data = read_json(record)
                    rows.append([
                        task, condition, data["test/success_rate"],
                        data.get("policy/avg_inference_time_ms"), str(record),
                    ])
        with (summary / "eval_summary.tsv").open("w", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow([
                "task", "condition", "success_rate", "inference_ms", "record"])
            writer.writerows(rows)
        print(f"Summary: {summary / 'eval_summary.tsv'}")


if __name__ == "__main__":
    main()
