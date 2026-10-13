#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential A -> B -> D -> C evaluation on physical GPU1 (RTX 4090).
# Checkpoint rule: highest retained top-k score; latest epoch breaks ties.

PROJECT_ROOT="${PASDP_PROJECT_ROOT:-/home/hy/dp_pro6000}"
PYTHON_BIN="${PASDP_PY:-/opt/miniconda3/envs/sim_torch27_v8/bin/python}"
TASK="${PASDP_TASK:-coffee_d0}"
SEED="${PASDP_SEED:-42}"
N_ENVS="${PASDP_EVAL_N_ENVS:-25}"
N_TEST="${PASDP_EVAL_N_TEST:-100}"
N_TEST_VIS="${PASDP_EVAL_N_TEST_VIS:-2}"
TEST_START_SEED="${PASDP_EVAL_TEST_START_SEED:-20000}"
TRAIN_N_ENVS="${PASDP_TRAIN_N_ENVS:-20}"
TRAIN_N_TEST="${PASDP_TRAIN_N_TEST:-30}"
TRAIN_TEST_START_SEED="${PASDP_TRAIN_TEST_START_SEED:-100000}"
N_DEMO="${PASDP_N_DEMO:-200}"
SKIP_DONE="${PASDP_SKIP_DONE:-1}"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
elif [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: bash run_pasdp_eval_abdc_4090.sh [--dry-run]"
    echo "Default evaluation: 100 test episodes, seeds 20000-20099, n_envs=25."
    echo "Environment overrides: PASDP_TASK, PASDP_SEED, PASDP_N_DEMO, PASDP_EVAL_N_ENVS, PASDP_EVAL_N_TEST, PASDP_EVAL_N_TEST_VIS, PASDP_EVAL_TEST_START_SEED, PASDP_TRAIN_N_ENVS, PASDP_TRAIN_N_TEST, PASDP_TRAIN_TEST_START_SEED, PASDP_SKIP_DONE"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "Unknown argument: $1" >&2
    exit 2
fi

cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT/third_party/robosuite:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline

TRAIN_ROOT="$PROJECT_ROOT/data/experiments/train"
EVAL_ROOT="$PROJECT_ROOT/data/experiments/eval"
LOG_ROOT="$PROJECT_ROOT/data/experiments/logs/eval"
EVAL_PROTOCOL="n_test${N_TEST}_seed${TEST_START_SEED}"
SELECTION_ROOT="$EVAL_ROOT/checkpoint_selection/$TASK/seed$SEED/$EVAL_PROTOCOL"

for value_name in SEED N_DEMO N_ENVS N_TEST N_TEST_VIS TEST_START_SEED TRAIN_N_ENVS TRAIN_N_TEST TRAIN_TEST_START_SEED; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer, got: $value" >&2
        exit 2
    fi
done
if (( N_ENVS < 1 || N_TEST < 1 || N_TEST_VIS > N_TEST )); then
    echo "Invalid evaluation sizes: n_envs=$N_ENVS n_test=$N_TEST n_test_vis=$N_TEST_VIS" >&2
    exit 2
fi

resolve_topk_checkpoint() {
    local stage="$1"
    local algo="$2"
    local group_root="$TRAIN_ROOT/$stage/$algo/$TASK/seed$SEED"
    local latest_path
    local checkpoint_dir
    local best_line
    local best_name

    if [[ ! -d "$group_root" ]]; then
        echo "Training group does not exist: $group_root" >&2
        return 1
    fi

    latest_path="$(find "$group_root" -type f -path '*/checkpoints/latest.ckpt' -printf '%T@ %p\n' | sort -k1,1nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$latest_path" ]]; then
        echo "No completed run checkpoint found under: $group_root" >&2
        return 1
    fi
    checkpoint_dir="$(dirname "$latest_path")"

    best_line="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'epoch*_score*.ckpt' -printf '%f\n' \
        | sed -E 's/^epoch([0-9]+)_score([-+0-9.eE]+)\.ckpt$/\2 \1 &/' \
        | sort -k1,1gr -k2,2nr \
        | head -n 1)"
    if [[ -z "$best_line" ]]; then
        echo "No top-k checkpoint found in: $checkpoint_dir" >&2
        return 1
    fi
    best_name="${best_line#* }"
    best_name="${best_name#* }"
    printf '%s/%s\n' "$checkpoint_dir" "$best_name"
}

checkpoint_sha256() {
    local checkpoint="$1"
    sha256sum "$checkpoint" | awk '{print $1}'
}

validate_training_config() {
    local checkpoint="$1"
    local run_dir
    local config_path
    run_dir="$(dirname "$(dirname "$checkpoint")")"
    config_path="$run_dir/.hydra/config.yaml"
    if [[ ! -s "$config_path" ]]; then
        echo "Resolved training config is missing: $config_path" >&2
        return 1
    fi
    "$PYTHON_BIN" - "$config_path" "$TASK" "$SEED" "$N_DEMO" "$TRAIN_N_ENVS" "$TRAIN_N_TEST" "$TRAIN_TEST_START_SEED" <<'PY'
import pathlib
import sys
import yaml

path, task, seed, n_demo, n_envs, n_test, test_start_seed = sys.argv[1:]
cfg = yaml.safe_load(pathlib.Path(path).read_text())
actual = {
    "task": str(cfg["task_name"]),
    "seed": int(cfg["training"]["seed"]),
    "n_demo": int(cfg["n_demo"]),
    "n_envs": int(cfg["task"]["env_runner"]["n_envs"]),
    "n_test": int(cfg["task"]["env_runner"]["n_test"]),
    "test_start_seed": int(cfg["task"]["env_runner"]["test_start_seed"]),
}
expected = {
    "task": task,
    "seed": int(seed),
    "n_demo": int(n_demo),
    "n_envs": int(n_envs),
    "n_test": int(n_test),
    "test_start_seed": int(test_start_seed),
}
if actual != expected:
    raise SystemExit(
        f"Training checkpoint protocol mismatch for {path}: "
        f"actual={actual}, expected={expected}"
    )
PY
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_eval() {
    local stage="$1"
    local algo="$2"
    local checkpoint="$3"
    local checkpoint_sha="$4"
    local nfe="${5:-}"
    local protocol_nfe="${6:-native}"
    local suffix="default"
    local output_dir
    local log_dir
    local console_log
    local -a command

    if [[ -n "$nfe" ]]; then
        suffix="nfe$nfe"
    fi
    output_dir="$EVAL_ROOT/$stage/$algo/$TASK/seed$SEED/$EVAL_PROTOCOL/$suffix"
    log_dir="$LOG_ROOT/$stage/$algo/$TASK/seed$SEED/$EVAL_PROTOCOL/$suffix"
    console_log="$log_dir/console.log"

    if [[ -s "$output_dir/eval_log.json" && "$SKIP_DONE" == "1" ]]; then
        echo "[SKIP] $stage/$algo/$EVAL_PROTOCOL/$suffix already has eval_log.json"
        return 0
    fi
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite incomplete or existing output: $output_dir" >&2
        return 1
    fi

    command=(
        "$PYTHON_BIN" -u eval_sim_rollout.py
        --checkpoint "$checkpoint"
        --output_dir "$output_dir"
        --device cuda:0
        --n-envs "$N_ENVS"
        --n-train 0
        --n-train-vis 0
        --n-test "$N_TEST"
        --n-test-vis "$N_TEST_VIS"
        --test-start-seed "$TEST_START_SEED"
    )
    if [[ -n "$nfe" ]]; then
        command+=(--num-inference-steps "$nfe")
    fi

    echo "[EVAL] $stage/$algo/$EVAL_PROTOCOL/$suffix"
    echo "  checkpoint: $checkpoint"
    if (( DRY_RUN )); then
        print_command "${command[@]}"
        return 0
    fi

    mkdir -p "$log_dir"
    set +e
    "${command[@]}" 2>&1 | tee "$console_log"
    local eval_status=${PIPESTATUS[0]}
    set -e
    if (( eval_status != 0 )); then
        echo "Evaluation failed ($eval_status): $stage/$algo/$EVAL_PROTOCOL/$suffix" >&2
        return "$eval_status"
    fi
    if [[ ! -s "$output_dir/eval_log.json" ]]; then
        echo "Evaluation returned success but eval_log.json is missing: $output_dir" >&2
        return 1
    fi

    "$PYTHON_BIN" - "$output_dir/eval_log.json" "$checkpoint" "$checkpoint_sha" "$stage" "$algo" "$TASK" "$SEED" "$N_DEMO" "$N_ENVS" "$N_TEST" "$N_TEST_VIS" "$TEST_START_SEED" "${nfe:-native}" "$protocol_nfe" <<'PY'
import json
import os
import pathlib
import sys

(
    log_path, checkpoint, checkpoint_sha, stage, algorithm, task, seed,
    n_demo, n_envs, n_test, n_test_vis, test_start_seed,
    requested_nfe, protocol_nfe,
) = sys.argv[1:]
path = pathlib.Path(log_path)
data = json.loads(path.read_text())
data.update({
    "evaluation/stage": stage,
    "evaluation/algorithm": algorithm,
    "evaluation/task_name": task,
    "evaluation/training_seed": int(seed),
    "evaluation/training_n_demo": int(n_demo),
    "evaluation/checkpoint_path": os.path.realpath(checkpoint),
    "evaluation/checkpoint_sha256": checkpoint_sha,
    "evaluation/checkpoint_rule": "max_topk_score_then_latest_epoch",
    "evaluation/n_envs": int(n_envs),
    "evaluation/n_train": 0,
    "evaluation/n_test": int(n_test),
    "evaluation/n_test_vis": int(n_test_vis),
    "evaluation/test_start_seed": int(test_start_seed),
    "evaluation/requested_nfe": requested_nfe if requested_nfe == "native" else int(requested_nfe),
    "evaluation/protocol_nfe": int(protocol_nfe) if protocol_nfe.isdigit() else protocol_nfe,
})
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
    echo "[DONE] $stage/$algo/$EVAL_PROTOCOL/$suffix -> $output_dir/eval_log.json"
}

create_reuse_record() {
    local stage="$1"
    local source_dir="$EVAL_ROOT/A/pasdp/$TASK/seed$SEED/$EVAL_PROTOCOL/default"
    local target_dir="$EVAL_ROOT/$stage/pasdp/$TASK/seed$SEED/$EVAL_PROTOCOL/default"
    local checkpoint="$2"
    local checkpoint_sha="$3"

    if (( DRY_RUN )); then
        echo "[REUSE] $stage/pasdp/$EVAL_PROTOCOL/default <- A/pasdp/$EVAL_PROTOCOL/default"
        return 0
    fi
    if [[ ! -s "$source_dir/eval_log.json" ]]; then
        echo "Cannot create reuse record before A-PASDP evaluation exists: $source_dir" >&2
        return 1
    fi
    if [[ -s "$target_dir/reuse_manifest.json" && "$SKIP_DONE" == "1" ]]; then
        echo "[SKIP] $stage/pasdp reuse record already exists"
        return 0
    fi
    if [[ -e "$target_dir" ]]; then
        echo "Refusing to overwrite existing reuse directory: $target_dir" >&2
        return 1
    fi
    mkdir -p "$target_dir"
    "$PYTHON_BIN" - "$target_dir/reuse_manifest.json" "$stage" "$source_dir" "$checkpoint" "$checkpoint_sha" "$TASK" "$SEED" <<'PY'
import json
import os
import pathlib
import sys

out, stage, source, checkpoint, checkpoint_sha, task, seed = sys.argv[1:]
payload = {
    "stage": stage,
    "algorithm": "pasdp",
    "task_name": task,
    "training_seed": int(seed),
    "reused": True,
    "source_evaluation": os.path.realpath(source),
    "source_checkpoint": os.path.realpath(checkpoint),
    "source_checkpoint_sha256": checkpoint_sha,
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    printf '# REUSED_FROM\n\nA-PASDP evaluation: `%s`\n\nCheckpoint: `%s`\n\nSHA256: `%s`\n' \
        "$source_dir" "$checkpoint" "$checkpoint_sha" > "$target_dir/REUSED_FROM.md"
    echo "[DONE] $stage/pasdp reuse record"
}

write_selection_manifest() {
    local output="$SELECTION_ROOT/checkpoints.tsv"
    if (( DRY_RUN )); then
        return 0
    fi
    mkdir -p "$SELECTION_ROOT"
    {
        printf 'id\tcheckpoint\tsha256\trule\n'
        printf 'A-PASDP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$A_PASDP_CKPT" "$A_PASDP_SHA"
        printf 'A-DP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$A_DP_CKPT" "$A_DP_SHA"
        printf 'A-SDP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$A_SDP_CKPT" "$A_SDP_SHA"
        printf 'A-Flow\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$A_FLOW_CKPT" "$A_FLOW_SHA"
        printf 'B-DP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$B_DP_CKPT" "$B_DP_SHA"
        printf 'B-SDP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$B_SDP_CKPT" "$B_SDP_SHA"
        printf 'B-Flow\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$B_FLOW_CKPT" "$B_FLOW_SHA"
        printf 'D-DP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$D_DP_CKPT" "$D_DP_SHA"
        printf 'D-SDP\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$D_SDP_CKPT" "$D_SDP_SHA"
        printf 'D-Flow\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$D_FLOW_CKPT" "$D_FLOW_SHA"
    } > "$output"
}

write_summary() {
    local summary_dir="$EVAL_ROOT/summary/$TASK/seed$SEED/$EVAL_PROTOCOL"
    if (( DRY_RUN )); then
        return 0
    fi
    mkdir -p "$summary_dir"
    "$PYTHON_BIN" - "$EVAL_ROOT" "$TASK" "$SEED" "$EVAL_PROTOCOL" "$summary_dir/eval_summary.json" "$summary_dir/eval_summary.tsv" <<'PY'
import json
import pathlib
import sys

root, task, seed, protocol, json_out, tsv_out = sys.argv[1:]
root = pathlib.Path(root)
rows = []
for path in sorted(root.glob(f"[ABDC]/*/{task}/seed{seed}/{protocol}/*/eval_log.json")):
    data = json.loads(path.read_text())
    rows.append({
        "stage": data.get("evaluation/stage"),
        "algorithm": data.get("evaluation/algorithm"),
        "protocol_nfe": data.get("evaluation/protocol_nfe"),
        "configured_inference_steps": data.get("evaluation/num_inference_steps"),
        "test_success_rate": data.get("test/success_rate", data.get("test/mean_score")),
        "test_avg_completion_step": data.get("test/avg_completion_step"),
        "test_median_completion_step": data.get("test/median_completion_step"),
        "policy_avg_inference_time_ms": data.get("policy/avg_inference_time_ms"),
        "checkpoint_sha256": data.get("evaluation/checkpoint_sha256"),
        "eval_log": str(path),
    })
pathlib.Path(json_out).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
with pathlib.Path(tsv_out).open("w") as f:
    columns = list(rows[0]) if rows else ["stage", "algorithm", "nfe", "test_success_rate", "eval_log"]
    f.write("\t".join(columns) + "\n")
    for row in rows:
        f.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")
PY
    echo "[SUMMARY] $summary_dir/eval_summary.json"
}

echo "Resolving completed training runs and top-k checkpoints..."
A_PASDP_CKPT="$(resolve_topk_checkpoint A pasdp)"
A_DP_CKPT="$(resolve_topk_checkpoint A dp)"
A_SDP_CKPT="$(resolve_topk_checkpoint A sdp)"
A_FLOW_CKPT="$(resolve_topk_checkpoint A flow)"
B_DP_CKPT="$(resolve_topk_checkpoint B dp)"
B_SDP_CKPT="$(resolve_topk_checkpoint B sdp)"
B_FLOW_CKPT="$(resolve_topk_checkpoint B flow)"
D_DP_CKPT="$(resolve_topk_checkpoint D dp)"
D_SDP_CKPT="$(resolve_topk_checkpoint D sdp)"
D_FLOW_CKPT="$(resolve_topk_checkpoint D flow)"

if (( DRY_RUN )); then
    A_PASDP_SHA=dry-run
    A_DP_SHA=dry-run
    A_SDP_SHA=dry-run
    A_FLOW_SHA=dry-run
    B_DP_SHA=dry-run
    B_SDP_SHA=dry-run
    B_FLOW_SHA=dry-run
    D_DP_SHA=dry-run
    D_SDP_SHA=dry-run
    D_FLOW_SHA=dry-run
else
    echo "Computing checkpoint SHA256 values..."
    A_PASDP_SHA="$(checkpoint_sha256 "$A_PASDP_CKPT")"
    A_DP_SHA="$(checkpoint_sha256 "$A_DP_CKPT")"
    A_SDP_SHA="$(checkpoint_sha256 "$A_SDP_CKPT")"
    A_FLOW_SHA="$(checkpoint_sha256 "$A_FLOW_CKPT")"
    B_DP_SHA="$(checkpoint_sha256 "$B_DP_CKPT")"
    B_SDP_SHA="$(checkpoint_sha256 "$B_SDP_CKPT")"
    B_FLOW_SHA="$(checkpoint_sha256 "$B_FLOW_CKPT")"
    D_DP_SHA="$(checkpoint_sha256 "$D_DP_CKPT")"
    D_SDP_SHA="$(checkpoint_sha256 "$D_SDP_CKPT")"
    D_FLOW_SHA="$(checkpoint_sha256 "$D_FLOW_CKPT")"
fi

write_selection_manifest

echo "Validating task, training seed, n_demo, training n_envs, training n_test, and test seed start for every checkpoint..."
for checkpoint in \
    "$A_PASDP_CKPT" "$A_DP_CKPT" "$A_SDP_CKPT" "$A_FLOW_CKPT" \
    "$B_DP_CKPT" "$B_SDP_CKPT" "$B_FLOW_CKPT" \
    "$D_DP_CKPT" "$D_SDP_CKPT" "$D_FLOW_CKPT"; do
    validate_training_config "$checkpoint"
done

TEST_END_SEED=$((TEST_START_SEED + N_TEST - 1))
echo "Training checkpoint protocol: n_envs=$TRAIN_N_ENVS n_test=$TRAIN_N_TEST test_start_seed=$TRAIN_TEST_START_SEED"
echo "Independent evaluation protocol: task=$TASK training_seed=$SEED n_demo=$N_DEMO n_envs=$N_ENVS n_train=0 n_test=$N_TEST n_test_vis=$N_TEST_VIS test_seeds=$TEST_START_SEED-$TEST_END_SEED"
echo "Physical GPU1: CUDA_VISIBLE_DEVICES=1, MUJOCO_EGL_DEVICE_ID=0, evaluator device=cuda:0"

# A
run_eval A pasdp "$A_PASDP_CKPT" "$A_PASDP_SHA" "" "steady1/cold41"
run_eval A dp "$A_DP_CKPT" "$A_DP_SHA" "" 10
run_eval A sdp "$A_SDP_CKPT" "$A_SDP_SHA" "" 8
run_eval A flow "$A_FLOW_CKPT" "$A_FLOW_SHA" "" 1

# B: PASDP reuses A-PASDP.
create_reuse_record B "$A_PASDP_CKPT" "$A_PASDP_SHA"
run_eval B dp "$B_DP_CKPT" "$B_DP_SHA" "" 1
run_eval B sdp "$B_SDP_CKPT" "$B_SDP_SHA" "" 1
run_eval B flow "$B_FLOW_CKPT" "$B_FLOW_SHA" "" 1

# D: PASDP reuses A-PASDP.
create_reuse_record D "$A_PASDP_CKPT" "$A_PASDP_SHA"
run_eval D dp "$D_DP_CKPT" "$D_DP_SHA" "" 40
run_eval D sdp "$D_SDP_CKPT" "$D_SDP_SHA" "" 1
run_eval D flow "$D_FLOW_CKPT" "$D_FLOW_SHA" "" 1

# C: all four budgets reuse exactly the same A-DP checkpoint and SHA256.
for nfe in 1 20 40 100; do
    run_eval C dp "$A_DP_CKPT" "$A_DP_SHA" "$nfe" "$nfe"
done

write_summary
if (( DRY_RUN )); then
    echo "Dry run completed; no evaluation outputs were written."
else
    echo "All A -> B -> D -> C evaluations completed."
fi
