#!/usr/bin/env bash
set -Eeuo pipefail

# Sequential formal A+B evaluation for the nine non-coffee Robomimic tasks.
# B-PASDP reuses A-PASDP, so this performs 9 * 7 = 63 real rollouts.

ROOT="${PASDP_PROJECT_ROOT:-/home/hy/dp_pro6000}"
PY="${PASDP_PY:-/opt/miniconda3/envs/sim_torch27_v8/bin/python}"
SEED="${PASDP_SEED:-42}"
N_DEMO="${PASDP_N_DEMO:-200}"
N_ENVS="${PASDP_EVAL_N_ENVS:-25}"
N_TEST="${PASDP_EVAL_N_TEST:-100}"
N_TEST_VIS="${PASDP_EVAL_N_TEST_VIS:-2}"
TEST_START_SEED="${PASDP_EVAL_TEST_START_SEED:-20000}"
TRAIN_N_ENVS="${PASDP_TRAIN_N_ENVS:-20}"
TRAIN_N_TEST="${PASDP_TRAIN_N_TEST:-30}"
TRAIN_TEST_START_SEED="${PASDP_TRAIN_TEST_START_SEED:-100000}"
PHYSICAL_GPU="${PASDP_EVAL_PHYSICAL_GPU:-1}"
SKIP_DONE="${PASDP_SKIP_DONE:-1}"
WAIT_INTERVAL="${PASDP_WAIT_INTERVAL_SECONDS:-30}"

TASKS=(
    threading_d0
    pick_place_d0
    coffee_preparation_d0
    hammer_cleanup_d0
    square_d0
    stack_d0
    stack_three_d0
    nut_assembly_d0
    three_piece_assembly_d0
)

DRY_RUN=0
CHECK_ONLY=0
WAIT_TRAINING=0

usage() {
    cat <<'EOF'
Usage: bash run_pasdp_eval_ab_remaining9_4090.sh [options]

Options:
  --wait-training  Wait until all 63 training .done markers exist, then evaluate.
  --check          Only check training/checkpoint readiness; do not evaluate.
  --dry-run        Print all evaluation commands without writing results.
  -h, --help       Show this help.

Default formal protocol: n_train=0, n_test=100, n_envs=25,
test seeds 20000-20099, training seed 42, n_demo=200.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --wait-training) WAIT_TRAINING=1 ;;
        --check) CHECK_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

for value_name in SEED N_DEMO N_ENVS N_TEST N_TEST_VIS TEST_START_SEED TRAIN_N_ENVS TRAIN_N_TEST TRAIN_TEST_START_SEED WAIT_INTERVAL; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "$value_name must be a non-negative integer, got: $value" >&2
        exit 2
    fi
done
if (( N_ENVS < 1 || N_TEST < 1 || N_TEST_VIS > N_TEST || WAIT_INTERVAL < 1 )); then
    echo "Invalid evaluation sizes or wait interval." >&2
    exit 2
fi

case "$PHYSICAL_GPU" in
    0)
        CUDA_VISIBLE_LIST=0
        EGL_DEVICE=1
        ;;
    1)
        CUDA_VISIBLE_LIST=1
        EGL_DEVICE=0
        ;;
    *)
        echo "PASDP_EVAL_PHYSICAL_GPU must be 0 or 1, got: $PHYSICAL_GPU" >&2
        exit 2
        ;;
esac

TRAIN_ROOT="$ROOT/data/experiments/train"
STATE_ROOT="$ROOT/data/experiments/run_state/remaining9_ab_seed${SEED}"
EVAL_ROOT="$ROOT/data/experiments/eval"
LOG_ROOT="$ROOT/data/experiments/logs/eval"
EVAL_PROTOCOL="n_test${N_TEST}_seed${TEST_START_SEED}"
SUMMARY_ROOT="$EVAL_ROOT/summary/remaining9/seed$SEED/$EVAL_PROTOCOL"

if [[ ! -x "$PY" ]]; then
    echo "Python environment not found: $PY" >&2
    exit 1
fi
if [[ ! -f "$ROOT/eval_sim_rollout.py" ]]; then
    echo "Evaluator not found: $ROOT/eval_sim_rollout.py" >&2
    exit 1
fi

training_marker() {
    local task="$1"
    local stage="$2"
    local algo="$3"
    find "$STATE_ROOT" -type f -name "${task}_${stage}_${algo}.done" -print -quit 2>/dev/null
}

list_missing_training() {
    local task stage algo marker
    for task in "${TASKS[@]}"; do
        for stage_algo in A:pasdp A:dp A:sdp A:flow B:dp B:sdp B:flow; do
            stage="${stage_algo%%:*}"
            algo="${stage_algo##*:}"
            marker="$(training_marker "$task" "$stage" "$algo")"
            if [[ -z "$marker" ]]; then
                printf '%s/%s/%s\n' "$task" "$stage" "$algo"
            fi
        done
    done
}

wait_for_training() {
    local missing
    while true; do
        missing="$(list_missing_training)"
        if [[ -z "$missing" ]]; then
            echo "All 63 training completion markers are present."
            return 0
        fi
        if (( ! WAIT_TRAINING )); then
            echo "Evaluation is not ready; missing training completion markers:" >&2
            printf '%s\n' "$missing" >&2
            echo "Run again after training finishes, or use --wait-training." >&2
            return 1
        fi
        echo "[$(date '+%F %T %Z')] Waiting for training completion:"
        printf '  %s\n' $missing
        sleep "$WAIT_INTERVAL"
    done
}

resolve_topk_checkpoint() {
    local task="$1"
    local stage="$2"
    local algo="$3"
    local group_root="$TRAIN_ROOT/$stage/$algo/$task/seed$SEED"
    local latest_path checkpoint_dir best_line best_name

    latest_path="$(find "$group_root" -type f -path '*/checkpoints/latest.ckpt' -printf '%T@ %p\n' 2>/dev/null \
        | sort -k1,1nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -z "$latest_path" ]]; then
        echo "No latest checkpoint found under: $group_root" >&2
        return 1
    fi
    checkpoint_dir="$(dirname "$latest_path")"
    best_line="$(find "$checkpoint_dir" -maxdepth 1 -type f -name 'epoch*_score*.ckpt' -printf '%f\n' \
        | sed -E 's/^epoch([0-9]+)_score([-+0-9.eE]+)\.ckpt$/\2 \1 &/' \
        | sort -k1,1gr -k2,2nr | head -n 1)"
    if [[ -z "$best_line" ]]; then
        echo "No top-k checkpoint found in: $checkpoint_dir" >&2
        return 1
    fi
    best_name="${best_line#* }"
    best_name="${best_name#* }"
    printf '%s/%s\n' "$checkpoint_dir" "$best_name"
}

validate_training_run() {
    local task="$1"
    local algo="$2"
    local checkpoint="$3"
    local run_dir config_path log_name
    run_dir="$(dirname "$(dirname "$checkpoint")")"
    config_path="$run_dir/.hydra/config.yaml"
    if [[ "$algo" == pasdp ]]; then
        log_name=train_logs.json.txt
    else
        log_name=logs.json.txt
    fi
    "$PY" - "$config_path" "$run_dir/$log_name" "$task" "$SEED" "$N_DEMO" "$TRAIN_N_ENVS" "$TRAIN_N_TEST" "$TRAIN_TEST_START_SEED" <<'PY'
import json
import pathlib
import sys
import yaml

config_path, log_path, task, seed, n_demo, n_envs, n_test, test_start_seed = sys.argv[1:]
config_path = pathlib.Path(config_path)
log_path = pathlib.Path(log_path)
if not config_path.is_file():
    raise SystemExit(f"Resolved training config is missing: {config_path}")
if not log_path.is_file():
    raise SystemExit(f"Training log is missing: {log_path}")

cfg = yaml.safe_load(config_path.read_text())
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
        f"Training protocol mismatch for {config_path}: "
        f"actual={actual}, expected={expected}"
    )

with log_path.open("rb") as stream:
    stream.seek(0, 2)
    pos = stream.tell()
    data = b""
    while pos > 0 and data.count(b"\n") < 2:
        size = min(8192, pos)
        pos -= size
        stream.seek(pos)
        data = stream.read(size) + data
lines = [line for line in data.splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"Training log is empty: {log_path}")
last = json.loads(lines[-1])
if int(last.get("epoch", -1)) != 250:
    raise SystemExit(
        f"Training log has not reached epoch 250: {log_path}; "
        f"last_epoch={last.get('epoch')}"
    )
PY
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

run_eval() {
    local task="$1"
    local stage="$2"
    local algo="$3"
    local checkpoint="$4"
    local checkpoint_sha="$5"
    local protocol_nfe="$6"
    local output_dir="$EVAL_ROOT/$stage/$algo/$task/seed$SEED/$EVAL_PROTOCOL/default"
    local log_dir="$LOG_ROOT/$stage/$algo/$task/seed$SEED/$EVAL_PROTOCOL/default"
    local console_log="$log_dir/console.log"
    local -a command=(
        "$PY" -u "$ROOT/eval_sim_rollout.py"
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

    if [[ -s "$output_dir/eval_log.json" && "$SKIP_DONE" == 1 ]]; then
        echo "[SKIP] $task $stage-$algo already evaluated"
        return 0
    fi
    if [[ -e "$output_dir" ]]; then
        echo "Refusing to overwrite incomplete or existing output: $output_dir" >&2
        return 1
    fi

    echo "[EVAL] $task $stage-$algo checkpoint=$(basename "$checkpoint")"
    if (( DRY_RUN )); then
        print_command "${command[@]}"
        return 0
    fi

    mkdir -p "$log_dir"
    set +e
    "${command[@]}" 2>&1 | tee "$console_log"
    eval_status=${PIPESTATUS[0]}
    set -e
    if (( eval_status != 0 )); then
        echo "Evaluation failed ($eval_status): $task $stage-$algo" >&2
        return "$eval_status"
    fi
    if [[ ! -s "$output_dir/eval_log.json" ]]; then
        echo "Evaluation returned success but eval_log.json is missing: $output_dir" >&2
        return 1
    fi

    "$PY" - "$output_dir/eval_log.json" "$checkpoint" "$checkpoint_sha" "$stage" "$algo" "$task" "$SEED" "$N_DEMO" "$N_ENVS" "$N_TEST" "$N_TEST_VIS" "$TEST_START_SEED" "$protocol_nfe" <<'PY'
import json
import os
import pathlib
import sys

(
    log_path, checkpoint, checkpoint_sha, stage, algorithm, task, seed,
    n_demo, n_envs, n_test, n_test_vis, test_start_seed, protocol_nfe,
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
    "evaluation/protocol_nfe": int(protocol_nfe) if protocol_nfe.isdigit() else protocol_nfe,
    "evaluation/reused": False,
})
tmp = path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
tmp.replace(path)
PY
    echo "[DONE] $task $stage-$algo -> $output_dir/eval_log.json"
}

create_pasdp_reuse() {
    local task="$1"
    local checkpoint="$2"
    local checkpoint_sha="$3"
    local source_dir="$EVAL_ROOT/A/pasdp/$task/seed$SEED/$EVAL_PROTOCOL/default"
    local target_dir="$EVAL_ROOT/B/pasdp/$task/seed$SEED/$EVAL_PROTOCOL/default"

    if (( DRY_RUN )); then
        echo "[REUSE] $task B-PASDP <- A-PASDP"
        return 0
    fi
    if [[ ! -s "$source_dir/eval_log.json" ]]; then
        echo "A-PASDP evaluation is missing for reuse: $source_dir" >&2
        return 1
    fi
    if [[ -s "$target_dir/reuse_manifest.json" && "$SKIP_DONE" == 1 ]]; then
        echo "[SKIP] $task B-PASDP reuse record already exists"
        return 0
    fi
    if [[ -e "$target_dir" ]]; then
        echo "Refusing to overwrite existing reuse directory: $target_dir" >&2
        return 1
    fi
    mkdir -p "$target_dir"
    "$PY" - "$target_dir/reuse_manifest.json" "$source_dir/eval_log.json" "$checkpoint" "$checkpoint_sha" "$task" "$SEED" <<'PY'
import json
import os
import pathlib
import sys

out, source_log, checkpoint, checkpoint_sha, task, seed = sys.argv[1:]
payload = {
    "stage": "B",
    "algorithm": "pasdp",
    "task_name": task,
    "training_seed": int(seed),
    "reused": True,
    "source_evaluation_log": os.path.realpath(source_log),
    "source_checkpoint": os.path.realpath(checkpoint),
    "source_checkpoint_sha256": checkpoint_sha,
}
pathlib.Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
    echo "[DONE] $task B-PASDP reuse record"
}

write_selection_manifest() {
    local task="$1"
    shift
    local output="$EVAL_ROOT/checkpoint_selection/$task/seed$SEED/$EVAL_PROTOCOL/checkpoints.tsv"
    if (( DRY_RUN || CHECK_ONLY )); then
        return 0
    fi
    mkdir -p "$(dirname "$output")"
    {
        printf 'id\tcheckpoint\tsha256\trule\n'
        while (( $# > 0 )); do
            printf '%s\t%s\t%s\tmax_topk_score_then_latest_epoch\n' "$1" "$2" "$3"
            shift 3
        done
    } > "$output"
}

write_summaries() {
    if (( DRY_RUN )); then
        return 0
    fi
    mkdir -p "$SUMMARY_ROOT"
    local tasks_csv
    tasks_csv="$(IFS=,; echo "${TASKS[*]}")"
    "$PY" - "$EVAL_ROOT" "$tasks_csv" "$SEED" "$EVAL_PROTOCOL" "$SUMMARY_ROOT" <<'PY'
import json
import pathlib
import sys

root, tasks_csv, seed, protocol, summary_root = sys.argv[1:]
root = pathlib.Path(root)
summary_root = pathlib.Path(summary_root)
tasks = tasks_csv.split(",")
rows = []

def metric(data, name):
    return data.get(name)

for task in tasks:
    task_rows = []
    for path in sorted(root.glob(f"[AB]/*/{task}/seed{seed}/{protocol}/default/eval_log.json")):
        data = json.loads(path.read_text())
        row = {
            "task": task,
            "stage": data.get("evaluation/stage"),
            "algorithm": data.get("evaluation/algorithm"),
            "reused": False,
            "protocol_nfe": data.get("evaluation/protocol_nfe"),
            "test_success_rate": metric(data, "test/success_rate"),
            "test_avg_completion_step": metric(data, "test/avg_completion_step"),
            "test_median_completion_step": metric(data, "test/median_completion_step"),
            "policy_avg_inference_time_ms": metric(data, "policy/avg_inference_time_ms"),
            "checkpoint_sha256": data.get("evaluation/checkpoint_sha256"),
            "record": str(path),
        }
        task_rows.append(row)

    reuse_path = root / "B" / "pasdp" / task / f"seed{seed}" / protocol / "default" / "reuse_manifest.json"
    if reuse_path.is_file():
        reuse = json.loads(reuse_path.read_text())
        source_path = pathlib.Path(reuse["source_evaluation_log"])
        source = json.loads(source_path.read_text())
        task_rows.append({
            "task": task,
            "stage": "B",
            "algorithm": "pasdp",
            "reused": True,
            "protocol_nfe": source.get("evaluation/protocol_nfe"),
            "test_success_rate": metric(source, "test/success_rate"),
            "test_avg_completion_step": metric(source, "test/avg_completion_step"),
            "test_median_completion_step": metric(source, "test/median_completion_step"),
            "policy_avg_inference_time_ms": metric(source, "policy/avg_inference_time_ms"),
            "checkpoint_sha256": reuse.get("source_checkpoint_sha256"),
            "record": str(reuse_path),
        })

    order = {("A", "pasdp"): 0, ("A", "dp"): 1, ("A", "sdp"): 2,
             ("A", "flow"): 3, ("B", "pasdp"): 4, ("B", "dp"): 5,
             ("B", "sdp"): 6, ("B", "flow"): 7}
    task_rows.sort(key=lambda row: order[(row["stage"], row["algorithm"])])
    rows.extend(task_rows)

columns = [
    "task", "stage", "algorithm", "reused", "protocol_nfe",
    "test_success_rate", "test_avg_completion_step",
    "test_median_completion_step", "policy_avg_inference_time_ms",
    "checkpoint_sha256", "record",
]

def write_pair(target, selected_rows):
    target.mkdir(parents=True, exist_ok=True)
    (target / "eval_summary.json").write_text(
        json.dumps(selected_rows, indent=2, sort_keys=True) + "\n"
    )
    with (target / "eval_summary.tsv").open("w") as stream:
        stream.write("\t".join(columns) + "\n")
        for row in selected_rows:
            stream.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")

write_pair(summary_root, rows)
for task in tasks:
    task_target = root / "summary" / task / f"seed{seed}" / protocol
    write_pair(task_target, [row for row in rows if row["task"] == task])
PY
    echo "[SUMMARY] $SUMMARY_ROOT/eval_summary.tsv"
}

wait_for_training

if (( ! DRY_RUN && ! CHECK_ONLY )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is not available." >&2
        exit 1
    fi
    gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    if (( gpu_count < 2 )); then
        echo "Expected two GPUs, but nvidia-smi reported $gpu_count." >&2
        exit 1
    fi
    if pgrep -f 'train_sim.py' >/dev/null; then
        echo "Refusing to start evaluation while train_sim.py is still running." >&2
        exit 1
    fi
fi

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_LIST"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE"
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROOT/third_party/robosuite:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1

test_end_seed=$((TEST_START_SEED + N_TEST - 1))
echo "Nine-task sequential A+B formal evaluation"
echo "physical_gpu=$PHYSICAL_GPU cuda_visible_devices=$CUDA_VISIBLE_DEVICES egl_device=$MUJOCO_EGL_DEVICE_ID"
echo "training_seed=$SEED n_demo=$N_DEMO n_envs=$N_ENVS n_test=$N_TEST test_seeds=$TEST_START_SEED-$test_end_seed"
echo "checkpoint_rule=max_topk_score_then_latest_epoch"

for task in "${TASKS[@]}"; do
    echo
    echo "========== $task =========="

    A_PASDP_CKPT="$(resolve_topk_checkpoint "$task" A pasdp)"
    A_DP_CKPT="$(resolve_topk_checkpoint "$task" A dp)"
    A_SDP_CKPT="$(resolve_topk_checkpoint "$task" A sdp)"
    A_FLOW_CKPT="$(resolve_topk_checkpoint "$task" A flow)"
    B_DP_CKPT="$(resolve_topk_checkpoint "$task" B dp)"
    B_SDP_CKPT="$(resolve_topk_checkpoint "$task" B sdp)"
    B_FLOW_CKPT="$(resolve_topk_checkpoint "$task" B flow)"

    for algo_ckpt in \
        "pasdp:$A_PASDP_CKPT" "dp:$A_DP_CKPT" "sdp:$A_SDP_CKPT" "flow:$A_FLOW_CKPT" \
        "dp:$B_DP_CKPT" "sdp:$B_SDP_CKPT" "flow:$B_FLOW_CKPT"; do
        algo="${algo_ckpt%%:*}"
        checkpoint="${algo_ckpt#*:}"
        validate_training_run "$task" "$algo" "$checkpoint"
    done

    if (( DRY_RUN || CHECK_ONLY )); then
        A_PASDP_SHA=dry-run
        A_DP_SHA=dry-run
        A_SDP_SHA=dry-run
        A_FLOW_SHA=dry-run
        B_DP_SHA=dry-run
        B_SDP_SHA=dry-run
        B_FLOW_SHA=dry-run
    else
        A_PASDP_SHA="$(sha256sum "$A_PASDP_CKPT" | awk '{print $1}')"
        A_DP_SHA="$(sha256sum "$A_DP_CKPT" | awk '{print $1}')"
        A_SDP_SHA="$(sha256sum "$A_SDP_CKPT" | awk '{print $1}')"
        A_FLOW_SHA="$(sha256sum "$A_FLOW_CKPT" | awk '{print $1}')"
        B_DP_SHA="$(sha256sum "$B_DP_CKPT" | awk '{print $1}')"
        B_SDP_SHA="$(sha256sum "$B_SDP_CKPT" | awk '{print $1}')"
        B_FLOW_SHA="$(sha256sum "$B_FLOW_CKPT" | awk '{print $1}')"
    fi

    write_selection_manifest "$task" \
        A-PASDP "$A_PASDP_CKPT" "$A_PASDP_SHA" \
        A-DP "$A_DP_CKPT" "$A_DP_SHA" \
        A-SDP "$A_SDP_CKPT" "$A_SDP_SHA" \
        A-Flow "$A_FLOW_CKPT" "$A_FLOW_SHA" \
        B-DP "$B_DP_CKPT" "$B_DP_SHA" \
        B-SDP "$B_SDP_CKPT" "$B_SDP_SHA" \
        B-Flow "$B_FLOW_CKPT" "$B_FLOW_SHA"

    if (( CHECK_ONLY )); then
        echo "[READY] $task: seven unique checkpoints validated"
        continue
    fi

    run_eval "$task" A pasdp "$A_PASDP_CKPT" "$A_PASDP_SHA" "steady1/cold41"
    run_eval "$task" A dp "$A_DP_CKPT" "$A_DP_SHA" 10
    run_eval "$task" A sdp "$A_SDP_CKPT" "$A_SDP_SHA" 8
    run_eval "$task" A flow "$A_FLOW_CKPT" "$A_FLOW_SHA" 1
    create_pasdp_reuse "$task" "$A_PASDP_CKPT" "$A_PASDP_SHA"
    run_eval "$task" B dp "$B_DP_CKPT" "$B_DP_SHA" 1
    run_eval "$task" B sdp "$B_SDP_CKPT" "$B_SDP_SHA" 1
    run_eval "$task" B flow "$B_FLOW_CKPT" "$B_FLOW_SHA" 1
done

if (( CHECK_ONLY )); then
    echo "All nine tasks and 63 unique training checkpoints are ready for evaluation."
elif (( DRY_RUN )); then
    echo "Dry run completed; no evaluation outputs were written."
else
    write_summaries
    echo "All 63 real evaluations completed; nine B-PASDP reuse records were added."
fi
