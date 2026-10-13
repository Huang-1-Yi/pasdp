#!/usr/bin/env bash
set -Eeuo pipefail

# Finish the only two pending runs in the remaining-nine-task A+B matrix.
# Use the dedicated launchers instead of invoking this file directly:
#   run_pasdp_train_pending_b_sdp_gpu0.sh
#   run_pasdp_train_pending_b_flow_gpu1.sh

ROOT=/home/hy/dp_pro6000
PY=/opt/miniconda3/envs/sim_torch27_v8/bin/python
SEED=42
N_DEMO=200
TASK=three_piece_assembly_d0
STATE_DIR="$ROOT/data/experiments/run_state/remaining9_ab_seed42/group1_2080ti"
LOG_DIR="$ROOT/data/experiments/logs/train_remaining9_ab/seed42/group1_2080ti/$TASK"

usage() {
    echo "Usage: bash $0 {b_sdp_gpu0|b_flow_gpu1} [--dry-run]" >&2
}

if (( $# < 1 )); then
    usage
    exit 2
fi

LANE=$1
shift
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then
    DRY_RUN=1
    shift
fi
if (( $# > 0 )); then
    usage
    exit 2
fi

case "$LANE" in
    b_sdp_gpu0)
        STAGE=B
        ALGORITHM=sdp
        EXPERIMENT=sdp_b_nfe1_robomimic
        PHYSICAL_GPU=0
        EGL_DEVICE=1
        CUDA_VISIBLE_LIST=0,1
        ;;
    b_flow_gpu1)
        STAGE=B
        ALGORITHM=flow
        EXPERIMENT=flow_b_nfe1_rgb_adapted_robomimic
        PHYSICAL_GPU=1
        EGL_DEVICE=0
        CUDA_VISIBLE_LIST=1,0
        ;;
    *)
        usage
        exit 2
        ;;
esac

DATASET="$ROOT/data/robomimic/datasets/$TASK/${TASK}_abs.hdf5"
MARKER="$STATE_DIR/${TASK}_${STAGE}_${ALGORITHM}.done"
LOG_FILE="$LOG_DIR/${STAGE}_${ALGORITHM}.log"
LOCK_FILE="$STATE_DIR/${TASK}_${STAGE}_${ALGORITHM}.lock"
OUTPUT_GROUP="$STAGE/$ALGORITHM/$TASK/seed$SEED"

if [[ ! -x "$PY" ]]; then
    echo "Python environment not found: $PY" >&2
    exit 1
fi
if [[ ! -e "$DATASET" ]]; then
    echo "Dataset not found: $DATASET" >&2
    exit 1
fi
if [[ -f "$MARKER" ]]; then
    echo "[SKIP] Already completed: $MARKER"
    exit 0
fi

"$PY" - "$ROOT/diffusion_policy/config/env_runner/robomimic/standard.yaml" <<'PY'
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
cfg = yaml.safe_load(path.read_text())
actual = {
    "n_train": int(cfg["n_train"]),
    "n_train_vis": int(cfg["n_train_vis"]),
    "n_test": int(cfg["n_test"]),
    "n_test_vis": int(cfg["n_test_vis"]),
    "n_envs": int(cfg["n_envs"]),
    "test_start_seed": int(cfg["test_start_seed"]),
}
expected = {
    "n_train": 10,
    "n_train_vis": 2,
    "n_test": 30,
    "n_test_vis": 2,
    "n_envs": 20,
    "test_start_seed": 100000,
}
if actual != expected:
    raise SystemExit(
        f"Frozen training rollout protocol mismatch in {path}: "
        f"actual={actual}, expected={expected}"
    )
PY

if (( ! DRY_RUN )); then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is not available." >&2
        exit 1
    fi
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if (( GPU_COUNT < 2 )); then
        echo "Expected two GPUs, but nvidia-smi reported $GPU_COUNT." >&2
        exit 1
    fi
    if pgrep -f 'eval_sim_rollout.py' >/dev/null; then
        echo "Refusing to start training while eval_sim_rollout.py is running." >&2
        exit 1
    fi
fi

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# The compute GPU is first, so the Hydra device cuda:0 remains the requested
# physical GPU. The other GPU stays visible because the EGL ordinal is based
# on the machine-wide DRM render-node order on this host.
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_LIST"
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE"
export PYTHONPATH="$ROOT/third_party/robosuite:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Lane=$LANE task=$TASK stage=$STAGE algorithm=$ALGORITHM"
echo "physical_gpu=$PHYSICAL_GPU egl_device=$EGL_DEVICE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "training_seed=$SEED n_demo=$N_DEMO output_group=$OUTPUT_GROUP"
echo "Training parameters are inherited unchanged from the frozen experiment config."

if (( DRY_RUN )); then
    printf '[DRY-RUN] %q -u train_sim.py --config-name=train experiment=%q task_name=%q n_demo=%q training.seed=%q training.resume=false output_root=data/experiments/train output_group=%q\n' \
        "$PY" "$EXPERIMENT" "$TASK" "$N_DEMO" "$SEED" "$OUTPUT_GROUP"
    exit 0
fi

mkdir -p "$STATE_DIR" "$LOG_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    echo "This pending run already has an active launcher: $LOCK_FILE" >&2
    exit 1
fi

{
    echo
    echo "===== fresh 4090 run $(date '+%F %T %Z') lane=$LANE ====="
} | tee -a "$LOG_FILE"

"$PY" -u train_sim.py --config-name=train \
    "experiment=$EXPERIMENT" \
    "task_name=$TASK" \
    "n_demo=$N_DEMO" \
    "training.seed=$SEED" \
    "training.resume=false" \
    "output_root=data/experiments/train" \
    "output_group=$OUTPUT_GROUP" \
    2>&1 | tee -a "$LOG_FILE"

touch "$MARKER"
echo "[DONE] $STAGE/$ALGORITHM/$TASK marker=$MARKER"
