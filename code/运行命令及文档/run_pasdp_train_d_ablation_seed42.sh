#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/hy/dp_pro6000
PY=/opt/miniconda3/envs/sim_torch27_v8/bin/python
SEED=42
N_DEMO=200
QUEUE=${1:-}
shift || true
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ -z "$QUEUE" || $# -ne 0 ]]; then
    echo "Usage: bash $0 {queue1|queue2|queue3|queue4} [--dry-run]" >&2
    exit 2
fi

case "$QUEUE" in
  queue1) GPU=0; EGL=1; TASKS=(coffee_d0 pick_place_d0 stack_three_d0) ;;
  queue2) GPU=0; EGL=1; TASKS=(threading_d0 hammer_cleanup_d0 nut_assembly_d0) ;;
  queue3) GPU=1; EGL=0; TASKS=(coffee_preparation_d0 square_d0) ;;
  queue4) GPU=1; EGL=0; TASKS=(stack_d0 three_piece_assembly_d0) ;;
  *) echo "Unknown queue: $QUEUE" >&2; exit 2 ;;
esac

CONDITIONS=(d1 d2 d12)
EXPERIMENTS=(
  pasdp_d1_schedule_ablation_robomimic
  pasdp_d2_loss_weight_ablation_robomimic
  pasdp_d12_schedule_loss_ablation_robomimic
)
STATE_ROOT="$ROOT/data/experiments/run_state/d_ablation_seed42/$QUEUE"
LOG_ROOT="$ROOT/data/experiments/logs/train_d_ablation/seed42/$QUEUE"

[[ -x "$PY" ]] || { echo "Python environment not found: $PY" >&2; exit 1; }
for task in "${TASKS[@]}"; do
  [[ -e "$ROOT/data/robomimic/datasets/$task/${task}_abs.hdf5" ]] || {
    echo "Dataset not found: $task" >&2; exit 1;
  }
done

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU,$EGL"
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID="$EGL" PYOPENGL_PLATFORM=egl
export PYTHONPATH="$ROOT/third_party/robosuite:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1 WANDB_MODE=offline PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$STATE_ROOT" "$LOG_ROOT"
if (( ! DRY_RUN )); then
  exec 9>"$STATE_ROOT/launcher.lock"
  flock -n 9 || { echo "$QUEUE already active" >&2; exit 1; }
fi

echo "queue=$QUEUE physical_gpu=$GPU egl_device=$EGL tasks=${TASKS[*]}"
for task in "${TASKS[@]}"; do
  for index in "${!CONDITIONS[@]}"; do
    condition=${CONDITIONS[$index]}
    experiment=${EXPERIMENTS[$index]}
    marker="$STATE_ROOT/${task}_${condition}.done"
    log="$LOG_ROOT/${task}_${condition}.log"
    if [[ -f "$marker" ]]; then
      echo "[SKIP] $task $condition"
      continue
    fi
    command=("$PY" -u train_sim.py --config-name=train
      "experiment=$experiment" "task_name=$task" "n_demo=$N_DEMO"
      "training.seed=$SEED" "training.resume=false"
      "output_root=data/experiments/train"
      "output_group=D_ablation/$condition/$task/seed$SEED")
    echo "[START] $task $condition"
    if (( DRY_RUN )); then printf '%q ' "${command[@]}"; printf '\n'; continue; fi
    "${command[@]}" 2>&1 | tee -a "$log"
    touch "$marker"
    echo "[DONE] $task $condition"
  done
done
