#!/usr/bin/env bash
set -Eeuo pipefail

# Common worker for the three A+B launch scripts. Prefer running one of:
#   run_pasdp_train_ab_group1_2080ti.sh
#   run_pasdp_train_ab_group2_4090a.sh
#   run_pasdp_train_ab_group3_4090b.sh

ROOT=/home/hy/dp_pro6000
PY=/opt/miniconda3/envs/sim_torch27_v8/bin/python
SEED=42
N_DEMO=200
STATE_ROOT="$ROOT/data/experiments/run_state/remaining9_ab_seed42"
LOG_ROOT="$ROOT/data/experiments/logs/train_remaining9_ab/seed42"

usage() {
    echo "Internal usage: bash $0 {group1|group2|group3} [--dry-run]" >&2
    echo "Run one of the three run_pasdp_train_ab_group*.sh launchers instead." >&2
}

if (( $# < 1 )); then
    usage
    exit 2
fi

GROUP=$1
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

case "$GROUP" in
    group1)
        GROUP_LABEL=group1_2080ti
        PHYSICAL_GPU=0
        EGL_DEVICE=1
        TASKS=(stack_three_d0 nut_assembly_d0 three_piece_assembly_d0)
        ;;
    group2)
        GROUP_LABEL=group2_4090a
        PHYSICAL_GPU=1
        EGL_DEVICE=0
        TASKS=(pick_place_d0 threading_d0 stack_d0)
        ;;
    group3)
        GROUP_LABEL=group3_4090b
        PHYSICAL_GPU=1
        EGL_DEVICE=0
        TASKS=(coffee_preparation_d0 hammer_cleanup_d0 square_d0)
        ;;
    *)
        usage
        exit 2
        ;;
esac

STAGES=(A A A A B B B)
ALGORITHMS=(pasdp dp sdp flow dp sdp flow)
EXPERIMENTS=(
    pasdp_a_native_robomimic
    dp_a_paper_robomimic
    sdp_a_paper_robomimic
    flow_a_rgb_adapted_robomimic
    dp_b_nfe1_robomimic
    sdp_b_nfe1_robomimic
    flow_b_nfe1_rgb_adapted_robomimic
)

if [[ ! -x "$PY" ]]; then
    echo "Python environment not found: $PY" >&2
    exit 1
fi

for task in "${TASKS[@]}"; do
    dataset="$ROOT/data/robomimic/datasets/$task/${task}_abs.hdf5"
    if [[ ! -e "$dataset" ]]; then
        echo "Dataset not found: $dataset" >&2
        exit 1
    fi
done

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

if (( ! DRY_RUN )) && pgrep -f 'eval_sim_rollout.py' >/dev/null; then
    echo "Refusing to start training while eval_sim_rollout.py is running." >&2
    echo "Wait for the 100-episode evaluation to finish, then rerun this launcher." >&2
    exit 1
fi

cd "$ROOT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# MuJoCo / robosuite 1.4.1 resolves MUJOCO_EGL_DEVICE_ID against the
# machine-wide DRM render-node order.  Keep the compute GPU first (so cuda:0
# remains the requested training GPU), but also expose the GPU whose CUDA PCI
# ordinal matches the EGL render-device ordinal when the two orders differ.
if [[ "$PHYSICAL_GPU" == "$EGL_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
else
    export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU,$EGL_DEVICE"
fi
export MUJOCO_GL=egl
export MUJOCO_EGL_DEVICE_ID=$EGL_DEVICE
export PYTHONPATH="$ROOT/third_party/robosuite:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$STATE_ROOT/$GROUP_LABEL" "$LOG_ROOT/$GROUP_LABEL"

if (( ! DRY_RUN )); then
    exec 9>"$STATE_ROOT/$GROUP_LABEL/launcher.lock"
    if ! flock -n 9; then
        echo "This group already has an active launcher: $GROUP_LABEL" >&2
        exit 1
    fi
fi

echo "Group=$GROUP_LABEL physical_gpu=$PHYSICAL_GPU egl_device=$EGL_DEVICE cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "Training seed=$SEED n_demo=$N_DEMO"
echo "Tasks: ${TASKS[*]}"
echo "Per task: A-PASDP -> A-DP -> A-SDP -> A-Flow -> B-DP -> B-SDP -> B-Flow"
echo "B-PASDP reuses A-PASDP and is not retrained. D and C are not run by this script."

run_one() {
    local task=$1
    local stage=$2
    local algorithm=$3
    local experiment=$4
    local output_group="$stage/$algorithm/$task/seed$SEED"
    local marker="$STATE_ROOT/$GROUP_LABEL/${task}_${stage}_${algorithm}.done"
    local log_dir="$LOG_ROOT/$GROUP_LABEL/$task"
    local log_file="$log_dir/${stage}_${algorithm}.log"

    if [[ -f "$marker" ]]; then
        echo "[SKIP] $stage/$algorithm/$task already completed: $marker"
        return 0
    fi

    echo "[START] $stage/$algorithm/$task experiment=$experiment"
    if (( DRY_RUN )); then
        printf '[DRY-RUN] %q -u train_sim.py --config-name=train experiment=%q task_name=%q n_demo=%q training.seed=%q training.resume=false output_root=data/experiments/train output_group=%q\n' \
            "$PY" "$experiment" "$task" "$N_DEMO" "$SEED" "$output_group"
        return 0
    fi

    mkdir -p "$log_dir"
    "$PY" -u train_sim.py --config-name=train \
        "experiment=$experiment" \
        "task_name=$task" \
        "n_demo=$N_DEMO" \
        "training.seed=$SEED" \
        "training.resume=false" \
        "output_root=data/experiments/train" \
        "output_group=$output_group" \
        2>&1 | tee -a "$log_file"

    touch "$marker"
    echo "[DONE] $stage/$algorithm/$task marker=$marker"
}

for task in "${TASKS[@]}"; do
    echo "========== $GROUP_LABEL / $task =========="
    for i in "${!EXPERIMENTS[@]}"; do
        run_one "$task" "${STAGES[$i]}" "${ALGORITHMS[$i]}" "${EXPERIMENTS[$i]}"
    done
done

if (( DRY_RUN )); then
    echo "$GROUP_LABEL dry run completed; no training was started."
else
    echo "$GROUP_LABEL completed all 21 A+B training runs."
fi
