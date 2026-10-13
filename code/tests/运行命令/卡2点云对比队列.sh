#!/usr/bin/env bash
set -euo pipefail

# 在一张 GPU 上串行执行 DP / PADP / Focal，避免三个 batch=128 进程争抢显存。
# 用法：
#   CUDA_VISIBLE_DEVICES=2 bash tests/运行命令/卡2点云对比队列.sh \
#     metaworld door-close 42
# 主轴全部覆盖后，最后补 seed 消融：
#   CUDA_VISIBLE_DEVICES=2 bash tests/运行命令/卡2点云对比队列.sh \
#     metaworld door-close 43 44
# 只补跑 Focal：
#   ALGOS=focal CUDA_VISIBLE_DEVICES=2 bash ... metaworld door-close 42

SUITE="${1:-metaworld}"
TASK_NAME="${2:-door-close}"
shift $(( $# >= 2 ? 2 : $# ))
SEEDS=("${@:-42}")

CONDA_ENV="${CONDA_ENV:-sim_torch27_v8}"
ALGOS="${ALGOS:-dp padp focal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

case "$SUITE" in
  metaworld)
    DATA_PATH="data/metaworld_${TASK_NAME}_expert.zarr"
    TASK_CONFIG="metaworld/${TASK_NAME}_pointcloud"
    ;;
  adroit)
    DATA_PATH="data/adroit_${TASK_NAME}_expert.zarr"
    TASK_CONFIG="adroit/${TASK_NAME}_pointcloud"
    ;;
  *)
    echo "suite 只能是 metaworld 或 adroit，收到：$SUITE" >&2
    exit 2
    ;;
esac

if [[ ! -e "$DATA_PATH" ]]; then
  echo "缺少数据：$PROJECT_ROOT/$DATA_PATH" >&2
  exit 2
fi

LOG_DIR="run_logs/pointcloud_compare/${SUITE}_${TASK_NAME}"
mkdir -p "$LOG_DIR"

echo "project=$PROJECT_ROOT"
echo "env=$CONDA_ENV gpu=$CUDA_VISIBLE_DEVICES suite=$SUITE task=$TASK_NAME"
echo "algorithms=$ALGOS seeds=${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
  for algo in $ALGOS; do
    case "$algo" in
      dp|padp|focal) ;;
      *) echo "未知算法：$algo" >&2; exit 2 ;;
    esac

    if [[ "$algo" == "focal" ]]; then
      experiment="focal_${SUITE}"
    else
      experiment="compare_${algo}_${SUITE}"
    fi
    log="$LOG_DIR/${algo}_${SUITE}_${TASK_NAME}_seed${seed}.log"
    echo "[$(date '+%F %T')] START algo=$algo seed=$seed log=$log"

    cmd=(
      conda run --no-capture-output -n "$CONDA_ENV"
      python -u train_sim.py --config-name=train
      experiment="$experiment" task="$TASK_CONFIG"
      training.seed="$seed" training.resume=false
      training.early_stopping.enabled=false
    )
    if [[ -n "$OUTPUT_ROOT" ]]; then
      cmd+=(output_root="$OUTPUT_ROOT")
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      printf 'DRY_RUN:'
      printf ' %q' "${cmd[@]}"
      printf ' > %q 2>&1\n' "$log"
      continue
    fi

    if "${cmd[@]}" > "$log" 2>&1; then
      echo "[$(date '+%F %T')] DONE  algo=$algo seed=$seed"
    else
      status=$?
      echo "[$(date '+%F %T')] FAIL  algo=$algo seed=$seed status=$status" >&2
      tail -n 100 "$log" >&2 || true
      exit "$status"
    fi
  done
done

echo "[$(date '+%F %T')] ALL DONE"
