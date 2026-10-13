#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/hy/dp_pro6000
PY=${PASDP_PY:-/opt/miniconda3/envs/sim_torch27_v8/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
case "${PASDP_EVAL_PHYSICAL_GPU:-1}" in
  0) export CUDA_VISIBLE_DEVICES=0,1 MUJOCO_EGL_DEVICE_ID=1 ;;
  1) export CUDA_VISIBLE_DEVICES=1,0 MUJOCO_EGL_DEVICE_ID=0 ;;
  *) echo 'PASDP_EVAL_PHYSICAL_GPU must be 0 or 1' >&2; exit 2 ;;
esac
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl WANDB_MODE=offline HYDRA_FULL_ERROR=1
export PYTHONPATH="$ROOT/third_party/robosuite:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
exec "$PY" "$ROOT/scripts/eval_pasdp_d.py" "$@"
