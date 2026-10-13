#!/usr/bin/env bash
set -euo pipefail

cd /home/hy/dp_pro6000
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam_real

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
export PYTHONPATH=/home/hy/dp_pro6000:/home/hy/FasterWAM:/home/hy/FasterWAM/src:${PYTHONPATH:-}
export DIFFSYNTH_MODEL_BASE_PATH=/home/hy/FasterWAM/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES="$({
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  } | sort -t, -k2,2n | head -n1 | cut -d, -f1 | tr -d ' ')"
fi
export CUDA_VISIBLE_DEVICES
echo "Selected physical GPU: ${CUDA_VISIBLE_DEVICES} (model device cuda:0)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_id="fold100_fasterwam_$(date +%Y%m%d_%H%M%S)"
run_dir="/home/hy/dp_pro6000/data/fasterwam_real/${run_id}"
mkdir -p "${run_dir}"

python -u -m diffusion_policy.envs.realword.fasterwam.run_fold100_fasterwam_franka \
  --output "${run_dir}" \
  --left_robot_ip 192.168.0.168 \
  --right_robot_ip 192.168.0.168 \
  --camera_serial_numbers 248622300313,352122274349,409122274331 \
  --camera_resolution 640x480 \
  --obs_resolution 640x480,640x480,640x480 \
  --device cuda:0 \
  --frequency 10 \
  --num_inference_steps 10 \
  --action_horizon 32 \
  --replan_steps 1 \
  2>&1 | tee "${run_dir}/console.log"
