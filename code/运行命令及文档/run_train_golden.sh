#!/usr/bin/env bash
# 一键启动 PADP_Golden_Standard 训练 (Square_D0, positionwise, sample pred)
#
# 使用 equidiff2 conda env,设置 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# 兼容 tensorboard 2.10.1 (旧 proto 代码) + protobuf 4.x。

set -euo pipefail

# 强制使用 equidiff2 的 python (避免 PATH 里其他 venv 抢走)
PY=/opt/miniconda3/envs/equidiff2/bin/python
if [ ! -x "$PY" ]; then
    echo "ERROR: equidiff2 python not found at $PY" >&2
    exit 1
fi

# 解决 tensorboard 2.10.1 在 protobuf 4.x 下 _pb2.py 老 API 不兼容的问题
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 锁 GPU:物理 GPU 0 被另一个进程占 19GB(只给 OOM),用物理 GPU 1
# (几乎空闲,684MB)。CUDA_VISIBLE_DEVICES=1 让 Python 进程只看到
# 物理 GPU 1,否则默认 ALL 会被调度到 GPU 0 一起 OOM。
# 用法:GPU=1 ./run_train_golden.sh     # 走物理 GPU 1
#      GPU=0 ./run_train_golden.sh     # 走物理 GPU 0 (默认会和 19GB 进程抢)
GPU_ID="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "[INFO] CUDA_VISIBLE_DEVICES=$GPU_ID (physical GPU $GPU_ID)"

cd /media/disk7t/PADP_v3/PADP_v3

exec "$PY" train_sim.py \
    --config-name=robomimic_padp_position_wise \
    task_name=square_d0 \
    n_demo=200 \
    n_obs_steps=1 \
    horizon=40 \
    window_exp_gamma=0.25 \
    policy.noise_schedule_mode=positionwise \
    dataloader.batch_size=64 \
    policy.pred_type=sample \
    policy.noise_scheduler.prediction_type=sample \
    exp_name=PADP_Golden_Standard \
    task.env_runner.n_test_vis=0 \
    task.env_runner.n_train_vis=0 \
    hydra.run.dir=outputs_train/PADP_Golden_Standard \
    "$@"
