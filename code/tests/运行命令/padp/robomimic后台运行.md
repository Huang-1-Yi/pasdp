# PADP：Robomimic 后台运行

状态：可运行。Robomimic encoder 与 `timm_convnext_tiny` 均做过训练启动验证。

## 1. 安装 YAML

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
cp tests/运行命令/padp/robomimic_padp_command_v1.yaml \
  diffusion_policy/config/robomimic_padp_command_v1.yaml
mkdir -p run_logs/padp/robomimic
```

## 2. 单任务后台启动函数

```bash
GPU=${GPU:-0}
RGB_MODEL=${RGB_MODEL:-robomimic}

run_one() {
  task="$1"
  dataset="data/robomimic/datasets/${task}/${task}_abs.hdf5"
  if [ ! -f "$dataset" ]; then
    echo "缺少数据：$dataset"
    return 1
  fi
  log="run_logs/padp/robomimic/padp_${task}_${RGB_MODEL}.log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
  conda run --no-capture-output -n sim_torch27_v8 \
  python -u train_sim.py \
    --config-name=robomimic_padp_command_v1 \
    task_name="$task" n_demo=200 n_obs_steps=1 n_action_steps=1 horizon=40 \
    window_exp_gamma=0.25 \
    policy.noise_schedule_mode=positionwise \
    policy.pred_type=sample \
    policy.noise_scheduler.prediction_type=sample \
    policy.rgb_model="$RGB_MODEL" \
    training.resume=false \
    name="padp_${task}_${RGB_MODEL}" \
    > "$log" 2>&1 &
  echo "task=$task pid=$! log=$log"
}
```

## 3. 12 个任务入口

```bash
run_one coffee_d0
run_one coffee_preparation_d0
run_one hammer_cleanup_d0
run_one kitchen_d0
run_one mug_cleanup_d0
run_one nut_assembly_d0
run_one pick_place_d0
run_one square_d0
run_one stack_d0
run_one stack_three_d0
run_one threading_d0
run_one three_piece_assembly_d0
```

ConvNeXt-Tiny 示例：

```bash
RGB_MODEL=timm_convnext_tiny
run_one coffee_d0
```

一次只启动符合显存预算的任务数。`HF_HUB_OFFLINE=1` 要求 ConvNeXt 权重已经缓存。

## 4. 查看与停止

```bash
tail -f run_logs/padp/robomimic/padp_coffee_d0_robomimic.log
pgrep -af 'train_sim.py.*robomimic_padp_command_v1'
nvidia-smi
```

使用启动时输出的 PID 执行 `kill <PID>`。
