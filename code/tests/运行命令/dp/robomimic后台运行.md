# DP：Robomimic 后台运行

状态：可运行。已验证 Robomimic encoder；ConvNeXt-Tiny 已完成训练启动验证。

## 1. 安装 YAML

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
cp tests/运行命令/dp/robomimic_dp_command_v1.yaml \
  diffusion_policy/config/robomimic_dp_command_v1.yaml
mkdir -p run_logs/dp/robomimic
```

## 2. 单任务后台启动函数

以下代码块先整体粘贴一次。默认 GPU 0、Robomimic raw encoder；切换 ConvNeXt 时执行
`RGB_MODEL=timm_convnext_tiny`。

```bash
GPU=${GPU:-0}
RGB_MODEL=${RGB_MODEL:-robomimic_raw}

run_one() {
  task="$1"
  dataset="data/robomimic/datasets/${task}/${task}_abs.hdf5"
  if [ ! -f "$dataset" ]; then
    echo "缺少数据：$dataset"
    return 1
  fi
  log="run_logs/dp/robomimic/dp_${task}_${RGB_MODEL}.log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
  conda run --no-capture-output -n sim_torch27_v8 \
  python -u train_sim.py \
    --config-name=robomimic_dp_command_v1 \
    task_name="$task" n_demo=200 n_obs_steps=2 n_action_steps=8 horizon=16 \
    policy.rgb_model="$RGB_MODEL" \
    training.resume=false \
    name="dp_${task}_${RGB_MODEL}" \
    > "$log" 2>&1 &
  echo "task=$task pid=$! log=$log"
}
```

## 3. 12 个任务入口

每次只选择需要的任务运行；先检查 GPU 显存，不要把12条同时粘贴到单卡。

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

使用已缓存的 ConvNeXt-Tiny：

```bash
RGB_MODEL=timm_convnext_tiny
run_one coffee_d0
```

若权重尚未缓存，首次前台运行时去掉 `HF_HUB_OFFLINE=1`；后台离线模式不会下载缺失权重。

## 4. 查看与停止

```bash
tail -f run_logs/dp/robomimic/dp_coffee_d0_robomimic_raw.log
pgrep -af 'train_sim.py.*robomimic_dp_command_v1'
nvidia-smi
```

启动时记录 `$!` 的 PID，停止时使用 `kill <PID>`，不要使用会误杀其他实验的
`pkill -f train_sim.py`。
