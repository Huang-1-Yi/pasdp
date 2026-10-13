# FocalPolicy：Robomimic 后台运行

状态：当前 Focal consistency-flow policy 的 Robomimic RGB 路径已完成烟测；正式12任务结果待跑。

## 1. 准备目录

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/focalpolicy/robomimic
```

## 2. 单任务后台启动函数

```bash
GPU=${GPU:-0}
ENCODER=${ENCODER:-robomimic_raw}
SEED=${SEED:-42}

run_one() {
  task="$1"
  dataset="data/robomimic/datasets/${task}/${task}_abs.hdf5"
  if [ ! -f "$dataset" ]; then
    echo "缺少数据：$dataset"
    return 1
  fi
  log="run_logs/focalpolicy/robomimic/focalpolicy_${task}_${ENCODER}_seed${SEED}.log"
  nohup env CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
  conda run --no-capture-output -n sim_torch27_v8 \
  python -u train_sim.py \
    --config-name=train experiment=focal_robomimic \
    encoder="rgb/$ENCODER" \
    task_name="$task" n_demo=200 \
    training.seed="$SEED" training.resume=false \
    training.early_stopping.enabled=false \
    name="focalpolicy_${task}_${ENCODER}_seed${SEED}" \
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

ConvNeXt-Tiny 属于“encoder 已验证、FocalPolicy 组合待验证”：

```bash
ENCODER=timm_convnext_tiny
run_one coffee_d0
```

第一阶段所有 policy/encoder/task 组合都只使用默认 `SEED=42`。四个实验主轴各走完一遍后，
才分别设置 `SEED=43`、`SEED=44` 补最终 seed 消融；不要只给 seed42 的优胜配置补 seed。

## 4. 查看与停止

```bash
tail -f run_logs/focalpolicy/robomimic/focalpolicy_coffee_d0_robomimic_raw_seed42.log
pgrep -af 'train_sim.py.*focal_robomimic'
nvidia-smi
```

使用启动时输出的 PID 执行 `kill <PID>`。
