# DP：Adroit 点云运行

DP 点云已经可运行。三任务使用相同 `12/2/4` 和 Focal XYZ encoder，动作维度由 task
自动切换：door=28、hammer=26、pen=24。

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/dp/adroit

nohup env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=compare_dp_adroit \
  task=adroit/door_pointcloud \
  training.seed=42 training.resume=false \
> run_logs/dp/adroit/dp_door_seed42.log 2>&1 &
```

Hammer/Pen 分别使用 `task=adroit/hammer_pointcloud`、
`task=adroit/pen_pointcloud`。三算法和多 seed 启动函数见
[点云三算法对比](../点云三算法对比.md)。
