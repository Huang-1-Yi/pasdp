# FocalPolicy：Adroit 点云复现

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/focalpolicy/adroit

nohup env CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=focal_adroit \
  task=adroit/door_pointcloud \
  training.seed=42 training.resume=false \
  training.early_stopping.enabled=false \
> run_logs/focalpolicy/adroit/focal_door_seed42.log 2>&1 &
```

Hammer/Pen 使用 `task=adroit/hammer_pointcloud`、`task=adroit/pen_pointcloud`。
第一阶段三个任务和各消融配置均只跑 seed42；全部主轴覆盖后再补 seed43/44。
完整三算法比较函数与多 seed 要求见 [点云三算法对比](../点云三算法对比.md)。
