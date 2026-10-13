# PADP：Adroit 点云运行

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/padp/adroit

nohup env CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=compare_padp_adroit \
  task=adroit/door_pointcloud \
  training.seed=42 training.resume=false \
> run_logs/padp/adroit/padp_door_seed42.log 2>&1 &
```

任务选择：`adroit/door_pointcloud`、`adroit/hammer_pointcloud`、
`adroit/pen_pointcloud`。公平对比必须和 DP/Focal 使用相同 seed、recipe 与 capacity；
详见 [点云三算法对比](../点云三算法对比.md)。
