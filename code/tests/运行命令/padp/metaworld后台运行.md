# PADP：MetaWorld 点云运行

公平对比入口是 `compare_padp_metaworld`，固定为和 DP/Focal 相同的点云 encoder、
`12/2/4`、head 宽度和训练预算。原始 `experiment=padp_metaworld` 的 `40/1/1` 仍保留，
但不能拿来直接归因三算法差异。

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/padp/metaworld

nohup env CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=compare_padp_metaworld \
  task=metaworld/door-close_pointcloud \
  training.seed=42 training.resume=false \
> run_logs/padp/metaworld/padp_door-close_seed42.log 2>&1 &
```

完整 smoke、三算法并行函数和多 seed 规范见
[点云三算法对比](../点云三算法对比.md)。
