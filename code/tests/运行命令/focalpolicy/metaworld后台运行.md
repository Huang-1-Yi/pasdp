# FocalPolicy：MetaWorld 点云复现

`focal_metaworld` 使用 Focal consistency flow、DCT 频域监督、Focal XYZ PointNet、
`12/2/4` 和 `capacity=base`（`[256,512,1024]`），是当前点云复现主入口。

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
mkdir -p run_logs/focalpolicy/metaworld

nohup env CUDA_VISIBLE_DEVICES=2 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=focal_metaworld \
  task=metaworld/door-close_pointcloud \
  training.seed=42 training.resume=false \
  training.early_stopping.enabled=false \
> run_logs/focalpolicy/metaworld/focal_door-close_seed42.log 2>&1 &
```

默认 `recipe=pointcloud`：batch 128、1000 epoch、每 100 epoch rollout/checkpoint；
2080Ti 与 Pro 6000D 使用相同默认值。
第一阶段只跑 seed42；所有环境、encoder、policy 和输入配置各完成一次后，再补 seed43/44。
完整 smoke 和批量函数见 [点云三算法对比](../点云三算法对比.md)。
