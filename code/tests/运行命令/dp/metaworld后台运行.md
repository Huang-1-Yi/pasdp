# DP：MetaWorld 点云运行

状态：已接入通用 observation factory；`door-close` 已通过真实 dataset、1 batch、验证和
1-step rollout。正式公平对比使用 `compare_dp_metaworld`。

前台正式运行：

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=compare_dp_metaworld \
  task=metaworld/door-close_pointcloud \
  training.seed=42 training.resume=false
```

后台运行：

```bash
mkdir -p run_logs/dp/metaworld
nohup env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=compare_dp_metaworld \
  task=metaworld/door-close_pointcloud \
  training.seed=42 training.resume=false \
> run_logs/dp/metaworld/dp_door-close_seed42.log 2>&1 &
```

切换场景只改 `task=metaworld/<场景>_pointcloud`。完整三算法函数、smoke 和多 seed
要求见 [点云三算法对比](../点云三算法对比.md)。
