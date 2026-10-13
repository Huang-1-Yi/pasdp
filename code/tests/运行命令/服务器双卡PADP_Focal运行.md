# 服务器双卡并行：PADP 卡 2、Focal 卡 1

本轮只比较 PADP 与 Focal：两个算法并行占用不同的 Pro 6000D，同一算法的多个 seed 在
自己的 GPU 上串行执行。

| 算法 | 物理 GPU | 默认输出 |
|---|---:|---|
| PADP | 2 | `data/outputs/padp/` |
| Focal | 1 | `data/outputs/focal/` |

两者固定使用相同 `capacity=base`、Focal XYZ encoder、`12/2/4`、batch 128、1000 epoch
以及每 100 epoch rollout/checkpoint。

## 1. 登录、同步与数据检查

```bash
ssh 10.242.5.12
cd /media/disk7t/dp_pro6000
git pull

conda env list | grep sim_torch27_v8
nvidia-smi -i 1,2
ls -ld data/metaworld_door-close_expert.zarr

mkdir -p data/outputs/padp data/outputs/focal run_logs/pointcloud_compare
test -w data/outputs/padp && test -w data/outputs/focal
```

数据整理好后按照 [数据软链接](数据软链接.md) 执行链接脚本。

## 2. 先 dry-run

```bash
ALGOS=padp DRY_RUN=1 CUDA_VISIBLE_DEVICES=2 CONDA_ENV=sim_torch27_v8 \
bash tests/运行命令/卡2点云对比队列.sh metaworld door-close 42

ALGOS=focal DRY_RUN=1 CUDA_VISIBLE_DEVICES=1 CONDA_ENV=sim_torch27_v8 \
bash tests/运行命令/卡2点云对比队列.sh metaworld door-close 42
```

脚本文件名中的“卡2”是历史命名；实际使用传入的 `CUDA_VISIBLE_DEVICES`，第二条会正确
使用物理卡 1。

## 3. 后台并行启动

PADP → 卡 2：

```bash
mkdir -p run_logs/pointcloud_compare
nohup env ALGOS=padp CUDA_VISIBLE_DEVICES=2 \
CONDA_ENV=sim_torch27_v8 HYDRA_FULL_ERROR=1 \
bash tests/运行命令/卡2点云对比队列.sh metaworld door-close 42 \
> run_logs/pointcloud_compare/queue_padp_gpu2_door-close.log 2>&1 &
echo "PADP pid=$!"
```

Focal → 卡 1：

```bash
nohup env ALGOS=focal CUDA_VISIBLE_DEVICES=1 \
CONDA_ENV=sim_torch27_v8 HYDRA_FULL_ERROR=1 \
bash tests/运行命令/卡2点云对比队列.sh metaworld door-close 42 \
> run_logs/pointcloud_compare/queue_focal_gpu1_door-close.log 2>&1 &
echo "Focal pid=$!"
```

第一阶段固定只跑 seed42。环境、encoder、policy 和输入主轴都走完一遍后，再把两条命令
末尾改成 `43 44`，补最终 seed 消融。

## 4. 查看进度

```bash
tail -f run_logs/pointcloud_compare/queue_padp_gpu2_door-close.log
tail -f run_logs/pointcloud_compare/queue_focal_gpu1_door-close.log

nvidia-smi -i 1,2
pgrep -af 'train_sim.py.*(compare_padp_metaworld|focal_metaworld)'
```

详细训练日志分别位于：

```text
run_logs/pointcloud_compare/metaworld_door-close/padp_*.log
run_logs/pointcloud_compare/metaworld_door-close/focal_*.log
```

Hydra 结果分别位于项目内：

```text
data/outputs/padp/<日期>/<实验目录>/
data/outputs/focal/<日期>/<实验目录>/
```

只有管理员已创建可写的外部目录时，才可以在两条命令中显式增加：

```text
OUTPUT_ROOT=/某个可写目录
```

不要使用当前无权限的 `/data/output`。

## 5. Adroit 切换

把两条命令中的：

```text
metaworld door-close
```

换成以下任一项：

```text
adroit door
adroit hammer
adroit pen
```
