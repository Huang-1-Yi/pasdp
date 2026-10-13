# DP / PADP / FocalPolicy 运行命令索引

本目录是实验启动资料库。新点云实验直接使用已经进入 Hydra 搜索路径的组合式
`--config-name=train experiment=compare_*`，不需要复制 YAML。各算法目录中保留的
`*_command_v1.yaml` 只用于旧 Robomimic 命令兼容。

正式默认值不按 GPU 型号分叉：所有 experiment 使用 `capacity=base`
（`down_dims=[256,512,1024]`）；Robomimic batch 固定为 64；MetaWorld/Adroit 点云
batch 固定为 128，并训练 1000 epoch、每 100 epoch rollout/checkpoint。只有显式指定
`capacity=smoke recipe=smoke_pointcloud` 时才会进入测试用小配置。

## 运行前统一链接数据

服务器数据集中存放在 `/media/disk7t/data`；脚本会优先自动识别该目录，不存在时兼容
`/home/hy/data`。进入项目后先执行：

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
```

新 clone 到 `/media/disk7t/dp_pro6000` 后必须重新执行；软链接属于本机状态，不会通过 Git 同步。

服务器根目录不同时可传 `DP_DATA_ROOT=/实际路径`。脚本幂等且拒绝覆盖已有真实数据；完整
规则见 [数据软链接.md](数据软链接.md)。各训练配置仍统一使用可迁移的 `data/...` 相对路径。

## 按物理 GPU 指定

Robomimic/Robosuite 同时涉及 CUDA 计算顺序和 EGL 渲染顺序，二者的数字编号
可能不同。`robomimic_square_policy队列.sh` 接受 `nvidia-smi -L` 显示的
`GPU_UUID`，再通过 PCI bus 自动将计算和 EGL 定位到同一块物理卡。不要再对
该脚本手动传裸的 `CUDA_VISIBLE_DEVICES=0/1/2`。

只检查映射而不启动训练：

```bash
GPU_MAPPING_ONLY=1 \
GPU_UUID=GPU-13d3ef42-6500-b4d3-a6e7-a78bb05375f1 \
bash tests/运行命令/robomimic_square_policy队列.sh q0
```

日志中的 `nvidia_index` 是 `nvidia-smi` 物理卡号，`cuda_visible_devices`和
`egl_index` 是脚本解析后传给两个底层的编号。训练始终使用逻辑 `cuda:0`，
它对应 `GPU_UUID` 指定的物理卡。

## 目录约定

每个算法目录均包含：

- 一个可复制到 `diffusion_policy/config/` 的主 YAML；
- `robomimic后台运行.md`：12 个代表任务；
- `metaworld后台运行.md`：12 个任务入口及数据存在性检查；
- `adroit后台运行.md`：3 个官方任务 × 4 个 seed，共12个实验入口；
- `yaml文件可选参数.md`：编码器状态、可粘贴 YAML 行和 CLI override。

## 当前真实可用性

| 算法 | Robomimic RGB | MetaWorld 点云 | Adroit 点云 |
|---|---|---|---|
| DP | 可运行；Robomimic 与 ConvNeXt-Tiny 路径做过训练启动 | 通用 encoder factory + 1 batch + rollout 已烟测 | 配置、动作维度和 Policy 已适配 |
| PADP | 可运行；Robomimic 与 ConvNeXt-Tiny 做过训练 | `40/1/1` 长实验及公平对比 `12/2/4` 均可运行 | door/hammer/pen 可运行 |
| FocalPolicy | Robomimic RGB 已烟测 | 点云 `12/2/4` 已通过 1 batch + rollout | door 已通过端到端 smoke，三任务可运行 |

## Square D0 policy 队列

`robomimic_square_policy队列.sh` 将 7 个固定 encoder 的 controlled policy 实验和
ACT/BCRNN 两个原生架构基线分成 q0、q2、q1a、q1b 四个队列。四队列
可同时启动，每个队列内部串行。完整 nohup 命令见项目根目录
`对比实验/07_Square_D0_policy首轮命令.md`。

三算法点云公平对比的统一入口、smoke、后台函数和 seed 规范见
[点云三算法对比.md](点云三算法对比.md)。
在服务器 `10.242.5.12` 的卡 2 上运行时，使用
[服务器10.242.5.12卡2运行.md](服务器10.242.5.12卡2运行.md) 和
`卡2点云对比队列.sh`，让三个算法串行占用该卡。

PADP 使用卡 2、Focal 使用卡 1 并行训练时，直接使用
[服务器双卡PADP_Focal运行.md](服务器双卡PADP_Focal运行.md)。

## 已有数据

- Robomimic 源目录：`/media/disk7t/data/robomimic`；任务以
  `data/robomimic/datasets/<task>/<task>_abs.hdf5` 是否存在为准。
- MetaWorld 源目录：`/media/disk7t/data/metaworld`；当前有 `door-close`、`hand-insert`、
  `peg-insert-side`、`pick-place`。
- Adroit 源目录：`/media/disk7t/data/adroit`；当前有 `door`、`hammer`、`pen`。

运行前必须检查数据文件，文档中的启动函数也会做存在性检查，避免后台任务启动后才报错。
