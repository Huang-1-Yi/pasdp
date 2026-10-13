# PADP 代码结构与功能 v3.2

> 代码快照：2026-07-31，`master` 提交 `ddf8a63`
> 项目根目录：`/home/hy/dp_pro6000`
> 本文以当前文件、Hydra 合成结果和单测为准，不以旧 checkpoint 或历史目录为准。

本文是 [PADP_代码结构与功能v3_1.md](PADP_代码结构与功能v3_1.md) 的后继版本，用于快速定位当前 DP / PADP / PADP-Unrolled / FocalPolicy / Streaming Diffusion / FlowPolicy / ACT / DP3 / Diffusion Transformer / BCRNN 的配置、数据、模型、训练和 rollout 链路。

---

## 1. v3.2 相对 v3.1 的主要变化

1. **新增 PADP Streaming-Unrolled Training（方向 A）**：增加了 `padp_unrolled_robomimic` 的 policy、dataset、workspace、temporal 和 experiment 完整配置链。它在训练时展开 K 个连续控制周期，逐步将上一周期预测作为下一周期输入，用来减轻 PADP 训练/推理 buffer 分布差异。
2. **Square policy 队列支持物理 GPU UUID**：`tests/运行命令/robomimic_square_policy队列.sh` 会通过 PCI bus 同时解决 CUDA 计算序号和 EGL render node 序号，避免裸 `CUDA_VISIBLE_DEVICES=0/1/2` 在异构多卡服务器上指错卡。
3. **Square 实验区分 controlled / native / full-method**：受控 policy 对比统一 encoder、capacity 和 temporal；ACT/BCRNN 等保留原生架构；PADP H40/O1/A1 和 Diffusion Transformer H16/O2/A8 可作为完整方法单独运行。
4. **纠正 v3.1 的 early stopping 描述**：当前 `base/train_common.yaml` 和正式 recipe 均默认 `training.early_stopping.enabled=false`；公平对比需跑满相同训练预算。早停 helper 仍然存在，但只在显式开启时生效。
5. **纠正 MetaWorld 任务数**：`config/task/metaworld/` 当前是 **44 个** `*_pointcloud.yaml`，不是 42 个。
6. **补全 PADP 点云 RTV8 数据层说明**：MetaWorld 和 Adroit 都有通用 SequenceSampler dataset 与 PADP RTV8 mapping-table dataset 两条路径。`compare_padp_*` 默认使用 PADP 专用 dataset 和 `shuffle=false` recipe。
7. **Robomimic runner 协议固定**：`standard` 是正式结果协议（14 train states + 50 test seeds + 32 envs）；`small` 只用于本机/小显存烟测，不得混入正式对比表。
8. **补记 PADP chunk 对齐验证**：PADP 在 O2/A4 等配置下从 `n_obs_steps-1` 开始执行动作，buffer 每次左移 `n_action_steps`，不再固定左移 1。`tests/test_padp_chunk_alignment.py` 对此进行回归测试。

---

## 2. 总体架构与责任边界

当前主链路是“Hydra 组合配置 + suite 无关 policy + shape-meta 驱动 encoder”：

```text
train_sim.py
  └─ config/train.yaml
      └─ experiment/<method>_<suite>.yaml
          ├─ task/        观测和动作 shape_meta
          ├─ dataset/     离线数据和采样协议
          ├─ env_runner/  在线 rollout 协议
          ├─ policy/      算法数学与推理逻辑
          ├─ encoder/     RGB / point cloud 编码器选择
          ├─ capacity/    预测头容量
          ├─ temporal/    H / O / A 时序协议
          ├─ recipe/      batch / epoch / rollout / checkpoint
          ├─ runtime/     CPU / CUDA
          └─ workspace/   训练编排、EMA、日志和 checkpoint
```

分层原则：

- `policy/` 按算法组织，不为 Robomimic / MetaWorld / Adroit 复制三份 policy。
- `model/` 只放网络与可复用模型组件，不放 loss 编排或 rollout 控制。
- `dataset/` 定义离线样本语义；`envs/` 定义在线交互和视频/点云 wrapper。
- `workspace/` 负责优化器、EMA、validation、rollout、checkpoint 和 W&B/JSON 日志。
- 输入模态差异交给 `shape_meta + encoder factory`，环境差异交给 `task + dataset + env_runner`。

---

## 3. 根目录

### 3.1 训练和评估入口

| 文件 | 当前作用 |
|---|---|
| [train_sim.py](train_sim.py) | **仿真主入口**。注册 `get_max_steps` / `get_ws_x_center` / `get_ws_y_center` / `eval` resolver，Hydra 合成后实例化 workspace 并执行 `run()`。 |
| [train.py](train.py) | 更通用的 Hydra 入口，只注册 `eval`；需要 `get_max_steps` 的 Robomimic 组合不应用它取代 `train_sim.py`。 |
| [eval_sim_rollout.py](eval_sim_rollout.py) | checkpoint 后的独立仿真闭环 rollout 入口，与训练 Workspace 共用 `common/rollout.py`。 |
| [eval_action_horizon.py](eval_action_horizon.py) | 离线 action-horizon 分析入口，调用 common 中的 batched/sequential 等价路径。 |
| [run_train_golden.sh](run_train_golden.sh) | PADP 历史黄金配置启动脚本。新对比实验优先用 `--config-name=train experiment=...`。 |

### 3.2 根目录中的文档和环境

| 路径 | 用途 |
|---|---|
| [README.md](README.md) | 安装、数据链接、训练、encoder 与 MetaWorld/Adroit 的主操作手册。 |
| [diffusion_policy/config/README.md](diffusion_policy/config/README.md) | Hydra 组合轴、旧 YAML 迁移和点云对比协议。 |
| [diffusion_policy/policy/README.md](diffusion_policy/policy/README.md) | policy 扁平化规则、Streaming 命名和当前输入边界。 |
| [代码简化计划.md](代码简化计划.md) | policy/model 分层已完成项与 workspace/config/archive 待办项。 |
| `conda_environment_sim_torch27_v4.yaml` | Robomimic 主回归环境清单。 |
| `conda_environment_sim_torch27_v6.yaml` | 可选 RGB encoder 按 provider 注释的核心环境。 |
| `conda_environment_sim_torch27_v8.yaml` | MetaWorld / Adroit / MuJoCo-Py 点云 rollout 环境。 |

### 3.3 主目录

| 目录 | 责任 |
|---|---|
| `diffusion_policy/` | 唯一活跃 Python 业务包。 |
| `diffusion_policy_archive/` | 历史死代码归档，不应被新 config 引用。 |
| `tests/` | 单测和可同步到服务器的运行命令。 |
| `third_party/` | 固定版本的 Robosuite / Robomimic / MetaWorld / mj_envs / mjrl / mujoco-py / PyTorch3D 等。 |
| `data/` | 数据软链接、缓存与 `outputs/`；已被 Git 忽略。 |
| `run_logs/` | `nohup` 标准输出；已被 Git 忽略。 |
| `对比实验/` | 实验协议、进度表和队列命令；当前 `.gitignore` 中默认忽略，不会随普通 `git add .` 同步。 |

---

## 4. `diffusion_policy/config`：Hydra 组合式配置

### 4.1 统一入口和配置轴

`config/train.yaml` 默认选择 `experiment=dp_robomimic`。推荐调用形式：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_robomimic \
  task_name=square_d0 n_demo=200 \
  training.resume=false
```

| 配置轴 | 目录 | 职责 |
|---|---|---|
| `base` | `config/base/` | 输出、EMA、默认 optimizer、训练、W&B、checkpoint 公共字段。 |
| `experiment` | `config/experiment/` | 可复现的薄预设，选定其余轴和 `output_group`。 |
| `task` | `config/task/` | `shape_meta`、任务名、点数和动作维度。 |
| `dataset` | `config/dataset/` | 标准 SequenceSampler、PADP RTV8 或 unrolled dataset。 |
| `env_runner` | `config/env_runner/` | Robomimic / MetaWorld / Adroit 评估环境和评估规模。 |
| `policy` | `config/policy/` | 算法数学参数，不定义实验容量。 |
| `encoder` | `config/encoder/` | RGB/点云 encoder 选择，合并到 `policy`。 |
| `capacity` | `config/capacity/` | `down_dims`、embedding、kernel、GroupNorm 等预测头容量。 |
| `temporal` | `config/temporal/` | `horizon / n_obs_steps / n_action_steps`，以及少量时序专用参数。 |
| `recipe` | `config/recipe/` | batch、worker、epoch、rollout/checkpoint 周期和 validation。 |
| `runtime` | `config/runtime/` | CPU/CUDA 运行配置。 |
| `workspace` | `config/workspace/` | 训练编排类的 `_target_`。 |
| `task_schema` | `config/task_schema/` | 可复用的 task shape 片段，目前主要是 MetaWorld 点云。 |

### 4.2 公共默认值

`base/train_common.yaml` 的当前关键语义：

- `training.device=cuda:0`：这是**进程可见卡中的逻辑卡 0**，不一定是 `nvidia-smi` 物理 GPU 0。
- `training.resume=true`：正式从零实验必须显式覆盖 `training.resume=false`。
- `training.optimizer_source=hydra`；ACT 改为 `model`，Diffusion Transformer 改为 `policy_method`。
- `training.early_stopping.enabled=false`；helper 阈值默认 1.0、patience 默认 3。
- EMA 默认开启，ACT 主动关闭。
- 默认输出是 `data/outputs/<output_group>/<date>/<time>_<name>_<task>`。
- checkpoint 默认保留 latest 和 `test_mean_score` top-5。

### 4.3 19 个活跃 experiment

| 类别 | Experiment | 核心定位 |
|---|---|---|
| Robomimic | `dp_robomimic` | DP-DDIM RGB 基线。 |
| Robomimic | `padp_robomimic` | PADP H40/O1/A1 + RGB RTV8 黄金基线。 |
| Robomimic | `padp_unrolled_robomimic` | PADP K=2 展开训练的**实验性烟测预设**。 |
| Robomimic | `focal_robomimic` | Focal consistency-flow RGB。 |
| Robomimic | `streaming_diffusion_robomimic` | 上游 TEDI / Streaming Diffusion 复现。 |
| Robomimic | `streaming_diffusion_optimized_robomimic` | 窗口加权和确定性 chunk 映射优化版。 |
| Robomimic | `flow_policy_robomimic` | Consistency Flow Policy。 |
| Robomimic | `diffusion_transformer_robomimic` | Transformer 预测头的 diffusion policy。 |
| Robomimic | `act_robomimic` | ACT / DETR-VAE 原生架构。 |
| Robomimic | `bcrnn_robomimic` | Robomimic 原生 BCRNN full-method。 |
| MetaWorld | `padp_metaworld` | PADP 原生 H40/O1/A1 点云。 |
| Adroit | `padp_adroit` | PADP 原生 H40/O1/A1 点云。 |
| MetaWorld | `focal_metaworld` | Focal 上游 temporal H12/O2/A4。 |
| Adroit | `focal_adroit` | Focal 上游 temporal H12/O2/A4。 |
| MetaWorld | `dp3_metaworld` | DP3 H16/O2/A8 点云 full-method。 |
| Controlled | `compare_dp_metaworld` / `compare_dp_adroit` | 固定 Focal XYZ encoder、base capacity、H12/O2/A4 的 DP。 |
| Controlled | `compare_padp_metaworld` / `compare_padp_adroit` | 同协议 PADP，但使用 PADP RTV8 dataset/recipe。 |

### 4.4 主要实验的默认组合

| 方法 | H/O/A | Encoder | Dataset | Workspace | Recipe / batch | 输出组 |
|---|---|---|---|---|---|---|
| DP Robomimic | 16/2/8 | `robomimic_raw` | standard image | DP | 251 epoch@200 demos / 64 | `dp` |
| PADP Robomimic | 40/1/1 | `robomimic` | RGB RTV8 | PADP | 251 / 64, shuffle=false | `padp` |
| PADP Unrolled | 40/1/1, K=2 | `robomimic` | RTV8 unrolled | PADP-Unrolled | 251 / 16, rollout=false, smoke | `padp` |
| Focal RGB | 16/2/4 | `robomimic_raw` | standard image | DP | 251 / 64 | `focal` |
| Streaming / optimized | 20/1/1 | `robomimic_raw` | standard image | DP | 251 / 64 | 各自独立 |
| FlowPolicy | 40/2/8 | `robomimic_raw` | standard image | DP | 251 / 64 | `flow_policy` |
| Diffusion Transformer | 16/2/8 | `robomimic_raw` | standard image | DP | 251 / 64 | `diffusion_transformer` |
| ACT | 10/1/10 | ACT ResNet18 | standard image | DP | 251 / 64 | `act` |
| BCRNN | 10/1/1 | Robomimic native | 10-step dataset sequence | BCRNN | 251 / 64 | `bcrnn` |
| DP/PADP/Focal point-cloud controlled | 12/2/4 | Focal XYZ | 算法所需 standard/RTV8 | DP/PADP | 1000 / 128 | `dp/padp/focal` |
| DP3 MetaWorld | 16/2/8 | DP3 PointNet | standard point cloud | DP | 1000 / 128 | `dp3` |

> `50000 / n_demo + 1` 在 `n_demo=200` 时解析为 251 epoch；`1000 / n_demo` 解析为每 5 epoch rollout/checkpoint。

### 4.5 task 与动作维度

- Robomimic / MimicGen RGB 契约位于 `task/robomimic/mimicgen_abs.yaml`，实际 task 名通过 `task_name=` 覆盖。
- MetaWorld 有 44 个点云 task，统一为 `1024×3 XYZ + 9D agent_pos -> 4D action`。
- Adroit 点云统一为 `512×3 XYZ + 24D agent_pos`，但不同手部任务的控制自由度不同：Door 28D、Hammer 26D、Pen 24D。这些维度来自各环境动作空间，应由 `task` 自动传导，不应在 policy 中写死。

---

## 5. `diffusion_policy/dataset`：数据层

### 5.0 RTV8 公共后端

`dataset/rtv8/` 只抽取 RGB、MetaWorld 和 Adroit 已有实现中逐项相同的协议：

| 文件 | 功能 |
|---|---|
| `mapping_table.py` | 唯一的 epoch-seeded balanced mapping 计算，以及 `dataset index -> episode/window/abs_start` 解析。 |
| `epoch_state.py` | `set_epoch`、worker threadpool、current-epoch mtime 同步、point-cloud mapping archive 和 validation view。 |
| `cache.py` | FileLock、MemoryStore/DirectoryStore 打开或创建、转换失败清理。 |

公共后端不读取观测或动作，不改 padding，不计算 normalizer。
RGB 特有 fingerprint/window 校验与三套 converter/sample decoder 仍由 suite adapter 负责。

### 5.1 Robomimic 四种 image dataset

| 文件 | 功能 |
|---|---|
| `replay_image_dataset.py` | DP/Focal/Streaming/Flow/Transformer/ACT 等默认通用 dataset。HDF5 转 ReplayBuffer，使用 `SequenceSampler`、`pad_before/pad_after`和 zip Zarr 缓存。 |
| `replay_image_dataset_balanced.py` | 标准 dataset 的平衡采样变体，目前没有独立正式 experiment 默认选它。 |
| `replay_image_dataset_padp.py` | PADP RTV8。为每个 episode 物理化首尾 padding，保存 `pos_info/episode_map`，每 epoch 构建 `[batch_size,max_cols,2]` 映射表，通过 DirectoryStore 直接挂载磁盘缓存。 |
| `replay_image_dataset_padp_unrolled.py` | 继承 RTV8 PADP dataset，对 K 步展开返回 K 帧连续 obs 和 `H+K-1` 帧 action；不改 RTV8 后端和 mapping 协议。 |

PADP RTV8 的 `shuffle=false` 是数据协议要求，不是遗漏：DataLoader 使用 SequentialSampler 保持二维映射表的 batch/列对应，不同 epoch 的映射表本身已重新打乱。RGB PADP 还支持 `uint8_to_gpu=true`，将 `/255` 归一化移到 GPU，降低 CPU 内存与传输压力。

### 5.2 MetaWorld / Adroit 点云 dataset

两个 suite 都存在两类 adapter：

- `*_pointcloud_dataset.py`：通用 Zarr + ReplayBuffer + SequenceSampler，给 DP/Focal/DP3 和 PADP 原生轨道使用。
- `*_pointcloud_dataset_padp.py`：将 RTV8 物理 padding、episode map、per-epoch mapping table 扩展到 `point_cloud + agent_pos`。`compare_padp_metaworld/adroit` 默认选它。

PADP 点云 recipe 因此也固定 `shuffle=false`，但训练预算与通用 `pointcloud` recipe 一致：batch 128、1000 epoch、每 100 epoch rollout/checkpoint。

### 5.3 PADP-Unrolled 样本契约

默认 `H=40, K=2`：

```text
obs:         [B, K, ...]       = [B, 2, ...]
action:      [B, H+K-1, D]     = [B, 41, D]
window_info: cycle 0 的 H 个位置元数据
```

接近 episode 末尾时使用最后一帧 obs/action/pos 复制填充，保证张量形状固定。

---

## 6. `diffusion_policy/envs`：环境、runner 和 wrapper

### 6.1 Robomimic / Robosuite

`envs/robomimic/robomimic_image_runner_padp.py` 是当前 DP/PADP/Focal/扩展 RGB policy 共用 runner。它通过 `robomimic_image_wrapper.py`、vector env 和 video wrapper 完成 rollout。

| Runner 配置 | n_train | n_test | n_envs | 用途 |
|---|---:|---:|---:|---|
| `env_runner=robomimic/standard` | 14 | 50 | 32 | Pro 6000D/服务器正式对比。 |
| `env_runner=robomimic/small` | 6 | 18 | 12 | 2080Ti/本机烟测，结果不可与 standard 混表。 |

### 6.2 MetaWorld

- `metaworld_wrapper.py`：选定 `*-v2-goal-observable` 任务，输出点云和状态。
- `metaworld_runner.py`：默认 20 个评估 episode、每 episode 最多 200 步、84 像素渲染。
- `task_schema/metaworld_pointcloud.yaml`：统一 1024 XYZ points、9D state、4D action。

### 6.3 Adroit

- `adroit.py`：dm_env/gym 适配层。
- `adroit_runner.py`：默认 20 个评估 episode、每 episode 最多 300 步。
- `rrl_local/`：Adroit 多相机/RRL 兼容工具。
- `gym_util/mujoco_point_cloud.py` 与 `point_sampling.py`：MuJoCo 深度到点云和下采样。
- `mjpc_wrapper.py` / `mjpc_diffusion_wrapper.py`：将 Adroit 封装为点云 gym env。

### 6.4 CUDA 与 EGL 不是同一套序号

Robosuite rollout 同时使用 CUDA 计算卡和 EGL render node。在服务器上，`nvidia-smi` 序号、PCI 排序后的 CUDA ordinal 和 `/dev/dri/renderD*` 序号可能不同。Square 队列脚本使用 `GPU_UUID` 和 PCI bus 动态导出：

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=<compute ordinal>,<egl ordinal>
MUJOCO_EGL_DEVICE_ID=<egl ordinal>
```

目标 compute GPU 永远放在 `CUDA_VISIBLE_DEVICES` 第一位，所以 policy/workspace 中的 `cuda:0` 仍指向用户指定的物理 GPU。

---

## 7. `diffusion_policy/model`：纯模型层

当前只保留五个责任明确的子目录：

```text
model/
├─ common/              EMA、normalizer、LR scheduler、rotation/shape/tensor 工具
├─ components/          Conv1d 原子层和 LowdimMaskGenerator
├─ diffusion_heads/     DP/PADP/DP3/Streaming 的 Conditional UNet
├─ encoders/            RGB / point cloud / voxel 编码器工厂
└─ transformer_heads/   Diffusion Transformer 与 ACT/DETR-VAE
```

`model/diffusion/`、`model/vision/`、`model/detr/`、`model/sdp_fp/` 已从活跃树删除。它们的有效内容已迁移到上述目录，无活跃链路的实现只留在 `diffusion_policy_archive/`。

### 7.1 diffusion heads

| 文件 | 用途 |
|---|---|
| `base_conditional_unet1d.py` | DP/PADP 共用 U-Net body，`use_diffusion_step` 控制 timestep embedding。 |
| `conditional_unet1d.py` | DP 薄子类，显式接收 diffusion timestep。 |
| `conditional_unet1d_padp.py` | PADP 薄子类，不向 forward 暴露 timestep。 |
| `conditional_unet1d_dp3.py` | DP3 使用的 Conditional UNet。 |
| `conditional_unet1d_streaming.py` | Streaming Diffusion 两个版本使用的 UNet。 |

`components/conv1d_components.py` 提供 `Conv1dBlock / Downsample1d / Upsample1d`；`components/mask_generator.py` 提供 `LowdimMaskGenerator`。

### 7.2 encoder 总工厂

`model/encoders/factory.py::build_obs_encoder()` 按 `shape_meta.obs.*.type` 检测 `rgb / point_cloud / voxel / low_dim`：

- 只有一个空间模态时直接返回该 encoder，保持历史 module key 层级。
- 存在多个空间模态时，用 `MultiModalObsEncoder` 融合。
- 特征拼接顺序为 `RGB -> point_cloud -> voxel -> low_dim`，low-dim 只追加一次。
- policy 通过 `obs_encoder.output_shape()` 自动取得输出维度并设置 global condition，切换 encoder 不需要手填 `obs_feature_dim`。

### 7.3 RGB encoder

`encoders/rgb/factory.py` 采用 lazy import，未选中 provider 时不会 import 可选包。

| 类别 | 已注册别名 |
|---|---|
| Robomimic | `robomimic_raw`、`robomimic` |
| Timm | `timm_resnet18`、`timm_convnext_tiny`、`timm_vit_tiny`、`timm_dino_vits16`、`timm_dinov2_vits14`、`timm_dinov3_vits16` |
| R3M | `r3m_resnet18`、`r3m_resnet34`、`r3m_resnet50` |
| CLIP | `clip_vit_b32`、`openclip_vit_b32` |

`robomimic_raw` 保留 DP 历史原始 module 布局，`robomimic` 保留 PADP wrapper 布局；两者数学上都是 Robomimic 视觉 encoder，但 checkpoint key 不相同。Timm/R3M/CLIP 属于按需安装 provider，必须固定版本或 commit。

### 7.4 point-cloud / voxel encoder

| 组件 | 定位 |
|---|---|
| `pointnet_obs_encoder.py` | 项目通用 PointNet，per-point MLP + permutation-invariant reduce。 |
| `focal_obs_encoder.py` | FocalPolicy 兼容的 XYZ/XYZRGB PointNet，low-dim state 独立 MLP。 |
| `dp3_encoder.py` | DP3 专用 encoder，由 DP3 policy 直接组装，不是普通 `encoder=` 消融的等价替换。 |
| `voxel/conv3d_obs_encoder.py` | shape-meta 驱动的 dense Conv3D encoder；尚无配套 voxel dataset/runner，不是端到端可用实验。 |

点云工厂别名：`pointnet`、`pointnet_xyz`、`pointnet_xyzrgb`、`pointnet_focal`、`pointnet_focal_xyz`、`pointnet_focal_xyzrgb`。正式点云 controlled 对比固定 `pointnet_focal_xyz`。

### 7.5 transformer heads

- `transformer_heads/diffusion_transformer.py`：Diffusion Transformer 序列预测网络。
- `transformer_heads/act/`：ACT 使用的 DETR-VAE、ResNet backbone、position encoding 和 Transformer。
- Focal/Flow 的 consistency-flow 时间调度属于 policy 数学辅助，位于 `policy/common/`，不放在 `model/`。

---

## 8. `diffusion_policy/policy`：算法层

### 8.1 活跃 policy

| 文件 | Hydra 公开 target / 实现 |
|---|---|
| `dp.py` | `DPPolicy` 是 `DiffusionUnetHybridImagePolicy` 的公开别名。 |
| `padp.py` | `PADPPolicy` 是 `SlidingWindowDiffusionPolicy` 的公开别名。 |
| `padp_unrolled.py` | `SlidingWindowDiffusionPolicyUnrolled`，继承 PADP。 |
| `focal_policy.py` | `FocalPolicy`，consistency-flow + DCT objective。 |
| `flow_policy.py` | `FlowPolicy` 是历史 `DiffusionUnetHybridImagePolicy` 实现类的公开别名。 |
| `streaming_diffusion_policy.py` | `StreamingDiffusionPolicy`，上游 TEDI 复现。 |
| `streaming_diffusion_policy_optimized.py` | `OptimizedStreamingDiffusionPolicy`，本项目优化版。 |
| `diffusion_transformer.py` | `DiffusionTransformerHybridImagePolicy`。 |
| `act.py` | `ACTPolicyWrapper`，内部组装 ACT/DETR-VAE。 |
| `dp3.py` | `DP3`。 |
| `robomimic_policy.py` | `RobomimicBCRNNPolicy` 是 `RobomimicImagePolicy` 的公开别名。 |

### 8.2 `policy/common`

| 文件 | 作用 |
|---|---|
| `base_policy.py` | 所有 policy 的 `predict_action/reset/set_normalizer` 抽象边界。 |
| `base_diffusion_policy.py` | 通用 normalizer 和 observation encoding 辅助。 |
| `schedulers.py` | TEDI 使用的 DDPM/DDIM scheduler。 |
| `padp_schedulers.py` | PADP 位置感知加噪函数。 |
| `flow_matching.py` | FlowPolicy 的 `ConsistencyFlowSchedule`。 |
| `focal_objectives.py` | 纯 PyTorch 正交 DCT matrix 和 horizon DCT，不需要 `torch-dct`。 |
| `focal_time_sampler.py` | FocalPolicy 的 logit-normal / joint-v0 时间对采样。 |

### 8.3 PADP 核心行为

PADP 在初始化时构建长度 `H+1` 的 cosine `alpha_bar` 表，并注册为 buffer。训练时：

1. encoder 将 `n_obs_steps` 观测编码为 global condition。
2. `padp_schedulers.add_position_noise()` 按窗口位置加噪。
3. `ConditionalUnet1DPadp` 一次预测整个 action window。
4. MSE 按 `linear / exponential / constant` window weights 加权。

加噪模式实际支持：

```text
positionwise (horizon alias)
linear
constant
random
chunkwise / chunk_wise
```

推理时，执行切片从 `action_start=n_obs_steps-1` 开始，取 `n_action_steps` 步；然后 buffer 左移同样的 `n_action_steps`，在尾部补新高斯噪声。这使 H40/O1/A1 保持旧行为，同时使 H12/O2/A4 的 observation/action/buffer 向前量一致。

### 8.4 PADP-Unrolled（实验性）

`SlidingWindowDiffusionPolicyUnrolled.compute_loss()` 展开 K 个 cycle：

- cycle 0 使用当前 target window 加噪。
- cycle `i>0` 在 teacher target 和 student 上一 cycle 预测之间采样。
- student 路径使用 `pred_prev[:,1:]` 左移一步，尾部补噪声，并 `detach()` 前一 cycle。
- self-rollout 概率从 0 线性增长到 1，默认在总 epoch 前 40% 完成 warmup。
- K 个 cycle 的 window-weighted loss 取平均。

当前默认 experiment 是**开发烟测状态**：`capacity=smoke`、batch16、`env_runner=small`、`training.enable_rollout=false`。其 Hydra 组合已验证，训练实现已接入；但在通过实际 1-batch 与标准 capacity + rollout 回归前，不应列入正式 PADP 成功率对比表。

---

## 9. `diffusion_policy/workspace`：训练编排

| Workspace | 主要责任 |
|---|---|
| `common/base_training_workspace.py` | suite-neutral 初始化骨架：seed、model/EMA、三种 optimizer source、normalizer、runner 与 device setup；不拥有 loss 或训练循环数学。 |
| `common/rollout.py` | 对 `env_runner.run(policy)` 的单次无变换调用，四个 Workspace 和独立 rollout CLI 共用。 |
| `common/action_horizon.py` | 从旧 DP/PADP `run_eval_2()` 原样迁出的 batched/sequential 动作位置分析。 |
| `train_policy_workspace.py` | suite-neutral 通用 policy workspace，承载 DP、Focal、Streaming、Flow、Transformer、ACT、DP3，支持三种 optimizer source。 |
| `train_padp_workspace.py` | suite-neutral PADP RTV8 专用训练，支持 dataset `set_epoch`、per-sample loss、`uint8_to_gpu`、EMA 和 rollout。 |
| `train_padp_unrolled_workspace.py` | 标准 PADP workspace 薄子类；只覆盖 self-rollout curriculum 与 runner enable hook。 |
| `robomimic/train_bcrnn_workspace.py` | Robomimic native BCRNN `train_on_batch/on_epoch_end` 协议，共用 rollout/checkpoint 周边能力。 |
| `rollout_early_stopping.py` | policy 无关的 rollout 连续成功计数 helper。 |
| `workspace_util.py` | encoder 冻结等共享辅助。 |

Workspace suite 去耦已完成：三个通用 workspace 位于 `workspace/` 根目录；`workspace/robomimic/` 只保留真正依赖 Robomimic native policy API 的 BCRNN adapter。

### 9.1 early stopping 当前语义

所有正式对比默认关闭早停：

```yaml
training:
  early_stopping:
    enabled: false
    monitor_key: test_mean_score
    threshold: 1.0
    patience: 3
```

显式开启后，连续 `patience` 次 rollout 满足阈值才停止，并先保存 checkpoint。为保证 policy/encoder/input 消融的预算一致，正式表不应在某个方法上单独开启早停。

---

## 10. 实验容量、时序和 runner 协议

### 10.1 capacity

| 配置 | `down_dims` | 用途 |
|---|---|---|
| `capacity=base` | `[256,512,1024]` | 所有正式 controlled 对比的默认。 |
| `capacity=large` | `[512,1024,2048]` | Focal 上游/full-method 复现轨道，不与 base 混表。 |
| `capacity=smoke` | `[32,64,128]` | 只验证链路和张量形状。 |

### 10.2 预算

- Robomimic：batch64；`n_demo=200` 时 251 epoch；每 5 epoch rollout/checkpoint。
- MetaWorld/Adroit 点云：batch128；1000 epoch；每 100 epoch rollout/checkpoint。
- Focal 上游复现：`capacity=large recipe=focal_upstream`，batch128、3000 epoch、每 200 epoch rollout。
- 烟测：`capacity=smoke recipe=smoke_pointcloud`，1 epoch / 1 train step / 1 val step，不产生正式 checkpoint。

### 10.3 controlled 与 full-method 不能混表

- **Controlled**：固定 suite、input、encoder、capacity、temporal、recipe、runner、seed，只换 policy。
- **Native/full-method**：保留方法原生 temporal、workspace、dataset、encoder 或 optimizer，用于回答“这个完整方法在本任务上能到多少”。

例如 Square 中，受控 PADP H16/O2/A4 和 PADP 原生 H40/O1/A1 是两个问题；Diffusion Transformer 受控 H16/O2/A4 + Hydra optimizer 与原生 H16/O2/A8 + policy-method optimizer 也必须分表。

---

## 11. 典型运行链路

### 11.1 PADP Robomimic 原生链路

```bash
HYDRA_FULL_ERROR=1 python -u train_sim.py --config-name=train \
  experiment=padp_robomimic \
  task_name=square_d0 n_demo=200 \
  env_runner=robomimic/standard \
  training.resume=false
```

```text
train_sim.py
 -> experiment/padp_robomimic
 -> dataset/robomimic/replay_image_padp (RTV8)
 -> encoder/rgb/robomimic
 -> policy/padp_positionwise
    -> policy.padp.PADPPolicy
    -> policy/common/padp_schedulers.py
    -> model/diffusion_heads/conditional_unet1d_padp.py
 -> workspace/padp
    -> train_padp_workspace.py
 -> env_runner/robomimic/standard
 -> data/outputs/padp/...
```

### 11.2 PADP-Unrolled 开发烟测

```bash
HYDRA_FULL_ERROR=1 python -u train_sim.py --config-name=train \
  experiment=padp_unrolled_robomimic \
  task_name=square_d0 n_demo=200 \
  training.resume=false
```

默认不实例化 env runner，只验证 K=2 连续样本、self-rollout curriculum 和 loss 链。要升级为正式实验，应单独新建正式 experiment 或显式覆盖 capacity/runner/rollout，并先完成服务器 smoke。

### 11.3 点云受控对比

```bash
# MetaWorld
python -u train_sim.py --config-name=train \
  experiment=compare_padp_metaworld \
  task=metaworld/door-close_pointcloud \
  training.resume=false

# Adroit
python -u train_sim.py --config-name=train \
  experiment=focal_adroit \
  task=adroit/hammer_pointcloud \
  training.resume=false
```

DP/PADP/Focal 在受控点云轨道均使用 H12/O2/A4、Focal XYZ encoder、base capacity、batch128 和 1000 epoch。

### 11.4 Square policy 队列

`tests/运行命令/robomimic_square_policy队列.sh` 当前队列：

| 入口 | 依次运行 |
|---|---|
| `q0` | controlled DP -> controlled Streaming -> native BCRNN |
| `q0_remaining` | controlled Streaming -> native BCRNN |
| `q2` | controlled PADP -> controlled Streaming Optimized |
| `q1a` | controlled Focal -> controlled Diffusion Transformer |
| `q1b` | controlled Flow -> native ACT |
| `native_padp` | PADP full-method H40/O1/A1 |
| `native_transformer` | Diffusion Transformer full-method H16/O2/A8 |
| `native_padp_transformer` | 上述两个 full-method 串行 |

使用 UUID 只检查 GPU 映射：

```bash
GPU_MAPPING_ONLY=1 \
GPU_UUID=<nvidia-smi -L 显示的 GPU-UUID> \
bash tests/运行命令/robomimic_square_policy队列.sh q0
```

### 11.5 数据软链接

```bash
DP_DATA_ROOT=/media/disk7t/data \
bash tests/运行命令/链接数据集.sh
```

脚本会把数据根一级目录链到项目 `data/`，并额外把 MetaWorld/Adroit Zarr 平铺为 config 期望的 `data/metaworld_<task>_expert.zarr` 和 `data/adroit_<task>_expert.zarr`。它拒绝覆盖已有真实文件，可重复执行。

---

## 12. 测试与当前验证状态

### 12.1 单测文件

| 文件 | 覆盖内容 |
|---|---|
| `test_composable_configs.py` | 核心 experiment 合成、44 个 MetaWorld task、点云公平协议、Focal 上游 recipe、policy 实例化。 |
| `test_policy_layout.py` | 扁平 policy 路径和旧路径清理。 |
| `test_modality_encoders.py` | RGB/点云/体素 encoder factory 和多模态融合。 |
| `test_focal_policy.py` | Focal forward、DCT objective 和 time sampler。 |
| `test_focal_env_utils.py` | MetaWorld/Adroit/Focal 环境工具。 |
| `test_metaworld_dataset.py` | MetaWorld 点云 dataset。 |
| `test_adroit_dataset.py` | Adroit 点云 dataset。 |
| `test_rollout_early_stopping.py` | 早停计数、阈值和禁用语义。 |
| `test_padp_chunk_alignment.py` | PADP O2/A4 等 chunk 下的动作切片和 buffer 左移对齐。 |

### 12.2 本文生成时的实测

已在 `sim_torch27_v8` 执行：

```bash
python -m unittest \
  tests.test_composable_configs \
  tests.test_policy_layout \
  tests.test_padp_chunk_alignment \
  tests.test_rollout_early_stopping
```

结果：**20 tests passed**。同时 `padp_unrolled_robomimic --cfg job --resolve` 已确认解析为 H40/O1/A1、K=2、batch16、shuffle=false、smoke capacity 和 `enable_rollout=false`。

该结果证明 import、Hydra 合成、张量契约和辅助控制正常，**不等价于 PADP-Unrolled 已完成 Pro 6000D 长训练或 standard rollout 验收**。

---

## 13. 新增与修改指南

### 13.1 新增 policy

1. 在 `diffusion_policy/policy/<algo>.py` 添加 suite 无关实现。
2. 通用 diffusion policy 优先继承 `policy/common/base_diffusion_policy.py`；非 diffusion policy 继承 `BasePolicy`。
3. 在 `config/policy/<algo>.yaml` 只写算法数学。
4. 在 `config/experiment/<algo>_<suite>.yaml` 选定 task/dataset/runner/encoder/capacity/temporal/recipe/workspace。
5. 为组合、import 和 1-batch loss 添加单测，然后再上服务器 rollout。

### 13.2 新增 encoder

1. 在 `model/encoders/rgb|pointcloud|voxel/` 增加 encoder。
2. 在对应 `factory.py` 注册 provider 和固定预设别名。
3. 可选依赖保持 lazy import，不得因为未使用的 encoder 导致主环境 import 失败。
4. 需要可复现默认时新增 `config/encoder/<modality>/<name>.yaml`。
5. 使用 `output_shape()` 让 policy 自动传导条件维度。

### 13.3 新增 task/suite

1. task 只定义 shape 契约和环境元数据，不复制 policy。
2. 数据差异写入 `config/dataset/<suite>/` 和 `dataset/<suite>/`。
3. rollout 差异写入 `config/env_runner/<suite>/` 和 `envs/<suite>/`。
4. 动作维度必须从 `shape_meta.action.shape` 传导，不得在 policy 根据 task 名硬编码。
5. Robomimic/MimicGen 新 task 如果使用 `${get_max_steps:...}`，还需要在 `train_sim.py::max_steps` 注册。

### 13.4 继续开发 PADP-Unrolled

当前 unrolled workspace 是从 PADP workspace 派生的独立文件，便于隔离回归风险。后续正式化前应补：

1. unrolled dataset K=1 与标准 PADP dataset 的数学等价测试。
2. K=2 的 obs/action 连续性与 episode 边界测试。
3. teacher/student 采样概率调度测试。
4. `capacity=base + env_runner=standard + enable_rollout=true` 的 1-batch + 1-rollout 服务器 smoke。
5. 与 PADP H40/O1/A1 统一 seed/预算的正式对比，并保持 early stopping 关闭。

---

## 14. 已知边界和维护约定

- 当前仓库只保证新组合式配置和从零训练，不承诺重构前 checkpoint/import 兼容。
- 通用 Workspace 已完成 suite 去耦；新增 MetaWorld/Adroit 等 suite 时不得复制 policy/workspace，只新增 task、dataset adapter 与 runner。
- `config/task/` 根层仍有 6 个历史快捷 task，在所有命令切到分层 task 前不删除。
- `tests/运行命令/卡2点云对比队列.sh` 仍是硬件命名的历史脚本，后续应重命名为 GPU 无关的 point-cloud queue，并引入与 Square 队列相同的 UUID/EGL 映射。
- DP3 的参数统计必须同时注明 capacity、点云 encoder 是否计入以及统计的是 predictor 还是整个 policy；不应将某次 smoke 日志中的约 1.5M 直接当作所有 DP3 配置的固定参数量。
- Timm encoder 默认 `share_rgb_model=true`；Robomimic encoder 的历史结构与空间信息保留方式不同。编码器消融必须把“backbone”、“多相机共享方式”和“特征汇聚”分开记录。
- 体素 encoder 仅是模型组件就绪，没有 dataset/runner 就不能宣称端到端可用。
- 正式对比的事实来源优先级为：**实际合成的 Hydra config > 源代码 > 本文 > 历史迁移文档**。

---

## 15. 快速导航

- 配置：[diffusion_policy/config/README.md](diffusion_policy/config/README.md)
- Policy 布局：[diffusion_policy/policy/README.md](diffusion_policy/policy/README.md)
- Encoder 工厂：[diffusion_policy/model/encoders/README.md](diffusion_policy/model/encoders/README.md)
- 操作手册：[README.md](README.md)
- 服务器命令：[tests/运行命令/README.md](tests/运行命令/README.md)
- 代码分层待办：[代码简化计划.md](代码简化计划.md)
- Focal 迁移笔记：[focalpolicy.md](focalpolicy.md)
- 历史对照：[PADP_代码结构与功能v3_1.md](PADP_代码结构与功能v3_1.md)
