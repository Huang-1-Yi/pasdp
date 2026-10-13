# PADP / DP-DDIM 复现指南

本仓库在 MimicGen / Robosuite 图像任务上训练 **Diffusion Policy (DP-DDIM 基线)** 与 **Position-Aware Diffusion Policy (PADP)**，并提供隔离的 MetaWorld 点云实验链。

- `diffusion_policy/config/experiment/dp_robomimic.yaml` — 原始 DP-DDIM 基线
- `diffusion_policy/config/experiment/padp_robomimic.yaml` — PADP(默认 `noise_schedule_mode=positionwise` 黄金标准)
- `diffusion_policy/config/experiment/padp_metaworld.yaml` — PADP + MetaWorld + Focal XYZ PointNet

主回归环境固定为 `equidiff_torch27_v4`；可选 RGB encoder 建议使用隔离的
`sim_torch27_v6` 环境；MetaWorld/Adroit 使用完整隔离的 `sim_torch27_v8`，避免
legacy MuJoCo 依赖影响正在运行的 v4 长回归。详见后文 §1.6、§1.9 和 §环境与依赖。

---

## AUBO 双臂实机测试

当前 T265 双臂遥操入口为 `demo_dual_teleop_aubo_v3.py`。
启动保持，按 H 归位（2 倍规划速度并恢复初始 I/O），再按 E 遥操。
高度下限由所选 JSON 决定，现场调试后确定。
详见 [AUBO v3 现场配置与改动记录](运行命令及文档/AUBO双臂遥操作_v3_现场配置与改动记录_2026-10-09.md)
及 [v3 使用说明](diffusion_policy/envs/realword/aubo_t265/README_v3.md)。

以下保留早期键盘程序的设备与操作记录。

当前设备映射：

| 设备 | 地址 |
|---|---|
| 左 AUBO | `192.168.3.200`，RPC `30004`，RTDE `30010` |
| 右 AUBO | `192.168.3.199`，RPC `30004`，RTDE `30010` |
| 右 WSG 夹爪 | `192.168.3.100:1000`，当前未接入本控制程序 |

### 离线手动安装 pyaubo_sdk

仓库已经保存适用于 `robodiff`（Python 3.9、Linux x86_64）的 wheel：

```text
third_party/aubo_sdk/wheels/pyaubo_sdk-0.26.0rc6-cp39-cp39-manylinux2014_x86_64.whl
```

在新环境或重新部署时执行：

```bash
cd /home/robot/lightumi/dp_pro6000
conda activate robodiff
python -m pip install \
  --no-index \
  --no-deps \
  third_party/aubo_sdk/wheels/pyaubo_sdk-0.26.0rc6-cp39-cp39-manylinux2014_x86_64.whl
```

验证安装与接口：

```bash
python -c 'import importlib.metadata as m, pyaubo_sdk; print(m.version("pyaubo-sdk")); print(pyaubo_sdk.__file__)'
python tests/aubo_keyboard_teleop.py
```

第二条命令只登录并读取两臂状态，不上电、不进入伺服、不发送运动目标。运行前关闭其他
可能登录 AUBO 控制柜的 RPC 客户端。只有两臂均显示 `mode=Running safety=Normal` 才继续。

### 首次实时键盘控制

确认现场急停可用、两臂周围无人员/线缆/障碍物后，先使用较低的连续速度：

```bash
cd /home/robot/lightumi/dp_pro6000
conda activate robodiff
python tests/aubo_keyboard_teleop.py \
  --enable-motion \
  --translation-speed-mm-s 20 \
  --rotation-speed-deg-s 10 \
  --robot-frequency 200 \
  --servo-t 0.02 \
  --keyboard-frequency 10 \
  --command-horizon 0.1 \
  --verbose
```

两臂已经是 `Running` 时不要追加 `--power-on`。程序显示 IP 后还会要求一次人工确认，确认
后两臂进入伺服并保持各自当前位姿，此时尚未产生位移。

这里沿用参考程序 `demo_udp_teleop_aubo.py` 的默认定义：程序启动时的双臂 TCP 位姿就是
本次会话的 home。因此请先用示教器把两臂放到期望的 home，再启动键盘程序；程序不会
猜测或写死一组关节角。

启动阶段会先等待两臂的有效 RTDE 实测状态，再用各自当前关节角发送一条零位移保持命令；
只有两条命令都被控制柜接受后才会进入键盘界面。SDK `0.26.0` 偶发返回 `-13
(Request was ignored)` 时，程序只会有限次数重新选择伺服模式并重试同一条保持命令，
不会自动解除保护停。

首次按以下顺序验证：

1. 首次修复验证先不要按运动键：确认打印的 TCP 和关节角不是全零，按 `P` 再读一次，
   然后按 `Q` 退出。
2. 再次运行后按 `1` 只选择左臂，短按 `W` 沿左臂基坐标 `+X` 移动，再短按 `S` 返回。
3. 按 `2` 只选择右臂，短按 `W` 沿右臂基坐标 `+X` 移动，再短按 `S` 返回。
4. 两边方向和响应都正确后，按 `3` 选择双臂，再测试一次 `W`、`S`。
5. 按 `H`，确认双臂分段返回程序启动时记录的 home。
6. 平移确认后再用 `I/K`、`J/L`、`U/O` 分别测试 `Rx/Ry/Rz` 旋转。

完整键位：`W/S=+X/-X`、`A/D=+Y/-Y`、`R/F=+Z/-Z`、
`I/K=+Rx/-Rx`、`J/L=+Ry/-Ry`、`U/O=+Rz/-Rz`。运动使用每台机械臂各自的
基坐标系；双臂模式发送相同的基坐标增量，不代表两臂在世界坐标中的方向必然相同。
`H` 会让两臂以当前配置的小步长返回启动 home；归位期间可用 `Ctrl-C` 或实体急停中断。

键盘程序不限制 TCP 相对启动位置的工作范围。它使用 `pynput` 跟踪按键按下/松开状态，
默认以 10 Hz 更新笛卡尔末端目标，同时以 200 Hz 执行插值、逆运动学和 `servoJoint`；
两种频率可分别通过 `--keyboard-frequency` 和 `--robot-frequency` 设置。程序不依赖终端
的按键自动重复；松开按键后不再追加目标，并保留控制器内部的最大速度和单周期关节增量
保护，避免 IK 解跳变直接形成过大的 `servoJoint` 指令。它不会自动解除 Protective Stop，
任一臂的逆解或 `servoJoint` 持续返回异常时会退出控制。软件退出不能代替实体急停。

更完整的 SDK 安装和控制说明见
[`third_party/aubo_sdk/README.md`](third_party/aubo_sdk/README.md)。

---

## 0. 数据软链接

数据默认集中存放在 `/home/hy/data/robomimic`、`/home/hy/data/metaworld` 和
`/home/hy/data/adroit`。项目配置继续使用 `data/...` 相对路径；克隆或更新代码后执行：

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
```

脚本可重复执行且不会覆盖现有真实文件。服务器数据根目录不同可使用
`DP_DATA_ROOT=/实际路径 bash tests/运行命令/链接数据集.sh`。详见
[统一数据软链接](tests/运行命令/数据软链接.md)。

---

## 1. 训练(快速开始)

### 1.1 准备数据

数据按任务名分目录统一存放,路径:

```
data/robomimic/datasets/<任务名>/<任务名>_abs.hdf5
```

例如 Coffee_D0:

```bash
ls -lh data/robomimic/datasets/coffee_d0/coffee_d0_abs.hdf5
```

默认软链为 `data/robomimic -> /home/hy/data/robomimic`。第一次训练会自动生成 Zarr
缓存与 epoch 映射文件，后续运行直接复用。

### 1.2 PADP 训练(默认黄金标准)

```bash
conda activate equidiff_torch27_v4
HYDRA_FULL_ERROR=1 python train_sim.py \
  --config-name=train experiment=padp_robomimic \
  task_name=coffee_d0 \
  n_demo=200 \
  n_obs_steps=1 \
  horizon=40 \
  window_exp_gamma=0.25 \
  policy.noise_schedule_mode=positionwise \
  dataloader.batch_size=64 \
  policy.pred_type=sample \
  policy.noise_scheduler.prediction_type=sample \
  exp_name=PADP_Golden_Standard
```

如需切换消融对照,只改 `policy.noise_schedule_mode` 即可,可选值:

| 取值 | 说明 |
|---|---|
| `positionwise` | 黄金标准,静态位置感知,O(1) 极速广播 |
| `constant`   | 整条轨迹共享同一噪声层级 |
| `linear`     | 噪声随位置严格线性递增 |
| `random`     | 每个位置独立随机采样 |
| `chunkwise`  | 分块噪声分配,块大小由 `policy.noise_chunk_size` 控制 |

### 1.3 DP-DDIM 基线训练

```bash
HYDRA_FULL_ERROR=1 python train_sim.py \
  --config-name=train experiment=dp_robomimic \
  task_name=coffee_d0 \
  n_demo=200 \
  exp_name=DP_DDIM_Baseline
```

`experiment/dp_robomimic.yaml` 默认 `horizon=16`。如需令数据窗口与文件名中的 40 对齐，显式追加 `horizon=40`，但对比旧实验时务必保持参数一致。

### 1.4 训练产物

训练输出默认写入:

```
data/outputs/YYYY.MM.DD/HH.MM.SS_<实验名>_<任务名>/
```

每个实验目录包含:

- `checkpoints/latest.ckpt` — 最新 checkpoint
- `checkpoints/epoch{epoch:03d}_score{test_mean_score:.4f}.ckpt` — topk 保留
- `train_logs.json.txt` — 每 epoch 一行 train/val/rollout 指标
- `wandb/` — 离线 W&B run 目录(可用 `wandb sync` 上传)

`training.resume=true` **只从当前 Hydra 输出目录下的 `checkpoints/latest.ckpt` 恢复**,不会自动搜索其他实验目录;需要跨实验续训时请改 `pretrained_model_path`。

### 1.5 GPU 选用与多卡并行训练

服务器 3 卡可同时跑训练。单次训练显式选用 GPU 用 `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=train experiment=padp_robomimic ...
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train_sim.py --config-name=train experiment=dp_robomimic ...
```

任务可不同 GPU 同时跑。同一 GPU 跑多个任务不会自动分配显存；使用较大的可选
encoder 时建议每张卡只保留一个活跃训练。确需同卡双开时先将
`dataloader.batch_size=32`，并观察第一个 batch 后的显存，而不是只看模型初始化是否成功。

“每张卡最多准备 7 个实验”应理解为排队顺序运行，不是 7 个训练进程同时占用一张卡。

### 1.6 可选 RGB encoder（工厂化、按需安装）

DP 和 PADP 都支持用一个 Hydra 参数切换图像编码器：

```bash
policy.rgb_model=timm_convnext_tiny
```

两个配置中的默认值分别是：

- DP：`policy.rgb_model=robomimic_raw`
- PADP：`policy.rgb_model=robomimic`

这两个默认值保持旧 checkpoint 的模块布局。切换 encoder 时，Policy 会调用
`obs_encoder.output_shape()` 取得实际输出维度，并自动设置扩散模型的条件维度；不需要手写
`obs_feature_dim`。但不同 encoder 生成的模型结构不同，checkpoint 不能直接混用。

需要区分“BCRNN encoder”和“完整 BCRNN policy”：

- `policy.rgb_model=robomimic_raw` 或 `robomimic` 只把 Robomimic `bc_rnn`
  配置构造出的视觉网络用作 encoder，策略头仍然是 DP/PADP；
- 真正的原始 BCRNN 对照使用 `experiment=bcrnn_robomimic`，包括 Robomimic
  原生 recurrent policy、损失和内部优化器。

前台运行完整 BCRNN policy：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py --config-name=train \
  experiment=bcrnn_robomimic \
  task_name=coffee_d0 n_demo=200 \
  training.resume=false \
  name=bcrnn_coffee_d0
```

BCRNN 使用 `horizon=10 / n_obs_steps=1 / n_action_steps=1`：训练数据按十步
序列送入 RNN，rollout 每次读取一帧并输出一步。它是单独的 policy，不能通过
`policy.rgb_model=...` 切换，也不参与 encoder factory 消融。

从旧 PADP_v3 仓库迁移的新增对照策略也使用同一个组合入口：

```text
experiment=streaming_diffusion_robomimic
experiment=streaming_diffusion_optimized_robomimic
experiment=act_robomimic
experiment=flow_policy_robomimic
experiment=dp3_metaworld
experiment=diffusion_transformer_robomimic
```

前四个和 Diffusion Transformer 默认是 Robomimic RGB，DP3 默认是 MetaWorld
点云。它们分别写入 `data/outputs/streaming_diffusion`、
`streaming_diffusion_optimized`、`act`、
`flow_policy`、`dp3` 和 `diffusion_transformer`。参数边界、烟测结论、真机候选与
暂缓迁移清单见 [v8策略迁移与真机候选.md](v8策略迁移与真机候选.md)。

所有正式组合式训练默认关闭 rollout 早停并跑满预算。如需诊断性早停，可显式覆盖：

```bash
training.early_stopping.enabled=true
training.early_stopping.patience=5
training.early_stopping.threshold=0.95
```

正式“全成功”实验应保留默认 `threshold=1.0`；低阈值只适合控制流烟测。

#### 1.6.1 已注册的模型别名

下表维度按当前 Coffee 双 RGB 相机、9 维 low-dim 状态计算。单相机维度是每路 RGB
特征宽度，总维度是两路特征拼接 low-dim 后的 Policy 观察维度。

| `policy.rgb_model` | 每路 RGB | 双相机总维度 | 额外依赖 | 备注 |
|---|---:|---:|---|---|
| `robomimic_raw` | 64 | 137 | 无 | DP 默认，保持历史原始布局 |
| `robomimic` | 64 | 137 | 无 | PADP 默认，保持历史 wrapper 布局 |
| `timm_resnet18` | 512 | 1033 | Timm | ImageNet 预训练，优先烟测 |
| `timm_convnext_tiny` | 768 | 1545 | Timm | ImageNet 预训练，显存高于 ResNet18 |
| `timm_vit_tiny` | 192 | 393 | Timm | ImageNet 预训练 ViT |
| `timm_dino_vits16` | 384 | 777 | Timm | DINO ViT-S/16 |
| `timm_dinov2_vits14` | 384 | 777 | Timm | DINOv2 ViT-S/14 |
| `timm_dinov3_vits16` | 384 | 777 | Timm | DINOv3；自动将输入调整为 80×80 |
| `r3m_resnet18` | 512 | 1033 | R3M | R3M 预训练 ResNet18 |
| `r3m_resnet34` | 512 | 1033 | R3M | R3M 预训练 ResNet34 |
| `r3m_resnet50` | 2048 | 4105 | R3M | 条件头和显存明显增大，不建议首轮并行 |
| `clip_vit_b32` | 512 | 1033 | OpenAI CLIP | 只使用视觉塔，不加载文本塔进 Policy |
| `openclip_vit_b32` | 512 | 1033 | OpenCLIP + Timm | ViT-B/32，LAION 权重 |

`timm`、`r3m`、`openai_clip`、`open_clip` 也是底层 provider 名，适合在
`rgb_encoder_kwargs` 中自行指定模型；日常实验优先使用上表固定预设，减少命令差异。

#### 1.6.2 配置项、默认传导与冻结

两个活跃 YAML 都包含以下控制项：

```yaml
policy:
  obs_encoder: null
  rgb_model: robomimic       # DP 配置中是 robomimic_raw
  rgb_encoder_kwargs: {}
  crop_shape: [76, 76]
  obs_encoder_group_norm: true
  eval_fixed_crop: true
```

参数优先级为“provider 构造默认值 < 工厂预设 < Policy 公共配置 <
`rgb_encoder_kwargs` 显式覆盖”。因此只切换模型时无需重复输入一串
`policy.rgb_encoder_kwargs.*`；裁剪、固定评估裁剪等公共设置会自动传入。

预训练 encoder 默认是可训练的。冻结 encoder 只需追加：

```bash
+policy.rgb_encoder_kwargs.frozen=true
```

例如冻结 R3M ResNet18：

```bash
policy.rgb_model=r3m_resnet18 \
+policy.rgb_encoder_kwargs.frozen=true
```

冻结后会关闭 encoder 梯度并保持其 `eval` 状态。其他常用显式覆盖：

```bash
# Timm 只使用网络结构，从随机权重训练；此时配置中的 GroupNorm 才会生效
+policy.rgb_encoder_kwargs.pretrained=false

# 禁用图像裁剪
+policy.rgb_encoder_kwargs.crop_shape=null
```

Hydra 语法注意：`rgb_model` 已存在于配置中，所以使用
`policy.rgb_model=...`，不要加 `+`；`frozen`、`pretrained` 等键默认不在空字典中，
命令行新增它们时要使用 `+policy.rgb_encoder_kwargs.<键>=...`。

#### 1.6.3 独立环境与按需依赖

`conda_environment_sim_torch27_v6.yaml` 的有效依赖只创建核心环境；文件末尾的 encoder
依赖全部是注释，不会被 `mamba env create/update` 自动安装。推荐先创建隔离环境：

```bash
mamba env create -f conda_environment_sim_torch27_v6.yaml
conda activate sim_torch27_v6
python -m pip check
```

已有 v6 环境可更新；备用和保底方案分别是：

```bash
mamba env update -f conda_environment_sim_torch27_v6.yaml -n sim_torch27_v6

# 仅用 v5 核心清单修复 v6，不会安装可选 encoder
mamba env update -f conda_environment_sim_torch27_v5.yaml -n sim_torch27_v6

# 环境已无法可靠求解时重建
mamba env remove -n sim_torch27_v6
mamba env create -f conda_environment_sim_torch27_v6.yaml
```

若当前 v4 已稳定运行某个 encoder，也可以继续使用，但不要在任何训练进程运行期间安装、
卸载或升级它们正在使用的包。无论使用哪个环境，都应通过该环境的 Python 调用 pip，
不要直接使用可能指向其他环境的 `pip`：

```bash
python -m pip install ...
# 未激活环境时：
conda run -n sim_torch27_v6 python -m pip install ...
```

每次只安装当前实验需要的一组，并在安装后运行 `python -m pip check`。
`--no-deps` 可以防止 pip 自动升级或降级核心环境，但前提是列出的运行依赖已经满足。

Timm（同时覆盖 ResNet、ConvNeXt、ViT、DINO）：

```bash
python -m pip install --no-deps timm==1.0.28 safetensors==0.8.0
python -m pip check

# 只卸载 Timm；safetensors 可能与 OpenCLIP 共用
python -m pip uninstall -y timm
```

R3M（固定 commit，不能跟随主分支）：

```bash
python -m pip install --no-deps \
  'git+https://github.com/facebookresearch/r3m.git@b2334e726887fa0206962d7984c69c5fb09cceab'
python -m pip check
python -m pip uninstall -y r3m
```

OpenAI CLIP（不要误装 PyPI 上名称含糊的 `clip` 包）：

```bash
python -m pip install --no-deps \
  ftfy==6.3.1 wcwidth==0.8.2 regex==2024.11.6 \
  'git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
python -m pip check
python -m pip uninstall -y clip
```

OpenCLIP（先安装上面的 Timm 组）：

```bash
python -m pip install --no-deps \
  open-clip-torch==3.3.0 ftfy==6.3.1 wcwidth==0.8.2 regex==2024.11.6
python -m pip check
python -m pip uninstall -y open-clip-torch
```

不要顺手卸载 `safetensors`、`ftfy`、`wcwidth` 或 `regex`，它们可能正被另一个
provider 共用。更完整的依赖注释见 `conda_environment_sim_torch27_v6.yaml` 文件末尾。

#### 1.6.4 首次联网预缓存与离线运行

每个“用户 + 缓存目录 + 模型”只需预缓存一次，不需要为 DP、PADP 或每张 GPU 分别执行。
以 Timm 为例：

```bash
python -c "import timm; timm.create_model('resnet18.a1_in1k', pretrained=True, num_classes=0, global_pool='avg')"
python -c "import timm; timm.create_model('convnext_tiny', pretrained=True, num_classes=0, global_pool='avg')"
python -c "import timm; timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=0, global_pool='avg', img_size=(84,84))"
python -c "import timm; timm.create_model('vit_small_patch16_224.dino', pretrained=True, num_classes=0, global_pool='avg', img_size=(84,84))"
python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, global_pool='avg', img_size=(84,84))"
python -c "import timm; timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=0, global_pool='avg', img_size=(80,80))"
```

DINOv3 权重可能需要先在 Hugging Face 获得授权。其他 provider 的预缓存：

```bash
# R3M：按实际要用的 resnet18 / resnet34 / resnet50 分别执行，缓存位于 ~/.r3m
python -c "import r3m; r3m.device='cpu'; r3m.load_r3m('resnet18')"

# OpenAI CLIP
python -c "import clip; clip.load('ViT-B/32', device='cpu', jit=False)"

# OpenCLIP
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cpu')"
```

只有预缓存成功后才在训练命令中设置 `HF_HUB_OFFLINE=1`。该变量主要控制 Hugging Face
缓存；R3M 的 `~/.r3m` 和 OpenAI CLIP 自己的缓存也必须提前存在。若首次运行提示找不到
权重，不要反复后台重启，先去掉离线变量，在前台完成一次下载和构建。

#### 1.6.5 前台训练示例

前台命令适合首次烟测，可立即看到下载、encoder 选择、维度或 OOM 错误。

DP + Timm ConvNeXt-Tiny：

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
python -u train_sim.py \
  --config-name=train experiment=dp_robomimic \
  task_name=coffee_d0 n_demo=200 \
  policy.rgb_model=timm_convnext_tiny \
  training.resume=false \
  name=dp_timm_convnext_tiny
```

PADP + 冻结的 R3M ResNet18：

```bash
CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
python -u train_sim.py \
  --config-name=train experiment=padp_robomimic \
  task_name=coffee_d0 n_demo=200 \
  n_obs_steps=1 horizon=40 window_exp_gamma=0.25 \
  policy.noise_schedule_mode=positionwise \
  policy.pred_type=sample \
  policy.noise_scheduler.prediction_type=sample \
  policy.rgb_model=r3m_resnet18 \
  +policy.rgb_encoder_kwargs.frozen=true \
  training.resume=false \
  name=padp_r3m_resnet18_frozen
```

把 `policy.rgb_model` 换成 §1.6.1 的任一已安装、已缓存别名即可。首次烟测建议先降低
成本，例如追加 `dataloader.batch_size=8 training.num_epochs=1`；通过后再启动正式实验。

#### 1.6.6 后台静默运行示例

先创建日志目录，再让每个实验使用唯一的 `name` 和日志文件：

```bash
mkdir -p run_logs

nohup env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
python -u train_sim.py \
  --config-name=train experiment=dp_robomimic \
  task_name=coffee_d0 n_demo=200 \
  policy.rgb_model=timm_resnet18 \
  training.resume=false \
  name=dp_timm_resnet18 \
  > run_logs/dp_timm_resnet18.log 2>&1 &

nohup env CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
python -u train_sim.py \
  --config-name=train experiment=padp_robomimic \
  task_name=coffee_d0 n_demo=200 \
  n_obs_steps=1 horizon=40 window_exp_gamma=0.25 \
  policy.noise_schedule_mode=positionwise \
  policy.pred_type=sample \
  policy.noise_scheduler.prediction_type=sample \
  policy.rgb_model=timm_convnext_tiny \
  training.resume=false \
  name=padp_timm_convnext_tiny \
  > run_logs/padp_timm_convnext_tiny.log 2>&1 &
```

如果要在 GPU 1、GPU 2 上各准备最多 7 个实验，使用两个顺序队列。下面示例中每张卡始终
只有一个训练进程；前一个命令退出后才启动下一个。先按 §1.6.3 安装所有涉及的 provider，
并按 §1.6.4 缓存全部权重。已有同卡训练结束前不要再启动队列。

```bash
mkdir -p run_logs

# GPU 1：依次运行 DP；按需要增删 models，一次最多列 7 个
nohup bash -c '
models=(
  robomimic_raw timm_resnet18 timm_convnext_tiny timm_vit_tiny
  r3m_resnet18 clip_vit_b32 openclip_vit_b32
)
for model in "${models[@]}"; do
  env CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
  python -u train_sim.py \
    --config-name=train experiment=dp_robomimic \
    task_name=coffee_d0 n_demo=200 \
    policy.rgb_model="${model}" \
    training.resume=false \
    name="dp_${model}" \
    > "run_logs/dp_${model}.log" 2>&1
done
' > run_logs/dp_gpu1_queue.log 2>&1 &

# GPU 2：依次运行 PADP；按需要增删 models，一次最多列 7 个
nohup bash -c '
models=(
  robomimic timm_resnet18 timm_convnext_tiny timm_vit_tiny
  r3m_resnet18 clip_vit_b32 openclip_vit_b32
)
for model in "${models[@]}"; do
  env CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 HYDRA_FULL_ERROR=1 \
  python -u train_sim.py \
    --config-name=train experiment=padp_robomimic \
    task_name=coffee_d0 n_demo=200 \
    n_obs_steps=1 horizon=40 window_exp_gamma=0.25 \
    policy.noise_schedule_mode=positionwise \
    policy.pred_type=sample \
    policy.noise_scheduler.prediction_type=sample \
    policy.rgb_model="${model}" \
    training.resume=false \
    name="padp_${model}" \
    > "run_logs/padp_${model}.log" 2>&1
done
' > run_logs/padp_gpu2_queue.log 2>&1 &
```

外层 `*_queue.log` 用于记录 shell 级错误；每个模型的训练输出仍写入自己的日志。默认
Robomimic 项不需要外部权重，其他项若缺包或缺缓存会退出并进入队列下一项，因此正式长跑前
仍应对每种 provider 做一次前台烟测。

管理和验收后台任务：

```bash
jobs -l
ps -fp <PID>
tail -n 100 run_logs/dp_timm_resnet18.log
tail -f run_logs/padp_timm_convnext_tiny.log
watch -n 2 nvidia-smi

# 优先用 SIGINT 让训练做正常清理；无响应时再考虑 SIGTERM
kill -INT <PID>
```

`nohup` 返回 PID 只说明进程已创建，不代表训练有效。至少确认日志已经出现以下内容，且
进度越过 `Training epoch 0: 0%` 并完成一个 batch：

```text
RGB encoder selection: requested=timm_convnext_tiny, ... model_name=convnext_tiny, ...
```

还应核对 `RGB encoder params` 和 `Obs feature dim/output_shape`。如果只看到 torchvision
旧接口 warning，它本身不是加载失败；真正的选择结果以上述工厂日志为准。

#### 1.6.7 实验隔离注意事项

- `training.resume=false` 用于新架构实验，避免误读旧结构 checkpoint；每个实验使用唯一
  `name`，同时保证输出目录和日志文件不冲突。
- 运行中的 Python 已加载其进程内代码和动态库，但在同一环境卸载包、覆盖源码或切换
  checkout 仍会给后续 import、worker 和复现实验带来风险。长回归期间应在单独 worktree/
  克隆目录与独立环境做 encoder 实验。
- encoder 越大，扩散条件层也可能随观察维度增大。能完成初始化不等于能完成第一个反向；
  以首个 batch 和 `nvidia-smi` 峰值作为显存烟测标准。
- 公平比较时固定任务、`n_demo`、seed、horizon、batch size、训练 epoch 和 rollout 设置，
  只改变 `policy.rgb_model`（以及事先声明的冻结策略）。
- R3M ResNet50 的总观察维度为 4105，优先排在 ResNet18/34、Timm ResNet18 和
  ConvNeXt-Tiny 之后；同卡不要与其他大模型并发。

---

### 1.7 RGB、点云、体素 encoder 分层（model 层已完成）

`diffusion_policy/model/encoders/` 现在按照 `shape_meta` 的类型和单步张量形状分为：

| 目录 | `shape_meta.type` | encoder 输入 | 默认模型 | 额外依赖 |
|---|---|---|---|---|
| `rgb/` | `rgb` | `[B,C,H,W]` | 现有 Robomimic 工厂 | 取决于所选 RGB provider |
| `pointcloud/` | `point_cloud` | `[B,N,C]` | `pointnet` | 无 |
| `voxel/` | `voxel` | `[B,C,D,H,W]` | `conv3d` | 无 |

三类 encoder 都返回 `[B,D]`，并通过 `output_shape()` 报告实际维度。顶层
`build_obs_encoder(shape_meta=...)` 会按 `type` 自动构建：只有一种空间模态时直接返回
具体 encoder；多种空间模态时分别编码，再按 `RGB -> point_cloud -> voxel -> low_dim`
拼接。纯 RGB 路径不增加新包装层，因此原有 DP/PADP checkpoint key 不变。

点云快捷预设是 `pointnet`、`pointnet_xyz`、`pointnet_xyzrgb`，以及独立复现的
`pointnet_focal`、`pointnet_focal_xyz`、`pointnet_focal_xyzrgb`；体素快捷预设是
`conv3d`、`voxel_cnn`、`voxel_cnn_small`。这里的默认体素 CNN 是支持任意正尺寸的普通
PyTorch Conv3d，并不是 PADP 中绑定 16/58/64 尺寸、`escnn` 群表示、in-hand RGB 和
末端位姿的专用等变 task encoder。完整接口和 Python 示例见
`diffusion_policy/model/encoders/README.md`。

RGB 两份主回归仍是 RGB + low-dim；MetaWorld 已另行接通 point-cloud dataset、
normalizer、runner 和 PADP Policy。体素端到端实验仍未接通，不能只把现有 YAML 的
`type` 改为 `voxel`。

---

### 1.8 FocalPolicy 的 Robomimic 隔离实验

FocalPolicy 已拆成共享 policy/encoder/temporal/capacity/recipe 模块和三个 suite
入口：`experiment=focal_robomimic`、`experiment=focal_metaworld` 和
`experiment=focal_adroit`。Robomimic 主消融保持标准 DP 的 Dataset、双相机
`robomimic_raw` encoder、`horizon=16`、`n_obs_steps=2`、`n_action_steps=4` 和
`down_dims=[256,512,1024]`，只把 DP Policy 换成 Focal consistency-flow Policy。
两者都不需要 `torch-dct` 或新的环境包。

`16/2/4` 比原先的 `40/1/1` 更接近上游 Focal 的 `12/2/4` 长预测短执行语义，同时
保留本项目标准 DP 的 horizon=16。主实验暂不把 head 改成上游
`[512,1024,2048]`，否则会同时改变 Policy 和模型容量；上游宽度只作为单独 capacity
ablation。论文复现轨道显式追加 `capacity=large recipe=focal_upstream`，不会污染公平消融默认值。

实验顺序固定为：先让每个 unique configuration 只跑 `training.seed=42`；环境、encoder、
policy 和输入主轴都完成一遍后，再统一补 seed43/44。最终统计使用三个 seed，但 seed 消融
不得提前占用第一阶段算力，也不得只给 seed42 的优胜配置补 seed。

前台运行：

```bash
HYDRA_FULL_ERROR=1 python -u train_sim.py \
  --config-name=train experiment=focal_robomimic \
  task_name=coffee_d0 n_demo=200 \
  training.resume=false \
  name=focal_flow_dp16_obs2_a4_coffee_d0
```

11GB 显卡先用小 batch 验证持续训练和 rollout：

```bash
HYDRA_FULL_ERROR=1 python -u train_sim.py \
  --config-name=train experiment=focal_robomimic \
  task_name=coffee_d0 n_demo=200 \
  dataloader.batch_size=8 val_dataloader.batch_size=8 \
  training.resume=false \
  name=focal_flow_dp16_obs2_a4_coffee_d0_bs8
```

完全匹配的 DP baseline 应覆盖其默认 `n_action_steps=8`，否则并非只替换 Policy：

```bash
HYDRA_FULL_ERROR=1 python -u train_sim.py \
  --config-name=train experiment=dp_robomimic \
  task_name=coffee_d0 n_demo=200 \
  horizon=16 n_obs_steps=2 n_action_steps=4 \
  policy.rgb_model=robomimic_raw \
  training.resume=false \
  name=dp_ddim_h16_obs2_a4_coffee_d0
```

正式对比时必须统一 batch size、seed、训练 update/样本预算和完整训练状态。
详细数学实现、烟测结果、后台命令及成功率判断见
`v8代码优化_工作2_FocalPolicy与Robomimic烟测.md`。

---

### 1.9 用 v8 YAML 直接创建可运行 MetaWorld / Adroit 的环境

`conda_environment_sim_torch27_v8.yaml` 是独立的 MetaWorld/Adroit 工作环境。它会直接
安装 Torch 2.7.1/cu128、Open3D CPU、MoviePy 1.0.3、现代 `mujoco==2.3.2`，并从本仓库 editable 安装
`mujoco-py`、MetaWorld、mjrl 和 mj_envs；不需要再逐个执行 pip 命令，也不安装
`torch-dct`。v8 另固定 `setuptools==80.9.0`，仅为兼容 W&B 视频链所用的
`imageio-ffmpeg==0.4.7`；`sim_torch27_v7` 是不含这组 legacy rollout 包的对照环境。

第一次创建前只需完成一次系统前置：

```bash
sudo apt-get install -y libglew-dev libegl1-mesa-dev
mkdir -p "$HOME/.mujoco"
cd "$HOME/.mujoco"
wget -c https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
  -O mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
test -f "$HOME/.mujoco/mujoco210/bin/libmujoco210.so"
```

然后回到仓库根目录，一条 YAML 命令创建完整 Python 环境：

```bash
cd /home/hy/dp_pro6000
mamba env create -f conda_environment_sim_torch27_v8.yaml
conda activate sim_torch27_v8
python -m pip check
```

YAML 的本地包路径是相对仓库根目录解析的，因此必须在 `/home/hy/dp_pro6000` 执行。
YAML 还为 v8 设置了 `MUJOCO_GL=egl`、MuJoCo 2.1 动态库路径和 NVIDIA driver 路径；
这些变量只作用于 v8，不会写入 `~/.bashrc`，也不会改变 v4/v7。不要把
`/usr/lib/x86_64-linux-gnu` 显式加入 `LD_LIBRARY_PATH`：其中的系统 cuDNN 可能遮蔽
Torch cu128 wheel 自带版本，最先在 BCRNN/LSTM 初始化时表现为
`cuDNN version incompatibility`。

若 v8 是按旧版 YAML 创建的，不必重建环境，执行一次下面的更新并重新激活即可：

```bash
conda env config vars set -n sim_torch27_v8 \
  LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia" \
  MUJOCO_GL=egl
conda deactivate
conda activate sim_torch27_v8
```

创建后验证 MetaWorld 基础环境：

```bash
python -c "import torch,mujoco,mujoco_py,open3d,metaworld; \
print(torch.__version__,mujoco.__version__,open3d.__version__)"

# BCRNN/ACT 等含 RNN 的策略还应验证实际加载的 cuDNN 和一次 GPU LSTM 前向。
python -c "import torch; print('cuDNN',torch.backends.cudnn.version()); \
x=torch.randn(2,3,4,device='cuda'); y,_=torch.nn.LSTM(4,8,batch_first=True).cuda()(x); \
print('LSTM PASS',tuple(y.shape))"

python -c "import metaworld; \
e=metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-close-v2-goal-observable'](); \
o=e.reset(); print('MetaWorld reset PASS',type(o)); e.close()"
```

离线数据用统一脚本从 `/home/hy/data/metaworld` 建立软链接；注意真实文件名是
`door-close_expert`，不是 `door-close-expert`：

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
```

前台烟雾训练：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python -u train_sim.py \
  --config-name=train experiment=padp_metaworld \
  training.debug=true training.resume=false \
  checkpoint.save_last_ckpt=false checkpoint.topk.k=0 \
  name=padp_metaworld_smoke
```

上面关闭 smoke checkpoint，避免 62.9M 参数模型在两轮 debug 中写出约 3GB 文件；
正式训练不要添加这两个 checkpoint override。

后台静默训练：

```bash
mkdir -p run_logs
nohup env CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
python -u train_sim.py \
  --config-name=train experiment=padp_metaworld \
  training.resume=false \
  name=padp_metaworld_door_close \
> run_logs/padp_metaworld_door_close.log 2>&1 &
```

首轮配置有意保留 PADP 的 `horizon=40`、`n_obs_steps=1`、`n_action_steps=1`、
positionwise loss，只切换为 `1024×3 XYZ`、9 维状态、4 维动作、Focal 点云 encoder 和
MetaWorld runner。这是 PADP 点云最小消融，不等于 Focal Flow 的 `12/2/4` 复现。

Adroit 无需再修改 v8 YAML。当前已接入 door、hammer、pen 三个任务，
它们共用 `512×3 XYZ + 24 维 agent_pos`，动作维度分别为 28、26、24。
离线数据默认从 `/home/hy/data/adroit` 建立软链接：

```bash
cd /home/hy/dp_pro6000
bash tests/运行命令/链接数据集.sh
```

PADP 前台 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py \
  --config-name=train experiment=padp_adroit \
  task=adroit/door_pointcloud \
  task.env_runner.eval_episodes=1 task.env_runner.max_steps=1 \
  training.debug=true training.resume=false \
  checkpoint.topk.k=0 checkpoint.save_last_ckpt=false \
  name=padp_adroit_door_smoke
```

FocalPolicy 前台 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 \
conda run --no-capture-output -n sim_torch27_v8 \
python -u train_sim.py \
  --config-name=train experiment=focal_adroit \
  task=adroit/door_pointcloud \
  task.env_runner.eval_episodes=1 task.env_runner.max_steps=1 \
  training.debug=true training.resume=false \
  checkpoint.topk.k=0 checkpoint.save_last_ckpt=false \
  name=focalpolicy_adroit_door_smoke
```

两份 Adroit 主配置均默认 `horizon=12`、`n_obs_steps=2`、
`n_action_steps=4`。PADP 与 Focal 都使用
`task=adroit/hammer_pointcloud` 或 `task=adroit/pen_pointcloud` 切换任务；`task_name`、
数据路径、runner 和动作维度都会由 task YAML 自动传导。
后台批量命令见 `tests/运行命令/padp/adroit后台运行.md` 和
`tests/运行命令/focalpolicy/adroit后台运行.md`。
完整环境 smoke、错误定位和回滚命令见 `v8升级指令.md`。

---

## 2. 评估与 rollout

rollout 由 `task.env_runner._target_: diffusion_policy.envs.robomimic.robomimic_image_runner_padp.RobomimicImageRunner` 完成,自动在训练过程中以 `rollout_every` 周期触发(默认 `1000/n_demo`,即 n_demo=200 时每 5 epoch 一次)。rollout 完成后写:

- `test_mean_score` — 平均成功率(进入 `train_logs.json.txt` 与 W&B)
- `rollouts/` — 渲染视频片段

**checkpoint topk 判据**:`monitor_key=test_mean_score`,保留 `k=5` 个最高分的 checkpoint。

### 2.1 单独跑评估

当前训练循环里 rollout 与训练耦合,无独立 eval 入口。要单独评估某 checkpoint,把训练命令改成只跑 rollout:

```bash
HYDRA_FULL_ERROR=1 python train_sim.py \
  --config-name=train experiment=padp_robomimic \
  task_name=coffee_d0 n_demo=200 n_obs_steps=1 horizon=40 \
  window_exp_gamma=0.25 policy.noise_schedule_mode=positionwise \
  dataloader.batch_size=64 \
  policy.pred_type=sample policy.noise_scheduler.prediction_type=sample \
  training.num_epochs=0 \
  training.pretrained_model_path=data/outputs/<某实验目录>/checkpoints/latest.ckpt \
  exp_name=PADP_EvalOnly
```

`num_epochs=0` + `pretrained_model_path` 会跳过训练直接进 rollout。后续会拆出独立 eval 入口(见 §后续重构阶段)。

### 2.2 12 个 MimicGen 任务清单

后续会训练以下 12 个任务,数据集均通过 `mimicgen_envs` 注册或 robosuite 内置提供:

```
stack_d0  coffee_d0  nut_assembly_d0       hammer_cleanup_d0
coffee_preparation_d0  mug_cleanup_d0      square_d0
threading_d0           stack_three_d0       three_piece_assembly_d0
pick_place_d0          kitchen_d0
```

env 依赖位置:

| 任务 | env 实现 | 第三方位置 |
|---|---|---|
| stack/coffee/nut_assembly/hammer_cleanup/mug_cleanup/threading/three_piece_assembly/pick_place | `mimicgen_envs` | `third_party/mimicgen/mimicgen_envs/envs/robosuite/` |
| coffee_preparation/stack_three | `mimicgen_envs` 注册的 hdf5 | 同上 |
| square | robosuite 内置 Square | `third_party/robosuite/` |
| kitchen | robosuite task zoo | `third_party/robosuite-task-zoo/` |

任务切换用 `task_name=<上述>`,配置文件通过 `task` defaults 自动选中 mimicgen_abs(_padp)。

---

## 3. 服务器部署

```bash
ssh 10.242.5.12
cd /media/disk7t
git clone git@gitee.com:huang-1-yi/dp_pro6000.git dp_pro6000
cd /media/disk7t/dp_pro6000

# 已有 v8 环境时，在新 clone 中重新绑定 editable 包，避免继续导入旧 checkout。
conda activate sim_torch27_v8
python -m pip install --no-deps \
  -e ./third_party/mimicgen \
  -e ./third_party/mujoco-py-2.1.2.14 \
  -e ./third_party/Metaworld \
  -e ./third_party/mjrl \
  -e ./third_party/mj_envs
python -m pip check

# 把集中数据映射到新 clone 的 data/ 下。
DP_DATA_ROOT=/media/disk7t/data bash tests/运行命令/链接数据集.sh
```

如果服务器尚无 v8 环境，在新 clone 根目录执行
`mamba env create -f conda_environment_sim_torch27_v8.yaml`。环境创建不访问 GitHub。
Conda/PyPI/PyTorch cu128 的基础包仍需可用镜像或本地缓存。

非交互 ssh 调用需用绝对路径:

```bash
ssh 10.242.5.12 'cd /media/disk7t/dp_pro6000 \
  && /opt/miniconda3/envs/sim_torch27_v8/bin/python train_sim.py ...'
```

挂载服务器代码到本机编辑用 sshfs(交互式输密码或 ssh-copy-id):

```bash
mkdir -p ~/mnt/server
sshfs 10.242.5.12:/media/disk7t/dp_pro6000 ~/mnt/server -o reconnect,follow_symlinks
```

卸载:`fusermount -u ~/mnt/server`。

若 Git 启用 sparse checkout,确保 `third_party` 没被排除。MimicGen 是 editable 安装,创建环境后不要移动或删除 `third_party/mimicgen`。

---

## 4. 环境与依赖

### 4.1 v4 低风险升级(当前生效)

`conda_environment_sim_torch27_v4.yaml` 是当前生效环境。与 v3 相比本轮升级 10 个包:

| 包 | v3 | v4 | 依据 |
|---|---|---|---|
| python | 3.9.18 | 3.10.13 | CLASS/iPhUMI 基线 |
| numpy | 1.23.3 | 1.26.4 | CLASS 已验证 |
| numba | 0.56.4 | 0.59.0 | numba 0.58 不支持 numpy 1.26 |
| scipy | 1.9.1 | 1.11.4 | UMI 已验证 |
| hydra-core | 1.2.0 | 1.3.2 | CLASS 已验证 |
| einops | 0.4.1 | 0.6.1 | CLASS/UMI 已验证 |
| zarr | 2.12.0 | 2.17.0 | 不进 zarr 3 |
| numcodecs | 0.10.2 | 0.12.1 | 配套 zarr 2.17 |
| tqdm | 4.64.1 | 4.66.4 | CLASS 已验证 |
| dill | 0.3.5.1 | 0.3.7 | UMI 已验证 |

保持不动:

- `torch==2.7.1+cu128` / `torchvision==0.22.1+cu128`(Pro 6000D 稳定核心)
- `diffusers==0.11.1` 与 `huggingface_hub==0.25.2`(高风险,单独升级)
- `gym==0.21.0`、`mujoco==2.3.2`(仿真链锁死)
- `robosuite 1.4.1` / `robomimic 0.3.0` / `robosuite-task-zoo 0.1` / `mimicgen_envs 0.1.0`(均从 `third_party` 本地安装,固定提交)
- `pytorch3d 0.7.9`(无扩展 wheel,`third_party/wheels/`)
- `pandas / matplotlib / tensorboard / tensorboardx / imagecodecs`(待后续代码重构后再精简)

**构建工具约束**:`egl_probe==1.0.2` 是 C++ 扩展,`setup.py` 调 `cmake --version` 检测;执行 `mamba env create` 前必须 `cmake` 在 PATH 可见(`apt install cmake` 或 `conda install cmake`)。完整约束与升级记录见 `v4升级指令.md`。

### 4.2 验证环境

```bash
conda activate equidiff_torch27_v4
python -m pip check

python - <<'PY'
from pathlib import Path
import ast, inspect
import torch, torchvision
import numpy as np, scipy, numba, zarr, hydra, einops
import robosuite, robomimic, mimicgen_envs
import pytorch3d.transforms, diffusers
from robomimic.envs import env_robosuite

print(f"Python {__import__('sys').version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__} | numba {numba.__version__}")
print(f"zarr {zarr.__version__} | hydra {hydra.__version__} | einops {einops.__version__}")
print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | diffusers {diffusers.__version__}")
print(f"robosuite {robosuite.__version__} | robomimic {robomimic.__version__}")
print(f"CUDA available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}")

ws = Path(inspect.getsourcefile(env_robosuite))
imports = {a.name for n in ast.walk(ast.parse(ws.read_text(encoding='utf-8'))) if isinstance(n,(ast.Import,ast.ImportFrom)) for a in n.names}
assert "open3d" not in imports, "robomimic wrapper 不应 import open3d"

texture = Path(robosuite.__file__).resolve().parent / "models/assets/textures/light-gray-floor-tile.png"
assert texture.is_file() and texture.stat().st_size > 0, texture
print("环境验证通过")
PY
```

### 4.3 本地依赖前置检查

```bash
test -f third_party/robosuite/setup.py
test -f third_party/robomimic/setup.py
test -f third_party/robosuite-task-zoo/setup.py
test -f third_party/mimicgen/setup.py
test -f third_party/wheels/pytorch3d-0.7.9-py3-none-any.whl
test -s third_party/robosuite/robosuite/models/assets/textures/light-gray-floor-tile.png
test -s third_party/mimicgen/mimicgen_envs/models/robosuite/assets/textures/ceramic.png
```

第三方版本/提交/本地修改清单见 `third_party/UPSTREAM_VERSIONS.yaml`。model 层已有不依赖
Open3D 的 PointNet / Conv3d encoder，但本地 Robomimic 的活跃 dataset/runner 仍面向
RGB/低维配置，且已停用 Open3D、点云与体素生成路径并关闭深度图渲染。若后续要跑真实
`point_cloud` / `voxels` / 深度 observation，仍需恢复或重写数据生成链并重新评估 Open3D。

---

## 5. 代码现状(v5 阶段1 已完成)

### 5.1 目录结构(已合并)

`env_runner/` + `gym_util/` + `env/` 已合并为单一 `diffusion_policy/envs/`:

```
diffusion_policy/envs/
  __init__.py
  base_image_runner.py
  robomimic/
    __init__.py
    robomimic_image_runner_padp.py    ← 活跃(两个 config 共用)
    robomimic_image_wrapper.py         ← 活跃
  gym_util/
    __init__.py
    async_vector_env.py / sync_vector_env.py / multistep_wrapper.py / video_recording_wrapper.py
```

历史死代码(44 .py + 整目录 block_pushing/kitchen/pusht + 1 个 bet 脚本,共 45 文件)已迁至 `diffusion_policy_archive/`,工作树顶上 `# ARCHIVED: <原路径> on <date>` 注释,稳定后可整体 `rm -rf`。

### 5.2 已落地的训练加速(v5 / v4 改动)

详见 `v5代码优化_v1.md`,实测发生在 `equidiff_torch27_v4` + Pro 6000D:

| 改动 | 文件 | 速度影响 |
|---|---|---|
| A.3 `threadpool_limits(1)` 移到 `worker_init_fn` | `replay_image_dataset_padp.py` + `train_padp_workspace.py` | 让 profile 干净,主路径 <0.1% |
| A.4 `current_epoch.txt` 改 `os.stat` mtime 比较 | `replay_image_dataset_padp.py` | 消除文件 IO 噪声 0.5-1.5% |
| A.5 W&B 每 100 step 记录一次(epoch 末合并) | `train_padp_workspace.py` | 离线日志序列化 1ms/batch → 1ms/100batch |
| B `uint8_to_gpu=true`(已经默认固化) | yaml + dataset + workspace + action-horizon sequential path | **实测提速 27.8%**(591 min → 427 min,251 epoch 全程) |

B 的数值回归(CPU vs uint8 GPU)在 251 epoch 完整对照下:
- epoch 0 `train_loss`:`|diff|=2.87e-5`(与离线核验 `max|diff|=0` 吻合)
- 末段(200-250)相对差异 0.28%,**收敛到同一吸引子**
- `test_mean_score` 均值 0.733 → 0.753(+2 pp,std 反而下降),51 次 rollout 无崩溃、无 NaN

B 改动文件清单:
1. `diffusion_policy/config/experiment/padp_robomimic.yaml` — `training.uint8_to_gpu: true`(默认固化为 GPU 路径)
2. `diffusion_policy/config/task/mimicgen_abs_padp.yaml` — `dataset.uint8_to_gpu: ${training.uint8_to_gpu}` 透传
3. `diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py` — `__init__` 加 `uint8_to_gpu` 参数,`__getitem__` 走 uint8 或 float32 分支
4. `diffusion_policy/workspace/train_padp_workspace.py` + `workspace/common/action_horizon.py` — 训练与离线 sequential 分析均保留 GPU 归一化
5. `diffusion_policy/envs/robomimic/robomimic_image_runner_padp.py` — 删诊断打印(env 路径不动,env 已给 float32 [0,1])

`robomimic_dp_ddim_sample_40_1.yaml` 链路完全未改,仍是原始 CPU `/255` 路径,可作为原始 DP baseline 对照。

### 5.3 后续重构阶段(占位,未完成)

| 阶段 | 文件 | 状态 |
|---|---|---|
| 阶段1 环境目录合并 | `v5代码优化_阶段1_环境目录合并.md` | ✅ 完成 |
| 阶段2 model 原子层抽离 | `v5代码优化_阶段2_model原子层抽离.md` | 已细化,待执行 |
| 阶段3 model 变体审计与子类化 | `v5代码优化_阶段3_model变体审计与子类化.md` | 占位大纲 |
| 阶段4 policy 基类化与 workspace 模块化 | `v5代码优化_阶段4_policy基类化与workspace模块化.md` | 占位大纲 |

每阶段执行原则与回写规则见 `v5代码优化_思路总纲.md`。

---

## 6. 常见问题

### 本地依赖提示不存在

确认命令在 `dp_pro6000` 根目录执行,并检查 `third_party` 是否完整拉取。YAML 中相对路径按当前项目目录解析;git sparse checkout 必须保留 `third_party`。

### MuJoCo 报 `PNG file load error` 或 `texplane`

Robosuite 运行时纹理未进入源码或已装 wheel。先检查:

```bash
test -s third_party/robosuite/robosuite/models/assets/textures/light-gray-floor-tile.png
```

失败说明 git 提交未含第三方图片资源,应 pull 到修过 `.gitignore` 的提交(见 `v4升级指令.md` 中 egl_probe/cmake 约束的修复历史)。文件恢复后旧环境仍需从完整源码强制重装 Robosuite:

```bash
python -m pip install --no-deps --force-reinstall ./third_party/robosuite
python - <<'PY'
from pathlib import Path
import robosuite
p = Path(robosuite.__file__).resolve().parent/"models/assets/textures/light-gray-floor-tile.png"
assert p.is_file() and p.stat().st_size > 0, p
print("Robosuite 纹理恢复:", p)
PY
```

### Hydra 提示 workspace 不可导入

看错误最内层的 `ModuleNotFoundError`,Hydra 最外层"workspace 不可导入"通常是包装信息。例如 `No module named 'pandas'` 表示环境依赖缺失,并非路径错误。完整错误开关必须写为 `HYDRA_FULL_ERROR=1`,不要漏掉开头的 `H`。

### `torch.cuda.is_available()` 为 `False`

`nvidia-smi` 检查驱动与 GPU 可见性,再确认装的是 `torch==2.7.1+cu128`。不要额外安装 Conda 版 `pytorch` / `pytorch-cuda` / `cudatoolkit` 覆盖该 wheel。若报 `forward compatibility was attempted on non supported HW`,多半内核模块与用户态库版本不一致,**重启电脑即可对齐**。

### 手动中断 rollout 后 `EGL_NOT_INITIALIZED`

`Ctrl+C` 后子进程销毁 EGL 上下文时可能打印。若正常训练和 rollout 期间未出现,只是强制中断后的清理提示。

### 进程退出阶段异常

`DataLoader worker killed by signal: Aborted` / `resource_tracker leaked semaphore` / `EGL_NOT_INITIALIZED` 在 v4 早先实跑时出现在训练/val/rollout 入口完成之后,不等同前向反向失败。后续单独处理(见 `v5代码优化_v1.md` §一 后续清理项)。

### 显存或内存不足

```bash
dataloader.batch_size=32 \
dataloader.num_workers=6 \
val_dataloader.num_workers=6 \
task.env_runner.n_envs=6
```

这些 override 降吞吐但不改网络结构。

### 本机(2080Ti)与 `EGL_PLATFORM_DEVICE`

本机驱动 580 若内核模块与用户态库不一致会报 `Cannot initialize a EGL device display`。**生产训练一律在服务器跑**(Pro 6000D / RTX 4090 EGL 头less 渲染正常)。本机仅做开发编辑与 import 静态检查。

---

## 7. 版本约束(产出实验的环境不要单独升级)

当前复现链的一组兼容约束:

- Python 3.10、NumPy 1.26.4、Numba 0.59、SciPy 1.11.4
- Gym 0.21、MuJoCo 2.3.2、Diffusers 0.11.1、Hugging Face Hub 0.25.2
- Torch 2.7.1+cu128 / Torchvision 0.22.1+cu128 单独升级是高风险动作,需先固化 DDIM 采样回归测试再做
- Zarr 2.17 / Numcodecs 0.12(不进 Zarr 3)

不建议在产出实验的环境中单独升级其中任意一个。升级路线见 `v4升级指令.md` 的低风险/高风险分档表。

---

## 8. 相关文件导航

| 文件 | 用途 |
|---|---|
| `conda_environment_sim_torch27_v4.yaml` | 当前生效环境(低风险升级) |
| `conda_environment_sim_torch27_v6.yaml` | 核心 v6 环境 + 各类可选 encoder 的分类安装/卸载注释 |
| `conda_environment_sim_torch27_v7.yaml` | Torch 2.7/cu128 干净对照环境 |
| `conda_environment_sim_torch27_v8.yaml` | 可直接创建的 MetaWorld/Adroit 点云 rollout 完整环境 |
| `v7代码优化_RGB编码器工厂化与第二backbone.md` | RGB 工厂实现、参数传导、兼容验证与后续实验记录 |
| `diffusion_policy/model/encoders/README.md` | RGB、点云、体素输入协议、工厂入口与端到端边界 |
| `v7代码优化_思路总纲.md` | v7 阶段总纲与长回归并行开发边界 |
| `v8算法结构迁移与复现.md` | FocalPolicy 等旧目录仓库到当前结构的路径、职责与复现门禁对照 |
| `v8代码优化_阶段0_上游基线与迁移判定.md` | v8 来源固化、活跃引用闭包和逐文件迁移判定规则 |
| `v8代码优化_阶段1_Focal点云编码器对比.md` | Focal 独立点云 encoder、现有 PointNet 对比和数值等价门禁 |
| `v8代码优化_工作2_FocalPolicy与Robomimic烟测.md` | FocalPolicy 采样、损失、推理、YAML、烟测及运行命令 |
| `v8代码优化_工作3_MetaWorld与Adroit环境复现计划.md` | legacy MuJoCo、third_party、点云 Dataset、串行 Runner、环境门禁和回退计划 |
| `v8升级指令.md` | Focal legacy 环境的最小依赖、隔离安装、分层 smoke、正式启动边界与回滚命令 |
| `v8代码优化_阶段3_点云数据集与配置迁移.md` | 离线点云数据、normalizer、shape_meta 和首批任务配置 |
| `v8代码优化_阶段4_环境点云生成与Runner迁移.md` | 在线点云生成、MetaWorld/Adroit runner 与环境隔离 |
| `v8代码优化_阶段5_Workspace训练与评估迁移.md` | 训练入口、EMA/scheduler/checkpoint 和端到端短回归 |
| `v8代码优化_阶段6_消融实验与长回归.md` | 公平消融矩阵、多 seed 评估与最终长回归门禁 |
| `v4升级指令.md` | v3→v4 升级过程与决策记录 |
| `v5代码优化_思路总纲.md` | 代码重构总纲,4 个阶段路线图 |
| `v5代码优化_阶段1_环境目录合并.md` | ✅ 已完成,环境目录合并执行记录 |
| `v5代码优化_阶段2_model原子层抽离.md` | 待执行 |
| `v5代码优化_阶段3_model变体审计与子类化.md` | 占位大纲 |
| `v5代码优化_阶段4_policy基类化与workspace模块化.md` | 占位大纲 |
| `v5代码优化_v1.md` | v4/v5 已实施改动与加速实测记录 |
| `third_party/UPSTREAM_VERSIONS.yaml` | 第三方固定提交与本地修改清单 |
| `diffusion_policy_archive/` | v5 阶段1 归档区,稳定后可 `rm -rf` |
