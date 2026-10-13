# 组合式训练配置

统一入口是 `train.yaml`。完整实验不再复制一份大 YAML，而是由 Hydra 从以下配置轴组合：

- `experiment`：可复现的薄预设，负责选择其余各轴及实验默认值；
- `task`：观测和动作的 `shape_meta` 契约；
- `dataset`：离线数据读取实现和采样参数，组合后仍挂载为 `task.dataset`；
- `env_runner`：在线评估实现，组合后仍挂载为 `task.env_runner`；
- `policy`：DP、PADP 等算法及其数学参数；
- `encoder`：RGB 或点云编码器，只合并到 `policy`；
- `capacity`：UNet 宽度、时间嵌入宽度、kernel 和 GroupNorm 分组，只合并到 `policy`；
- `temporal`：`horizon/n_obs_steps/n_action_steps`；
- `recipe`：batch、epoch、验证和 checkpoint 预算；
- `runtime`：CPU/CUDA 及 worker 等运行资源。

`policy` 与 `capacity` 不再重复定义字段：`policy` 只描述算法，`capacity` 是模型容量的
唯一真源。所有正式 `experiment` 都显式选择 `capacity=base`；`large` 和 `smoke` 只能
作为命令行显式消融/烟测覆盖。

实验输出也由 experiment 显式选择 `output_group`，默认目录为：

```text
data/outputs/dp/
data/outputs/padp/
data/outputs/focal/
```

因此 Focal 即使复用 DP workspace，也不会保存到 DP 目录。需要把结果写到项目外的数据盘
时只覆盖 `output_root=/绝对路径`，不要修改算法分组。

## 四个旧入口的替代命令

| 已删除入口 | 新命令 |
|---|---|
| `robomimic_dp_ddim_robomimic.yaml` | `python -u train_sim.py --config-name=train experiment=dp_robomimic` |
| `robomimic_padp_robomimic.yaml` | `python -u train_sim.py --config-name=train experiment=padp_robomimic` |
| `robomimic_padp_metaworld.yaml` | `python -u train_sim.py --config-name=train experiment=padp_metaworld` |
| `robomimic_padp_adroit.yaml` | `python -u train_sim.py --config-name=train experiment=padp_adroit` |

原有根级 override 仍可使用，例如 `task_name=coffee_d0 n_demo=200`、
`training.resume=false`、`policy.noise_schedule_mode=constant`。

## FocalPolicy 模块化入口

FocalPolicy 不再复制完整大 YAML。共享数学、encoder、时间窗、容量和训练预算分别由配置组
选择：

| 配置轴 | 默认选择 | 职责 |
|---|---|---|
| `policy` | `focal_flow` | consistency-flow objective、sampler、DCT 权重 |
| `encoder` | 随 suite 选择 | RGB 或 Focal XYZ 观测编码 |
| `temporal` | 随 suite 选择 | Robomimic `16/2/4`；点云 `12/2/4` |
| `capacity` | `base` | controlled 宽度 `[256,512,1024]` |
| `recipe` | 随 suite 选择 | dataloader、训练与 rollout 预算 |

三个正式 suite 入口为：

```text
experiment=focal_robomimic
experiment=focal_metaworld
experiment=focal_adroit
```

旧配置迁移关系：

| 已删除的大 YAML | 模块化替代 |
|---|---|
| `robomimic_focal_flow_dp16_v1.yaml` | `experiment=focal_robomimic` |


| `robomimic_focalpolicy_adroit.yaml` | `experiment=focal_adroit` |

复现上游点云容量与训练预算时追加 `capacity=large recipe=focal_upstream`。该组合属于
full-method 复现轨道，不能与 `capacity=base` 的 controlled 结果混表。

## 单轴消融示例

PADP Robomimic 改用 ConvNeXt-Tiny，其他轴不变：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_robomimic \
  encoder=rgb/timm_convnext_tiny \
  task_name=coffee_d0 n_demo=200 \
  training.resume=false
```

Adroit 切换任务，动作维度和数据路径随 task 自动传导：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_adroit \
  task=adroit/hammer_pointcloud \
  training.resume=false
```

MetaWorld 的 44 个 FocalPolicy 场景已经迁入 `task/metaworld/`。例如切换到
Hand Insert 或 Peg Insert Side：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_metaworld \
  task=metaworld/hand-insert_pointcloud

python -u train_sim.py --config-name=train \
  experiment=padp_metaworld \
  task=metaworld/peg-insert-side_pointcloud
```

这些任务配置统一使用当前离线数据和在线 runner 已验证的 `1024×3 XYZ` 契约。
配置存在不代表本机已经存在对应 Zarr；数据路径为
`data/metaworld_<task_name>_expert.zarr`，可用 `tests/运行命令/链接数据集.sh`
接入 `/home/hy/data/metaworld` 中已有的场景。

完整 MetaWorld task 选择如下（每项前面加 `task=metaworld/`）：

```text
assembly_pointcloud              basketball_pointcloud
bin-picking_pointcloud           box-close_pointcloud
button-press-wall_pointcloud     button-press_pointcloud
coffee-pull_pointcloud           coffee-push_pointcloud
dial-turn_pointcloud             disassemble_pointcloud
door-close_pointcloud            door-lock_pointcloud
door-open_pointcloud             door-unlock_pointcloud
drawer-close_pointcloud          faucet-close_pointcloud
faucet-open_pointcloud           hammer_pointcloud
hand-insert_pointcloud           handle-press_pointcloud
handle-pull-side_pointcloud      handle-pull_pointcloud
lever-pull_pointcloud            peg-insert-side_pointcloud
peg-unplug-side_pointcloud       pick-out-of-hole_pointcloud
pick-place-wall_pointcloud       pick-place_pointcloud
plate-slide-back-side_pointcloud plate-slide-back_pointcloud
plate-slide-side_pointcloud      plate-slide_pointcloud
push-wall_pointcloud             push_pointcloud
reach-wall_pointcloud            reach_pointcloud
shelf-place_pointcloud           soccer_pointcloud
stick-pull_pointcloud            stick-push_pointcloud
sweep-into_pointcloud            sweep_pointcloud
window-close_pointcloud          window-open_pointcloud
```

Adroit task 为 `adroit/door_pointcloud`、`adroit/hammer_pointcloud` 和
`adroit/pen_pointcloud`。

临时切换为通用 PointNet：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_adroit \
  encoder=pointcloud/pointnet
```

只检查最终合成配置而不启动训练：

```bash
python -u train_sim.py --config-name=train \
  experiment=padp_adroit task=adroit/pen_pointcloud \
  --cfg job --resolve
```

本机 CPU 诊断可追加 `runtime=cpu`；正式训练默认使用 `runtime=cuda`。

## 点云 DP / PADP / Focal 对比

MetaWorld：

```bash
experiment=compare_dp_metaworld
experiment=compare_padp_metaworld
experiment=focal_metaworld
```

Adroit：

```bash
experiment=compare_dp_adroit
experiment=compare_padp_adroit
experiment=focal_adroit
```

六个 preset 默认固定 `encoder=pointcloud/focal_xyz`、`capacity=base` 和
`recipe=pointcloud`。controlled 对比中的 DP、PADP、Focal 统一使用 `12/2/4`；
原生 temporal 只能放入另列的 full-method 轨道。点云正式训练统一使用 batch 128、1000 epoch、每 100 epoch
rollout/checkpoint；链路烟测统一追加
`capacity=smoke recipe=smoke_pointcloud`。正式命令、后台函数和多 seed 规范见
`tests/运行命令/点云三算法对比.md`。
