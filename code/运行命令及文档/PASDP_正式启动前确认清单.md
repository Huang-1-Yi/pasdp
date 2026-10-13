# PASDP 正式启动前确认清单

更新时间：2026-08-24
用途：在启动正式训练与 evaluation 之前，逐项冻结实验范围与执行规则。

请按“已确认 / 待确认 / 备注”填写。若某项尚未决定，请直接写“待确认”，不要留空。

## 1. 当前已完成的前置事实

- 仓库已同步到最新提交：`c16a739`
- 预检已通过：Git、环境、GPU、磁盘、数据、Hydra compose、保活方式
- 目标 10 个 Robomimic 任务数据文件均可读
- 固定实验顺序已冻结：A → B → D → C
- 固定唯一训练配置数已冻结：10
- 训练与 evaluation 目录将使用 `data/experiments/`，不复用历史 `data/outputs/`

## 2. 需要最终确认的事项

### 2.1 正式任务列表

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| 是否跑全部 10 个任务 | 已确认 | 全部 10 个任务 |
| coffee_d0 | 已可用 |  |
| threading_d0 | 已可用 |  |
| pick_place_d0 | 已可用 |  |
| coffee_preparation_d0 | 已可用 |  |
| hammer_cleanup_d0 | 已可用 |  |
| square_d0 | 已可用 |  |
| stack_d0 | 已可用 |  |
| stack_three_d0 | 已可用 |  |
| nut_assembly_d0 | 已可用 |  |
| three_piece_assembly_d0 | 已可用 |  |

### 2.2 Training seeds

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| seed 42 | 已知可用 | 已确认 |
| seed 10000 | 待确认 | 已确认 |
| seed 1 | 待确认 | 已确认 |

请明确另外两个固定 training seeds，且三种方法、四个阶段都必须共用同一组 seeds。

### 2.3 GPU 与并行度

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| GPU0 分配 | 可用 | 已确认，单独串行运行 |
| GPU1 分配 | 可用 | 已确认，单独串行运行 |
| 是否允许并行 | 待确认 | 不并行，串行 |
| 若并行，是否按 stage 并行 | 待确认 | 不适用 |
| 若并行，是否按 task 并行 | 待确认 | 不适用 |
| 若串行，单卡还是双卡轮转 | 待确认 | 双卡分开串行，各自独立顺序执行 |

### 2.4 Checkpoint 选择规则

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| A/B/D 训练后 evaluation 使用 latest.ckpt | 待确认 | 否 |
| 还是统一使用 top-k 规则选出的 checkpoint | 待确认 | 已确认 |
| 若使用 top-k，是否固定同一条规则 | 待确认 | 已确认，统一同一条 top-k 规则 |
| C 组是否必须复用同一个 A-DP checkpoint | 已冻结 |  |

### 2.5 Pilot 策略

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| 是否先只跑 coffee_d0 + seed42 pilot | 待确认 | 已确认 |
| pilot 只做 A 阶段 | 建议 |  |
| pilot 成功后再扩展到 B/D/C | 建议 |  |

### 2.6 Evaluation 协议

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| n_test = 50 | 已冻结 | 否，改为 100 |
| n_test_vis = 2 | 已冻结 |  |
| n_train = 0 | 已冻结 |  |
| n_train_vis = 0 | 已冻结 |  |
| n_envs = 32 | 已冻结 | 否，改为 50 |
| 是否允许覆盖默认值 | 不允许 | 允许正式 evaluation 覆盖为 100 demo / 2 vis / 50 envs |

### 2.7 训练公共协议

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| n_demo = 200 | 已冻结 |  |
| epochs = 251（0–250） | 已冻结 |  |
| rollout_every = 5 | 已冻结 |  |
| checkpoint_every = 5 | 已冻结 |  |
| UNet channels = [256,512,1024] | 已冻结 |  |
| RGB crop = 84×84 -> crop76 | 已冻结 |  |
| PASDP 使用 RTV8 loader + shuffle=false | 已冻结 |  |
| PASDP 需要记录 mapping_manifest.npz | 已冻结 |  |

### 2.8 目录与保留规则

| 项目 | 当前状态 | 你的确认 |
|---|---|---|
| 训练目录使用 data/experiments/train | 已冻结 |  |
| evaluation 目录使用 data/experiments/eval | 已冻结 |  |
| logs 使用 data/experiments/logs | 已冻结 |  |
| manifests 使用 data/experiments/manifests | 已冻结 |  |
| 不覆盖已有正式目录 | 已冻结 |  |
| 失败 run 保留，不删除重跑 | 已冻结 |  |

## 3. 建议的回复格式

你可以直接按下面格式回复：

```text
任务：coffee_d0, threading_d0, ...
seeds：42, x, y
GPU：A/B 或 单卡串行
并行度：允许/不允许
checkpoint：latest.ckpt / top-k 规则
pilot：是/否
evaluation：n_train=0, n_train_vis=0, n_test=100, n_test_vis=2, n_envs=25, test_start_seed=20000
```

## 4. 回复后将进入的执行顺序

1. 先按用户确认的 pilot 或首个任务启动。
2. 确认首个任务日志持续更新后，再启动下一项。
3. 严格按 A → B → D → C 顺序推进。
4. 每个 run 结束后登记 checkpoint、日志、状态和是否复用。

## 5. 当前待办

- 等待用户最终确认以上事项
- 确认后正式启动 A 阶段
