# PASDP 固定实验矩阵执行计划

更新时间：2026-09-12

> 2026-09-12 当前阶段：training seed 42 的十任务A+B训练及100-episode正式评估均已完成。C组仅`coffee_d0`完成，其余九任务待36次评估；D组仅`coffee_d0`有checkpoint和评估，其余任务暂缓并待重新设计。下文原始全矩阵和多seed安排保留为后续扩展参考。

## 1. 已冻结的执行口径

- 任务数：10 个，全部运行
- 训练 seeds：`42`、`10000`、`1`
- 执行顺序：`A → B → D → C`
- GPU 方式：GPU0 跑 A/B，GPU1 跑 D/C；两张卡各自串行推进，彼此不做同卡并行
- A/B/D 训练后 evaluation：统一使用 top-k 规则选出的 checkpoint
- C：复用同一个 A-DP checkpoint，最后跑
- Pilot：先只跑 `coffee_d0 + seed42`
- 训练期 rollout：保持训练协议冻结值不变
- 正式 evaluation：`n_test=100`、`n_test_vis=2`、`n_train=0`、`n_train_vis=0`、`n_envs=25`、测试种子 `20000–20099`

## 2. 推荐任务切分

默认按 5/5 平分到两张卡，方便两条串行链同时推进：

- GPU0：`coffee_d0`、`threading_d0`、`pick_place_d0`、`coffee_preparation_d0`、`hammer_cleanup_d0`
- GPU1：`square_d0`、`stack_d0`、`stack_three_d0`、`nut_assembly_d0`、`three_piece_assembly_d0`

如果你想换顺序，只要交换下面两个任务列表即可，不影响矩阵规则。

## 3. 通用环境变量模板

两张卡都沿用同一套基础环境变量，只改 `CUDA_VISIBLE_DEVICES` 和 `PASDP_TASK`：

```bash
cd /home/hy/dp_pro6000
export PASDP_PY=/opt/miniconda3/envs/sim_torch27_v8/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64
export HYDRA_FULL_ERROR=1
export WANDB_MODE=offline
```

## 4. 统一的训练与 evaluation 规则

### 4.1 训练阶段

- A：`pasdp_a_native_robomimic`、`dp_a_paper_robomimic`、`sdp_a_paper_robomimic`、`flow_a_rgb_adapted_robomimic`
- B：`dp_b_nfe1_robomimic`、`sdp_b_nfe1_robomimic`、`flow_b_nfe1_rgb_adapted_robomimic`，PASDP 复用 A-PASDP
- D：`dp_d_pasdp_matched_robomimic`、`sdp_d_pasdp_matched_robomimic`、`flow_d_pasdp_matched_rgb_adapted_robomimic`，PASDP 复用 A-PASDP
- 每个 seed 内部必须严格按 `A → B → D → C` 顺序推进

### 4.2 evaluation 阶段

- A/B/D：训练完成后，统一挑选 top-k 规则下的 checkpoint 再评估
- C：最后跑，且只复用同一个 A-DP checkpoint
- evaluation 协议：`n_test=100`、`n_test_vis=2`、`n_train=0`、`n_train_vis=0`、`n_envs=25`、测试种子 `20000–20099`

## 5. GPU0 串行计划

GPU0 负责所有任务的 A/B 阶段。对于每个 task/seed，先完成 A，再完成 B。

### 5.1 GPU0 第 1 个任务：coffee_d0

#### seed 42

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=1
export PASDP_TASK=coffee_d0
export PASDP_SEED=42
```

先跑 A，再跑 B。A 产出的 A-PASDP 和 A-DP checkpoint 会供 GPU1 的 D/C 使用；B 只在 GPU0 上继续执行，不重复启动 PASDP。

#### seed 10000

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=1
export PASDP_TASK=coffee_d0
export PASDP_SEED=10000
```

先跑 A，再跑 B。

#### seed 1

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=1
export PASDP_TASK=coffee_d0
export PASDP_SEED=1
```

先跑 A，再跑 B。

### 5.2 GPU0 后续任务

依次串行推进：`threading_d0` -> `pick_place_d0` -> `coffee_preparation_d0` -> `hammer_cleanup_d0`

每个任务都按同样的 seed 顺序：`42 -> 10000 -> 1`，每个 seed 内按 `A -> B`。

## 6. GPU1 串行计划

GPU1 负责所有任务的 D/C 阶段。对于每个 task/seed，先等待对应 A 阶段完成并拿到 A-PASDP 与 A-DP checkpoint，再完成 D，最后完成 C。

### 6.1 GPU1 第 1 个任务：square_d0

#### seed 42

```bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=0
export PASDP_TASK=square_d0
export PASDP_SEED=42
```

先跑 D，再跑 C。D 中 PASDP 复用 GPU0 产出的 A-PASDP checkpoint；C 复用 GPU0 产出的 A-DP checkpoint。

#### seed 10000

```bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=0
export PASDP_TASK=square_d0
export PASDP_SEED=10000
```

先跑 D，再跑 C。

#### seed 1

```bash
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_EGL_DEVICE_ID=0
export PASDP_TASK=square_d0
export PASDP_SEED=1
```

先跑 D，再跑 C。

### 6.2 GPU1 后续任务

依次串行推进：`stack_d0` -> `stack_three_d0` -> `nut_assembly_d0` -> `three_piece_assembly_d0`

每个任务都按同样的 seed 顺序：`42 -> 10000 -> 1`，每个 seed 内按 `D -> C`。

## 7. 启动前建议

- 先在 GPU0 上只跑 `coffee_d0 + seed42` 作为 pilot
- 观察日志持续增长、checkpoint 生成、Hydra resolved config 正常后，再让 GPU1 进入对应 D/C 阶段
- A/B 训练结束后统一取 top-k checkpoint 做 evaluation
- D 放在 GPU1 的每个 task/seed 最后，且必须复用同一个 A-PASDP checkpoint
- C 放在 GPU1 的每个 task/seed 最后，且必须复用同一个 A-DP checkpoint

## 8. 注意事项

- 不要改写训练期 rollout 的 `n_train=10,n_test=30,n_envs=20,test_start_seed=100000`；正式 evaluation 单独使用 `n_train=0,n_test=100,n_envs=25,test_start_seed=20000`
- 不要在命令行覆盖 `horizon`、`n_obs_steps`、`n_action_steps`、`prediction_type`、`num_train_timesteps`、`num_inference_steps`
- PASDP 的 RTV8 loader 与 `shuffle=false` 必须保留
- 失败 run 保留，不覆盖重跑

## 9. 两张卡的最终分工摘要

- GPU0：A + B
- GPU1：D + C
- 同一个 task/seed 的 A 先在 GPU0 完成后，GPU1 再消费 A-PASDP 和 A-DP checkpoint 进入 D/C
- 不使用 for 循环也可以，用 `&&` 串联每一步命令即可

## 9. 连续串行执行方式

如果你希望每张 GPU 自动连续跑几种算法，不需要写 `for` 循环，直接用一条串行命令链即可。

推荐方式：

- 先准备好一个 GPU 对应的任务、seed 和算法顺序
- 每个步骤都单独写一条命令
- 用 `&&` 把命令串起来，前一个成功后才会自动进入下一个
- 如果某一步失败，后面的步骤不会继续执行

示例逻辑：

```bash
cmd_A && cmd_B && cmd_D && cmd_C
```

对于本次正式矩阵，若只是做某个 task/seed 的连续执行，可以写成：

```bash
python train_sim.py ... A ... && \
python train_sim.py ... B ... && \
python train_sim.py ... D ... && \
python eval_sim_rollout.py ... C ...
```

这样就能满足“每个 GPU 自动连续跑几种算法”，不必使用循环结构。
