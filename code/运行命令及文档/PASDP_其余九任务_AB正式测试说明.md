# PASDP 其余九任务 A/B 正式测试说明

更新时间：2026-09-12

## 1. 测试目标

对除 `coffee_d0` 以外的九个 Robomimic 任务执行 A/B 正式独立评估。测试沿用已完成 `coffee_d0` pilot 的正式流程，不重新训练、不修改 checkpoint，只测训练完成后的策略效果。

每个任务实际评估以下七个唯一模型：A-PASDP、A-DP、A-SDP、A-Flow、B-DP、B-SDP、B-Flow。B-PASDP 与 A-PASDP 是同一个训练模型，因此复用 A-PASDP 的 checkpoint 和评估结果，不重复 rollout。

九任务合计执行 63 次真实评估，并生成 9 条 B-PASDP 复用记录，共形成 72 条逻辑 A/B 结果。

## 2. 冻结测试协议

| 参数 | 值 |
|---|---|
| Training seed | 42 |
| `n_demo` | 200 |
| Evaluation `n_train / n_train_vis` | 0 / 0 |
| Evaluation `n_test / n_test_vis` | 100 / 2 |
| Evaluation `n_envs` | 25 |
| Evaluation test seeds | `20000–20099` |
| Training monitor seeds | `100000–100029`，仅用于选择 checkpoint，不与正式测试种子重叠 |
| Checkpoint规则 | 保留的 top-k 中 monitor score 最高者；同分取 epoch 较晚者 |
| 执行方式 | 单GPU、逐任务、逐方法严格串行 |
| 默认GPU | 物理 GPU1；`CUDA_VISIBLE_DEVICES=1`、`MUJOCO_EGL_DEVICE_ID=0` |

方法顺序固定为：

1. A-PASDP
2. A-DP
3. A-SDP
4. A-Flow
5. B-PASDP复用记录
6. B-DP
7. B-SDP
8. B-Flow

任务顺序固定为：`threading_d0`、`pick_place_d0`、`coffee_preparation_d0`、`hammer_cleanup_d0`、`square_d0`、`stack_d0`、`stack_three_d0`、`nut_assembly_d0`、`three_piece_assembly_d0`。

## 3. 启动前条件

脚本要求九任务的 63 个唯一训练全部具有 `.done` 标记，并验证选中训练目录的日志最后一个 epoch 为 250。只存在 `latest.ckpt` 或中间 top-k checkpoint 不算训练完成。

截至 2026-09-12，九任务 63/63 个唯一训练均已生成 `.done` 标记，完整日志均到达 epoch 250；串联评估的全部 checkpoint 已通过预检。运行前仍可复查：

```bash
cd /home/hy/dp_pro6000
bash run_pasdp_eval_ab_remaining9_4090.sh --check
```

检查命令应逐任务输出 `[READY]`，最终显示63个唯一训练 checkpoint 均可评估。若以后训练目录发生变化或文件缺失，检查会明确列出问题并退出，不会启动评估。

## 4. 推荐启动命令

本轮已于 2026-09-12 19:07 全部执行完成。以下命令仅用于日后断点复查；已完成项会被 `PASDP_SKIP_DONE=1` 跳过：

```bash
cd /home/hy/dp_pro6000
conda activate sim_torch27_v8
bash run_pasdp_eval_ab_remaining9_4090.sh
```

如果希望现在就启动一个等待队列，训练全部完成后自动开始评估：

```bash
cd /home/hy/dp_pro6000
conda activate sim_torch27_v8
bash run_pasdp_eval_ab_remaining9_4090.sh --wait-training
```

等待模式每30秒检查一次完成标记。若训练失败而没有生成 `.done`，评估脚本会继续等待，此时可按 `Ctrl+C` 停止等待，修复训练后重新执行。

## 5. 断点续测和输出

脚本默认 `PASDP_SKIP_DONE=1`。如果评估中断，重新执行同一命令时，已经存在完整 `eval_log.json` 的项目会自动跳过，不会重复测试或覆盖结果。

若某个评估目录已经创建但缺少 `eval_log.json`，脚本会拒绝覆盖并停下。应先检查对应 `console.log` 确认失败原因，再决定是否手动移走该不完整目录。

单项结果写入：

```text
data/experiments/eval/<A或B>/<算法>/<task>/seed42/n_test100_seed20000/default/eval_log.json
```

控制台日志写入：

```text
data/experiments/logs/eval/<A或B>/<算法>/<task>/seed42/n_test100_seed20000/default/console.log
```

九任务统一汇总写入：

```text
data/experiments/eval/summary/remaining9/seed42/n_test100_seed20000/eval_summary.tsv
data/experiments/eval/summary/remaining9/seed42/n_test100_seed20000/eval_summary.json
```

每个任务还会生成一份独立汇总，位于：

```text
data/experiments/eval/summary/<task>/seed42/n_test100_seed20000/
```

## 6. 可选设置

默认使用物理 GPU1。若需要改用当前机器的物理 GPU0：

```bash
PASDP_EVAL_PHYSICAL_GPU=0 bash run_pasdp_eval_ab_remaining9_4090.sh
```

本机映射会自动改为 `CUDA_VISIBLE_DEVICES=0`、`MUJOCO_EGL_DEVICE_ID=1`。不要在两个终端同时运行同一串联脚本，否则可能同时选择相同输出目录并造成冲突。

如需只打印将要执行的命令，可在所有训练完成后运行：

```bash
bash run_pasdp_eval_ab_remaining9_4090.sh --dry-run
```

## 7. 结果解释约束

- 训练期 `test/success_rate` 来自 `100000–100029` 共30个 monitor seeds；正式结果来自 `20000–20099` 共100个独立测试种子，两者必须分开记录。
- checkpoint 对所有方法使用同一选择规则，不能在看到正式测试结果后改选 checkpoint。
- B-PASDP 是 A-PASDP 的复用项，不得统计为一次新的训练或一次新的 rollout。
- 九任务仍只有一个 training seed 42。本轮100-episode测试可以比较任务覆盖情况，但不能把单 training seed 的结果表述为多 seed 均值和标准差。
- 汇总结果已补入 `实验记录.md` 与 `PASDP_仿真实验_结果一览表.md`。

## 8. 实际完成结果

| Task | A-PASDP | A-DP | A-SDP | A-Flow | B-PASDP | B-DP | B-SDP | B-Flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `threading_d0` | 0.9000 | 0.7900 | 0.7900 | 0.7400 | 0.9000 | 0.0000 | 0.8100 | 0.6800 |
| `pick_place_d0` | 0.1900 | 0.2700 | 0.3700 | 0.0100 | 0.1900 | 0.0000 | 0.2300 | 0.0000 |
| `coffee_preparation_d0` | 0.8300 | 0.7900 | 0.7500 | 0.7500 | 0.8300 | 0.0000 | 0.5500 | 0.2700 |
| `hammer_cleanup_d0` | 1.0000 | 0.9400 | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 0.1500 | 0.9900 |
| `square_d0` | 0.9100 | 0.8500 | 0.8900 | 0.7900 | 0.9100 | 0.0000 | 0.2700 | 0.5400 |
| `stack_d0` | 0.9300 | 0.9500 | 0.9800 | 0.9600 | 0.9300 | 0.0000 | 0.8600 | 0.6000 |
| `stack_three_d0` | 0.5000 | 0.8100 | 0.7100 | 0.5400 | 0.5000 | 0.0000 | 0.6700 | 0.2500 |
| `nut_assembly_d0` | 0.5500 | 0.5100 | 0.5500 | 0.2400 | 0.5500 | 0.0000 | 0.0700 | 0.0100 |
| `three_piece_assembly_d0` | 0.5400 | 0.6700 | 0.5800 | 0.2600 | 0.5400 | 0.0000 | 0.4900 | 0.0800 |
| **9-task macro mean** | **0.7056** | **0.7311** | **0.7356** | **0.5867** | **0.7056** | **0.0000** | **0.4556** | **0.3800** |

完成数为 63/63 次真实评估和 9/9 条 B-PASDP 复用记录，共72/72条逻辑结果。原始 TSV/JSON 及逐任务汇总路径见第5节。
