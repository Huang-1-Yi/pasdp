# PASDP 服务器实验进度

更新时间：2026-09-12

## 当前状态

| 项目 | 当前值 |
|---|---|
| 项目目录 | `/home/hy/dp_pro6000` |
| Conda环境 | `/opt/miniconda3/envs/sim_torch27_v8` |
| Python / Torch | 3.10.13 / 2.7.1+cu128 |
| GPU0 | RTX 4090；单卡运行时 CUDA 0 / EGL 1 |
| GPU1 | RTX 4090；单卡运行时 CUDA 1 / EGL 0 |
| MuJoCo | `/home/hy/.mujoco/mujoco210` |
| 当前训练/评估进程 | 未发现；本次受限会话无法读取 `nvidia-smi` |
| 当前阶段 | 十任务 seed42 的 A/B 训练与100-episode正式evaluation全部完成；下一批可运行C组36项；D组暂缓 |

## 数据集状态

| Task | 数据文件 | 状态 |
|---|---|---|
| coffee_d0 | `data/robomimic/datasets/coffee_d0/coffee_d0_abs.hdf5` | 可用；A/B/C/D seed42评估完成 |
| threading_d0 | `data/robomimic/datasets/threading_d0/threading_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| pick_place_d0 | `data/robomimic/datasets/pick_place_d0/pick_place_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| coffee_preparation_d0 | `data/robomimic/datasets/coffee_preparation_d0/coffee_preparation_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| hammer_cleanup_d0 | `data/robomimic/datasets/hammer_cleanup_d0/hammer_cleanup_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| square_d0 | `data/robomimic/datasets/square_d0/square_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| stack_d0 | `data/robomimic/datasets/stack_d0/stack_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| stack_three_d0 | `data/robomimic/datasets/stack_three_d0/stack_three_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| nut_assembly_d0 | `data/robomimic/datasets/nut_assembly_d0/nut_assembly_d0_abs.hdf5` | 可用；A/B seed42评估完成 |
| three_piece_assembly_d0 | `data/robomimic/datasets/three_piece_assembly_d0/three_piece_assembly_d0_abs.hdf5` | 可用；A/B seed42评估完成 |

## seed42 A/B/C/D进度

| 范围 | 完成数 | 状态 |
|---|---:|---|
| 十任务A训练 | 40/40 | 完成 |
| 十任务B新增训练 | 30/30 | 完成；B-PASDP复用A-PASDP |
| 十任务A正式evaluation | 40/40 | 完成 |
| 十任务B正式evaluation | 30/30真实运行 + 10/10复用 | 完成 |
| C组正式evaluation | 4/40 | `coffee_d0`完成；其余九任务待36次 |
| D组正式evaluation | 4/40逻辑项 | 仅`coffee_d0`完成；其余九任务无D checkpoint |
| A/B/C/D逻辑结果合计 | 88/160 | 单训练种子seed42 |

## 实际公共训练参数

| 参数 | 值 |
|---|---|
| Task / seed / demos | 十个Robomimic任务 / `42 / 200` |
| Epoch / rollout / checkpoint | 251 / 每5 epoch / 每5 epoch |
| Batch / workers | 64 / 8 |
| `n_envs` | 20 |
| `n_train / n_train_vis / train_start_idx` | 10 / 2 / 0 |
| `n_test / n_test_vis / test_start_seed` | 30 / 2 / 100000 |
| UNet / crop | `[256,512,1024] / 76×76` |
| 输出根目录 | `data/experiments/train` |

## 冻结的正式独立评估协议

| 参数 | 值 |
|---|---|
| `n_train / n_train_vis` | 0 / 0 |
| `n_test / n_test_vis` | 100 / 2 |
| `n_envs` | 25 |
| Final test seeds | `20000–20099` |
| 与训练监控种子关系 | 不重叠；训练监控仍为 `100000–100029` |
| 新输出协议目录 | `n_test100_seed20000` |

## 下一步

| 顺序 | 范围 | 数量 | 状态 |
|---:|---|---:|---|
| 1 | 其余九任务C组，每任务NFE 1/20/40/100 | 36次真实评估 | 可直接运行，无需新训练 |
| 2 | 其余九任务D组 | 若沿用旧设计：27训练 + 27评估 + 9复用 | 暂缓，先确定D组新设计 |
| 3 | training seeds `10000` 和 `1` | 完整矩阵扩展 | 尚未开始 |

## 文档

| 内容 | 文件 |
|---|---|
| 实验配置 | `PASDP_仿真实验_运行参数一览表.md` |
| 训练与评估结果 | `PASDP_仿真实验_结果一览表.md` |
| 实际执行明细 | `PASDP_服务器实验进度_实际执行.md` |
| 综合实验记录 | `实验记录.md` |
| 2080 Ti启动脚本 | `run_pasdp_train_ab_group1_2080ti.sh` |
| 4090终端1启动脚本 | `run_pasdp_train_ab_group2_4090a.sh` |
| 4090终端2启动脚本 | `run_pasdp_train_ab_group3_4090b.sh` |
| 公共执行器 | `run_pasdp_train_remaining9_seed42.sh` |
| 其余九任务A/B正式评估 | `run_pasdp_eval_ab_remaining9_4090.sh` |
| 其余九任务A/B汇总 | `data/experiments/eval/summary/remaining9/seed42/n_test100_seed20000/eval_summary.tsv` |
