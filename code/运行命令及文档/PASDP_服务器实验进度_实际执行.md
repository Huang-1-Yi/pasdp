# PASDP 服务器实验进度（实际执行）

更新时间：2026-09-12

## 当前执行状态

| 项目 | 状态 |
|---|---|
| 服务器 | `/home/hy/dp_pro6000` |
| 当前范围 | 十个Robomimic任务 / training seed 42 / n_demo 200 |
| A/B唯一训练 | 70/70完成，均到epoch 250 |
| D唯一训练 | 3/30完成；仅`coffee_d0` |
| B-PASDP / D-PASDP | 复用A-PASDP，不重复训练 |
| A/B 100-episode evaluation | 70/70真实评估 + 10/10 PASDP复用；全部完成 |
| C组NFE 1/20/40/100 | 4/40完成；仅`coffee_d0` |
| D组100-episode evaluation | 3/30真实评估 + 1/10 PASDP复用；仅`coffee_d0` |
| 正式评估口径 | `n_envs=25`，`n_test=100`，test seeds `20000–20099`；原30-episode结果已保留 |
| 当前训练/评估进程 | 未发现；本次受限会话无法读取 `nvidia-smi` |

## 实际训练协议

| 参数 | 实际值 |
|---|---|
| Task / training seed | 十个Robomimic任务 / `42` |
| `n_demo` | 200 |
| Epoch | 0–250，共251 |
| Batch / DataLoader workers | 64 / 8 |
| Rollout / checkpoint interval | 5 / 5 |
| `n_envs` | 20 |
| `n_train / n_train_vis / train_start_idx` | 10 / 2 / 0 |
| `n_test / n_test_vis / test_start_seed` | 30 / 2 / 100000 |
| Backbone / crop | `[256,512,1024] / 76×76` |
| 当前硬件 | GPU0 / GPU1 均为RTX 4090 |
| EGL绑定 | 物理GPU0：CUDA 0 / EGL 1；物理GPU1：CUDA 1 / EGL 0 |

## `coffee_d0` pilot训练完成清单

| ID | Experiment | Final epoch | Final monitor test success | 状态 |
|---|---|---:|---:|---|
| A-PASDP | `pasdp_a_native_robomimic` | 250 | 0.9667 | 完成 |
| A-DP | `dp_a_paper_robomimic` | 250 | 0.9000 | 完成 |
| A-SDP | `sdp_a_paper_robomimic` | 250 | 0.9000 | 完成 |
| A-Flow | `flow_a_rgb_adapted_robomimic` | 250 | 0.9000 | 完成 |
| B-DP | `dp_b_nfe1_robomimic` | 250 | 0.0000 | 完成 |
| B-SDP | `sdp_b_nfe1_robomimic` | 250 | 0.8000 | 完成 |
| B-Flow | `flow_b_nfe1_rgb_adapted_robomimic` | 250 | 0.6000 | 完成 |
| D-DP | `dp_d_pasdp_matched_robomimic` | 250 | 0.0000 | 完成 |
| D-SDP | `sdp_d_pasdp_matched_robomimic` | 250 | 0.0000 | 完成 |
| D-Flow | `flow_d_pasdp_matched_rgb_adapted_robomimic` | 250 | 0.0000 | 完成 |

## 十任务A/B正式评估完成清单

| 任务范围 | 真实评估 | PASDP复用 | 逻辑结果 | 完成时间 |
|---|---:|---:|---:|---|
| `coffee_d0` A/B | 7/7 | 1/1 | 8/8 | 2026-08-29 |
| 其余九任务 A/B | 63/63 | 9/9 | 72/72 | 2026-09-12 19:07 |
| **合计** | **70/70** | **10/10** | **80/80** | **完成** |

## 输出与记录

| 文档/目录 | 路径 |
|---|---|
| 配置一览表 | `PASDP_仿真实验_运行参数一览表.md` |
| 结果一览表 | `PASDP_仿真实验_结果一览表.md` |
| 训练根目录 | `data/experiments/train` |
| PASDP RTV8 manifest | `data/robomimic/datasets/coffee_d0/coffee_d0_abs.rtdp_v8_maps.n200.h40.obs1.act1.b64/mapping_manifest.npz` |
| 独立评估汇总 | `data/experiments/eval/summary/coffee_d0/seed42/eval_summary.tsv` |
| 100-episode评估汇总 | `data/experiments/eval/summary/coffee_d0/seed42/n_test100_seed20000/eval_summary.tsv` |
| 其余九任务A/B汇总 | `data/experiments/eval/summary/remaining9/seed42/n_test100_seed20000/eval_summary.tsv` |
| 综合实验记录 | `实验记录.md` |
| 剩余9任务训练脚本 | `run_pasdp_train_remaining9_seed42.sh` |

## 待执行

| 顺序 | 项目 | 状态 |
|---:|---|---|
| 1 | `coffee_d0`按统一top-k规则选择A/B/D checkpoint | 完成 |
| 2 | `coffee_d0` A/B/D标准独立evaluation | 完成 |
| 3 | `coffee_d0`写入B-PASDP、D-PASDP复用记录 | 完成 |
| 4 | `coffee_d0` C组同一A-DP checkpoint的NFE 1/20/40/100 evaluation | 完成 |
| 5 | `coffee_d0 / seed42` 使用 `20000–20099` 的100-episode补充evaluation | 完成 |
| 6 | 其余9个任务，training seed 42，A+B共63次训练 | 完成 |
| 7 | 其余9个任务A+B正式100-episode evaluation | 完成；63次真实评估+9复用 |
| 8 | 其余9个任务C组NFE 1/20/40/100 evaluation | 待36次；可直接运行 |
| 9 | 重新设计D组逐因素消融 | 待决定；其余9任务无D checkpoint |
| 10 | training seeds `10000` 和 `1` 扩展 | 待后续决定 |
