# PASDP 实验配置一览表

更新时间：2026-09-12

## 训练实验配置（coffee_d0 / seed 42）

| ID | Hydra experiment | 物理 GPU | CUDA / EGL | Policy | `T_o` | `H` | `T_a` | `T_train` | `T_infer/config` | NFE | Pred | Sampler | Encoder / modality | UNet channels | Crop | Dataset loader | Batch | Workers | Shuffle | Epochs | LR | Optimizer | Scheduler / warmup | EMA | Rollout every | Checkpoint every | `n_envs` | `n_train` | `n_train_vis` | `train_start_idx` | `n_test` | `n_test_vis` | `test_start_seed` | Max steps | `n_demo` | Output group |
|---|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-PASDP | `pasdp_a_native_robomimic` | GPU0 2080 Ti | 0 / 1 | PASDP | 1 | 40 | 1 | 40 | 40（position depth） | steady 1 / cold 41 | sample | position-wise + sliding buffer | Robomimic RGB | `[256,512,1024]` | 76×76 | RTV8 PADP | 64 | 8 | false | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `A/pasdp/coffee_d0/seed42` |
| A-DP | `dp_a_paper_robomimic` | GPU0 2080 Ti | 0 / 1 | DP | 2 | 16 | 8 | 100 | 10 | 10 | epsilon | DDIM | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `A/dp/coffee_d0/seed42` |
| A-SDP | `sdp_a_paper_robomimic` | GPU0 2080 Ti | 0 / 1 | SDP | 2 | 20 | 8 | 100 | grid 16 | 8 | epsilon | DDIMTEDi + buffer | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `A/sdp/coffee_d0/seed42` |
| A-Flow | `flow_a_rgb_adapted_robomimic` | GPU0 2080 Ti | 0 / 1 | FlowPolicy RGB | 2 | 8 | 4 | continuous | 1 | 1 | velocity | Euler | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `A/flow/coffee_d0/seed42` |
| B-DP | `dp_b_nfe1_robomimic` | GPU1 4090 | 1 / 0 | DP | 2 | 16 | 1 | 100 | 1 | 1 | epsilon | DDIM | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `B/dp/coffee_d0/seed42` |
| B-SDP | `sdp_b_nfe1_robomimic` | GPU1 4090 | 1 / 0 | SDP | 2 | 20 | 1 | 100 | grid 19 | 1 | epsilon | DDIMTEDi + buffer | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `B/sdp/coffee_d0/seed42` |
| B-Flow | `flow_b_nfe1_rgb_adapted_robomimic` | GPU1 4090 | 1 / 0 | FlowPolicy RGB | 2 | 4 | 1 | continuous | 1 | 1 | velocity | Euler | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `B/flow/coffee_d0/seed42` |
| D-DP | `dp_d_pasdp_matched_robomimic` | GPU1 4090 | 1 / 0 | DP adapted | 1 | 40 | 1 | 40 | 40 | 40 | sample | DDIM | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `D/dp/coffee_d0/seed42` |
| D-SDP | `sdp_d_pasdp_matched_robomimic` | GPU1 4090 | 1 / 0 | SDP adapted | 1 | 40 | 1 | 40 | grid 40 | 1 | sample | DDIMTEDi + buffer | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `D/sdp/coffee_d0/seed42` |
| D-Flow | `flow_d_pasdp_matched_rgb_adapted_robomimic` | GPU1 4090 | 1 / 0 | FlowPolicy RGB | 1 | 40 | 1 | continuous | 1 | 1 | velocity | Euler | Robomimic RGB | `[256,512,1024]` | 76×76 | Standard | 64 | 8 | true | 251 | 1e-4 | AdamW | cosine / 500 | yes | 5 | 5 | 20 | 10 | 2 | 0 | 30 | 2 | 100000 | 400 | 200 | `D/flow/coffee_d0/seed42` |

## Checkpoint 复用与已执行评估配置

| ID | 类型 | Source checkpoint | `T_o` | `H` | `T_a` | `T_train` | Pred | Eval NFE | `n_envs` | `n_test` | Test seeds | 独立评估状态 |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|
| B-PASDP | 复用 | A-PASDP epoch210_score1.0000 | 1 | 40 | 1 | 40 | sample | steady 1 / cold 41 | 20 | 30 | 100000–100029 | 复用完成 |
| D-PASDP | 复用 | A-PASDP epoch210_score1.0000 | 1 | 40 | 1 | 40 | sample | steady 1 / cold 41 | 20 | 30 | 100000–100029 | 复用完成 |
| C-DP-NFE1 | Evaluation | A-DP epoch215_score1.0000 | 2 | 16 | 8 | 100 | epsilon | 1 | 20 | 30 | 100000–100029 | 完成 |
| C-DP-NFE20 | Evaluation | 同上 | 2 | 16 | 8 | 100 | epsilon | 20 | 20 | 30 | 100000–100029 | 完成 |
| C-DP-NFE40 | Evaluation | 同上 | 2 | 16 | 8 | 100 | epsilon | 40 | 20 | 30 | 100000–100029 | 完成 |
| C-DP-NFE100 | Evaluation | 同上 | 2 | 16 | 8 | 100 | epsilon | 100 | 20 | 30 | 100000–100029 | 完成 |

## 冻结的100-episode正式评估配置（已执行）

| 范围 | `n_train` | `n_train_vis` | `n_test` | `n_test_vis` | `n_envs` | Test seeds | 输出协议目录 | 状态 |
|---|---:|---:|---:|---:|---:|---|---|---|
| A/B/C/D，`coffee_d0 / seed42` | 0 | 0 | 100 | 2 | 25 | 20000–20099 | `n_test100_seed20000` | 16/16逻辑项完成 |
| A/B，其余九任务 / seed42 | 0 | 0 | 100 | 2 | 25 | 20000–20099 | `n_test100_seed20000` | 63次真实评估 + 9条复用完成 |
| C，其余九任务 / seed42 | 0 | 0 | 100 | 2 | 25 | 20000–20099 | `n_test100_seed20000` | 待36次；复用各任务A-DP checkpoint |
| D，其余九任务 / seed42 | 0 | 0 | 100 | 2 | 25 | 20000–20099 | `n_test100_seed20000` | 暂缓；无D checkpoint，待重定设计 |

本表上方的 `100000–100029` 行是已经完成的30-episode pilot 历史记录；正式100-episode评估使用不重叠的新种子，不覆盖或改写历史结果。
