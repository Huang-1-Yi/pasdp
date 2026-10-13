# PASDP 实验结果一览表

更新时间：2026-09-12

## 训练结果（coffee_d0 / seed 42）

| ID | 状态 | Start | 时长(h) | Final epoch | Global step | Final train loss | Final val loss | Final action MSE | Final train success | Final monitor test success | Best monitor test success | Avg completion step | Median completion step | Inference ms | Checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A-PASDP | 完成 | 2026-08-27 22:21 | 10.71 | 250 | 257678 | 0.002967 | — | — | 1.0000 | 0.9667 | 1.0000 | 235.17 | 221.0 | 16.87 | `data/experiments/train/A/pasdp/coffee_d0/seed42/2026.08.27/22.21.55_pasdp_a_native_h40_o1_a1_sample_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| A-DP | 完成 | 2026-08-28 09:04 | 7.31 | 250 | 165157 | 0.007526 | 0.010512 | 0.004801 | 1.0000 | 0.9000 | 1.0000 | 250.53 | 231.5 | 102.42 | `data/experiments/train/A/dp/coffee_d0/seed42/2026.08.28/09.04.56_dp_a_paper_h16_o2_a8_t100_i10_epsilon_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| A-SDP | 完成 | 2026-08-28 16:23 | 7.93 | 250 | 162145 | 0.007044 | 0.010905 | 0.009841 | 1.0000 | 0.9000 | 1.0000 | 245.73 | 232.5 | 85.81 | `data/experiments/train/A/sdp/coffee_d0/seed42/2026.08.28/16.23.24_sdp_a_paper_h20_o2_a8_t100_grid16_epsilon_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| A-Flow | 完成 | 2026-08-29 00:19 | 8.33 | 250 | 168169 | 1.392e-6 | 1.439e-6 | 0.005772 | 0.8000 | 0.9000 | 0.9667 | 247.87 | 222.5 | 18.48 | `data/experiments/train/A/flow/coffee_d0/seed42/2026.08.29/00.19.25_flow_a_rgb_h8_o2_a4_flow1_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| B-DP | 完成 | 2026-08-27 22:23 | 5.15 | 250 | 159635 | 0.007506 | 0.009817 | 0.628828 | 0.0000 | 0.0000 | 0.0000 | 400.00 | 400.0 | 21.27 | `data/experiments/train/B/dp/coffee_d0/seed42/2026.08.27/22.23.02_dp_b_nfe1_h16_o2_a1_t100_epsilon_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| B-SDP | 完成 | 2026-08-28 03:32 | 5.27 | 250 | 156623 | 0.005555 | 0.023210 | 0.212506 | 0.7000 | 0.8000 | 0.8000 | 327.83 | 327.5 | 21.57 | `data/experiments/train/B/sdp/coffee_d0/seed42/2026.08.28/03.32.27_sdp_b_nfe1_h20_o2_a1_t100_grid19_epsilon_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| B-Flow | 完成 | 2026-08-28 08:48 | 6.14 | 250 | 168922 | 1.353e-6 | 1.631e-6 | 0.005451 | 0.6000 | 0.6000 | 0.7000 | 315.07 | 300.0 | 19.65 | `data/experiments/train/B/flow/coffee_d0/seed42/2026.08.28/08.48.34_flow_b_nfe1_rgb_h4_o2_a1_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| D-DP | 完成 | 2026-08-27 22:23 | 8.95 | 250 | 140559 | 0.000357 | 0.005722 | 0.000222 | 0.0000 | 0.0000 | 0.0000 | 400.00 | 400.0 | 322.43 | `data/experiments/train/D/dp/coffee_d0/seed42/2026.08.27/22.23.49_dp_d_pasdp_matched_h40_o1_a1_t40_i40_sample_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| D-SDP | 完成 | 2026-08-28 07:20 | 5.64 | 250 | 140559 | 0.000396 | 0.003516 | 0.027709 | 0.0000 | 0.0000 | 0.0000 | 400.00 | 400.0 | 24.31 | `data/experiments/train/D/sdp/coffee_d0/seed42/2026.08.28/07.20.45_sdp_d_pasdp_matched_h40_o1_a1_t40_grid40_sample_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |
| D-Flow | 完成 | 2026-08-28 12:59 | 5.99 | 250 | 140559 | 1.179e-6 | 1.755e-6 | 0.002194 | 0.0000 | 0.0000 | 0.0000 | 400.00 | 400.0 | 17.61 | `data/experiments/train/D/flow/coffee_d0/seed42/2026.08.28/12.59.03_flow_d_pasdp_matched_rgb_h40_o1_a1_flow1_coffee_d0_seed42_coffee_d0/checkpoints/latest.ckpt` |

## 独立评估结果

| ID | Checkpoint | Evaluation episodes | NFE | Success rate | Avg completion step | Median completion step | Inference ms | 状态 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A-PASDP | epoch210_score1.0000 | 30 | steady 1 / cold 41 | 0.9000 | 242.40 | 217.0 | 19.14 | 完成 |
| A-DP | epoch215_score1.0000 | 30 | 10 | 0.9333 | 231.77 | 217.5 | 83.61 | 完成 |
| A-SDP | epoch155_score1.0000 | 30 | 8 | 1.0000 | 225.57 | 222.5 | 75.20 | 完成 |
| A-Flow | epoch135_score0.9667 | 30 | 1 | 0.9000 | 231.40 | 214.0 | 20.45 | 完成 |
| B-PASDP | 复用A-PASDP epoch210 | 30 | steady 1 / cold 41 | 0.9000 | 242.40 | 217.0 | 19.14 | 复用完成 |
| B-DP | epoch020_score0.0000 | 30 | 1 | 0.0000 | 400.00 | 400.0 | 21.61 | 完成 |
| B-SDP | epoch250_score0.8000 | 30 | 1 | 0.7667 | 313.67 | 322.5 | 20.49 | 完成 |
| B-Flow | epoch220_score0.7000 | 30 | 1 | 0.8000 | 287.73 | 269.5 | 19.70 | 完成 |
| D-PASDP | 复用A-PASDP epoch210 | 30 | steady 1 / cold 41 | 0.9000 | 242.40 | 217.0 | 19.14 | 复用完成 |
| D-DP | epoch020_score0.0000 | 30 | 40 | 0.0000 | 400.00 | 400.0 | 270.33 | 完成 |
| D-SDP | epoch020_score0.0000 | 30 | 1 | 0.0000 | 400.00 | 400.0 | 20.92 | 完成 |
| D-Flow | epoch020_score0.0000 | 30 | 1 | 0.0000 | 400.00 | 400.0 | 19.42 | 完成 |

## C组DP inference budget结果

| ID | Source checkpoint | NFE | Success rate | Avg completion step | Median completion step | Inference ms | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| C-DP-NFE1 | A-DP epoch215；30 episodes | 1 | 0.0000 | 400.00 | 400.0 | 24.86 | 完成 |
| C-DP-NFE20 | 同一checkpoint | 20 | 0.9333 | 230.53 | 215.0 | 145.67 | 完成 |
| C-DP-NFE40 | 同一checkpoint | 40 | 0.9333 | 234.20 | 216.5 | 273.28 | 完成 |
| C-DP-NFE100 | 同一checkpoint | 100 | 0.9333 | 231.00 | 216.5 | 666.87 | 完成 |

## 100-episode正式评估

| ID | Episodes | NFE | Success rate | Avg completion step | Median completion step | Inference ms | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| A-PASDP | 100 | steady 1 / cold 41 | 0.9700 | 223.61 | 216.0 | 18.68 | 完成 |
| A-DP | 100 | 10 | 0.9400 | 227.51 | 216.0 | 81.02 | 完成 |
| A-SDP | 100 | 8 | 0.9600 | 232.65 | 229.0 | 70.12 | 完成 |
| A-Flow | 100 | 1 | 0.8900 | 242.30 | 218.0 | 17.62 | 完成 |
| B-PASDP | 100 | steady 1 / cold 41 | 0.9700 | 223.61 | 216.0 | 18.68 | 复用A-PASDP |
| B-DP | 100 | 1 | 0.0000 | 400.00 | 400.0 | 20.67 | 完成 |
| B-SDP | 100 | 1 | 0.8000 | 313.97 | 322.0 | 20.25 | 完成 |
| B-Flow | 100 | 1 | 0.8100 | 273.49 | 244.0 | 18.69 | 完成 |
| D-PASDP | 100 | steady 1 / cold 41 | 0.9700 | 223.61 | 216.0 | 18.68 | 复用A-PASDP |
| D-DP | 100 | 40 | 0.0000 | 400.00 | 400.0 | 271.97 | 完成 |
| D-SDP | 100 | 1 | 0.0000 | 400.00 | 400.0 | 20.30 | 完成 |
| D-Flow | 100 | 1 | 0.0000 | 400.00 | 400.0 | 19.30 | 完成 |
| C-DP-NFE1 | 100 | 1 | 0.0000 | 400.00 | 400.0 | 20.61 | 完成 |
| C-DP-NFE20 | 100 | 20 | 0.9600 | 223.24 | 215.0 | 144.51 | 完成 |
| C-DP-NFE40 | 100 | 40 | 0.9700 | 222.96 | 217.0 | 275.93 | 完成 |
| C-DP-NFE100 | 100 | 100 | 0.9400 | 225.67 | 214.0 | 667.20 | 完成 |

## 十任务 A/B 100-episode正式评估

| Task | A-PASDP | A-DP | A-SDP | A-Flow | B-PASDP | B-DP | B-SDP | B-Flow |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `coffee_d0` | 0.9700 | 0.9400 | 0.9600 | 0.8900 | 0.9700 | 0.0000 | 0.8000 | 0.8100 |
| `threading_d0` | 0.9000 | 0.7900 | 0.7900 | 0.7400 | 0.9000 | 0.0000 | 0.8100 | 0.6800 |
| `pick_place_d0` | 0.1900 | 0.2700 | 0.3700 | 0.0100 | 0.1900 | 0.0000 | 0.2300 | 0.0000 |
| `coffee_preparation_d0` | 0.8300 | 0.7900 | 0.7500 | 0.7500 | 0.8300 | 0.0000 | 0.5500 | 0.2700 |
| `hammer_cleanup_d0` | 1.0000 | 0.9400 | 1.0000 | 0.9900 | 1.0000 | 0.0000 | 0.1500 | 0.9900 |
| `square_d0` | 0.9100 | 0.8500 | 0.8900 | 0.7900 | 0.9100 | 0.0000 | 0.2700 | 0.5400 |
| `stack_d0` | 0.9300 | 0.9500 | 0.9800 | 0.9600 | 0.9300 | 0.0000 | 0.8600 | 0.6000 |
| `stack_three_d0` | 0.5000 | 0.8100 | 0.7100 | 0.5400 | 0.5000 | 0.0000 | 0.6700 | 0.2500 |
| `nut_assembly_d0` | 0.5500 | 0.5100 | 0.5500 | 0.2400 | 0.5500 | 0.0000 | 0.0700 | 0.0100 |
| `three_piece_assembly_d0` | 0.5400 | 0.6700 | 0.5800 | 0.2600 | 0.5400 | 0.0000 | 0.4900 | 0.0800 |
| **Macro success** | **0.7320** | **0.7520** | **0.7580** | **0.6170** | **0.7320** | **0.0000** | **0.4900** | **0.4230** |
| **Mean inference ms** | **21.70** | **82.98** | **73.74** | **18.99** | **21.70** | **22.91** | **23.68** | **21.90** |

## A/B/C/D 正式评估完成度（seed 42）

| 实验 | 十任务目标 | 已完成 | 未完成 | 状态 |
|---|---:|---:|---:|---|
| A | 40次真实评估 | 40 | 0 | 完成 |
| B | 30次真实评估 + 10条复用 | 40 | 0 | 完成 |
| C | 40次真实评估 | 4 | 36 | 可直接评估 |
| D | 30次真实评估 + 10条复用 | 4 | 36 | 其余九任务无D checkpoint，待重定设计 |
| **合计** | **160条逻辑结果** | **88** | **72** | **当前仅training seed 42** |
