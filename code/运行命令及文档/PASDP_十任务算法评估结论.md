# PASDP 十任务算法评估结论

更新时间：2026-09-14

## 1. 结果口径

| 参数 | 数值 |
|---|---|
| 任务数 | 10个Robomimic任务 |
| Training seed | 42 |
| `n_demo` | 200 |
| 每项正式评估 | 100 episodes |
| 正式测试种子 | `20000–20099` |
| `n_train / n_test / n_envs` | `0 / 100 / 25` |
| Checkpoint规则 | 训练期monitor score最高，同分取epoch较晚者 |

本文中的算法效果以100-episode独立评估为准，不与训练期30-episode rollout混用。

## 2. 十任务整体结果

| 算法 | Macro success rate | Mean inference time | 整体判断 |
|---|---:|---:|---|
| A-SDP | **0.7580** | 73.74 ms | A组平均成功率最高，但推理较慢 |
| A-DP | 0.7520 | 82.98 ms | 成功率接近最高，平均推理最慢 |
| A-PASDP | 0.7320 | 21.70 ms | 成功率与DP/SDP接近，推理约快3.4–3.8倍 |
| A-Flow | 0.6170 | **18.99 ms** | 推理最快，但成功率明显降低 |
| B-PASDP | **0.7320** | 21.70 ms | 低NFE组最好；逐任务复用A-PASDP结果 |
| B-SDP | 0.4900 | 23.68 ms | NFE=1仍有一定闭环能力，但退化明显 |
| B-Flow | 0.4230 | 21.90 ms | 简单任务可保持较高成功率，复杂任务退化严重 |
| B-DP | 0.0000 | 22.91 ms | NFE=1在十任务上均为0/100 |

B-PASDP不是新训练的模型，而是复用A-PASDP checkpoint和正式评估结果。PASDP的NFE口径为稳态每步1次网络评估，冷启动需41次；不应将冷启动也表述为总NFE=1。

## 3. 逐任务成功率

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
| **Macro mean** | **0.7320** | **0.7520** | **0.7580** | **0.6170** | **0.7320** | **0.0000** | **0.4900** | **0.4230** |

## 4. 各算法结论

### 4.1 PASDP

PASDP在 `coffee_d0`、`threading_d0`、`coffee_preparation_d0`和`square_d0`上取得A组最高成功率，在`hammer_cleanup_d0`和`nut_assembly_d0`上并列最高。其十任务平均成功率比A-DP低0.020，比A-SDP低0.026，但平均推理速度分别约快3.8倍和3.4倍。因此当前证据支持PASDP具有较好的成功率—推理速度折中，不支持宣称其单种子平均成功率绝对最高。

### 4.2 DP

A-DP取得0.7520的平均成功率，在`stack_three_d0`和`three_piece_assembly_d0`上为A组最好。但它需要10步DDIM推理，平均推理时间为82.98 ms。B-DP将推理压缩至NFE=1后，十任务全部为0/100，说明普通epsilon-prediction DP不能在当前配置下直接压缩为单步去噪。

### 4.3 SDP

A-SDP的十任务平均成功率为0.7580，是A组最高；在`pick_place_d0`和`stack_d0`上取得最高成功率。B-SDP在NFE=1时仍有0.4900，明显好于B-DP，但较A-SDP下降0.268。SDP对低NFE比DP更稳定，但正常多步配置的推理成本仍较高。

### 4.4 Flow

A-Flow的平均推理时间为18.99 ms，是A组最快，但平均成功率只有0.6170。B-Flow平均成功率降至0.4230，任务差异很大：在`hammer_cleanup_d0`上为0.9900，但在`pick_place_d0`上为0，在`nut_assembly_d0`上仅为0.0100。Flow提供最低推理延迟，但当前数据下的跨任务稳定性弱于PASDP。

## 5. C组和D组结论

### 5.1 C组DP inference budget

`coffee_d0` 上复用同一A-DP checkpoint的结果为：

| NFE | Success rate | Inference time |
|---:|---:|---:|
| 1 | 0.0000 | 20.61 ms |
| 20 | 0.9600 | 144.51 ms |
| 40 | 0.9700 | 275.93 ms |
| 100 | 0.9400 | 667.20 ms |

NFE=1完全失效；NFE从20增加至40或100没有稳定成功率收益，但推理延迟持续增加。该结论目前仅来自`coffee_d0`，其余九任务还需完成36次C组正式评估。

### 5.2 D组PASDP-matched配置

`coffee_d0` 的D-DP、D-SDP和D-Flow正式成功率均为0/100。D组同时修改观察历史、预测horizon、动作步数、训练/推理时间步及prediction target，因此只能得出“旧的复合PASDP-matched配置在coffee任务上失败”，不能将失败归因于某一单独参数。其余九任务尚无D组checkpoint，应先重新确定D组设计，不盲目复制旧配置。

## 6. 任务难度现象

| 任务类别 | 任务 | 现象 |
|---|---|---|
| 较容易 | `hammer_cleanup_d0`、`stack_d0`、`coffee_d0` | A组主要方法成功率接近0.9或更高 |
| 中等 | `threading_d0`、`square_d0`、`coffee_preparation_d0` | A组大多数方法保持中高成功率 |
| 较困难 | `pick_place_d0`、`nut_assembly_d0`、`three_piece_assembly_d0` | 最好A组成功率仅为0.37、0.55和0.67 |

`stack_three_d0`的方法敏感性较强：A-DP为0.81，A-PASDP为0.50，不宜只根据单一平均值掩盖逐任务差异。

## 7. 当前可支持的论文结论

> 在training seed 42和十个Robomimic任务上，PASDP以约21.70 ms的平均推理时间取得0.732的平均成功率。与多步DP和SDP相比，其成功率分别低0.020和0.026，但推理约快3.8倍和3.4倍。在低NFE对比中，PASDP明显优于单步DP、SDP和Flow基线，表现出更好的闭环成功率与推理效率折中。

上述表述必须保留“单个training seed 42”限定。当前不能报告三种子均值±标准差，也不能将D组`coffee_d0`的零成功率外推到其他任务。

## 8. 结果来源

| 内容 | 路径 |
|---|---|
| `coffee_d0` 100-episode汇总 | `data/experiments/eval/summary/coffee_d0/seed42/n_test100_seed20000/eval_summary.tsv` |
| 其余九任务A/B汇总 | `data/experiments/eval/summary/remaining9/seed42/n_test100_seed20000/eval_summary.tsv` |
| 实验综合记录 | `实验记录.md` |
| 实验结果表 | `PASDP_仿真实验_结果一览表.md` |
