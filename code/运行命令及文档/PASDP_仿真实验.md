# PASDP 标准化仿真实验总表

> 整理日期：2026-08-24（Asia/Shanghai）
> 当前阶段：已实现 PASDP RTV8 任务自适应 batch 计划与精确 scheduler；未启动正式训练或长时间 evaluation。
> 执行顺序：**A → B → D → C**。
> 命名约定：论文和对外材料使用 `PASDP`；当前 Python 模块、类名和兼容 checkpoint 目标继续使用 `padp/PADP`。
> 事实来源：当前工作树中的 10 个实验 YAML、policy/workspace/runner/evaluator 实现，以及本项目另外三份现行文档。若文档与代码冲突，以 Hydra resolved config 和实际调用链为准。

基线实验配置与配套文档来自提交 `3e656f7（chore: 统一PASDP仿真实验协议）`。当前工作树在此基础上新增 RTV8 manifest、逐 epoch `set_epoch()` 和按真实 optimizer update 数计划 scheduler，尚未提交。正式服务器运行仍必须记录实际 Git commit、工作树是否干净、resolved config 和配置文件 SHA256。

验证结果：10/10 个唯一训练配置均成功完成 Hydra compose；RTV8/workspace 单元测试与相关回归测试通过；使用 `coffee_d0` 真实 episode 结构的非训练集成验证生成 251 个 epoch，batch 数范围为 1003–1048，总 batch 数为 257679。这些数字只是当前 `coffee_d0` 数据的调用链验证，不是写死参数，也不是正式论文结果。

---

## 第一部分：论文实验计划表

### 1.1 字段定义

| 字段 | 当前代码含义 |
|---|---|
| `T_o` | `n_obs_steps`，policy 使用的 observation history |
| `H` | `horizon`，预测动作序列长度 |
| `T_a` | `n_action_steps`，每次 policy call 后交给环境执行的动作数 |
| `T_train` | diffusion scheduler 的 `num_train_timesteps`；FlowPolicy 为连续 flow time，不适用 |
| `T_inference/config` | DP 的 DDIM steps、SDP 的 diffusion-level grid、PASDP 的 position depth 配置值，或 FlowPolicy 的 Euler steps |
| NFE | 一次 `predict_action()` 中主生成网络的 forward 次数；当前代码没有统一运行时 counter，表中数值为调用链静态推导 |
| Pred | 模型训练目标：`epsilon`、`sample` 或 consistency-flow velocity |

### 1.2 总体实验计划

| 顺序 | 实验组 | 研究问题 | 条件数 | 每 task/seed 新训练数 | checkpoint 复用 | evaluation |
|---:|---|---|---:|---:|---|---|
| 1 | A：Paper/native policy configuration | 各方法按论文/原生 policy 参数，在统一 RGB backbone、crop 和训练协议下比较 | 4 | 4 | A-PASDP 供 B/D-PASDP 复用；A-DP 供 C 复用 | 四个条件各做标准 test-only evaluation |
| 2 | B：Matched NFE=1、`T_a=1` | 在每控制周期目标 NFE=1 且执行一步时比较 | 4 | 3 | B-PASDP 复用 A-PASDP | PASDP、DP、SDP、Flow-RGB 四行；PASDP 结果可复用 |
| 3 | D：PASDP policy 参数对齐 | 对齐 `T_o=1,H=40,T_a=1`；diffusion 方法同时对齐 `T_train=40,Pred=sample` | 4 | 3 | D-PASDP 复用 A-PASDP | 机制对照；DP/SDP 必须标为 adapted control |
| 4 | C：DP inference budget | 同一 A-DP checkpoint 仅在 evaluation 改 DDIM steps | 4 | 0 | 每个 task/seed 的四档复用完全相同的 A-DP checkpoint | NFE=`1/20/40/100`，保持 `T_a=8` |

每个 task、每个 training seed 只有 **10 个唯一训练配置**：A 四个、B 三个、D 三个；C 不训练。正式协议要求三个相同 training seeds，因此训练总数为：

```text
10 × 正式任务数 × 3
```

任务列表和另外两个 training seeds 仍属于服务器预检后的用户确认项，不能由执行 AI 自行决定。若最终采用当前代码中已有的 10 个 D0 任务，则训练量为 300 个 run；在用户冻结任务列表前，这只是按公式计算的规模，不代表已经授权启动。

### 1.3 唯一训练配置清单

| Train ID | Hydra experiment | 论文行/复用关系 |
|---|---|---|
| T1 | `pasdp_a_native_robomimic` | A-PASDP；B-PASDP、D-PASDP 复用 |
| T2 | `dp_a_paper_robomimic` | A-DP；C 四档复用 |
| T3 | `sdp_a_paper_robomimic` | A-SDP |
| T4 | `flow_a_rgb_adapted_robomimic` | A-FlowPolicy (RGB adaptation) |
| T5 | `dp_b_nfe1_robomimic` | B-DP |
| T6 | `sdp_b_nfe1_robomimic` | B-SDP |
| T7 | `flow_b_nfe1_rgb_adapted_robomimic` | B-FlowPolicy (RGB adaptation) |
| T8 | `dp_d_pasdp_matched_robomimic` | D-DP-sample (adapted) |
| T9 | `sdp_d_pasdp_matched_robomimic` | D-SDP-sample (adapted) |
| T10 | `flow_d_pasdp_matched_rgb_adapted_robomimic` | D-FlowPolicy (RGB adaptation) |

### 1.4 当前代码解析出的固定公共协议

| 类别 | 字段 | resolved 值 | 代码来源/说明 |
|---|---|---:|---|
| 数据 | `n_demo` | 200 | 10 个 experiment YAML |
| 数据 | dataset seed / validation ratio | `42 / 0.02` | 两种 Robomimic dataset config；dataset seed 不等于 training seed |
| 输入 | 原始 RGB | 双相机 `3×84×84` | `task/robomimic/mimicgen_abs.yaml` |
| 输入 | crop | `[76,76]` | 10 个 resolved policy；训练 random crop、evaluation fixed crop |
| 模型 | UNet `down_dims` | `[256,512,1024]` | 10 个 resolved policy 均为 `capacity/base` |
| 模型 | diffusion embed / kernel / groups | `128 / 5 / 8` | `capacity/base.yaml` |
| 优化 | optimizer | AdamW，lr=`1e-4`，betas=`[0.95,0.999]`，eps=`1e-8`，weight decay=`1e-6` | `train_common.yaml` |
| 优化 | batch / workers | `64 / 8` | recipe；PASDP 和其他方法相同 |
| 优化 | LR scheduler / warm-up | cosine / 500 steps | `train_common.yaml` |
| 优化 | EMA | 启用 | `training.use_ema=true` |
| 训练 | epochs | `251`（epoch 0–250） | `50000 / n_demo + 1`，`n_demo=200` |
| 训练 | early stopping | 关闭 | 两个 recipe 均覆盖为 false |
| 训练 | rollout / checkpoint interval | 每 5 epochs | `1000 / n_demo` |
| 训练 | rollout/checkpoint 触发数 | 51 | epoch 0、5、…、250 |
| checkpoint | last / top-k | `latest.ckpt` + top-5 | top-k 监控 `test_mean_score`，mode=`max` |
| 训练期 rollout | train/test/envs | `10 / 30 / 20` | `env_runner/robomimic/standard.yaml`；test seeds `100000–100029` |
| 训练期录像 | train/test | `2 / 2` | 是上述 episode 的子集，不是新增 episode |
| 独立 evaluation | train/test/envs | `0 / 100 / 25` | 正式命令显式覆盖；test seeds `20000–20099` |
| 独立 evaluation 录像 | train/test | `0 / 2` | 正式命令行协议 |
| 输出 | 训练根目录 | `data/experiments/train/...` | 服务器提示词通过 `output_root/output_group` 覆盖；代码默认仍是 `data/outputs` |
| 输出 | 评估根目录 | `data/experiments/eval/...` | 服务器提示词显式传入 `--output_dir` |

PASDP 的唯一刻意保留的数据管线差异是：使用 RTV8 dataset、`shuffle=false` 和 `uint8_to_gpu=true`；其他三个方法使用标准 Robomimic dataset、`shuffle=true`。两者读取同一 task/demo HDF5，batch、worker、dataset seed 和 validation ratio 相同。该差异是当前 PASDP 自校正训练映射的实现要求，不能用命令行强行统一。

---

## 第二部分：四算法配置表

### 2.1 四算法实现与数据入口

| 论文名称 | 当前 policy target | Workspace | Dataset target | RGB encoder | Loader |
|---|---|---|---|---|---|
| PASDP | `diffusion_policy.policy.padp.PADPPolicy` | `TrainPADPWorkspace` | `replay_image_dataset_padp.RobomimicReplayImageDataset` | `robomimic` | RTV8，`shuffle=false` |
| DP | `diffusion_policy.policy.dp.DPPolicy` | `TrainPolicyWorkspace` | `replay_image_dataset.RobomimicReplayImageDataset` | `robomimic_raw` | 标准，`shuffle=true` |
| SDP | `diffusion_policy.policy.streaming_diffusion_policy.StreamingDiffusionPolicy` | `TrainPolicyWorkspace` | 同 DP | `robomimic_raw` | 标准，`shuffle=true` |
| FlowPolicy (RGB adaptation) | `diffusion_policy.policy.flow_policy.FlowPolicy` | `TrainPolicyWorkspace` | 同 DP | `robomimic_raw` | 标准，`shuffle=true` |

FlowPolicy 当前是将 consistency-flow 机制适配到统一 Robomimic RGB 输入，不是上游 point-cloud native pipeline；论文行名必须保留 `RGB adaptation` 限定。

### 2.2 算法固定机制参数

| 方法 | Scheduler / 生成机制 | 当前固定参数 | NFE 口径 |
|---|---|---|---|
| PASDP | position-wise direct prediction + sliding buffer；配置对象使用 DDIMScheduler，但推理不是 DDIM loop | beta=`0.0001→0.02`，schedule=`squaredcos_cap_v2`，`clip_sample=true`，positionwise，chunk=1，window weights=exponential，gamma=0.2 | 初始化循环 `H+T_o-1=40` 次，随后本次预测再 forward 1 次；cold=41、steady=1 |
| DP | 标准 DDIM loop | 同一 beta/schedule/clip，`set_alpha_to_one=true`，`steps_offset=0` | scheduler timesteps 长度等于主网络 forward 数 |
| SDP | `DDIMTEDiScheduler` + streaming action buffer | temporal weights=`0.2/0/0/0.8`，`buffer_init=zero`，其余 beta/schedule/clip 同表 | `num_inference_steps` 是 level grid，不等于单次 forward 数；while-loop 每轮 forward 一次 |
| FlowPolicy | consistency flow + uniform Euler | eps=`1e-2`，segments=`2`，boundary=`1`，delta=`1e-2`，alpha=`1e-5`，Euler steps=`1` | `consistency_flow.num_inference_step=1`，因此 NFE=1 |

### 2.3 A/B/D/C 精确 policy 参数

下表的 NFE 是当前实现的静态调用链推导，不是运行时 counter 输出。

| ID | Hydra experiment / checkpoint | `T_o` | `H` | `T_a` | `T_train` | `T_inference/config` | 静态 NFE | Pred | 新训练 |
|---|---|---:|---:|---:|---|---|---|---|---|
| A-PASDP | `pasdp_a_native_robomimic` | 1 | 40 | 1 | 40 | position grid=40；字段不控制 loop | cold=41；steady=1 | sample | 是 |
| A-DP | `dp_a_paper_robomimic` | 2 | 16 | 8 | 100 | DDIM=10 | 10 | epsilon | 是 |
| A-SDP | `sdp_a_paper_robomimic` | 2 | 20 | 8 | 100 | level grid=16 | 8 | epsilon | 是 |
| A-Flow-RGB | `flow_a_rgb_adapted_robomimic` | 2 | 8 | 4 | N/A，连续 flow | Euler=1 | 1 | velocity / consistency flow | 是 |
| B-PASDP | 复用 A-PASDP | 1 | 40 | 1 | 40 | position grid=40 | cold=41；steady=1 | sample | 否 |
| B-DP | `dp_b_nfe1_robomimic` | 2 | 16 | 1 | 100 | DDIM=1 | 1 | epsilon | 是 |
| B-SDP | `sdp_b_nfe1_robomimic` | 2 | 20 | 1 | 100 | level grid=19 | 1 | epsilon | 是 |
| B-Flow-RGB | `flow_b_nfe1_rgb_adapted_robomimic` | 2 | 4 | 1 | N/A，连续 flow | Euler=1 | 1 | velocity / consistency flow | 是 |
| D-PASDP | 复用 A-PASDP | 1 | 40 | 1 | 40 | position grid=40 | cold=41；steady=1 | sample | 否 |
| D-DP-sample | `dp_d_pasdp_matched_robomimic` | 1 | 40 | 1 | 40 | DDIM=40 | 40 | sample | 是 |
| D-SDP-sample | `sdp_d_pasdp_matched_robomimic` | 1 | 40 | 1 | 40 | level grid=40 | 1 | sample | 是 |
| D-Flow-RGB | `flow_d_pasdp_matched_rgb_adapted_robomimic` | 1 | 40 | 1 | N/A，连续 flow | Euler=1 | 1 | velocity / consistency flow | 是 |
| C-DP-NFE1 | A-DP checkpoint | 2 | 16 | 8 | 100 | eval DDIM=1 | 1 | epsilon | 否 |
| C-DP-NFE20 | 同一个 A-DP checkpoint | 2 | 16 | 8 | 100 | eval DDIM=20 | 20 | epsilon | 否 |
| C-DP-NFE40 | 同一个 A-DP checkpoint | 2 | 16 | 8 | 100 | eval DDIM=40 | 40 | epsilon | 否 |
| C-DP-NFE100 | 同一个 A-DP checkpoint | 2 | 16 | 8 | 100 | eval DDIM=100 | 100 | epsilon | 否 |

关键边界：

- A-DP 使用论文值 `T_inference=10`；不是上游仓库后续 YAML 的 8。
- A-SDP 的 grid=16 在 `H20/O2/A8` 两个 executable chunks 下，每次需要 8 次主网络 forward。
- B-SDP 不能把 grid 直接写成 1。`H20/O2/A1` 有 19 个 executable chunks，grid=19 才使当前 chunk 每 call 下降一级，即静态 NFE=1。
- A-Flow 上游名义 `H4/O2/A4` 从 `start=T_o-1` 只能切出 3 步；当前代码采用最小 UNet-compatible `H8/O2/A4`，保证 runner 真正执行 4 步。
- D 不是 Matched NFE。它对齐 PASDP policy 参数；D-DP 的 NFE 是 40，D-DP/D-SDP 的 `sample` 目标都属于 adapted control。
- C 与 B-DP 不是同一对照：C 保持 A-DP 的 `T_a=8` 并复用 checkpoint；B-DP 使用 `T_a=1` 的独立训练配置。

---

## 第三部分：实验支持状态表

状态定义：`READY` 表示代码入口具备；`PARTIAL` 表示可运行但论文级记录不完整；`MISSING` 表示当前代码没有；`PENDING` 表示必须在服务器或由用户冻结。

| # | 能力 | 当前状态 | 当前代码证据 / 执行约束 |
|---:|---|---|---|
| 1 | A/B/D 10 个训练配置 | READY | 本轮 10/10 Hydra compose 成功；resolved 参数见第二部分 |
| 2 | 统一训练入口 | READY | `train_sim.py --config-name=train experiment=<name>` |
| 3 | task / `n_demo` / training seed override | READY | `task_name=... n_demo=200 training.seed=...` |
| 4 | checkpoint 保存 | READY | `latest.ckpt` + top-5，监控 `test_mean_score` |
| 5 | A/B/D 单 checkpoint evaluation | READY | `eval_sim_rollout.py --checkpoint --output_dir` |
| 6 | C 同 checkpoint NFE override | READY | `--num-inference-steps` 已实现，且拒绝 `T_inference>T_train`；JSON 写入最终 steps |
| 7 | 100 test episode success rate | READY | runner 输出 `test/success_rate`；正式命令传 `n_train=0,n_test=100,test_start_seed=20000` |
| 8 | completion step | READY | runner 输出 `test/avg_completion_step` 与 `test/median_completion_step` |
| 9 | 训练/评估/算法分目录 | READY（命令约定） | `output_root/output_group` 与 evaluator `--output_dir` 支持；目录纪律由服务器提示词执行，不是代码强制 |
| 10 | 历史 smoke 调用链 | READY（非论文结果） | 现行记录为 A 4/4、B 3/3、D 3/3、C 四档通过；参数统一后另有 10/10 compose。不得把 smoke 指标写入论文 |
| 11 | `success_count` | MISSING | runner 只写 rate，没有显式 count |
| 12 | 100 个 per-episode success records | PARTIAL | JSON 按种子记录 max reward，可由 `20000–20099` 推导逐 episode 成败，但没有独立 success bool 数组 |
| 13 | 统一 observed-NFE counter | MISSING | 当前 NFE 只能从 policy loop 静态推导；`eval_log.json` 不含 observed forwards |
| 14 | PASDP cold/steady 分离记录 | PARTIAL | 调用链可确定 cold=41、steady=1，但没有自动 counter 或分项 JSON |
| 15 | 论文级 latency | MISSING | runner 的 `perf_counter` 计时为 batch rollout，无统一 warm-up、CUDA 前后同步；`policy/avg_inference_time_ms` 不可用于论文 |
| 16 | 单次完整 manifest JSON | PARTIAL | evaluator 只写 runner metrics、inference steps 和 action steps；第四部分标准需服务器 AI 另写 manifest |
| 17 | 三 training seeds 自动 mean/std 聚合 | MISSING | 当前无聚合脚本；std 的 `ddof` 也尚未由用户冻结 |
| 18 | 完整批量矩阵/失败恢复 | PARTIAL | 服务器提示词定义交互流程和状态，但没有仓库内专用 matrix runner |
| 19 | 正式 checkpoint 选择规则 | PENDING | 必须统一冻结 `latest` 或同一 top-k 规则；禁止逐算法挑最好 checkpoint |
| 20 | 正式任务列表和三个 seeds | PENDING | 由用户在服务器预检后确认；不能沿用示例值替代正式决定 |
| 21 | 数据完整性 | PENDING | 当前机器只可见 `coffee_d0` 与 `nut_assembly_d0` 两个 HDF5；服务器必须逐任务核验路径、大小和可读性 |
| 22 | 可复现代码快照 | READY | 当前实验 YAML 已纳入 `3e656f7` 并同步远端；正式 run 仍须把实际 commit、dirty 状态和 resolved-config hash 写入 manifest |
| 23 | PASDP 逐 epoch RTV8 映射与精确 LR 周期 | READY（当前未提交实现） | 生成映射时写入 `mapping_manifest.npz`；workspace 每 epoch 调用 `set_epoch()`，cosine scheduler 使用全部 epoch 的真实 optimizer update 总数；正式配置不再限制 560 batches |

### 3.1 当前总体判断

核心训练和 evaluation 链已经具备 A、B、D、C 四组实验的执行能力，可以进入服务器预检和用户确认阶段；尚不能直接声称“论文结果链完整”。正式结果之前至少还要解决或明确处理：

1. 任务、三个 training seeds 和 checkpoint 选择规则；
2. 服务器数据、GPU、磁盘与服务器实际拉取的代码快照核验；
3. observed-NFE、PASDP cold/steady 和论文级 latency 的记录方式；
4. success count/per-episode 记录及三 seed 聚合；
5. 单次 manifest 和最终表的可追溯填充。

现阶段可以运行 success-rate 主实验，但不能把现有 runner latency 当作论文 Time(ms)，也不能把静态 NFE 当成已经由实验测得的 observed NFE。

---

## 第四部分：单次运行 JSON 标准

### 4.1 保存位置与角色

每个 `task × training_seed × experiment condition` 使用一个 manifest：

```text
data/experiments/manifests/<task>/seed<seed>/<stage>_<algorithm>[_nfeX].json
```

该 JSON 是服务器执行层的标准，不是当前 `eval_sim_rollout.py` 已自动完整输出的格式。AI 必须读取 resolved config、checkpoint 和 `eval_log.json` 后填写；无法从代码或产物确认的字段必须为 `null`，不得猜测。

### 4.2 标准 JSON 模板

```json
{
  "schema_version": "pasdp.single_run.v1",
  "identity": {
    "stage": null,
    "experiment_id": null,
    "method_paper_name": null,
    "hydra_experiment": null,
    "task_name": null,
    "training_seed": null,
    "dataset_seed": 42,
    "n_demo": 200,
    "status": "PENDING"
  },
  "code": {
    "repository": "/home/hy/dp_pro6000",
    "git_commit": null,
    "git_dirty": null,
    "git_diff_sha256": null,
    "resolved_config_path": null,
    "resolved_config_sha256": null
  },
  "data": {
    "dataset_path": null,
    "dataset_size_bytes": null,
    "dataset_sha256": null,
    "validation_ratio": 0.02,
    "loader_target": null,
    "shuffle": null
  },
  "policy": {
    "target": null,
    "workspace_target": null,
    "T_o": null,
    "H": null,
    "T_a": null,
    "T_train": null,
    "T_inference_config": null,
    "prediction_type": null,
    "sampler": null,
    "configured_nfe": null,
    "static_expected_nfe": null,
    "observed_nfe": null,
    "pasdp_cold_start_nfe": null,
    "pasdp_steady_state_nfe": null,
    "down_dims": [256, 512, 1024],
    "crop_shape": [76, 76],
    "rgb_model": null
  },
  "training": {
    "num_epochs": 251,
    "batch_size": 64,
    "num_workers": 8,
    "optimizer": "AdamW",
    "learning_rate": 0.0001,
    "use_ema": true,
    "early_stopping": false,
    "rollout_every": 5,
    "checkpoint_every": 5,
    "train_rollout": {
      "n_train": 14,
      "n_train_vis": 2,
      "n_test": 50,
      "n_test_vis": 2,
      "n_envs": 32
    },
    "start_time": null,
    "end_time": null,
    "elapsed_seconds": null,
    "output_dir": null,
    "console_log": null
  },
  "checkpoint": {
    "selection_rule": null,
    "path": null,
    "size_bytes": null,
    "sha256": null,
    "source_manifest": null,
    "reused": false
  },
  "evaluation": {
    "model_weights": "ema",
    "n_train": 0,
    "n_train_vis": 0,
    "n_test": 50,
    "n_test_vis": 2,
    "n_envs": 32,
    "output_dir": null,
    "eval_log_path": null,
    "num_inference_steps_recorded": null,
    "n_action_steps_recorded": null
  },
  "metrics": {
    "test_mean_score": null,
    "test_success_rate": null,
    "test_success_count": null,
    "test_episode_success": null,
    "test_avg_completion_step": null,
    "test_median_completion_step": null,
    "raw_runner_avg_inference_time_ms": null,
    "raw_runner_latency_paper_eligible": false,
    "paper_latency": {
      "protocol": null,
      "samples": null,
      "mean_ms": null,
      "median_ms": null,
      "p95_ms": null,
      "cold_start_ms": null,
      "steady_state_ms": null
    }
  },
  "hardware": {
    "hostname": null,
    "gpu_model": null,
    "gpu_uuid": null,
    "cuda_visible_devices": null,
    "cuda_version": null,
    "pytorch_version": null,
    "conda_env": null
  },
  "error": null,
  "notes": []
}
```

### 4.3 填写规则

| 字段 | 规则 |
|---|---|
| `identity` | stage/condition 来自第一部分；task 和 training seed 来自用户确认，不能使用示例代替 |
| `policy` | 必须从该 run 的 resolved config 填写；NFE 同时区分 configured、static expected 和 observed |
| `checkpoint` | C 四档的 SHA256 必须完全相同；B/D-PASDP 用 `reused=true` 和 `source_manifest` 指向 A-PASDP |
| `evaluation` | 当前 resolved config 均启用 EMA，evaluator 会选择 EMA model；steps 字段读取 `eval_log.json`，协议数量来自正式 CLI |
| `test_success_rate` | 读取 `test/success_rate`；不要根据 console 手抄 |
| `test_success_count` / per-episode | 当前代码不产生，保持 `null`，除非后续经授权增加真实记录能力 |
| runner latency | 可保存原始诊断值，但 `raw_runner_latency_paper_eligible` 必须为 false |
| paper latency | 只有完成统一 batch=1、固定输入、warm-up、CUDA synchronize 协议后才能填写 |
| `status` | 仅使用 `PENDING/RUNNING/TRAINED/EVALUATED/FAILED/REUSED` |
| 失败记录 | 不覆盖旧目录；填写 `error`、日志和失败输出目录，新 run 使用新的 manifest |

---

## 第五部分：最终论文汇总表

### 5.1 当前代码覆盖的 10 个 D0 任务候选

| 论文显示名 | task id | `max_steps` |
|---|---|---:|
| Coffee | `coffee_d0` | 400 |
| Threading | `threading_d0` | 400 |
| Pick Place | `pick_place_d0` | 1000 |
| Coffee Preparation | `coffee_preparation_d0` | 800 |
| Hammer Cleanup | `hammer_cleanup_d0` | 500 |
| Square | `square_d0` | 400 |
| Stack | `stack_d0` | 400 |
| Stack Three | `stack_three_d0` | 400 |
| Nut Assembly | `nut_assembly_d0` | 500 |
| Three-Piece Assembly | `three_piece_assembly_d0` | 500 |

这些 task id 和 max steps 均存在于当前 `train_sim.py`。最终正式任务子集仍以用户在服务器预检后的确认结果为准；未被确认的任务不得因为出现在本表就自动启动。

### 5.2 论文主汇总矩阵

每个任务单元格填写三个 training seeds 的独立 50-test-episode success rate 的 `mean ± std`。当前尚未运行正式实验，全部结果保持 `待运行`。`std` 的 `ddof` 尚未冻结，因此不得提前计算。

| Group / Method | Coffee | Thread | PickPlace | CoffeePrep | Hammer | Square | Stack | Stack3 | Nut | ThreePiece | NFE | Paper latency |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A-PASDP | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | cold 41 / steady 1（静态） | 待标准测量 |
| A-DP | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 10（静态） | 待标准测量 |
| A-SDP | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 8（静态） | 待标准测量 |
| A-Flow-RGB | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| B-PASDP（复用 A） | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | cold 41 / steady 1（静态） | 待标准测量 |
| B-DP | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| B-SDP | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| B-Flow-RGB | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| D-PASDP（复用 A） | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | cold 41 / steady 1（静态） | 待标准测量 |
| D-DP-sample | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 40（静态） | 待标准测量 |
| D-SDP-sample | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| D-Flow-RGB | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| C-DP-NFE1 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 1（静态） | 待标准测量 |
| C-DP-NFE20 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 20（静态） | 待标准测量 |
| C-DP-NFE40 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 40（静态） | 待标准测量 |
| C-DP-NFE100 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 待运行 | 100（静态） | 待标准测量 |

### 5.3 汇总规则

1. 每个表格单元先保存三个 training seeds 的三个独立 success rates，再计算 `mean ± std`；不得将 150 个 episode 合并后只报一个比例。
2. `ddof=0` 或 `ddof=1` 必须在正式聚合前由用户统一冻结，并写入所有汇总产物。
3. 所有行使用相同 checkpoint 选择规则；C 四档额外要求 checkpoint SHA256 完全相同。
4. B-PASDP、D-PASDP 与 A-PASDP 协议完全相同时复用结果，并在 manifest 中保留来源，不能复制后伪装为独立训练。
5. A-Flow、B-Flow、D-Flow 的论文名称统一写 `FlowPolicy (RGB adaptation)`。
6. D-DP/D-SDP 必须写 `sample-adapted control`，不能与 A 组论文原生 baseline 混称。
7. 在 observed-NFE counter 未实现前，NFE 列必须标注“静态推导”；不能写成实验测得。
8. 在标准 latency 协议完成前，Paper latency 保持“待标准测量”；禁止抄入 runner 的 `policy/avg_inference_time_ms`。
9. 当前代码输出的正式可用主指标是 `test/success_rate`、`test/avg_completion_step`、`test/median_completion_step`；缺失字段保持空值，不做推测性补全。

---

### 5.4 审计结论

当前代码已经具备按 **A → B → D → C** 执行 10 个唯一训练配置和 4 个 C 组 budget evaluation 的核心能力。正式实验可以进入服务器预检，但仍需由用户冻结任务、三个 training seeds、checkpoint 规则和 std 定义，并在运行层建立完整 manifest。success-rate 主实验可按现有入口执行；observed NFE、PASDP cold/steady latency、论文级 batch=1 latency、success_count/per-episode 和自动三 seed 聚合仍不是当前代码的完整能力。
