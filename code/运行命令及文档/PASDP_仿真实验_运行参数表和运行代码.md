# PASDP 仿真实验：运行参数表和运行代码

> 更新时间：2026-08-24
> 运行顺序：**A → B → D → C**
> 本页只保留实际要运行的配置与命令。示例任务为 `coffee_d0`；正式运行时替换 `PASDP_TASK`，并分别运行 3 个 training seeds。
> 10 个唯一训练配置共用同一训练、checkpoint 和 rollout 协议；唯一刻意保留的数据输入差异是 PASDP 使用 RTV8 loader 与 `shuffle=false`，这是自校正训练的必要条件。

## 1. 精简参数表

| 顺序 | ID | Hydra experiment | `T_o` | `H` | `T_a` | `T_train` | `T_inference/config` | 实际 NFE | Pred | 训练关系 |
|---:|---|---|---:|---:|---:|---|---|---:|---|---|
| 1 | A-PASDP | `pasdp_a_native_robomimic` | 1 | 40 | 1 | 40 | position grid=40 | steady=1；cold=41 | sample | 新训练；B/D-PASDP 复用 |
| 2 | A-DP | `dp_a_paper_robomimic` | 2 | 16 | 8 | 100 | DDIM=10 | 10 | epsilon | 新训练；C 四档复用同一 checkpoint |
| 3 | A-SDP | `sdp_a_paper_robomimic` | 2 | 20 | 8 | 100 | level grid=16 | 8 | epsilon | 新训练 |
| 4 | A-Flow-RGB | `flow_a_rgb_adapted_robomimic` | 2 | 8 | 4 | continuous | Euler=1 | 1 | velocity | 新训练；RGB adaptation |
| 5 | B-PASDP | 复用 `pasdp_a_native_robomimic` | 1 | 40 | 1 | 40 | position grid=40 | steady=1；cold=41 | sample | **不重复训练** |
| 6 | B-DP | `dp_b_nfe1_robomimic` | 2 | 16 | 1 | 100 | DDIM=1 | 1 | epsilon | 新训练 |
| 7 | B-SDP | `sdp_b_nfe1_robomimic` | 2 | 20 | 1 | 100 | level grid=19 | 1 | epsilon | 新训练 |
| 8 | B-Flow-RGB | `flow_b_nfe1_rgb_adapted_robomimic` | 2 | 4 | 1 | continuous | Euler=1 | 1 | velocity | 新训练 |
| 9 | D-PASDP | 复用 `pasdp_a_native_robomimic` | 1 | 40 | 1 | 40 | position grid=40 | steady=1；cold=41 | sample | **不重复训练** |
| 10 | D-DP-sample | `dp_d_pasdp_matched_robomimic` | 1 | 40 | 1 | 40 | DDIM=40 | 40 | sample | 新训练；adapted control |
| 11 | D-SDP-sample | `sdp_d_pasdp_matched_robomimic` | 1 | 40 | 1 | 40 | level grid=40 | 1 | sample | 新训练；adapted control |
| 12 | D-Flow-RGB | `flow_d_pasdp_matched_rgb_adapted_robomimic` | 1 | 40 | 1 | continuous | Euler=1 | 1 | velocity | 新训练；adapted control |
| 13–16 | C-DP-NFE{1,20,40,100} | 无训练 config | 2 | 16 | 8 | 100 | DDIM={1,20,40,100} | {1,20,40,100} | epsilon | **同一个 A-DP checkpoint，仅 evaluation override** |

唯一训练配置总数为 **10**：A 四个、B 三个、D 三个。C 不训练新模型。

说明：FlowPolicy 上游名义 `H4/O2/A4` 在其 `start=T_o-1` 切片下只能返回3步。A-Flow-RGB 为保证统一 runner 实际执行4步，使用最小 UNet-compatible `H8/O2/A4`。所有方法统一使用 `base=[256,512,1024]` 和 `84×84 -> crop76`；PASDP 保留自校正所需的 RTV8 数据输入。

## 2. 通用环境

```bash
cd /home/hy/dp_pro6000

export PASDP_PY=/opt/miniconda3/envs/sim_torch27_v8/bin/python
export PYTHONPATH=/home/hy/dp_pro6000/third_party/robosuite:${PYTHONPATH:-}
export PASDP_TASK=coffee_d0
export PASDP_SEED=42
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
# 本机 CUDA 与 EGL 枚举顺序相反：CUDA 0=2080Ti，而 EGL 1=2080Ti。
export MUJOCO_EGL_DEVICE_ID=1
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=/home/hy/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda-12.8/lib64
export HYDRA_FULL_ERROR=1
```

正式实验固定 `n_demo=200`。完成 seed 42 后，把 `PASDP_SEED` 改成另外两个预先确定的 training seeds，逐项重复；不同方法必须使用相同的 seed 集合。

正式训练公共协议：`n_demo=200`，251 epochs（0–250），关闭早停；每 5 epochs 触发 rollout 和 checkpoint，均为 51 个触发点。`latest.ckpt` 每次覆写，top-k 最多保留 5 个，因此全新运行目录最终最多有 6 个 checkpoint 文件。每次训练期周期 rollout 统一为 10 个 train initial states + 30 个 checkpoint-validation seeds，视频子集分别为 2/2，`n_envs=20`，validation seeds 为 `100000–100029`。

本机三路训练直接使用配置中的 `n_envs=20`，不在命令行覆盖环境数量。物理 GPU1（4090）必须使用 `CUDA_VISIBLE_DEVICES=1 MUJOCO_EGL_DEVICE_ID=0`。

PASDP 的 251 个 epoch 数与其他方法一致，但 RTV8 每 epoch 的 batch 数由当前 task 的 episode 映射自动决定。映射生成时会把逐 epoch batch 数和总数写入 `mapping_manifest.npz`；训练时逐 epoch 切换对应映射，cosine scheduler 按实际 optimizer update 总数走完周期。不使用 coffee 专用常数，也不把 PASDP 截断为每 epoch 560 batches。旧格式映射会在首个 PASDP 进程中一次性重建，之后各训练 seed 直接复用。

训练后的标准独立 evaluation 只计算 test protocol：`n_train=0,n_train_vis=0,n_test=100,n_test_vis=2,n_envs=25,test_start_seed=20000`，即测试种子 `20000–20099`。它与训练 checkpoint 监控使用的 `100000–100029` 不重叠。

## 3. A：论文/原生配置，先运行

```bash
$PASDP_PY -u train_sim.py --config-name=train \
  experiment=pasdp_a_native_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=dp_a_paper_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=sdp_a_paper_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=flow_a_rgb_adapted_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED
```

## 4. B：Matched NFE=1，其次运行

B-PASDP 直接复用 A-PASDP，不再启动训练。只运行下面三个新 checkpoint：

```bash
$PASDP_PY -u train_sim.py --config-name=train \
  experiment=dp_b_nfe1_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=sdp_b_nfe1_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=flow_b_nfe1_rgb_adapted_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED
```

## 5. D：PASDP policy 参数对齐，然后运行

D-PASDP 直接复用 A-PASDP。D-DP 与 D-SDP 是 `sample` adapted controls，必须分别新训练：

```bash
$PASDP_PY -u train_sim.py --config-name=train \
  experiment=dp_d_pasdp_matched_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=sdp_d_pasdp_matched_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED

$PASDP_PY -u train_sim.py --config-name=train \
  experiment=flow_d_pasdp_matched_rgb_adapted_robomimic \
  task_name=$PASDP_TASK n_demo=200 training.seed=$PASDP_SEED
```

## 6. C：最后运行 DP inference budget

先把 `PASDP_A_DP_CKPT` 指向当前 task/seed 的 A-DP `latest.ckpt` 或选定的 top-k checkpoint。四档必须使用**完全相同的 checkpoint**，并保持 `T_a=8`：

```bash
export PASDP_A_DP_CKPT=/absolute/path/to/A-DP/checkpoints/latest.ckpt

for PASDP_NFE in 1 20 40 100; do
  $PASDP_PY -u eval_sim_rollout.py \
    --checkpoint "$PASDP_A_DP_CKPT" \
    --output_dir "data/experiments/eval/C/dp/${PASDP_TASK}/seed${PASDP_SEED}/n_test100_seed20000/nfe${PASDP_NFE}" \
    --device cuda:0 \
    --n-envs 25 --n-train 0 --n-train-vis 0 \
    --n-test 100 --n-test-vis 2 --test-start-seed 20000 \
    --num-inference-steps "$PASDP_NFE"
done
```

Evaluator 会拒绝 `T_inference > T_train`，并把最终值写入 `eval_log.json` 的 `evaluation/num_inference_steps`。A-DP 的 `T_train=100`，所以 1/20/40/100 四档均合法。

## 7. A/B/D checkpoint 的标准 evaluation 模板

```bash
export PASDP_CKPT=/absolute/path/to/checkpoints/latest.ckpt
export PASDP_EVAL_OUT=data/experiments/eval/<stage>/<algorithm>/${PASDP_TASK}/seed${PASDP_SEED}/n_test100_seed20000/default

$PASDP_PY -u eval_sim_rollout.py \
  --checkpoint "$PASDP_CKPT" \
  --output_dir "$PASDP_EVAL_OUT" \
  --device cuda:0 \
  --n-envs 25 --n-train 0 --n-train-vis 0 \
  --n-test 100 --n-test-vis 2 --test-start-seed 20000
```

## 8. 已有验证记录

历史 smoke 协议：`coffee_d0`、`n_demo=10`、batch=1、1 个 train step、1 个 validation step、1 env × 1 step rollout、禁用 checkpoint 与在线 logging。C 另用小容量 A-DP smoke checkpoint 验证 evaluator override。这些只是运行链验证，不是论文结果。

| 项目 | 最终结果 |
|---|---|
| A-PASDP / A-DP / A-SDP / A-Flow-RGB | 4/4 通过 |
| B-DP / B-SDP / B-Flow-RGB | 3/3 通过；B-PASDP 复用已通过的 A-PASDP |
| D-DP / D-SDP / D-Flow-RGB | 3/3 通过；D-PASDP 复用已通过的 A-PASDP |
| C：同 checkpoint NFE=1/20/40/100 | 4/4 evaluation smoke 通过，JSON 分别记录 1/20/40/100 |

首次 smoke 发现两项问题：SDP 对 `84×84` 输入使用 crop84 会触发 random-crop 断言；Flow 的名义 H4/O2/A4 只能返回3步。最终协议统一改为 crop76，并使用 H8/O2/A4 的 A-Flow-RGB。进程退出时 robosuite 仍可能打印 `EGL_NOT_INITIALIZED` 析构警告，但不影响训练、rollout 或进程退出码。

本次将 UNet/crop/训练协议统一后，10/10 配置已通过 Hydra compose，两组仓库 unittest 分别为 11/11 和 6/6。这些变更不改变 policy 调用链，不要为此强制重跑全部 smoke；服务器上正式运行前做 Hydra compose 与数据/GPU 预检即可。

## 9. 配置文件

- A：[PASDP](diffusion_policy/config/experiment/pasdp_a_native_robomimic.yaml)、[DP](diffusion_policy/config/experiment/dp_a_paper_robomimic.yaml)、[SDP](diffusion_policy/config/experiment/sdp_a_paper_robomimic.yaml)、[Flow-RGB](diffusion_policy/config/experiment/flow_a_rgb_adapted_robomimic.yaml)
- B：[DP](diffusion_policy/config/experiment/dp_b_nfe1_robomimic.yaml)、[SDP](diffusion_policy/config/experiment/sdp_b_nfe1_robomimic.yaml)、[Flow-RGB](diffusion_policy/config/experiment/flow_b_nfe1_rgb_adapted_robomimic.yaml)
- D：[DP-sample](diffusion_policy/config/experiment/dp_d_pasdp_matched_robomimic.yaml)、[SDP-sample](diffusion_policy/config/experiment/sdp_d_pasdp_matched_robomimic.yaml)、[Flow-RGB](diffusion_policy/config/experiment/flow_d_pasdp_matched_rgb_adapted_robomimic.yaml)
- C evaluator：[eval_sim_rollout.py](eval_sim_rollout.py)
