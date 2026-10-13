# PASDP 最新优化与执行交接

> 更新时间：2026-08-24（Asia/Shanghai）
>
> 用途：在新电脑或新 AI 会话中继续 PASDP 仿真实验。
>
> 当前阶段：代码优化已提交；服务器正在运行 `coffee_d0 + seed42` 的 A/B/D 训练 pilot；独立 evaluation 和 C 组当前暂缓。

> **新服务器覆盖说明（2026-08-27）**：当前机器的项目目录为
> `/home/hy/dp_pro6000`，GPU 为 RTX 2080 Ti + RTX 4090。下文关于 RTX 6000D、
> 旧 PID 和旧服务器正在运行 pilot 的内容仅是历史快照，不代表当前机器状态。当前新机器
> 尚未启动训练；环境状态和启动门禁以 `PASDP_服务器实验进度.md` 的 2026-08-27
> 新服务器状态为准。

---

## 1. 使用方法与事实优先级

新 AI 应先完整阅读本文，再按第 12 节读取引用文档。不同资料冲突时，使用以下优先级：

1. 服务器上当时的 `ps` / `nvidia-smi` / console log / resolved config；
2. 用户最新明确指令；
3. 本交接文档；
4. 当前 Git commit 中的实际代码和 Hydra resolved config；
5. 其他计划、进度和历史审计文档。

本文中的服务器运行状态来自用户在 2026-08-24 21:41 提供的终端输出，并非由当前本机直接连接服务器复核。继续操作前必须在服务器再做一次实时检查。

---

## 2. 一页状态摘要

| 项目 | 最新状态 |
|---|---|
| 远端仓库 | `origin = git@gitee.com:huang-1-yi/dp_pro6000.git` |
| 最新文档提交 | `ca57144 docs: add PASDP execution planning artifacts` |
| 最新训练代码提交 | `c16a739 feat: 完善PASDP任务自适应训练批次计划` |
| `ca57144` 对训练代码的影响 | 无；该提交只改文档，所以已由 `c16a739` 启动的训练不需重启 |
| 当前服务器项目目录 | `/home/hy/dp_pro6000` |
| 当前 pilot | `coffee_d0 + training seed 42` |
| 当前运行 | A-PASDP、B-DP、D-DP 三条链在物理 GPU1 并发 |
| GPU0 | 正被其他 `lerobot_v06` 任务占用，不属于本轮 PASDP |
| GPU1 | RTX 6000D，约 57.5 GiB / 85.7 GiB，97% utilization，约 74℃ |
| 当前 evaluation | 暂缓，不启动 |
| C 组 | 无训练；本质是复用 A-DP checkpoint 的 evaluation NFE sweep，当前完全暂缓 |
| 当前首要任务 | 不再启动新进程；完成并核验 `coffee_d0 + seed42` 的 10 个唯一训练 |

---

## 3. 本轮代码优化结果

### 3.1 RTV8 任务自适应映射计划

`diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py` 已实现：

- 生成每个 epoch 的 RTV8 mapping 时，直接记录该 epoch 的 batch 数；
- 将 251 个 epoch 的 `epoch_batch_counts` 和 `total_batches` 汇总到 `mapping_manifest.npz`；
- fingerprint 覆盖 schema version、batch size、dataset seed、train episode IDs 和 `episode_map`；
- mapping 与 manifest 使用临时文件 + `os.replace()` 原子发布，中断时不留下半写文件；
- 旧格式 mapping 缺少新 schema 时会一次性重建；
- manifest 有效后，后续训练 seed 直接复用，不再反复扫描计算所有 epoch。

### 3.2 逐 epoch 映射切换

`diffusion_policy/workspace/train_padp_workspace.py` 在每个 epoch 开始时：

1. 调用 dataset `set_epoch(epoch_index)`；
2. 加载该 epoch 对应 mapping；
3. 核验 `len(train_dataloader)` 与 manifest 记录一致；
4. 不一致时直接报错，避免静默使用错误映射。

日志新增：

- `rtv8/epoch_mapping_batches`；
- `rtv8/epoch_training_batches`；
- `rtv8/total_training_batches`；
- `rtv8/total_optimizer_steps`。

### 3.3 精确 LR scheduler 周期

- 不再用 `len(epoch0) × num_epochs` 估算 PASDP 全程；
- 不再限制每 epoch 560 batches；
- scheduler 根据当前 task 的 manifest 总 batch 数和 gradient accumulation 计算真实 optimizer updates；
- 不对 `coffee_d0` 写死常数，其他 task 会生成各自的计划。

### 3.4 代码结构整理

- 将只有一个生产调用点的 optimizer-step 计算直接放回 scheduler 阶段；
- 将只有一个调用点的 epoch 映射激活与长度校验直接放回 epoch 开头；
- 保留 `_resolve_epoch_batch_plan()`，因为它同时承担 manifest 回退、长度校验和 debug cap，不是简单函数；
- 保留 PADP-Unrolled 虚方法钩子和 manifest 生命周期函数；
- workspace 与 dataset 已使用 `######## 功能 ########` 标记主要阶段。

### 3.5 验证状态

- RTV8/workspace 直接相关测试通过；
- 相关 Hydra composable config 测试通过；
- 最终相关与配置测试合计 24 项通过；
- `py_compile` 与 `git diff --check` 通过；
- 使用 `coffee_d0` 真实 episode 结构的非训练集成验证：251 epochs，batch 数范围 1003–1048，总 batch 数 257679；
- 上述数字只是当时 `coffee_d0` 调用链验证，不是论文结果，也不是所有 task 共用常数。

---

## 4. 已冻结的训练矩阵

### 4.1 唯一训练配置

| Stage | 方法 | experiment | 是否新训练 |
|---|---|---|---:|
| A | PASDP | `pasdp_a_native_robomimic` | 是 |
| A | DP | `dp_a_paper_robomimic` | 是 |
| A | SDP | `sdp_a_paper_robomimic` | 是 |
| A | Flow-RGB | `flow_a_rgb_adapted_robomimic` | 是 |
| B | PASDP | 复用 A-PASDP | 否 |
| B | DP | `dp_b_nfe1_robomimic` | 是 |
| B | SDP | `sdp_b_nfe1_robomimic` | 是 |
| B | Flow-RGB | `flow_b_nfe1_rgb_adapted_robomimic` | 是 |
| D | PASDP | 复用 A-PASDP | 否 |
| D | DP | `dp_d_pasdp_matched_robomimic` | 是 |
| D | SDP | `sdp_d_pasdp_matched_robomimic` | 是 |
| D | Flow-RGB | `flow_d_pasdp_matched_rgb_adapted_robomimic` | 是 |
| C | DP NFE 1/20/40/100 | 复用同一 A-DP checkpoint | 否，仅 evaluation |

因此，每个 `task × training seed` 只有 **10 个唯一训练**：A 4 个 + B 3 个 + D 3 个。

### 4.2 公共训练协议

| 项目 | 固定值 |
|---|---|
| `n_demo` | 200 |
| epochs | 251（epoch 0–250） |
| early stopping | 关闭 |
| rollout | 每 5 epochs，共 51 个触发点 |
| checkpoint | 每 5 epochs，`latest.ckpt + top-5`，全新目录最多 6 个文件 |
| 训练期 rollout | `n_train=10,n_train_vis=2,n_test=30,n_test_vis=2,n_envs=20,test_start_seed=100000` |
| UNet channels | `[256,512,1024]` |
| RGB | `84×84 -> crop76`（train random / eval center） |
| training seeds | `42, 10000, 1` |
| PASDP dataset | RTV8 loader + `shuffle=false` |
| 其他方法 dataset | 标准 Robomimic loader + standard shuffle |

PASDP 与其他方法的 epoch 数一致，但不强行使每 epoch batch 数相同。PASDP 必须保留全量 RTV8 自校正映射；标准 DP/SDP/Flow 的有效窗口数也会因 horizon/action 配置不同而变化。

### 4.3 任务集

1. `coffee_d0`
2. `threading_d0`
3. `pick_place_d0`
4. `coffee_preparation_d0`
5. `hammer_cleanup_d0`
6. `square_d0`
7. `stack_d0`
8. `stack_three_d0`
9. `nut_assembly_d0`
10. `three_piece_assembly_d0`

---

## 5. 服务器最新执行快照

### 5.1 GPU 和进程

用户于 2026-08-24 21:41 提供的 `nvidia-smi` 显示：

| 物理 GPU | 状态 |
|---|---|
| GPU0 | RTX 6000D，约 29.8 GiB，98%，84℃；主要是 8 个 `lerobot_v06` Python 计算进程，不属于 PASDP |
| GPU1 | RTX 6000D，约 57.5 GiB，97%，74℃；正在运行 PASDP 实验 |

GPU1 存在 3 个主计算进程：

- PID `295016`，约 2.8 GiB；
- PID `304146`，约 7.7 GiB；
- PID `311204`，约 8.9 GiB。

同时存在三组各约 32 个 EGL/Robosuite 环境子进程，每个约 387 MiB，合计约 96 个环境进程。三个主 PID 与 A/B/D 的精确对应需用服务器 `ps -o pid,ppid,etime,cmd` 再确认，不应只根据显存大小硬推。

### 5.2 最新日志进度

| 训练链 | 用户看到的最新状态 |
|---|---|
| `A/pasdp/coffee_d0/seed42` | 进入训练期 rollout；`Eval Coffee_D0Image 1/2`，约 `264/400` |
| `B/dp/coffee_d0/seed42` | `Training epoch 0`，约 `542/875` |
| `D/dp/coffee_d0/seed42` | `Training epoch 10`，约 `298/799` |

用户在 `tail -f` 界面按下的 `Ctrl+C` 只停止了 `tail`；后续 `nvidia-smi` 仍可见主训练进程，所以训练并未被终止。

`B-DP=875 batches/epoch` 与 `D-DP=799 batches/epoch` 不同本身不证明异常；两组 temporal/horizon 配置不同，标准 dataset 的有效窗口数因此不同。

### 5.3 当前执行方式与原计划的差异

原计划文档写的是 GPU0/GPU1 分开且单卡串行。最新实况是：

- GPU0 已被其他项目占用；
- A、B、D 的当前首个训练同时在 GPU1 并发；
- GPU1 已达 97% utilization，不应再追加任务；
- 此并发方式会拉长单个 run 的墙钟时间并增加 EGL 显存占用，但在没有 OOM、NaN 或进程失败的前提下，不必因此立即中断已经运行的 seed42 pilot；
- 训练墙钟时间不能用作论文 latency。论文 latency 未来必须在独占 GPU 和统一协议下另行测量。

---

## 6. 当前立即应执行的计划

### 6.1 现在不做的事

- 不再启动任何新训练进程；
- 不启动独立 evaluation；
- 不启动 C；
- 不在 seed42 尚未全部完成时进入 seed10000；
- 不为了优化而修改正在运行实验所使用的代码或配置；
- 不删除、覆盖或重命名当前 run 目录。

### 6.2 立即检查启动链

服务器上执行：

```bash
cd /home/hy/dp_pro6000

git rev-parse HEAD
git status --short

nvidia-smi
ps -o pid,ppid,lstart,etime,cmd -p 295016,304146,311204
ps -ef --forest | grep -E 'train_sim.py|eval_sim_rollout.py'
```

必须确认三条 launcher 后面的内容：

- A 链只包含 A 的 4 个唯一训练；
- B 链只包含 B 的 3 个唯一训练；
- D 链只包含 D 的 3 个唯一训练；
- 不包含 `eval_sim_rollout.py`；
- 不会在另外两条 stage 链尚未完成时自动进入下一个 seed。

如果已启动的 shell `&&` 链中包含 evaluation 或未来 seed，不要盲目 kill 主训练 PID。先确认 launcher shell 与 child process 关系，再决定如何在当前 run 正常结束后阻止后续命令。

### 6.3 日志监控

优先用非持续命令，避免把 `Ctrl+C` 是否影响训练搞混：

```bash
tail -n 30 data/experiments/logs/train/A/pasdp/coffee_d0/seed42/console.log
tail -n 30 data/experiments/logs/train/B/dp/coffee_d0/seed42/console.log
tail -n 30 data/experiments/logs/train/D/dp/coffee_d0/seed42/console.log

grep -R -nE 'Traceback|CUDA out of memory|NaN|Killed' \
  data/experiments/logs/train/A \
  data/experiments/logs/train/B \
  data/experiments/logs/train/D
```

同时检查：

- console log 文件大小和修改时间持续增长；
- 进程没有变成 zombie；
- GPU1 没有 OOM/Xid；
- loss 没有 NaN/Inf；
- 运行目录没有与其他 stage/seed 冲突。

### 6.4 `coffee_d0 + seed42` pilot 完成条件

必须完成共 10 个唯一训练：

- A：PASDP、DP、SDP、Flow-RGB；
- B：DP、SDP、Flow-RGB；
- D：DP、SDP、Flow-RGB。

每个 run 至少核验：

1. 进程正常完成，无 traceback/OOM/NaN；
2. resolved config 中的 stage、experiment、task、seed 正确；
3. 最后到达 epoch 250；
4. `latest.ckpt` 与 top-k checkpoint 存在且大小合理；
5. 日志和 checkpoint 都在 `data/experiments/` 的正确 stage/task/seed 目录；
6. 记录实际 Git commit 和 dirty 状态。

A-PASDP 另外检查：

1. console 出现 `RTV8 training plan: epochs=251`；
2. 日志记录 `rtv8/epoch_mapping_batches` 和 `rtv8/total_optimizer_steps`；
3. mapping 目录存在 `mapping_manifest.npz`；
4. epoch 不同时确实切换 mapping；
5. 没有人为 `training.max_train_steps=560`。

### 6.5 后续顺序

只有在 `coffee_d0 + seed42` 的 A/B/D 共10个训练全部完成并核验后，才能继续：

```text
coffee_d0 seed42
→ coffee_d0 seed10000
→ coffee_d0 seed1
→ 下一个 task
```

每个 seed 都要等 A/B/D 的 10 个唯一训练全部完成，再统一更换 seed。

---

## 7. 当前可见的优化空间（尚未实现）

### 7.1 最大显存开销：rollout runner 生命周期

每个训练进程会在 workspace 开始时创建 runner，直到整个训练结束才关闭。当前训练协议使用 `n_envs=20`；GPU1 同时运行 B、D 时会驻留两组 runner，因此仍需关注 CUDA/EGL 显存压力。

后续可选方案：

1. 只在 rollout epoch 延迟创建 runner，rollout 后立即 close；
2. 保持现有 runner 生命周期，但限制每张 GPU 只有一个训练进程；
3. GPU0 的其他项目结束后，将新启动任务调整为每卡一个，不迁移正在运行的进程。

方案 1 可显著释放非 rollout epoch 的驻留显存，但会增加环境重建时间，而且涉及所有方法的训练生命周期。必须单独审计、smoke 并从新 seed/task 边界启用，不能影响已启动的 seed42。

### 7.2 运行调度优化优先于代码优化

当前 GPU1 已经 97% utilization。这说明再追加模型并不能提高有效吞吐，反而会增加：

- CUDA/EGL 显存压力；
- CPU 与 HDF5/Zarr I/O 争用；
- 单 run 墙钟时间；
- OOM、Xid 或环境子进程失败风险。

在没有完整的单进程/三进程总吞吐对比前，不要只根据某个 epoch 的速度决定长期并发度。

### 7.3 暂时不应改变的参数

不应为了节省资源临时修改：

- `n_envs=20`；
- 训练期 `n_train=10,n_test=30,test_start_seed=100000`；
- `rollout_every=5`；
- `checkpoint_every=5`；
- 251 epochs；
- batch size 64；
- PASDP RTV8 loader / `shuffle=false`。

这些是已冻结的训练协议。若后续修改 runner 生命周期，应保持 rollout seeds、episode 数、env 数和 checkpoint 选择语义不变。

---

## 8. 文档冲突与过时信息

### 8.1 进度文档已过时

[`PASDP_服务器实验进度_实际执行.md`](PASDP_服务器实验进度_实际执行.md) 仍写着“等待启动”和“尚未开始正式训练”。这已被 21:41 的服务器快照覆盖；当前 seed42 pilot 已经启动。

[`PASDP_服务器实验进度.md`](PASDP_服务器实验进度.md) 是更早的预检快照，不能用于判断当前 GPU 和进程状态。

### 8.2 GPU 分工计划内部矛盾

[`PASDP_固定实验矩阵执行计划.md`](PASDP_固定实验矩阵执行计划.md) 同时写了：

- GPU0/GPU1 按任务 5/5 分配；
- GPU0 负责所有 A/B，GPU1 负责所有 D/C。

两种分法不能同时成立。当前实况也不是上述任一方案，而是 GPU0 运行其他项目、GPU1 并发 A/B/D。后续新启动任务前必须另行确认调度方案。

### 8.3 C 和 evaluation 当前暂缓

C 没有新训练，只是 A-DP checkpoint 的 NFE `1/20/40/100` evaluation。用户当前要求“先只保留 train”，因此：

- 现在不应出现 `eval_sim_rollout.py`；
- 现在不应执行 C；
- top-k checkpoint 仍正常保存，留待未来 evaluation；
- 未来恢复 evaluation 前，再统一确认 test-only 协议。

此处的历史暂缓要求已于 2026-08-29 被后续决定取代。现已冻结正式 evaluation 为 `n_train=0,n_test=100,n_envs=25,test_start_seed=20000`，测试种子 `20000–20099`；训练期 checkpoint 监控仍为 `n_train=10,n_test=30,n_envs=20,test_start_seed=100000`。

### 8.4 命名约定

论文、表格和对外说明使用 `PASDP`；现有 Python 模块、类名、配置兼容字段和 checkpoint 内部仍可使用 `padp/PADP`。当前不做大范围命名重构。

---

## 9. 新电脑上的继续流程

### 9.1 拉取仓库

```bash
git clone git@gitee.com:huang-1-yi/dp_pro6000.git
cd dp_pro6000
git switch master
git pull --ff-only origin master
git log -3 --oneline --decorate
git status --short
```

不要为了更新文档而终止服务器已运行的 `c16a739` 进程；`ca57144` 不包含训练代码变更。

### 9.2 给新 AI 的推荐首条消息

```text
请先读取 PASDP_最新优化与执行交接.md，再读取其第12节列出的引用文档。

当前服务器的最新已知状态是：coffee_d0 + seed42 的 A/B/D 三条训练链在物理 GPU1 并发，GPU0 被其他 lerobot 任务占用；evaluation 和 C 暂缓。

首轮只做实时状态审计：检查 Git commit、dirty 状态、nvidia-smi、train_sim/eval_sim 进程树、三条 launcher 后续命令、console log、train_logs、checkpoint 和 mapping_manifest。不修改代码，不停止进程，不启动新任务，不启动 evaluation。

请先向我报告实际进度、异常、是否会自动进入下一 seed/evaluation，然后再给出下一步建议。
```

### 9.3 新 AI 首轮报告格式

```text
Git：<commit / dirty / remote sync>
GPU0：<其他任务 / 可用资源>
GPU1：<显存 / utilization / 温度>
A：<当前 algorithm / epoch / batch / PID / log mtime>
B：<当前 algorithm / epoch / batch / PID / log mtime>
D：<当前 algorithm / epoch / batch / PID / log mtime>
launcher：<是否包含后续 seed / evaluation>
PASDP manifest：<存在 / fingerprint / epochs / total batches>
checkpoint：<已生成列表>
异常：<traceback / OOM / NaN / Xid / none>
建议：<继续等待 / 需用户决策>
```

---

## 10. 未来 evaluation 恢复时的边界

当前不执行本节，只保留边界信息：

- A/B/D 使用统一 top-k checkpoint 选择规则，不逐方法挑最优；
- B-PASDP 和 D-PASDP 复用 A-PASDP；
- C 的 NFE 1/20/40/100 必须使用同一个 A-DP checkpoint；
- C 只 override inference steps，不重新训练；
- 论文 latency 必须独占 GPU，不能使用当前并发训练的墙钟时间；
- 独立 evaluation 协议已确认：`n_train=0,n_test=100,n_envs=25,test_start_seed=20000`。

---

## 11. 进度更新模板

后续每次交接至少追加以下内容：

```markdown
### YYYY-MM-DD HH:MM 执行快照

- Git commit：
- 工作树：
- GPU0：
- GPU1：
- 当前 task / seed：
- A：
- B：
- D：
- 已完成 run：
- 已生成 checkpoint：
- PASDP manifest：
- 异常：
- launcher 后续内容：
- 用户最新决策：
- 下一步：
```

---

## 12. 引用文档

1. [`PADP_代码结构与功能v3_2.md`](PADP_代码结构与功能v3_2.md)

   项目架构、Hydra 配置轴、dataset/policy/workspace/runner 责任边界和当时的验证状态。其 RTV8 章节早于本轮 manifest/scheduler 优化，相冲突时以本文和当前代码为准。

2. [`PASDP_仿真实验_运行参数表和运行代码.md`](PASDP_仿真实验_运行参数表和运行代码.md)

   A/B/D/C 的精简参数和标准命令。其独立 evaluation 段当前不执行。

3. [`PASDP_仿真实验_运行参数一览表.md`](PASDP_仿真实验_运行参数一览表.md)

   论文/native、NFE=1、PASDP 参数对齐和 DP inference-budget 的完整参数定义。

4. [`PASDP_仿真实验.md`](PASDP_仿真实验.md)

   实验总表、能力审计、manifest JSON 标准和论文汇总矩阵。其“未启动正式训练”描述已过时。

5. [`PASDP_服务器实验进度_实际执行.md`](PASDP_服务器实验进度_实际执行.md)

   服务器预检、数据、环境和用户冻结项。进度仍停留在启动前，不是最新运行状态。

6. [`PASDP_服务器实验进度.md`](PASDP_服务器实验进度.md)

   更早的服务器预检原始记录，可用于查环境版本、GPU 型号和 10 个 dataset 路径。

7. [`PASDP_服务器实验执行提示词.md`](PASDP_服务器实验执行提示词.md)

   服务器 AI 的安全边界、目录规范、训练完成判定、checkpoint 和结果登记标准。当前忽略其 evaluation 执行段，但保留质量约束。

8. [`PASDP_固定实验矩阵执行计划.md`](PASDP_固定实验矩阵执行计划.md)

   冻结任务、seed 和阶段的计划草案。其 GPU 5/5 任务分配与 A/B、D/C stage 分配互相矛盾，不得直接照抄启动。

9. [`PASDP_正式启动前确认清单.md`](PASDP_正式启动前确认清单.md)

   用户已确认的 10 任务、3 seeds、top-k、pilot 和目录约束。其 GPU 串行假设已被当前实况替代，evaluation 数量在恢复前需重新确认。

---

## 13. 当前交接结论

1. RTV8 manifest、逐 epoch mapping 切换和精确 scheduler 优化已完成并推送；
2. 服务器已从“等待启动”进入 `coffee_d0 + seed42` 正式 pilot 训练；
3. A/B/D 当前在同一物理 GPU1 并发，GPU0 被其他项目占用；
4. 现在不再追加进程，不启动 evaluation/C，不提前更换 seed；
5. 先核对 launcher 后续命令，再完成和验收 seed42 的10个唯一训练；
6. 最大的后续优化候选是 rollout runner/EGL 环境的生命周期，但在 seed42 运行期间不修改。
