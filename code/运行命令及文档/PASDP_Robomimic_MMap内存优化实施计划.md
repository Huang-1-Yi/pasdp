# PASDP Robomimic MMap 内存优化实施计划

更新日期：2026-08-31

## 1. 目标与边界

本计划针对 Robomimic RGB 训练在多进程 `spawn` 模式下的内存放大问题，目标是在保持以下实验口径不变的前提下，使多个训练任务共享同一份磁盘数据，避免每个 DataLoader worker 复制整份内存缓存。

- `training.seed=42`
- `n_demo=200`
- batch size 64
- train workers 8
- validation workers 8
- `n_envs=20`
- train/test episode 数、种子和 rollout 周期
- horizon、observation/action steps、模型、损失函数和 optimizer

本计划不将 `/dev/shm` 扩容作为主方案，不修改正在运行的 Hydra 快照，不删除现有 HDF5、ZIP Zarr、RTV8 DirectoryStore 或 checkpoint。

## 2. 当前实现

### 2.1 DP / SDP / Flow 基线数据路径

相关文件：

- `diffusion_policy/dataset/robomimic/replay_image_dataset.py`
- `diffusion_policy/config/dataset/robomimic/replay_image.yaml`
- `diffusion_policy/workspace/train_policy_workspace.py`
- `diffusion_policy/config/recipe/dp_robomimic.yaml`

当前流程：

```text
<task>_abs.hdf5
  -> <task>_abs.hdf5.200..zarr.zip
  -> ReplayBuffer.copy_from_store(..., zarr.MemoryStore())
  -> Dataset 保持完整 MemoryStore
  -> spawn 创建 train workers x 8 + val workers x 8
  -> worker 序列化获得 Dataset / MemoryStore 状态
  -> __getitem__ 在 CPU 将 uint8 图像转 float32 / 255
  -> batch 传入 GPU
```

主要问题：

1. `use_cache=true` 只减少了 HDF5 重复转换时间，未实现磁盘懒读取。
2. ZIP Zarr 打开后会被整体复制到 `zarr.MemoryStore`。
3. `diffusion_policy/envs/gym_util/async_vector_env.py` 强制全局使用 `spawn`，DataLoader worker 不能依赖 `fork` 的 copy-on-write。
4. train/val 各 8 个 worker，且 `persistent_workers=true`，一条训练最多长期保留 16 组 worker Dataset 状态。
5. RGB 在 worker 中转为 `float32`，使单个样本和预取 batch 的 CPU / IPC 体积扩大约 4 倍。
6. `train_policy_workspace.py` 在训练循环之前创建 env runner，20 个 MuJoCo 进程在非 rollout 阶段也保持存活。

### 2.2 PASDP 数据路径

相关文件：

- `diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py`
- `diffusion_policy/dataset/rtv8/cache.py`
- `diffusion_policy/config/dataset/robomimic/replay_image_padp.yaml`
- `diffusion_policy/workspace/train_padp_workspace.py`
- `diffusion_policy/config/recipe/padp_robomimic.yaml`

PASDP 已经具备两项目标特性：

- RTV8 缓存直接使用磁盘 `zarr.DirectoryStore`，不会把 ZIP 整体复制到 `MemoryStore`。
- `training.uint8_to_gpu=true` 时，Dataset 返回 RGB `uint8`，`train_padp_workspace.py` 在 GPU 上执行 `float().div_(255.0)`。

因此首期改造应主要针对 DP / SDP / Flow 基线，PASDP 只做接口一致性和内存基准验证，不重写 RTV8 映射逻辑。

### 2.3 实测资源现象

2026-08-31 在两条 4090 训练同时运行时：

- 物理内存 125 GiB，已用约 105–107 GiB。
- swap 2 GiB 已用尽。
- 每条训练存在 16 个 DataLoader worker 和 20 个 MuJoCo 环境进程。
- 第三条 Group 1 任务启动后触发系统 OOM，内核曾杀死 VS Code 进程。

与当前基线 ZIP 缓存对应的理论数据复制量：

| Task | ZIP Zarr 约大小 | 16 workers 理论放大 |
|---|---:|---:|
| `stack_three_d0` | 228 MiB | 3.6 GiB |
| `coffee_preparation_d0` | 931 MiB | 14.5 GiB |
| `pick_place_d0` | 1.6 GiB | 25.6 GiB |

表中数字不等于进程的完整 RSS，但可以解释实测中数十 GiB 的 DataLoader 额外开销。

## 3. 目标实现

### 3.1 目标数据流

```text
<task>_abs.hdf5
  -> 一次性转换为只读、无压缩 uint8 / float32 NPY 数组
  -> 主进程只保存 manifest、路径、shape、dtype 和 sampler 索引
  -> 每个 spawned worker 使用 np.load(..., mmap_mode="r") 懒打开同一文件
  -> worker 只切取当前 sample / window，不复制完整 Dataset
  -> RGB 以 uint8 batch 进入 GPU
  -> workspace 在 GPU 上执行 float32 / 255
```

数据不放入 `/dev/shm`。多个 worker 对同一只读文件的 mmap 由 Linux page cache 共享；内存紧张时，干净的文件缓存页可直接回收，无需先写入 swap。

### 3.2 目标缓存格式

每个 task 生成一个独立且带指纹的目录，示例：

```text
data/robomimic/datasets/<task>/
  <task>_abs.n200.abs1.rotrotation_6d.shape<hash>.mmap/
    manifest.json
    episode_ends.npy
    action.npy
    robot0_eef_pos.npy
    robot0_eef_quat.npy
    robot0_gripper_qpos.npy
    agentview_image.npy
    robot0_eye_in_hand_image.npy
```

类型约束：

- RGB：`uint8`，shape 为 `[N,H,W,C]`。
- action / low-dimensional observation：`float32`。
- episode boundaries / index：`int64`。
- 文件全部只读，不允许 worker 就地修改。
- manifest 记录 schema version、source HDF5 指纹、`n_demo`、shape meta、rotation representation、array shape/dtype 和转换完成状态。

不使用“每帧一个文件”的无压缩 Zarr chunk，避免大量小文件和 inode / metadata 开销。

### 3.3 Dataset 进程模型

目标 Dataset 必须实现以下约束：

1. `__init__` 不打开所有 memmap，只加载小型 manifest 和 sampler 元数据。
2. `_ensure_open()` 在主进程或 worker 首次访问时用 `mmap_mode="r"` 打开数组。
3. `__getstate__()` 在 spawn 序列化前清空打开的 handle，只传递路径和元数据。
4. `__setstate__()` 不将整份数据复制进 worker，handle 在 worker 内懒打开。
5. `get_validation_dataset()` 只更换 episode mask / sampler，继续指向同一组只读文件。
6. `__getitem__()` 只复制当前 sample/window，解决 PyTorch 对只读 NumPy view 的可写性警告，不物化整个 array。
7. normalizer 优先读取 manifest 中的预计算 statistics；缺失时才流式扫描 memmap，不生成完整副本。

### 3.4 目标效果

| 维度 | 当前实现 | 目标实现 | 预期效果 |
|---|---|---|---|
| 数据常驻位置 | worker 私有 MemoryStore | NVMe 只读 mmap | 去除整份数据集的 worker 私有副本 |
| 系统缓存 | 匿名内存，只能 swap/OOM | 可回收的文件页缓存 | 提高 `MemAvailable`，减少 swap-in/out |
| RGB CPU dtype | float32 | uint8 | sample、prefetch 和 IPC 体积约降为 1/4 |
| 图片解码 | JPEG2K 样本时解码 | 转换期解码一次 | 减少 worker CPU 开销 |
| 磁盘占用 | 较小 | 较大 | 使用充足 NVMe 换取 RAM/CPU |
| 冷启动 | 整体加载后训练 | 按需读取 | 首 epoch 可能受 NVMe 影响 |
| 热启动 | 内存复制后快 | OS page cache 共享 | 多任务时避免重复缓存 |
| 训练语义 | 当前标准 | 目标保持不变 | 通过 sample/batch hash 和短训练对齐验证 |

不承诺仅通过架构设计就达到固定 GiB 数值，最终效果必须用 PSS、`MemAvailable`、swap I/O 和 batch throughput 实测。

## 4. 修改计划

### 4.1 新增 mmap 缓存转换层

新增：

- `diffusion_policy/dataset/robomimic/mmap_cache.py`
- `diffusion_policy/scripts/build_robomimic_mmap_cache.py`

转换器责任：

1. 按 episode 顺序第一遍扫描 HDF5，统计 `n_steps`、episode boundaries 和各 key 的 shape/dtype。
2. 使用 `numpy.lib.format.open_memmap` 为每个 key 一次性创建目标 `.npy`。
3. 第二遍按 episode 顺序写入，不使用 `list + np.concatenate` 构造全量中间副本。
4. 图像保持 HDF5 中的精确 `uint8`像素。
5. action 复用现有 absolute-action / rotation conversion 逻辑，不新建第二套数学实现。
6. 计算并写入 normalizer statistics、source fingerprint 和 array checksum。
7. 使用 `FileLock`、临时目录和原子 rename；只有 manifest 标记 complete 后才允许 Dataset 使用。
8. 转换脚本支持单 task、全 10 tasks、`--verify-only` 和断点不覆盖检查。

### 4.2 新增只读 mmap Dataset

新增：

- `diffusion_policy/dataset/robomimic/replay_image_dataset_mmap.py`

修改/复用：

- `diffusion_policy/dataset/robomimic/replay_image_dataset.py`
- `diffusion_policy/common/sampler.py`

计划：

1. 新 Dataset 保留现有 `SequenceSampler`、padding、validation mask、absolute action 和 normalizer 语义。
2. 抽取基线 Dataset 中与 storage 无关的 key 分类、action conversion 和 normalizer 逻辑，避免新旧 backend 出现数学分叉。
3. 实现 `_ensure_open()`、`close()`、`__getstate__()` 和 `__setstate__()`。
4. RGB 返回 `uint8 [T,C,H,W]`；low-dimensional observation 和 action 继续返回 `float32`。
5. 仅对当前 window 做必要的 contiguous/copy，不对整个 memmap 调用 `np.asarray(...).copy()` 或 `astype()`。
6. validation Dataset 复用同一 manifest 和 mmap 路径，仅建立不同 sampler。
7. 保留原 `RobomimicReplayImageDataset` 作为 rollback backend，首期不删除 ZIP Zarr 路径。

### 4.3 新增可选 Dataset 配置

新增：

- `diffusion_policy/config/dataset/robomimic/replay_image_mmap.yaml`

配置必须显式包含：

```yaml
_target_: diffusion_policy.dataset.robomimic.replay_image_dataset_mmap.RobomimicReplayImageDatasetMMap
dataset_path: ${dataset_path}
n_demo: ${n_demo}
shape_meta: ${task.shape_meta}
horizon: ${horizon}
pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
pad_after: ${eval:'${n_action_steps}-1'}
n_obs_steps: ${dataset_obs_steps}
abs_action: ${task.abs_action}
rotation_rep: rotation_6d
seed: 42
val_ratio: 0.02
uint8_to_gpu: true
```

首期只通过 Hydra override 选择新 Dataset，不直接改动已冻结的 experiment defaults。完成全部验证后，再将以下基线 experiment 统一切换到 `replay_image_mmap`：

- `diffusion_policy/config/experiment/dp_a_paper_robomimic.yaml`
- `diffusion_policy/config/experiment/sdp_a_paper_robomimic.yaml`
- `diffusion_policy/config/experiment/flow_a_rgb_adapted_robomimic.yaml`
- `diffusion_policy/config/experiment/dp_b_nfe1_robomimic.yaml`
- `diffusion_policy/config/experiment/sdp_b_nfe1_robomimic.yaml`
- `diffusion_policy/config/experiment/flow_b_nfe1_rgb_adapted_robomimic.yaml`

### 4.4 在基线 workspace 中增加 GPU 侧 uint8 归一化

修改：

- `diffusion_policy/workspace/train_policy_workspace.py`
- 可选新增 `diffusion_policy/workspace/common/image_batch.py`

计划：

1. 复用 `train_padp_workspace.py` 已验证的语义：先 non-blocking 传入 GPU，再对 RGB `uint8` 执行 `float().div_(255.0)`。
2. 归一化必须同时覆盖 train、validation 和 sample probe 路径。
3. 只根据 dtype 和 shape-meta RGB key 处理图像，不误改 action、low-dimensional observation 或 `window_info`。
4. 将共同帮助函数放入 `workspace/common`，再让 policy/PADP workspace 共用，减少两条路径漂移。
5. 保留 `uint8_to_gpu=false` 回退开关，用于数值对齐。

### 4.5 DataLoader 配置处理

修改：

- `diffusion_policy/config/recipe/dp_robomimic.yaml`
- `diffusion_policy/config/recipe/padp_robomimic.yaml`

本计划的主对照保持 train/val workers 均为 8，以确认内存降低来自 storage backend，而不是 worker 数减少。

分两步测试：

1. 等价对照：完全保持现有 `num_workers=8`、`pin_memory=true`、`persistent_workers=true`。
2. 独立资源对照：在不改 worker 数的情况下测试 `prefetch_factor=1` 和 `pin_memory=false`；不将该结果与 mmap backend 本身的效果混在一起。

### 4.6 PASDP / RTV8 对齐

审核但首期不改变数据语义：

- `diffusion_policy/dataset/robomimic/replay_image_dataset_padp.py`
- `diffusion_policy/dataset/robomimic/replay_image_dataset_padp_unrolled.py`
- `diffusion_policy/dataset/rtv8/cache.py`
- `diffusion_policy/workspace/train_padp_workspace.py`

检查项：

1. DirectoryStore handle 在 spawn worker 中是否为独立只读打开，不序列化不必要的 runtime cache。
2. `uint8_to_gpu=true` 在 train/val/sample 路径上的处理是否与基线新帮助函数一致。
3. RTV8 epoch mapping、manifest、SequentialSampler 和 `window_info` 不得因通用化而改变。
4. PASDP 继续使用现有 DirectoryStore，不强制转换为基线 mmap schema。

### 4.7 可选的 rollout runner 生命周期优化

仅在 mmap 完成后仍不能满足三训练并行内存阈值时进入本阶段。

可能修改：

- `diffusion_policy/workspace/common/base_training_workspace.py`
- `diffusion_policy/workspace/train_policy_workspace.py`
- `diffusion_policy/workspace/train_padp_workspace.py`

目标为增加可选的 `persistent` / `per_rollout` runner 生命周期：

- `persistent`：保持现行为，作为默认和回退。
- `per_rollout`：只在 rollout epoch 创建 env runner，用 `try/finally` 确保完成或异常时关闭。

该选项可进一步释放非 rollout 阶段的 20 个 MuJoCo 进程，但会增加 EGL 上下文创建次数和 rollout 启动时间，必须独立验证，不与 mmap 首期同时上线。

## 5. 验证计划

### 5.1 静态与单元验证

新增：

- `tests/test_robomimic_mmap_cache.py`
- `tests/test_robomimic_mmap_dataset.py`
- `tests/test_robomimic_mmap_spawn.py`

覆盖：

1. HDF5 转换后每个 array 的 shape、dtype、episode boundaries 和 checksum。
2. 随机抽取至少 1,000 个索引，比较旧 Dataset 和 mmap Dataset 的 RGB、low-dimensional observation 和 action。
3. 覆盖 episode 头尾 padding、validation split、single/dual-arm action 和 rotation conversion。
4. 旧 CPU `float32/255` 与新 GPU `uint8 -> float32/255` 的最大绝对误差。
5. `spawn` 下 8 train + 8 val worker 能同时只读打开，不将完整 array 序列化到 worker。
6. worker 退出后 handle 和临时资源正常释放。
7. 不完整、fingerprint 不匹配或 source HDF5 已变更的缓存必须明确拒绝，不静默使用。

### 5.2 性能与内存基准

新增：

- `diffusion_policy/scripts/benchmark_robomimic_dataset_memory.py`

每个基准固定 task、seed、batch size、worker 数、sample 数和冷/热缓存状态，记录：

- Dataset 构造时间。
- 首 batch 延迟。
- 稳态 batches/s 和 samples/s。
- 主进程及 worker 的 RSS/PSS。
- `MemAvailable`、page cache、swap used、`si/so`。
- GPU utilization 和 data wait 时间。
- 进程退出后的内存回收情况。

对照矩阵：

| Backend | Train workers | Val workers | Pin memory | Prefetch |
|---|---:|---:|---|---:|
| MemoryStore | 8 | 8 | true | default 2 |
| MMap | 8 | 8 | true | default 2 |
| MMap | 8 | 8 | false | 1 |

接受条件：

1. 相同 8+8 workers 下，DataLoader 进程合计 PSS 相比 MemoryStore 至少下降 50%。
2. 单任务热缓存稳态 throughput 相比旧 backend 回退不超过 15%。
3. 持续 30 分钟基准中 swap 不增长，无持续 swap-in/out。
4. Dataset 和 worker 退出后私有内存可回收，无孤儿进程和 leaked semaphore 增长。

### 5.3 训练烟测

依次执行：

1. `stack_three_d0`：1 batch，无 rollout。
2. `stack_three_d0`：2 epoch，1 次 validation，1 次 rollout，`n_envs=1`。
3. `stack_three_d0`：保持正式 8+8 workers、`n_envs=20`，运行至少 2 个 rollout 周期。
4. `pick_place_d0`：使用最大的已知基线缓存，运行至少 30 分钟。
5. 两训练并行烟测。
6. 三训练并行烟测，使用与正式组相同的 2080 Ti / 4090 / 4090 绑定。

三训练并行接受条件：

- 连续运行至少 60 分钟。
- 无 kernel OOM、无 IDE/desktop 进程被杀、无 EGL 初始化失败。
- `MemAvailable` 稳态不低于 20 GiB。
- swap 不持续增长，`vmstat` 不出现持续 `si/so`。
- 三个任务的 log、checkpoint 和 completion marker 均正常更新。

## 6. 分阶段上线

### 阶段 0：冻结现场

- 不修改正在运行的 Group 2 / Group 3 进程。
- 保留其 `.hydra/config.yaml`、log 和 checkpoint。
- Group 1 在新 backend 完成烟测前不按原配置重启。

### 阶段 1：转换器与只读 backend

- 新增 converter、manifest 和 mmap Dataset。
- 首先转换 `stack_three_d0`，再转换 `pick_place_d0`。
- 保留旧 ZIP Zarr，不覆盖或删除。

### 阶段 2：数值对齐

- 完成 sample/batch hash、padding、split、normalizer 和 GPU 归一化验证。
- 任一像素、action、index 或 split 不一致时停止上线。

### 阶段 3：单任务性能验收

- 在 8+8 workers 下对比 MemoryStore 和 mmap。
- 分开记录 storage backend 效果与 pin/prefetch 效果。

### 阶段 4：并行烟测

- 先两任务，后三任务。
- 达到三训练接受条件后，才允许 Group 1 恢复长时间训练。

### 阶段 5：切换正式配置

- 基线 A/B experiment 统一切换到 `replay_image_mmap`。
- 更新实验配置一览表，记录 storage backend、RGB CPU/GPU dtype 和 cache fingerprint。
- 已完成的旧运行不追溯修改；新旧 backend 不得在同一运行目录中 resume。

### 阶段 6：可选的 runner 生命周期优化

- 只在 mmap 仍不足以支持三训练时执行。
- 独立 commit、独立基准、独立回滚开关。

## 7. 回滚与安全约束

1. 保留 `replay_image.yaml` 和原 `RobomimicReplayImageDataset`，通过 Dataset config 即可回退。
2. mmap cache 使用新目录，不覆盖 HDF5、ZIP Zarr 或 RTV8 DirectoryStore。
3. 构建失败的临时目录不能被 Dataset 识别为有效缓存。
4. 不在正在运行的训练目录中替换数据文件。
5. 新 backend 首次正式运行必须写入 backend schema version 和 cache fingerprint。
6. 出现数值不一致、throughput 超阈值回退、swap 持续增长、OOM、EGL 错误或孤儿 worker 时立即回退到旧 Dataset 并停止三任务并行。

## 8. 预计修改文件清单

| 类型 | 相对路径 | 修改内容 |
|---|---|---|
| 新增 | `diffusion_policy/dataset/robomimic/mmap_cache.py` | mmap schema、manifest、lock、atomic build、statistics |
| 新增 | `diffusion_policy/dataset/robomimic/replay_image_dataset_mmap.py` | 只读懒打开 Dataset，spawn-safe 序列化 |
| 新增 | `diffusion_policy/scripts/build_robomimic_mmap_cache.py` | 单 task / 全 task 构建和验证命令 |
| 新增 | `diffusion_policy/config/dataset/robomimic/replay_image_mmap.yaml` | 可选 mmap Dataset 配置 |
| 修改 | `diffusion_policy/dataset/robomimic/replay_image_dataset.py` | 抽取可复用语义，保留 MemoryStore rollback |
| 修改 | `diffusion_policy/workspace/train_policy_workspace.py` | RGB uint8 GPU 归一化，train/val/sample 覆盖 |
| 可选新增 | `diffusion_policy/workspace/common/image_batch.py` | policy/PADP 共享的 uint8 处理函数 |
| 审核 | `diffusion_policy/workspace/train_padp_workspace.py` | 共享函数对齐，不改 RTV8 语义 |
| 修改 | `diffusion_policy/config/recipe/dp_robomimic.yaml` | 增加 `uint8_to_gpu`；worker 数保持 8+8 |
| 审核 | `diffusion_policy/config/recipe/padp_robomimic.yaml` | 保持现有 worker 和 uint8 语义 |
| 延后修改 | 六个 A/B baseline experiment YAML | 验收后统一切换 Dataset default |
| 新增 | `tests/test_robomimic_mmap_cache.py` | 转换、manifest、fingerprint、不完整缓存拒绝 |
| 新增 | `tests/test_robomimic_mmap_dataset.py` | sample/batch、padding、split、dtype、normalizer 对齐 |
| 新增 | `tests/test_robomimic_mmap_spawn.py` | 8+8 spawn worker 只读共享和资源回收 |
| 新增 | `diffusion_policy/scripts/benchmark_robomimic_dataset_memory.py` | PSS、throughput、swap、cold/warm cache 基准 |

## 9. 完成标准

以下条件全部满足后，才可将 mmap 记为正式训练 backend：

- 旧/新 Dataset 的样本、batch、padding、split、action 和 normalizer 验证通过。
- 8+8 workers 下 DataLoader 合计 PSS 至少下降 50%。
- 单任务热缓存 throughput 回退不超过 15%。
- 三训练、`n_envs=20` 条件下持续运行 60 分钟且无 OOM/EGL 错误。
- `MemAvailable >= 20 GiB`，swap 不持续增长。
- 缓存指纹、backend schema 和实际 Hydra config 可追溯。
- 旧 MemoryStore backend 仍可通过配置回退。
- 实验记录中明确标记数据 backend 变更，不将它误写为模型或算法改动。
