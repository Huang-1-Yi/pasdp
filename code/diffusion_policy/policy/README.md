# Policy 目录约定

`policy/` 按算法组织，不按 Robomimic、MetaWorld、Adroit 等环境复制实现。环境差异由
`task/dataset/env_runner` 配置处理，输入差异由 `model/encoders` 处理。

## 活跃布局

- 外层单文件：可直接选择的 DP、PADP、ACT、DP3、Flow、Diffusion Transformer、
  Robomimic BCRNN、Streaming Diffusion 及其优化版。
- `focal_policy.py`：可直接选择的 FocalPolicy；其 DCT objective 和 time sampler
  分别位于 `common/focal_objectives.py` 和 `common/focal_time_sampler.py`。
- `common/`：`BasePolicy`、`BaseDiffusionPolicy`、TEDI/PADP scheduler 和
  FlowPolicy consistency-flow schedule 等非独立辅助件；该目录不得依赖具体 suite。

旧 `policy/robomimic/`、根目录 `base_image_policy.py`、`schedulers*.py` 和旧基类别名
已经删除。本仓库只支持当前组合式配置和从零训练，不保证加载重构前 checkpoint。

## Streaming 命名

- `streaming_diffusion_policy.py`：对上游
  `tedi_unet_hybrid_image_policy.py` 的复现与本项目接口适配。
- `streaming_diffusion_policy_optimized.py`：在复现版上加入窗口损失权重和确定性
  chunk timestep 映射的优化版。

对应 Hydra 入口分别为：

```text
experiment=streaming_diffusion_robomimic
experiment=streaming_diffusion_optimized_robomimic
```

## 当前输入边界

| Policy | 已验证输入 | 定位 |
|---|---|---|
| DP / PADP / Focal | RGB、点云 | controlled 主对比 |
| Robomimic BCRNN | RGB + low-dim | 原生 Robomimic full-method |
| ACT | RGB + low-dim | full-method |
| DP3 | 点云 + low-dim | 原生点云 full-method |
| Streaming / Flow / Transformer | Robomimic RGB | 扩展轨道，跨模态前需继续解耦 |

服务器已验证新布局下 DP、PADP、Focal、BCRNN、ACT、DP3、Flow、Diffusion
Transformer、Streaming Diffusion 及其优化版均能完成最小训练/rollout。
