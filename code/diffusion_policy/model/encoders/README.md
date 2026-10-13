# Observation encoder 分层

本目录按照 `shape_meta.obs.<key>.type` 和张量形状拆成三条独立分支：

| 模态 | `type` | 单步输入 | 默认实现 | 工厂参数 |
|---|---|---|---|---|
| RGB | `rgb` | `[B,C,H,W]` | Robomimic / timm / R3M / CLIP | `rgb_model` |
| 点云 | `point_cloud` | `[B,N,C]` | PointNet MLP + 对称池化 | `pointcloud_model` |
| 体素 | `voxel` | `[B,C,D,H,W]` | Conv3d + 自适应池化 | `voxel_model` |

Policy 会先把时间维展平，所以 encoder 本身都只处理单步 batch，并统一返回
`[B,D]`；`output_shape()` 返回 `(D,)`。未写 `type` 的 observation 继续按
`low_dim` 处理。

## 独立构建

```python
from diffusion_policy.model.encoders.pointcloud.factory import (
    build_pointcloud_obs_encoder,
)

encoder = build_pointcloud_obs_encoder(
    pointcloud_model="pointnet_xyz",  # 从 [N,6] 中只取前三维 XYZ
    shape_meta=shape_meta,
    encoder_kwargs={"feature_dim": 256},
)
```

FocalPolicy 的独立点云编码器使用 `pointnet_focal`；显式通道预设为
`pointnet_focal_xyz` 和 `pointnet_focal_xyzrgb`。它会把 `agent_pos` 先经 state MLP
再与 PointNet 特征拼接，默认输出 `(128,)`。详细复现边界见仓库根目录
`v8代码优化_阶段1_Focal点云编码器对比.md`。

体素对应 `build_voxel_obs_encoder("conv3d", ...)`。`pointnet`、`conv3d` 和
`voxel_cnn_small` 都只依赖当前核心 PyTorch 环境，不需要 Open3D、PyTorch3D 或
`escnn`。

## 按 shape_meta 自动分派

```python
from diffusion_policy.model.encoders import build_obs_encoder

encoder = build_obs_encoder(
    shape_meta=shape_meta,
    rgb_model="robomimic_raw",
    pointcloud_model="pointnet",
    voxel_model="conv3d",
)
```

只有一种空间模态时直接返回对应 encoder。纯 RGB 路径不会增加包装层，因此旧 DP / PADP
checkpoint key 不变。有多种空间模态时，工厂会分别编码并按
`RGB -> point_cloud -> voxel -> low_dim` 拼接；low-dim 只拼一次。

## 与 PADP 参考代码的边界

- PointNet 保留了 PADP / DP3 的逐点 MLP 和 max pooling，但去掉了固定 key、任务名、
  `termcolor` 和 PyTorch3D 依赖。
- 默认体素分支不是 PADP 的 `escnn` 等变网络。它保持相同的 `[C,D,H,W]` 输入契约，
  但支持任意正尺寸；PADP 等变版本只覆盖 16/58/64，并且还把 in-hand RGB、末端位姿和
  特定群表示绑在同一个 task encoder 中，不能当作通用体素 backbone 原样搬入。
- 当前两个活跃训练 YAML 仍是 RGB 数据链。点云或体素端到端训练还需要对应 dataset、
  normalizer、runner 和 Policy 配置；仅把 `type` 改掉并不会自动生成三维 observation。
