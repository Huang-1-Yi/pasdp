# FocalPolicy YAML 可选参数与编码器

## 当前兼容性

FocalPolicy 通过通用 observation factory 构建 encoder，因此代码接口支持 RGB、点云和体素。
当前只把“单独 encoder 已验证”与“FocalPolicy 组合已验证”区分开：

| 选择值 | encoder 状态 | FocalPolicy 组合状态 |
|---|---|---|
| `robomimic_raw` | 已验证 | Robomimic RGB 已烟测 |
| `timm_convnext_tiny` | 已验证 | 待烟测 |
| `pointnet_focal_xyz` | MetaWorld/Adroit 已验证 | MetaWorld 1 batch + rollout 已通过，长训练待执行 |
| 其他 selector | 已注册 | 待测试 |

## 可粘贴配置

优先使用 Hydra 配置组，不要把 encoder 字段复制进 experiment：

```bash
encoder=rgb/robomimic_raw
encoder=rgb/timm_convnext_tiny
encoder=pointcloud/focal_xyz
```

只有开发新 encoder 配置组时才粘贴下面的 YAML。

Robomimic RGB：

```yaml
policy:
  obs_encoder: null
  rgb_model: robomimic_raw
  rgb_encoder_kwargs: {}
```

ConvNeXt-Tiny：

```yaml
policy:
  obs_encoder: null
  rgb_model: timm_convnext_tiny
  rgb_encoder_kwargs: {}
```

点云：

```yaml
policy:
  obs_encoder: null
pointcloud_model: pointnet_focal_xyz
pointcloud_encoder_kwargs:
  point_feature_dim: 64
  state_mlp_size: [64, 64]
  use_layer_norm: true
  final_norm: layernorm
  include_low_dim: true
```

## 上游 Focal 典型时序与容量

```yaml
horizon: 12
n_obs_steps: 2
n_action_steps: 4
policy:
  down_dims: [256, 512, 1024]
  freq_weight: 1.0e-4
  eps: 1.0e-2
  time_scale: 99.0
```

`focal_metaworld`、`focal_adroit` 默认使用 `capacity=base`，即
`[256,512,1024]`；对应的 DP/PADP
对比 preset 也使用相同宽度。显存烟测必须显式指定 `capacity=smoke`，不能将 smoke
结果作为性能结果。

论文/上游配置使用：

```bash
capacity=large recipe=focal_upstream
```

对应 `[512,1024,2048]`、3000 epoch、每200 epoch rollout/checkpoint，必须与
公平消融结果分表报告。

## 待测试 selector

```yaml
# RGB
rgb_model: timm_resnet18
rgb_model: timm_vit_tiny
rgb_model: timm_dino_vits16
rgb_model: timm_dinov2_vits14
rgb_model: timm_dinov3_vits16
rgb_model: r3m_resnet18
rgb_model: r3m_resnet34
rgb_model: r3m_resnet50
rgb_model: clip_vit_b32
rgb_model: openclip_vit_b32

# 点云
pointcloud_model: pointnet_xyz
pointcloud_model: pointnet_xyzrgb
pointcloud_model: pointnet_focal_xyzrgb

# 体素
voxel_model: conv3d
```

冻结 encoder：

```yaml
training:
  freeze_encoder: true
```
