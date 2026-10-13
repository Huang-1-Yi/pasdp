# PADP YAML 可选参数与编码器

## 当前兼容性

| 类别 | 选择值 | 状态 |
|---|---|---|
| Robomimic RGB | `robomimic` | 已测试可用；PADP checkpoint 使用 wrapper 布局 |
| timm ConvNeXt | `timm_convnext_tiny` | 已测试能进入训练 |
| Focal XYZ PointNet | `pointnet_focal_xyz` | MetaWorld 已长训练；Adroit 已通过 dataset/loss/1-step rollout |
| 其他 RGB/点云/voxel | 下表 | 待测试 |

## 可粘贴的 Robomimic RGB 配置

```yaml
policy:
  obs_encoder: null
  rgb_model: robomimic
  rgb_encoder_kwargs: {}
  crop_shape: [76, 76]
  obs_encoder_group_norm: true
  eval_fixed_crop: true
```

ConvNeXt-Tiny：

```yaml
policy:
  obs_encoder: null
  rgb_model: timm_convnext_tiny
  rgb_encoder_kwargs: {}
  crop_shape: [76, 76]
  obs_encoder_group_norm: true
  eval_fixed_crop: true
```

MetaWorld 已验证点云：

```yaml
policy:
  obs_encoder: null
  pointcloud_model: pointnet_focal_xyz
  pointcloud_encoder_kwargs: {}
```

## 已注册但待测试

```yaml
# RGB，每次选择一个
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

# 点云，每次选择一个
pointcloud_model: pointnet_xyz
pointcloud_model: pointnet_xyzrgb
pointcloud_model: pointnet_focal_xyzrgb

# 体素
voxel_model: conv3d
```

## PADP 消融参数

```yaml
n_obs_steps: 1
n_action_steps: 1
horizon: 40
window_exp_gamma: 0.25
policy:
  noise_schedule_mode: positionwise
  noise_chunk_size: 1
  pred_type: sample
  noise_scheduler:
    prediction_type: sample
```

允许的噪声模式：`positionwise`、`constant`、`linear`、`random`、`chunkwise`。

上面的 `40/1/1` 是 PADP 原始消融窗口。与点云 Focal/DP 公平对比时应使用
`experiment=compare_padp_metaworld` 或 `compare_padp_adroit`，它们固定为共同的
`12/2/4`、Focal XYZ encoder 和 `capacity=base`（`[256,512,1024]`）。

冻结 encoder：

```yaml
training:
  freeze_encoder: true
```

Robomimic RGB 路径保持 `training.uint8_to_gpu: true`；MetaWorld/Adroit float32 点云必须改成：

```yaml
training:
  uint8_to_gpu: false
```
