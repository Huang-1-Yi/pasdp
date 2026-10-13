# DP YAML 可选参数与编码器

## 当前兼容性

| 类别 | 选择值 | 状态 |
|---|---|---|
| Robomimic RGB | `robomimic_raw` | 已测试可用；DP 应优先使用 raw 布局 |
| timm ConvNeXt | `timm_convnext_tiny` | 已测试能进入训练 |
| 点云 | `pointnet_focal_xyz` | DP 已接入通用 factory；MetaWorld 1 batch + rollout 已烟测 |
| 其他 timm/R3M/CLIP | 见下表 | 待做 DP 组合烟测 |

## 可直接粘贴到 DP YAML 的已验证 RGB 行

Robomimic：

```yaml
policy:
  obs_encoder: null
  rgb_model: robomimic_raw
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

对应命令行仅需：

```bash
policy.rgb_model=robomimic_raw
policy.rgb_model=timm_convnext_tiny
```

## 已注册但待测试的 RGB selector

```yaml
# 每次只保留其中一行
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
```

点云 task 下可通过下面配置选择 Focal XYZ encoder：

```yaml
pointcloud_model: pointnet_focal_xyz
```

组合式命令优先使用：

```bash
experiment=compare_dp_metaworld encoder=pointcloud/focal_xyz
experiment=compare_dp_adroit encoder=pointcloud/focal_xyz
```

## 常用训练 override

```yaml
training:
  resume: false
  freeze_encoder: false
  seed: 42
dataloader:
  batch_size: 64
```

冻结 encoder：

```bash
training.freeze_encoder=true
```

安装 timm（不改其他依赖）：

```bash
conda run -n sim_torch27_v8 python -m pip install --no-deps \
  timm==1.0.28 safetensors==0.8.0
```
