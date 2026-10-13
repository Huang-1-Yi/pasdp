# 固定版本第三方依赖

本目录把仿真训练依赖的指定版本源码和 PyTorch3D wheel 与主工程一起保存。服务器拉取完整仓库后，不再需要单独访问 GitHub、切换提交或执行 MimicGen 的 editable 安装。

版本、来源与本地修改记录见 `UPSTREAM_VERSIONS.yaml`。各项目的许可证保留在对应源码目录中。

主仓库和 Robosuite 上游的 `.gitignore` 原本会忽略 PNG/JPG 等图片。这里的图片包含 MuJoCo 运行时纹理，并非训练输出，必须随 Git 提交；缺失时会出现 `PNG file load error` 或 `texplane` 错误。

## 安装方式

推荐直接在仓库根目录运行：

```bash
mamba env create -f conda_environment_sim_torch27_v3.yaml
conda activate equidiff_torch27_v3
```

v3 环境文件会从以下本地路径安装：

- `robosuite`
- `robomimic`
- `robosuite-task-zoo`
- `mimicgen`
- `wheels/pytorch3d-0.7.9-py3-none-any.whl`

其中 MimicGen 由 YAML 自动执行本地 editable 安装。原因是该固定提交的 `setup.py` 构建普通 wheel 时会漏掉 `mimicgen_envs/models` 和模型资源；editable 安装与上游原用法一致，但不再需要用户手工执行额外命令。

这些本地源码不再依赖 GitHub 网络，但 Conda、PyPI 和 PyTorch CUDA wheel 的基础依赖仍需要镜像或已有缓存。

## 本地 Robomimic 修改

pointW 固定提交的 `env_robosuite.py` 会在模块顶层导入 Open3D，并在每次 RGB observation 中无条件生成点云和体素。当前工程只运行 `robomimic_dp_ddim_sample_40_1.yaml` 与 `robomimic_padp_position_wise_v3.yaml`，两者只使用 RGB 和低维 observation，因此本地版本整段注释停用了上述点云路径并关闭深度图渲染。原始实现仍保留在代码原位置，便于后续参考和恢复。

这意味着 v3 环境不安装 Open3D，也不支持依赖 `point_cloud`、`voxels` 或深度 observation 的配置。若以后恢复点云任务，需要恢复上游实现并单独加入 Open3D。

## 更新规则

不要直接覆盖某个子目录后继续使用旧的版本清单。升级第三方项目时至少需要：

1. 确认新的上游提交和许可证；
2. 重新应用并复核本地补丁；
3. 更新 `UPSTREAM_VERSIONS.yaml`；
4. 重建环境并完成训练、rollout 与 eval 冒烟测试。
