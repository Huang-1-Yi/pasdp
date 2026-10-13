卡住了，当前输出：“(base) hy@hy:~/dp_pro6000$ bash diffusion_policy/envs/realword/fasterwam/run_fold100.sh
Selected physical GPU: 1 (model device cuda:0)
[Device Check] FasterWAM 真机启动前检查 ...
[Device Check] RealSense detected=['248622300313', '352122274349', '409122274331']
[Device Check] camera role=base serial=248622300313: OK
[Device Check] camera role=left_wrist serial=352122274349: OK
[Device Check] camera role=right_wrist serial=409122274331: OK
[Device Check] left robot 192.168.0.168:4242: OK
[Device Check] right robot 192.168.0.168:4242: same endpoint, skipped
[Device Check] FasterWAM 真机启动前检查通过。
已初始化
已初始化
”


# dp_pro6000 + FasterWAM 真机入口

本目录是 `diffusion_policy.envs.realword.fasterwam` 部署子包，与上层共享
真机环境、相机和控制器。Python 入口从项目根目录通过 `python -m` 启动，
以支持包内相对导入；不要直接按 `.py` 文件路径执行。

FasterWAM 模型代码、权重和语言缓存仍位于外部 `/home/hy/FasterWAM`，
运行环境仍为 Linux 上的 `fasterwam_real` Conda 环境。

这个目录直接沿用 `dp_pro6000` 的双臂 Franka 真机框架，只替换动作来源：

```text
原版：T265/编码器 -> 14维绝对动作 -> DualFrankaRealEnv.exec_actions()
现在：FasterWAM   -> 14维绝对动作 -> DualFrankaRealEnv.exec_actions()
```

以下底层文件没有为本入口修改：

- `diffusion_policy/envs/realword/dual_franka_real_env.py`
- `diffusion_policy/envs/realword/franka_interpolation_controller.py`
- `diffusion_policy/envs/realword/multi_realsense.py`
- `diffusion_policy/envs/realword/single_realsense.py`
- 原始 `demo_dual_udp_teleop_franka_v3.py`

## 文件

- `__init__.py`：轻量包入口，导入包时不加载模型或初始化设备。
- `run_fold100_fasterwam_franka.py`：由原双臂遥操脚本复制得到的真机入口。
- `policy_adapter.py`：唯一的模型侧适配层，负责加载 FasterWAM、整理 observation、返回 `[H,14]` 动作。
- `run_fold100.sh`：可直接运行的启动脚本。
- `smoke_model_load.py`：只加载模型和固定语言缓存，不启动机械臂或相机。

## 已固定的模型资源

```text
checkpoint:
  /home/hy/FasterWAM/checkpoints/fold100_action/step_010000_merged.pt
dataset stats:
  /home/hy/FasterWAM/checkpoints/fold100_action/dataset_stats.json
text embedding:
  /home/hy/FasterWAM/checkpoints/fold100_action/text_embeddings/c5e39e32e704fe7a533eb4296adeffb7e910f86911713503866679c5afc646d6.t5_len128.wan22ti2v5b.pt
instruction:
  fold the T-shirt
```

运行时设置了 `create_text_embedding_if_missing=False`，因此只会读取现有语言
embedding，不会加载 T5，也不会临时重新编码文本。

基础 Wan/DiT/VAE 权重固定从 `/home/hy/FasterWAM/checkpoints` 读取，并设置
`DIFFSYNTH_SKIP_DOWNLOAD=true`；部署时不会尝试从 Hugging Face 临时下载。

## 启动

确认 `192.168.0.168:4242` 的双臂服务运行、三台相机连接、机械臂已归位后：

```bash
cd /home/hy/dp_pro6000
bash diffusion_policy/envs/realword/fasterwam/run_fold100.sh
```

脚本默认通过 `nvidia-smi` 选择已用显存最少的物理 GPU；如需手动指定：

```bash
CUDA_VISIBLE_DEVICES=0 bash diffusion_policy/envs/realword/fasterwam/run_fold100.sh
```

启动后默认是 `WAM PAUSED`，不会执行模型生成动作。模型与设备全部 ready 后：

- `E`：开始/暂停 FasterWAM；
- `C`：开始录制并开启 FasterWAM；
- `S`：停止录制并暂停 FasterWAM；
- `H/R/P`：保留原脚本的三种归位方式；
- `G`：手动切换两个夹爪；
- `1/2/3`：切换相机画面；
- `Q`：退出。

入口带单实例锁 `/tmp/dp_pro6000_fasterwam_real.lock`，重复启动会直接拒绝，
避免多份程序争抢 RealSense 或同时控制机械臂。

## 模型加载 smoke

```bash
cd /home/hy/dp_pro6000
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam_real
export PYTHONPATH=/home/hy/dp_pro6000:/home/hy/FasterWAM:/home/hy/FasterWAM/src:$PYTHONPATH
export DIFFSYNTH_MODEL_BASE_PATH=/home/hy/FasterWAM/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export CUDA_VISIBLE_DEVICES=1
python -m diffusion_policy.envs.realword.fasterwam.smoke_model_load
```

预期输出包含：

```text
FASTERWAM_MODEL_READY ...
MODEL_LOAD_SMOKE_OK=true ...
```

这个 smoke 会使用之前保存的三相机预览图和只读状态执行一次完整动作前向，
但不会连接相机、机械臂或发送任何运动命令。

## 停止用户 `hy` 的全部 Python 进程

下面是全量停止命令，会同时终止 `hy` 名下的训练、推理、Jupyter、Python
启动的终端以及其他 Python 任务；它不是只停止 FasterWAM。不要在仍需保留其他
Python 任务时执行。

先查看将受影响的进程：

```bash
pgrep -u hy -af '[p]ython'
```

先发送可正常清理资源的 `TERM`：

```bash
pkill -TERM -u hy -f '[p]ython'
```

等待数秒后检查；只有仍有无法退出的残留进程时，才强制停止：

```bash
pgrep -u hy -af '[p]ython'
pkill -KILL -u hy -f '[p]ython'
```

## 2026-09-17：Conda OpenCV 环境修复与联合验证

`fasterwam_real` 原先使用 pip 的 `opencv-python 4.6.0.66`。在整套
`dp_pro6000 + FasterWAM` 启动时，相机子进程会在首帧 ready 之前卡住，表现为
终端只打印两次“已初始化”，模型也尚未开始加载。三路相机单独及同时采集均正常，
原 `DualFrankaRealEnv`、`MultiRealsense`、`SingleRealsense` 和
`MultiCameraVisualizer` 文件也没有修改。

现已将该环境切换为 Conda `py-opencv=4.6`（OpenMP 构建）：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate fasterwam_real
pip uninstall -y opencv-python
conda install --override-channels \
  -c https://mirrors.ustc.edu.cn/anaconda/pkgs/main \
  -y py-opencv=4.6 opencv=4.6
```

验证版本：

```text
cv2: 4.6.0 (Conda, Parallel framework: OpenMP)
numpy: 1.26.4
torch: 2.7.1+cu128
GPU: NVIDIA GeForce RTX 4090
```

整套联合启动已经通过，关键输出为：

```text
[DualFrankaRealEnv] ready realsense_ready=True ... multi_cam_vis_alive=True env_ready=True
FASTERWAM_MODEL_READY checkpoint=/home/hy/FasterWAM/checkpoints/fold100_action/step_010000_merged.pt ...
准备完成：dp_pro6000 双臂真机框架 + FasterWAM。
```

验证时自动选择物理 GPU 1，模型加载后显存约为 12.4 GiB。程序默认保持
`WAM PAUSED`，未发送动作。不要在启动脚本中全局设置
`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1` 等变量；原 dp 相机子进程已有自己的
线程控制逻辑。
