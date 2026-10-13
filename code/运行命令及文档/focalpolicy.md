> **2026-07-28 本地审计说明**：下文是旧 AI 生成的原始草案，保留用于追溯，
> 但不要直接整段执行。当前工程的权威流程是 `v8升级指令.md` 和
> `conda_environment_sim_torch27_v8.yaml`。已确认的差异包括：不安装整份
> FocalPolicy、不安装 `torch-dct`、不预先降级 `setuptools`，mujoco-py 编译也
> 不需要 `CUDA_HOME` 或 `TORCH_CUDA_ARCH_LIST`；第三方源码统一从本项目
> `third_party/` 以 `--no-deps` 安装。

基于刚才在服务器上跑通的命令，换成 2080Ti 本机版可复现流程（历史草案）

# ============ 0. 前置：系统级依赖（只需做一次） ============
# 0.1 MuJoCo 2.1.0 C 库（免 sudo, 装到家目录）
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xzf mujoco210.tar.gz && rm mujoco210.tar.gz

# 0.2 mujoco-py 编译需要 glew 头文件（唯一硬 sudo）
sudo apt-get install -y libglew-dev libegl1-mesa-dev


# ============ 1. 用 v6 yaml 建环境（大改前确认一下文件名） ============
cd /home/hy/dp_pro6000
mamba env create -f conda_environment_sim_torch27_v6.yaml -n focal_py310
mamba activate focal_py310
# 检查 torch 在 2080Ti (sm_75) 上可用:
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"


# ============ 2. 装第三方源码包（全部 --no-deps, 不写进 yaml） ============
# 2.1 build tools (mujoco-py 编译要)
python -m pip install --no-deps setuptools==59.5.0 'Cython<3' patchelf==0.17.2.0

# 2.2 FocalPolicy editable
cd /home/hy/focalpolicy/FocalPolicy && python -m pip install --no-deps -e .

# 2.3 torch-dct (cu128 自带 sm_75 cuFFT, 2080Ti 直接 GPU 可用)
python -m pip install --no-deps torch-dct==0.1.6

# 2.4 mujoco-py (需 CUDA toolkit 12.8 + mujoco210)
cd /home/hy/focalpolicy/third_party/mujoco-py-2.1.2.14
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
# RTX 2080Ti = sm_75
export TORCH_CUDA_ARCH_LIST='7.5'
python -m pip install --no-deps --no-build-isolation -e .
# 验证:
MUJOCO_GL=egl python -c "import mujoco_py; print('mujoco_py ok')"

# 2.5 Metaworld (py3.10 兼容性风险点, --no-deps 绕开 setup.py 版本约束)
cd /home/hy/focalpolicy/third_party/Metaworld && python -m pip install --no-deps -e .
# 验证 py3.10 兼容:
MUJOCO_GL=egl LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia python -c "
import metaworld
env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE['door-close-v2-goal-observable']()
env.reset(); print('metaworld ok')
"

# 2.6 rrl-dependencies (仅 Adroit 任务需要)
cd /home/hy/focalpolicy/third_party/rrl-dependencies
python -m pip install --no-deps -e mj_envs/.
python -m pip install --no-deps -e mjrl/.
# 验证:
MUJOCO_GL=egl LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia python -c "
from mjrl.utils.gym_env import GymEnv
from focal_policy_3d.env.adroit import AdroitEnv
print('adroit ok')
"


# ============ 3. 常用 env 包装（写进 ~/.focal_env 或手动 export） ============
# 不写 ~/.bashrc, 仅在跑 FocalPolicy 时设:
export FOCAL_ENV="MUJOCO_GL=egl LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/x86_64-linux-gnu:/usr/lib/nvidia:$LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0"


# ============ 4. 训练（沿用服务器跑通的 launcher, 已在仓库内） ============
# launcher_focal_train.py / run_train.sh 已就位, scp 过来或重新生成:
cd /home/hy/focalpolicy
# 把服务器跑通的两个脚本拉到本机 (如果还没):
# scp 10.242.5.12:/media/disk7t/policy_raw/focalpolicy/{launcher_focal_train.py,run_train.sh} .

# 一行启动后台训练:
eval $FOCAL_ENV && ./run_train.sh door_close_local 100 10 10 128 8
tail -f /home/hy/focalpolicy/data/outputs/run_door_close_local.log



# 三大风险点（命令里已标）
2.4 mujoco-py 编译：若 TORCH_CUDA_ARCH_LIST='7.5' 不对、或本机 /usr/lib/nvidia 不存在，builder.py _ensure_set_env_var 会报 missing path。错误信息里"Now values"行告诉我们缺啥，加进 LD_LIBRARY_PATH 即可。
2.5 Metaworld py3.10 兼容：很可能命中 from collections import Mapping、collections.Iterable 等已删 API。错误会指向具体文件，按提示改 collections.abc 即可（不改 yaml）。
2.6 Adroit：若你只跑 MetaWorld，2.6 可跳过。Adroit 兼容性更碎，先跑 MetaWorld 验证主链。
