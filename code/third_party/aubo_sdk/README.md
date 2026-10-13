# AUBO SDK 手动安装与双臂控制

本目录以压缩包形式保存 AUBO C/C++ SDK，避免在 Git 工作区中展开近千个 SDK 文件：

- SDK 版本：`0.26.0-rc.7+f8b4ae3`
- 平台：Linux x86_64
- 压缩包：`aubo_sdk_0.26.0-rc7.zip`
- 解压后包含：`include/`、`lib/`、`share/example/` 和 `VERSION`

当前双臂映射：

| 设备 | IP | 端口 |
| --- | --- | --- |
| 左 AUBO | `192.168.3.200` | RPC `30004`、RTDE `30010` |
| 右 AUBO | `192.168.3.199` | RPC `30004`、RTDE `30010` |
| 右 WSG 夹爪 | `192.168.3.100` | TCP `1000`，当前未接入双臂控制代码 |

## 1. Python SDK 与本目录的关系

仅复制本目录不能提供 Python 模块。Python 控制程序还需要安装
`pyaubo_sdk` wheel。官方手册说明 Linux x86 支持 Python 3.6～3.11；本项目的
`robodiff` 环境是 Python 3.9，因此应优先使用同时满足以下条件的 wheel：

- Linux x86_64；
- CPython 3.9，文件名通常包含 `cp39`；
- 尽量与本目录的 `0.26.0-rc.7` SDK/控制柜版本一致。

公开 PyPI 的最高稳定版是 `0.24.1`；加上预发布版本后，`0.26.0` 系列最高为
`0.26.0rc6`，暂时没有与本目录完全一致的 `0.26.0rc7`。本机 Python 3.9、Linux
x86_64 应选择：

```text
pyaubo_sdk-0.26.0rc6-cp39-cp39-manylinux2014_x86_64.whl
```

它和 C/C++ SDK 只相差一个 RC 版本，是当前最接近的公开 Python wheel。安装后仍须先
做只读接口检查，再决定是否进入运动模式。

## 2. 手动安装 pyaubo_sdk wheel

进入项目环境：

```bash
cd /home/robot/lightumi/dp_pro6000
conda activate robodiff
python --version
python -c 'import platform; print(platform.machine(), platform.system())'
```

期望输出为 Python 3.9、`x86_64 Linux`。

如果让 pip 自动下载并选择上述 Python 3.9 Linux x86_64 wheel：

```bash
python -m pip install --pre 'pyaubo-sdk==0.26.0rc6'
```

如果已经从 PyPI 手动下载到 `~/下载/`，则执行：

```bash
python -m pip install ~/下载/pyaubo_sdk-0.26.0rc6-cp39-cp39-manylinux2014_x86_64.whl
```

安装后验证：

```bash
python -c 'import importlib.metadata as m; import pyaubo_sdk; print(m.version("pyaubo-sdk")); print(pyaubo_sdk.__file__)'
```

如果导入时报找不到 `libaubo_sdk.so`，先把 C/C++ SDK 解压到仓库外的固定目录：

```bash
mkdir -p /home/robot/.local/aubo_sdk-0.26.0-rc7
unzip -q \
  /home/robot/lightumi/dp_pro6000/third_party/aubo_sdk/aubo_sdk_0.26.0-rc7.zip \
  -d /home/robot/.local/aubo_sdk-0.26.0-rc7
```

然后在同一个终端执行：

```bash
export AUBO_SDK_ROOT=/home/robot/.local/aubo_sdk-0.26.0-rc7
export LD_LIBRARY_PATH="${AUBO_SDK_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
python -c 'import pyaubo_sdk; print(pyaubo_sdk.__file__)'
```

上述 `export` 只对当前终端有效。先不要写入 shell 启动文件，等实机验证通过后再决定
是否长期配置。

不要把 `aubo_sdk_0.26.0-rc7.zip` 交给 pip 安装：它是 C/C++ SDK 压缩包；Python
模块仍应使用上面的 `.whl` 安装。

## 3. 运行前网络检查

电脑控制网口应位于 `192.168.3.0/24`。检查两台机械臂：

```bash
ping -c 2 192.168.3.200
ping -c 2 192.168.3.199
nc -vz -w 2 192.168.3.200 30004
nc -vz -w 2 192.168.3.200 30010
nc -vz -w 2 192.168.3.199 30004
nc -vz -w 2 192.168.3.199 30010
```

运行 Python 状态检查或控制程序前，关闭其他可能登录控制柜的 AUBO RPC 客户端，避免
“最后登录的客户端成为控制者”导致控制权切换。

## 4. 只读状态检查

以下命令会登录两台控制柜并读取模式、Safety 状态、关节角和 TCP 位姿；不会调用
`poweron()`、不会进入伺服模式，也不会发送运动目标：

```bash
cd /home/robot/lightumi/dp_pro6000
conda activate robodiff
python tests/aubo_keyboard_teleop.py
```

只有左右两臂都显示预期的机器人名称、`Running` 和正常 Safety 状态后，才进入运动测试。

## 5. 双臂键盘控制

首次建议用较低的连续速度进行验证：

```bash
cd /home/robot/lightumi/dp_pro6000
conda activate robodiff
python tests/aubo_keyboard_teleop.py \
  --enable-motion \
  --translation-speed-mm-s 20 \
  --rotation-speed-deg-s 10 \
  --robot-frequency 200 \
  --servo-t 0.02 \
  --keyboard-frequency 10 \
  --command-horizon 0.1 \
  --verbose
```

程序会再次显示左右 IP，并要求人工确认后才进入伺服保持。默认不允许程序自动上电或松
刹车；如果机械臂不是 `Running`，程序会退出。

控制器会等待有效 RTDE 状态，并先发送当前关节角的零位移保持命令。只有两臂都接受该
命令后才报告 ready；接口偶发返回 `-13 (Request was ignored)` 时，仅有限次数重新选择
伺服模式并重试同一保持命令，不会自动解除 Protective Stop。首次修复验证先只按 `P`
确认状态不是全零，然后按 `Q` 退出；再次运行后再从单臂低速平移开始。

与参考程序一致，默认 home 是程序启动时读到的 TCP 位姿。启动前先用示教器把双臂放到
期望 home；进入键盘控制后按 `H`，两臂会按当前平移/旋转步长分段返回该启动 home。

只有在现场人员确认确实需要由程序上电时，才追加：

```bash
--power-on
```

键盘映射：

| 按键 | 功能 |
| --- | --- |
| `1` / `2` / `3` | 选择左臂 / 右臂 / 双臂 |
| `W` / `S` | 基坐标系 `+X` / `-X` |
| `A` / `D` | 基坐标系 `+Y` / `-Y` |
| `R` / `F` | 基坐标系 `+Z` / `-Z` |
| `I` / `K` | `+Rx` / `-Rx` |
| `J` / `L` | `+Ry` / `-Ry` |
| `U` / `O` | `+Rz` / `-Rz` |
| `H` | 双臂分段返回程序启动时记录的 home |
| `P` | 打印两臂状态 |
| `Space` | 以当前实测位姿重新保持 |
| `Q` 或 `Esc` | 退出伺服模式并断开连接，不执行下电 |

安全限制：

- 键盘程序不限制 TCP 相对启动位置的工作范围；
- 使用 `pynput` 跟踪按下/松开，默认以 10 Hz 更新末端目标，不依赖终端按键连发；
- `--robot-frequency` 独立设置 IK/`servoJoint` 循环，默认 200 Hz；
- `--servo-t 0.02` 沿用官方 5 ms `servoJoint` 示例中的 `t` 参数；
- 控制速度比例默认为 20%；
- 控制器保留最大笛卡尔速度和单周期关节增量保护，避免 IK 跳变造成突发运动；
- 不会自动解除 Protective Stop；
- 任一机械臂的逆解或 `servoJoint` 返回异常时，控制进程退出伺服模式。

软件退出不能替代实体急停。首次运动时必须有人在现场，保持急停可立即触达，并确保两臂
及末端工具周围没有人员、线缆或障碍物。

## 6. 相关代码

- `diffusion_policy/envs/realword/aubo_interpolation_controller.py`：单臂 RPC、RTDE、
  逆解和 `servoJoint` 控制。
- `diffusion_policy/envs/realword/dual_aubo_real_env.py`：左右控制柜协调与双臂动作接口。
- `tests/aubo_keyboard_teleop.py`：只读状态检查和有界键盘控制入口。
