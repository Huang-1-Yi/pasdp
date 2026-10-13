# 双 Aubo + T265 遥操

按 Franka v3 结构组织的独立入口为 `demo_dual_teleop_aubo_v2.py`，
搭配 `aubo_interpolation_controller_v2.py` 和 `dual_aubo_real_env_v2.py`。
阅读顺序、按键和离线对照见 [README_v2.md](README_v2.md)。

入口为项目根目录 `demo_dual_teleop_aubo.py`。保留原框架的共享内存输入、
机器人插值控制、RealSense RGB 摄像头、视频和 Zarr episode 录制。
现场包中的坐标标定、四元数相对映射、六关节回零逻辑已提取到主框架，
运行时不再引用 `aubo_t265_teleop_bundle` 的 Python 代码或目录。
原 `demo_dual_udp_teleop_franka_v3.py` 保留。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `demo_dual_teleop_aubo.py` | CLI、启动、键盘、预览与录制调度 |
| `config.py`、`config/dual.json` | IP、设备编号、标定、限速、相机与 I/O 配置 |
| `calibration.py` | 重力与前方构建基坐标方向、T265 手势标定、安装变化校验 |
| `pose_mapper.py` | 四元数相对姿态、左右轴符号、平滑与位移/旋转限幅 |
| `teleop_input.py` | 复用原 sender，按真实设备采样时间判断输入有效性 |
| `runtime.py` | 暂停、遥操、回零状态机与独立输出开关 |
| `workspace.py` | 使用实测 `world_from_base` 检查共同世界坐标中的 TCP 边界 |
| `../aubo_interpolation_controller.py` | xyz+rotvec 转 SDK RPY、RTDE、IK、servoJoint、回零和 I/O |
| `../dual_aubo_real_env.py` | 双臂生命周期、动作检查、观测对齐、相机与 episode |
| `../aubo_homing.py`、`../aubo_gripper.py` | 六关节五次曲线回零、数字输出适配 |
| `sdk_paths.py`、`sdk/` | 局部加载现场 SDK，不改变系统 Python 安装 |

## 运行环境

实机使用 Linux x86_64、CPython 3.9。`sdk/` 从现场包保留了
`pyaubo_sdk 0.26.0rc6` 与支持 T265 的 `pyrealsense2 2.53.1.4623`；
它们优先于当前 Python 环境中的同名包加载。
包内包含 Linux 原生 `.so`，Windows 上可进行离线测试和配置检查，不能运行实机入口。
使用现有 `conda_environment_real.yaml` 环境中的 NumPy、SciPy、atomics、pynput、
OpenCV、PyAV、Zarr、numcodecs、threadpoolctl；sender 还需要 `pyserial`。
SDK 来源及文件摘要见 `sdk/SOURCE.json`。
键盘监听需要桌面会话；`--headless` 仅关闭相机窗口。

从项目根目录运行：

```bash
# 无网络、SDK、摄像头和输出操作
python demo_dual_teleop_aubo.py --check-config
python demo_dual_teleop_aubo.py --check-config --enable-io

# 默认只读：读取机器人状态和 T265，打印映射目标，不启动伺服
python demo_dual_teleop_aubo.py --calibrate-on-start

# 运动：先做当前跟踪会话的方向标定，然后启动双臂并保持暂停
python demo_dual_teleop_aubo.py --enable-motion --calibrate-on-start

# 接线完成后，显式启用三路控制柜输出，并录制新的 Aubo 数据集
python demo_dual_teleop_aubo.py --enable-motion --enable-io \
  --calibrate-on-start --output data/aubo_demo
```

也可运行 `bash diffusion_policy/envs/realword/aubo_t265/scripts/run_dual_teleop.sh ...`。
用 `TELEOP_PYTHON=/path/to/python` 指定环境；脚本从任意工作目录定位项目。
`--config` 可传独立配置，配置中的标定和 home 文件路径相对于该 JSON。
`--duration` 限定运行秒数，`--no-cameras` 关闭 RGB 摄像头。
默认 `cameras.serial_numbers=[]`，配置实际 RGB 序列号后才会启动相机；不得填 T265 序列号。

## 安装坐标与标定

左臂 `192.168.3.200` / T265 `224622111506`；
右臂 `192.168.2.200` / T265 `121222110870`。
左右标定与六关节 home 来自现场包，并按 IP、序列号绑定。
这些记录是现场配置快照，现场仍需核对安装方向和 home 对应的实际姿态。

公开位姿始终为各机器人自身基坐标中的 `[x,y,z,rx,ry,rz]`，单位 m/rad；
`rx,ry,rz` 是旋转向量。只有 SDK 边界转换成 RPY。
人的前/左/上方向由 `operator_basis_base=B`、`operator_basis_t265=C` 定义，
映射旋转为 `B @ C.T`。重力用于确定“上”，不能单独确定“前”。
左臂沿用现场实际代码的平移与旋转 X/Y 反号，右臂三轴正号，均可分别配置。

`--calibrate-on-start` 在同一个 sender、同一个 T265 跟踪会话中收集左右各两次
水平手势，更新 C 与映射矩阵，随后继续使用该会话。每次手势保持姿态，移动约
10–30 cm 并保持至采集结束；检查实际动作方向后再扩大运动范围。
T265 重启后的跟踪世界偏航可能变化，无法仅通过机器人重力检测。
直接复用已保存 C 时需要确认方向，推荐新会话使用此选项。

机械安装发生变化时，用实测的机器人基坐标“人前方”向量更新对应臂：

```bash
python demo_dual_teleop_aubo.py --calibrate left --forward-base X Y Z
```

此命令读取安装参数和 T265，不启动伺服。独立标定命令会关闭跟踪会话，
后续运行推荐再用 `--calibrate-on-start` 对齐当前 T265 世界。
按 Space 的会话锚点只重新对齐手与当前 TCP，不替代安装标定。

两台机器人的共同世界平移外参尚未提供。当前默认启用的是 `calibrated_base` 模式（见文末说明）：
每臂以自己的基座为原点沿保存的标定 Z 限制高度，下限读取所选 JSON 的 `max_downward_distance`，余量为 0。
v2 的实时目标、RTDE 与归位/伺服关节目标均有高度检查，详见 `README_v2.md`。
若改用统一世界边界，仍必须为两臂填实测的 4×4 `world_from_base`，并设置
`minimum_world_z` 或 `lower_world_xyz` / `upper_world_xyz`。
检查对象为 TCP，不提供双臂、躯干、连杆与工具的碰撞规划。
默认参数对齐现场 `run_quaternion_home_dual_teleop.sh` 和 `current_dual_config/dual.json`。
两臂 `no_range_limit=true`，关闭相对起点平移/旋转范围限幅，速度为 1 cm/s、10°/s。
`motion.max_translation=0.05`、`max_rotation_deg=10` 仅在对应臂 `no_range_limit=false` 时生效。

`config/dual.json` 的 `motion` 配置遥操映射：比例 0.8/1、位置/旋转滤波系数 0.35/0.7、
死区 0.0002 m。滤波系数按 `teleop_frequency` 转换为时间常数，50 Hz 时约 0.04643/0.01661 s。
`controller` 独立配置机器人插值与伺服：平移插值速度上限 0.25 m/s、旋转 10°/s，
`servo_t=0.02`、`servo_a=3.5`、`servo_v=2`、`lookahead_time=0.05`、`gain=120`、
`speed_fraction=0.4`、`max_joint_step=0.05` rad、`launch_timeout=12` s。
`command_latency=0.04` 是目标调度提前量，与 `servo_t` 分开设置。

现场兼容配置使用 `max_joint_speed=null`、`max_following_error=null`、`reject_joint_jumps=false`，
不启用原包没有的额外关节速度检查、笛卡尔跟踪误差检查与跳变拒绝；仍按原包的关节步长限幅。
可填写正数阈值、启用跳变拒绝。公共控制器类的默认检查保持原有值，配置仅影响此入口。

## 按键与三路 I/O

| 按键 | 行为 |
| --- | --- |
| Space | 重新锚定后恢复遥操 / 暂停并结束当前 episode |
| H | 两臂按现场记录六关节姿态回零；回零结束后保持暂停 |
| V | 左臂控制柜 DO00 在高/低间切换，置高用于吸取 |
| B | 左臂控制柜 DO01 在高/低间独立切换 |
| N | 右臂控制柜 DO00 在高/低间切换 |
| C / S | 开始 / 保存 episode；开始时必须正在遥操且指定 `--output` |
| Backspace | 删除最近 episode（录制中先结束并删除该 episode） |
| 1 / 2 / 3 | 切换 RGB 相机视图 |
| Q / Esc | 保存当前有效 episode 并退出 |

V/B/N 每次物理按下切换一次，过滤系统按住按键产生的重复事件。
I/O 需要 `--enable-io` 且正在遥操；暂停和回零时不接受开关操作。
两路左臂输出可以同时为高，不假设它们互斥，也不自动联动。
启动只读取输出电平；进入遥操后保持读到的电平，按键才改变指定输出。
暂停、输入丢失、退出均保持已接受的输出电平，不自动释放吸取物。
软件不设置工具端电压或更改引脚方向。
输出状态来自控制柜锁存值与命令后的读回，不是吸力、夹爪宽度或抓取成功反馈。

配置默认禁用两臂输出，预留 `mode=independent`、左 `pins=[0,1]`、右 `pins=[0]`。
`--enable-io` 将类型设为 `standard_digital`；需完成接线再使用。
适配层还支持 `tool_digital` 以及 `mode=binary` 的自定义开/闭电平表。
[AUBO IoControl API](https://docs.aubo-robotics.cn/arcs_api/en/group__IoControl.html)
为控制柜标准数字输出接口依据。

输入采用设备采样时间，不把重复发布旧 pose 当成有效更新；
任一 T265 超时或置信度低于 2，双臂暂停并结束 episode，恢复后按 Space 重锚定。
回零期间 Space 或输入丢失会退出并停止双臂。
任一机器人进程退出、RTDE 超时或命令看门狗超时都会停止本次双臂会话；
IK 跳变拒绝及额外跟踪误差检查由 `controller` 配置决定。
回零是受速度/加速度限制的关节轨迹，需要现场确认路径可通过。
本启动器用 IP 文件锁防止自己的两个实例重复控制同一控制柜。

## 录制格式

`replay_buffer.zarr` 保存对齐的观测、动作和 stage；
`videos/<episode>/` 保存每台 RGB 相机的 MP4 与 `metadata.json`。
metadata 保存实际标定内容、配置、动作维度、按通道展开的动作顺序及 I/O 状态类型。
默认只有视频；`--record-images` 还在 Zarr 中保存 RGB `img_camera_<i>`。

未启用 I/O：12 维 `[left_pose6, right_pose6]`。
当前三路 I/O 启用后：15 维
`[left_pose6, left_DO00, left_DO01, right_pose6, right_DO00]`，各输出为 0/1。
观测 `left_gripper_state` 为两路电平，`right_gripper_state` 为一路电平。
机器人子进程内部用位掩码传输多路电平，公共动作与录制数据按通道展开。
自定义两臂 binary 配置仍为 14 维。
这三种格式使用不同数据目录，不能与原 Franka 连续夹爪宽度数据混合训练。
已有目录的字段、形状、类型不一致时拒绝追加，不自动清空数据。

## 离线验证

```bash
python -m unittest discover -s diffusion_policy/envs/realword/aubo_t265/tests -v
python demo_dual_teleop_aubo.py --check-config
bash -n diffusion_policy/envs/realword/aubo_t265/scripts/run_dual_teleop.sh
```

测试使用机器人和 I/O 替身，不连接网络、T265 或控制柜。
包括 SO(3) 坐标与方向绑定、四元数连续性与限幅、独立设备超时、关节回零、
双臂动作预检查、输出开关与录制 schema。实机 SDK 加载、双臂伺服与摄像头录制
仍需要在现场 Linux 环境验证。


## 当前高度参考：固定标定 Z 轴（2026-10-09）

两份默认配置现选择 `workspace.reference="calibrated_base"`。每臂从对应
`left_calib.json` / `right_calib.json` 的 `operator_basis_base` 读取第三列：

```text
z_base = operator_basis_base[:, 2]
h = z_base · p_base
p_goal = p_base + max(0, minimum_height(side) - h) * z_base
最后覆盖该臂的归位记录姿态
```

高度零点仍是各臂 BASE 原点，下限读取所选 JSON 并由现场调试确定；
没有新增桌面原点或重算水平姿态。当前标定 Z 与保存的重力反方向一致，
但高度轴现在固定取自标定文件，不使用启动读取的重力覆盖，也不逐帧更新。
重力读取和安装核对仍执行。两种模式共用目标修正、真实低反馈恢复和姿态覆盖顺序。

旧的 `gravity_base` 模式继续支持，显式选择它才会在启动时用柜体安装重力更新方向；
`world` 模式保持原有 4×4 外参计算。每轮仍是点积和必要的向量加法，
不增加 IK/FK/SDK 调用。`tests/test_calibrated_height.py` 离线验证新模式、
左右独立投影、固定轴、控制器恢复和姿态锁定。

## v3 当前归位与现场参数（2026-10-09）

- 启动保持当前状态，按 H 完成首次关节归位后，按 E 开启固定记录姿态遥操。
- v3 归位倍率为 2：规划时长减半；后续 H 使用固定姿态 XYZ 归位。
- 启动和 H 均恢复左 DO00/DO01=LOW、右 DO00=HIGH；H 完成时确认实际输出锁存。
- 高度以所选 JSON 为准，不在测试或控制器中固定默认数值；两份配置可以不同。
  本次 `dual.json` 快照为左 −580 mm、右 −535 mm，后续现场调试继续确定。

详见 `README_v3.md` 和 [本次现场配置与改动记录](../../../../运行命令及文档/AUBO双臂遥操作_v3_现场配置与改动记录_2026-10-09.md)。
