# Aubo v2：按 Franka 框架阅读与运行

`demo_dual_teleop_aubo_v2.py` 使用 Franka v3 的主流程组织方式，
数值控制仍采用现场 Aubo 四元数遥操逻辑。v1 入口和控制器保留。

## 先读入口，再读环境和控制器

| Franka 参考 | Aubo v2 对应 | 作用 |
| --- | --- | --- |
| `demo_dual_udp_teleop_franka_v3.py` | 根目录 `demo_dual_teleop_aubo_v2.py` | 初始化、观测、按键、目标、动作下发、定时 |
| `_capture_teleop_anchor` | v2 同名函数及 `refresh_anchor` | 当前手部与实际 TCP 建立锚点 |
| `_target_from_anchor` | v2 同名函数 | 调用已验证的 Aubo 四元数映射器 |
| `DualFrankaRealEnv` | `../dual_aubo_real_env_v2.py` | 左右控制器、观测对齐、相机和 episode |
| `FrankaInterface` | `../aubo_interpolation_controller_v2.py` 的 `AuboInterface` | 机器人 SDK 通信边界 |
| `FrankaInterpolationController.run` | `AuboInterpolationControllerV2.run` | 控制子进程内的插值与下发循环 |

入口 `run()` 按编号展开：

1. 核对安装并启动共享内存 T265 sender。
2. 创建双臂 v2 环境，传入同一份 JSON 的伺服参数。
3. 初始化映射器，以及 `teleop_enabled`、`homing`、`lock_rot`、`is_recording`、`stage`。
4. 每轮获取输入和观测。
5. 处理按键。
6. 明确区分回零、暂停和遥操；遥操分支计算 `left_target`、`right_target`。
7. 构造 action，调用 `env.set_teleop_target(action, stage)` 覆盖最新位置目标。
8. 更新图像和诊断。
9. `precise_wait` 等待下一周期。

v1 的 `TeleopSession.tick()` 没有用于 v2 主循环，目标构造和下发可以直接阅读。
`QuaternionPoseMapper` 继续复用，避免重写已通过对照的坐标变换和滤波。

## 控制器如何对应 Franka

```text
父进程：env.set_teleop_target
    → 左/右 set_teleop_target
    → 单槽共享位置快照（覆盖旧目标，不传未来到达时间）
父进程：env.set_gripper
    → SharedMemoryQueue（逐条保留 I/O 事件，不随位置合并）
子进程：AuboInterpolationControllerV2.run
    → _process_commands：STOP/HOLD/HOME/I/O/heartbeat，以及最新位置快照
    → 实时位置直接取最新 xyz + 固定归位姿态；策略 waypoint 使用插值器
    → _target_joints：RTDE 核对、回零/保持，或 RPY 转换 + IK
    → _send_joints：servoJoint 与返回码处理
    → _publish_state：真实反馈写入 SharedMemoryRingBuffer
父进程：env.get_obs → 观测对齐、图像与录制
```

Franka 在接口中发送笛卡尔阻抗目标，Aubo 则先做 IK，再发送六关节伺服目标。
因此不能直接复制 Franka 的 1000 Hz、七关节 home、夹爪宽度或工具变换。
倾斜安装使用原 Aubo 的左右标定 B/C、轴符号和 `xyz+rotvec` 公开位姿约定。

v2 控制器继承 v1 的初始化、进程 start/stop、共享内存队列/缓冲区、
命令入队和 SDK/RTDE 启动辅助方法；`AuboInterface`、控制状态和 `run()` 在 v2 文件中实现。
v2 环境复用 v1 的观测/录制逻辑，通过显式 `controller_class` 注入 v2 控制器。
这个工厂参数默认仍选择 v1，不修改模块全局变量，不切换原入口的控制器。

## 按键与运行

| 按键 | v2 行为 |
| --- | --- |
| E | 开启/暂停遥操，开启时重新锚定 |
| Z | 重新对齐手部与实际 TCP，保持当前遥操开关状态 |
| T | 确认强制姿态锁定并刷新 XYZ 锚点，不解锁旋转 |
| R | 开始录制，并重新锚定、开启遥操 |
| S | 保存当前 episode 并暂停遥操 |
| H | 按现场六关节记录回零，完成后暂停 |
| Space | stage 加一；新 episode 从 0 开始 |
| A（或 V） | 左 DO00/DO01 推进四步循环 00 → 01 → 11 → 10 → 00 |
| B | 左数采松开信号，不推进循环 |
| 数采按钮 C+D | 每次按下切换右 DO00；程序响应 C 一次并忽略同次 D |
| Backspace | 删除当前/最近 episode，沿用环境的空 episode 处理规则 |
| 1/2/3 | 切换 RGB 相机预览 |
| Q/Esc | 退出并释放资源，保存有效录制 |

Space 在 v2 中用于 stage，遥操开关改为 E。默认没有 E 键的五秒阻塞倒计时；
保留 Aubo sender 的配置启动延迟。H 使用现有 Aubo home 记录。
Franka 的 R/P 依赖其另一组关节返回位和笛卡尔 home，G 依赖夹爪宽度控制，
这些键没有套用到 Aubo；左柜两路输出按 A/V 四步循环切换，B 松开信号忽略。
右侧数采设备每次按下连续发送 C、D，程序只响应 C 一次来切换右柜 DO00，忽略同次的 D。

```bash
# 离线检查，不加载硬件 SDK
python demo_dual_teleop_aubo_v2.py --check-config

# 只读预览，E 开始查看目标；使用当前跟踪会话做方向标定
python demo_dual_teleop_aubo_v2.py --calibrate-on-start

# 启动双臂 v2，仍初始暂停
python demo_dual_teleop_aubo_v2.py --enable-motion --calibrate-on-start

# 接线完成后启用三路输出和录制；R 开始，S 保存
python demo_dual_teleop_aubo_v2.py --enable-motion --enable-io \
  --calibrate-on-start -o data/aubo_v2_demo
```

默认参数仍从 `config/dual.json` 读取，现场为 50 Hz 遥操、50 Hz 控制/反馈、
10 Hz 观测。实时入口不使用 `command_latency`，不应用映射器速度限制、
Cartesian 速度限制和关节步长/速度限制，也不新增加速度约束。
映射器滤波系数仍读取所选配置；`dual_no_smoothing.json` 的系数为 1，关闭该滤波。
两臂姿态仍固定为各自归位记录。SDK 伺服模式和柜体速度比例读取原配置。

HOLD/STOP 抢占位置目标；HOLD 取消未来排程的 I/O，但保留独立即时 I/O 事件。
位置快照也参与看门狗，旧快照不会反复续期。RTDE、回零与输入失效检查继续执行。
目标覆盖不等于已验证真机平滑；本次没有增加速度/加速度轨迹整形。

策略控制仍使用 `exec_actions(actions, timestamps, stages)` 和 `PoseTrajectoryInterpolator`。
v2 排程的更新和采样统一使用 `now`，避免未来起点截断造成采样前跳；
只丢弃已经到期的 waypoint，不因距离下个周期不足而提前忽略。
伺服与入口循环使用累计周期截止时间，超时跳过错过的周期，不突发补发。

## 沿标定 Z 方向的 TCP 高度限制

两份配置现已启用 `workspace.reference="calibrated_base"`。
每臂以自己的基座原点为零，不使用共同世界坐标的平移外参：

```text
up_base = operator_basis_base[:, 2]
height = dot(up_base, TCP_xyz_base)
downward_distance = -height
```

`max_downward_distance` 按 left/right 配置正数米值，每臂满足 `height >= -max_downward_distance[side]`，允许等号。
`target_margin=0.0`：取消目标余量，每臂所有高度检查统一使用该臂下限；非零余量配置会被拒绝。
这些单位均为米。限制对象是柜体当前 TCP，而不是机械臂的所有连杆。
倾斜安装时不能直接截断 BASE 的 Z；这里只修正标定 Z 方向分量，保留水平分量，最终使用归位记录姿态。
启动时从各臂标定矩阵读取 Z 方向；安装重力仅作核对，不覆盖边界方向。

- 遥操：覆盖目标提交前投影到所选配置中的对应臂下限，显示和录制保存修正后的目标。
- 策略：严格姿态锁定时归一化整批 waypoint（高度修正后覆盖姿态）；未锁定时仍拒绝越界动作。
- 实际反馈：启动进入伺服前和每轮控制都读取真实 TCP；低于该臂下限时进入向上恢复，不截断反馈值。
- 归位：检查归位终点，并在每个待发关节目标上做 SDK `forwardKinematics` 检查。
- 所有 v2 伺服目标：IK/关节步长处理后，再做 FK 高度检查；失败或越界阻止下发。
- 停止请求：若发生在 FK 通信过程中，返回后不会继续发送伺服指令。

这增加了每轮一次 FK 通信，可能影响实际 50 Hz 周期，需要现场测量。
这是软件采样检查，不是连续运动的安全认证。
若必须保证实际 TCP 在任何瞬间都不穿过平面，还需配置 AUBO 控制柜安全平面、
工具 TCP 和与实际速度相符的停止余量。不得把“目标不越界”当作“实际绝不会越界”。
本次只有离线验证，没有连接机械臂或修改柜体安全配置。

离线回归见 `tests/test_gravity_height.py`，覆盖倾斜坐标、目标投影、双臂录制、
实际反馈、FK 错误、归位中间点和停止抢占。旧的统一世界模式仍需要实测 `world_from_base`。

## 离线对照

```bash
python -m unittest discover -s diffusion_policy/envs/realword/aubo_t265/tests -v
```

`test_latest_teleop.py` 检查最新位置覆盖、I/O 事件顺序、STOP/HOLD 抢占、
看门狗、取消软件限速、目标排队 21 ms 后仍被处理以及轨迹更新时的采样连续性。
所有机械臂/SDK 使用替身，不连接真机。历史 `测试/aubo_v2_compare/results`
记录的是旧版排程行为，不能作为当前实时覆盖模式的数值验收结果。


## 实际 TCP 低于各臂下限时自动上移

仅在启用机械臂运动后执行。只读预览不会移动机器人。
初始暂停或运行中检测到低高度时，锁存该时刻的水平位置，姿态最终覆盖为归位记录，沿安装重力反方向建立该臂下限的恢复目标。
例如右臂实际高度 −551 mm，恢复目标为向上移动 1 mm；左臂 −571 mm 同样抬升 1 mm；继续显示、录制真实反馈。

恢复目标优先于普通遥操、旧插值轨迹和关节归位。v2 使用 IK → FK 目标核验 → servoJoint；v3 使用 servoCartesian。
到达边界后退出恢复：暂停时保持已接受的恢复位置；旧插值历史重置为恢复目标；实时人控继续采用最新目标。
归位若被恢复打断，则在恢复后从当前实测状态重新建立固定姿态 XYZ 归位轨迹。

恢复期间不吞并独立 I/O 事件，也不刷新命令看门狗。STOP、反馈失效、IK/FK/SDK 错误仍能终止控制。
待发目标的高度检查依然严格使用对应臂下限，余量为 0。该自动恢复目前经离线验证，未进行真机运动验证。


## 严格使用归位记录姿态（2026-10-09）

“水平参考”采用 `config/home_positions.json` 中各臂的 `tcp_pose_aubo_rpy`。
初始化时转换成 BASE 下的 rotvec，保存在 `DualWorkspace.locked_orientations`，
运行期间不随 T265 旋转、实际反馈或重力采样重算。此处不重新标定工具水平。

```text
人控最新目标 / 策略插值 / 暂停 / 抬升恢复 / 固定姿态归位
                     ↓
       先沿重力反方向将低高度修正到各臂下限
                     ↓
          最后覆盖各臂归位记录的固定姿态
                     ↓
 v2: IK → FK 核验高度和姿态 → servoJoint
 v3: SDK 边界再次依序覆盖 → servoCartesian
```

- 高度是 BASE 位移与安装“向上”向量的点积，不是直接写 BASE 的 Z 分量。
- 姿态锁与高度检查开关独立。T 键只确认锁定并刷新 XYZ 锚点，不解锁旋转。
- 启用运动后，初始暂停、普通暂停、低高度恢复均生成固定参考姿态的目标。
  初始实际姿态若与参考不同，会发送纠正目标；反馈仍显示真实姿态。
- H 归位在严格锁定下使用 `LockedCartesianHome`：从实际 XYZ 到关节记录 FK
  得到的终点 XYZ，以五次进度规划位置，每帧姿态固定。最终关节解允许不同于记录。
  未设置姿态锁的底层调用仍保留原有 `JointHome`。
- 恢复抬升优先于归位，期间冻结归位进度；恢复后从新鲜实际状态重建同类型归位。
- SDK 返回 0 才推进归位时间；队列满、STOP、独立 I/O 和看门狗处理保持有效。
- 策略 action chunk 保留时间排程，位姿副本先修正高度、再覆盖姿态，
  I/O 列与阶段不变；录制使用修正后的动作，不修改调用方输入数组。
- v2 严格锁定下 IK 失败立即退出，不回退旧倾斜关节目标，也不缩放关节步长
  生成偏离姿态的中间目标。FK 核验使用旋转差，`1e-4 rad` 只容纳数值误差。

`tests/test_strict_orientation.py` 覆盖这几条路径，结合既有恢复、相机首帧、
实时目标、独立 I/O 和 STOP 测试进行离线回归。命令姿态锁定不等于真实跟随误差为零；
本次未启动真机运动。


本次离线结果（2026-10-09）：133 项回归通过；v2 `dual.json` 与 v3
`dual_no_smoothing.json` 的 `--check-config` 均通过。


## 左右臂独立下限与标定矩阵（2026-10-09）

高度由实际选用的 JSON 独立设置，现场调试后确定；代码和测试不要求固定数值。
本次提交时的配置快照如下，后续以文件实际内容为准：

| 配置 | 左臂下限 | 右臂下限 |
| --- | --- | --- |
| `dual.json` | −580 mm | −535 mm |
| `dual_no_smoothing.json` | −570 mm | −550 mm |

以下展示 `dual.json` 本次快照；`max_downward_distance` 为正数，单位米：

```json
"workspace": {
  "enabled": true,
  "reference": "calibrated_base",
  "max_downward_distance": {"left": 0.58, "right": 0.535},
  "target_margin": 0.0
}
```

修改 `left` 或 `right` 就可以分别设置下限，重新启动程序生效。旧的单个数值
仍表示两臂共用下限；字典形式必须包含 left/right，两值都有限且大于零。
`DualWorkspace.minimum_height(side)` 是目标投影、真实反馈检查、恢复判断和日志
使用的统一查询接口；两个控制器都按自己的 `workspace_side` 查询。
目标低于所选配置中的对应臂下限时，沿标定 Z 投影到该下限；等号允许。
低实际反馈向上恢复到对应下限，姿态最后覆盖归位记录。

### 旧 gravity_base 实现与标定矩阵的关系

当前 `left_calib.json` / `right_calib.json` 的 `operator_basis_base` 是 3×3 旋转矩阵，
列向量表示操作者坐标轴在 BASE 中的方向。两份文件的第三列均与
`-gravity/||gravity||` 一致，所以以各自基座原点为零时两种高度算法等价：

```text
R = operator_basis_base    # BASE ← OPERATOR，注意变换方向
p_operator = R.T @ p_base
p_operator[2] = R[:, 2] · p_base = up_base · p_base
```

矩阵已经补偿倾斜安装，重力投影同样补偿倾斜；它不使用 BASE 本身的 z 分量。
旧 gravity_base 模式启动时读取并核验安装重力，一次更新 up_base；新默认模式固定读取标定 Z。
若实时安装值与记录略有差别，投影方向可能与保存矩阵 Z 列略有差别。

当前配置的两个 `world_from_base` 都是 null，未启用含平移原点的 4×4 世界变换。
若需要桌面或共同世界原点的绝对高度，要提供各臂 `T_WORLD_from_BASE`，并使用
`reference="world"`；此时 `p_world = R_world_base @ p_base + t_world_base`。
缺少平移原点时，基座相对高度不能直接解释为桌面/统一世界的绝对高度。
独立下限修改保留基座原点；当前高度方向改用保存的标定 Z，仍未重新定义高度零点。

本次独立下限修改：142 项离线回归通过，两份配置检查通过；未执行真机运动。


## 当前高度参考：固定标定 Z 轴（2026-10-09）

两份默认配置现选择 `workspace.reference="calibrated_base"`。每臂从对应
`left_calib.json` / `right_calib.json` 的 `operator_basis_base` 读取第三列：

```text
z_base = operator_basis_base[:, 2]
h = z_base · p_base
p_goal = p_base + max(0, minimum_height(side) - h) * z_base
最后覆盖该臂的归位记录姿态
```

高度零点仍是各臂 BASE 原点，下限从所选配置的 `workspace.max_downward_distance` 读取；
没有新增桌面原点或重算水平姿态。当前标定 Z 与保存的重力反方向一致，
但高度轴现在固定取自标定文件，不使用启动读取的重力覆盖，也不逐帧更新。
重力读取和安装核对仍执行。两种模式共用目标修正、真实低反馈恢复和姿态覆盖顺序。

旧的 `gravity_base` 模式继续支持，显式选择它才会在启动时用柜体安装重力更新方向；
`world` 模式保持原有 4×4 外参计算。每轮仍是点积和必要的向量加法，
不增加 IK/FK/SDK 调用。`tests/test_calibrated_height.py` 离线验证新模式、
左右独立投影、固定轴、控制器恢复和姿态锁定。


本次标定 Z 切换：149 项离线回归通过，两份配置检查通过；未执行真机运动。
