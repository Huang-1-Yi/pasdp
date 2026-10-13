# AUBO v3：原生 TCP 伺服

v3 是本次最新 v2 的独立代码快照。新增入口、环境和控制器，不覆盖 v1/v2，
也不从 v2 入口或控制器继承。共享 SDK、RTDE、共享内存、标定、映射、配置、
归位和录制工具继续复用。配置仍默认为 `config/dual.json`。

## 控制链差异

```text
v2：最新 TCP → 独立 SDK inverseKinematics → FK 高度核验 → servoJoint
v3：最新 TCP → TCP 高度核验 → rotvec_to_aubo → servoCartesian

严格锁定 H：LockedCartesianHome → 高度修正 → 固定姿态 → 对应版本伺服
未设姿态锁 H：JointHome → FK 高度核验 → servoJoint
策略：exec_actions → 原有 PoseTrajectoryInterpolator → v3 TCP 下发
```

- 两臂遥操姿态固定为各自归位记录，XYZ、坐标标定、滤波、按键和 I/O 不变。
- 实时人控继续使用覆盖式最新目标，不使用 action chunk 或绝对到达时间。
- 频率、`servo_t`、伺服模式 1、柜体速度比例和映射器滤波系数不变。
- TCP 位姿是 BASE 中的 xyz（米）+ rotvec（弧度）；SDK 边界转换为 xyz+RPY。
- 普通实时和策略位姿不再单独调用 IK/FK，不生成 Python 关节目标。
- SDK 初始化的关节预热仍使用 `servoJoint`。进入控制循环后，严格锁定模式的
  暂停和 H 归位使用固定姿态的 TCP 目标；未设置姿态锁时保留原关节分支及 FK 核验。
- TCP 模式暂停保持最后成功接收的位姿；队列满或错误目标不成为保持目标。
- TCP 转关节归位时，从新鲜 RTDE 实测关节起步，避免旧关节种子造成跳变。
- 当前代码不提供 v2 的“最后命令关节角”约束：`max_joint_step`、
  `max_joint_speed`、`reject_joint_jumps` 无法用于直接 TCP 下发；
  `max_following_error` 仅用于关节归位/关节保持。

## 文件

- 根目录 `demo_dual_teleop_aubo_v3.py`：独立入口。
- `../dual_aubo_real_env_v3.py`：创建左右 v3 控制器，动作/观测/录制格式不变。
- `../aubo_interpolation_controller_v3.py`：TCP/关节分支、暂停/归位衔接及返回码。
- `tests/test_servo_cartesian_v3.py`：使用 SDK 替身的离线验证。

## 返回码与停止

`servoCartesian` 返回 0 才更新最后接受的 TCP；返回 2（队列满）保留原目标；
返回 -13 最多连续恢复三次。其他错误（包括控制器内部 IK 错误）立即报错并
退出控制进程，通过环境健康检查停止会话，不静默改回 `servoJoint`。
v2 在严格锁定时也不容忍 IK 失败回退旧关节目标；未锁定时保留原临时失败容忍。

STOP/HOLD、输入失效、命令看门狗、RTDE 新鲜度与真实 TCP 高度检查继续执行。
v2/v3 均从所选配置读取各臂高度边界（允许等号），`target_margin=0.0`，不增加固定余量。
低于边界的遥操目标会投影到该臂下限；实际反馈低于该下限则优先向上恢复到边界，不修改反馈值。
直接 TCP 路径检查真实反馈和待发目标，不能检查控制器内部每个运动中间点。
机器人安全平面仍由控制柜配置，不把软件目标检查视为连续运动保证。

## 运行

使用当前已安装 SDK 的 `robodiff`（Linux x86_64 Python 3.9）：

```bash
cd /home/hy/Desktop/dp_pro6000
conda activate robodiff

# 离线配置检查：不连接机器人，不开启 T265
python demo_dual_teleop_aubo_v3.py --check-config

# 离线测试：SDK/机械臂全部使用替身
python -m unittest diffusion_policy.envs.realword.aubo_t265.tests.test_servo_cartesian_v3 -v

# 真机入口：初始暂停，E 开启遥操；R 开始录制，S 保存并暂停
python demo_dual_teleop_aubo_v3.py --enable-motion --enable-io --output data/aubo_v3_demo

# 如需沿用已有无滤波配置，显式选择；不改其余参数
python demo_dual_teleop_aubo_v3.py \
  --config diffusion_policy/envs/realword/aubo_t265/config/dual_no_smoothing.json \
  --enable-motion --enable-io --output data/aubo_v3_no_smoothing
```

默认只读预览仍会连接机器人读取状态和启动 T265；`--check-config` 才是纯离线。
v1/v2/v3 使用同一组柜体锁，不要同时运行运动入口。

## 平滑与验证范围

本次只替换普通控制的 SDK 下发链，不改人控更新频率、不添加轨迹整形、
不取消既有滤波/策略插值。`servoCartesian` 内部处理 IK，不代表不均匀下发
或跳变目标会自动变平滑。现场仍需比较调用耗时、实际周期和跟随表现。

官方接口：
[MotionControl.servoCartesian](https://docs.aubo-robotics.cn/arcs_api/en/group__MotionControl.html)。
文档说明主要有效参数是 pose 和 t；本次完整保留六参数调用格式。
本次离线验证不证明控制柜已经成功执行 TCP 伺服，也未启动机器人运动。

## 当前高度边界

各臂下限从所选配置的 `workspace.max_downward_distance` 读取，`target_margin=0.0`。
数值由现场调试决定，测试不锁定某组默认值；`dual.json` 与无平滑配置可独立调整。
默认沿保存的标定 Z 计算高度，实时目标低于配置下限时投影到下限，最终使用归位记录姿态。
真实反馈不作伪造或截断：实际低于对应臂下限时沿标定 Z 恢复到下限；恰好位于下限则允许。
OpenCV 模块名保持 `cv2`，`v3` 仅是本次程序版本与窗口名称。


## 实际低高度恢复

启用运动后，初始暂停和正常控制阶段均检测实际高度。低于该臂下限时锁存实际水平位置，姿态最终覆盖为归位记录，
暂时优先发送向上恢复到该臂下限的 servoCartesian 目标，不增加独立 IK/FK 调用。
恢复完成后，暂停保持恢复位置；旧插值历史重置；若归位被打断，从当前实测状态重建固定姿态 XYZ 归位轨迹。
STOP、独立 I/O、命令看门狗、RTDE 新鲜度和 SDK 返回码处理仍生效。只读预览不会触发运动。
离线回归见 `tests/test_height_recovery.py`；本次未进行真机运动验证。


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


## 启动保持、手动 H 归位（2026-10-09）

运动入口现在采用操作者选择的 **启动保持 → H 归位 → E 遥操** 流程。
不增加启动姿态偏差检查或阈值。启动保持实测关节，不在首帧强制切换归位记录姿态；
首次 H 使用原有 JointHome 渐进归位，到位后才启用固定归位姿态。
后续 H 仍使用 LockedCartesianHome 保持姿态的 XYZ 归位；策略 API 的默认启动方式不变。

初次归位允许末端从当前真实姿态转到归位记录姿态。启动保持和首次关节归位期间，
高度边界、实际低高度恢复、STOP、独立 I/O、RTDE 新鲜度和命令看门狗继续执行。
固定姿态工作空间只在控制子进程中暂时使用副本，父进程参考记录不修改。
首次归位结束前，E/R/Z/T 不开启遥操或录制；IO 与退出仍独立处理。

本次只读采样：右臂与记录姿态相差约 27.48°，该目标 IK 最大关节差约 120°；
高度约 −315 mm，高于 −550 mm 下限，没有触发高度修正。
这些数据与启动首帧目标跳变被柜体拒绝的情况相符，未重放实际运动。
`servoCartesian` −5 不只表示参数格式错误，也可能是目标位置或速度越界。
后续 −5 日志包含目标/实际 TCP、位置与旋转差及 servo_t，不修改参数或自动重试越界目标。

官方返回码说明：[MotionControl.servoCartesian](https://docs.aubo-robotics.cn/arcs_api/en/group__MotionControl.html)。


本次手动归位修改：156 项离线回归通过，v3 配置检查通过；仅进行只读状态与 IK 查询，未发出运动指令。

## 启动输出的异步反馈确认（2026-10-09）

v3 启动将左 DO00/DO01 设为 LOW、右 DO00 设为 HIGH，并等待最多 2 秒确认实际输出锁存。
右臂初始电平已按要求反转；后续 C/D 按键仍切换 DO00，高电平的下一次切换为 LOW。
此前两臂初始均为 LOW。旧实现只在写入后立即读取一次；若此时柜体仍返回旧电平，发布的缓存会一直不变，
可能出现 `Startup LOW was not confirmed ... ['right']`，即使柜体稍后已经 LOW。

现在 v2/v3 控制循环在输出尚未确认时，以最多 10 Hz 继续只读查询锁存。
确认后停止查询，不重复写输出、不等待睡眠、不修改 RTDE 运动状态读取。
数字输出事件继续独立排队；不会用命令目标冒充实际反馈。
超时报错会附上未确认臂每个配置引脚的 `actual` 和 `target`，便于区分真实电平
未切换与启动反馈问题。输出锁存仍不代表夹爪或吸盘的物理到位反馈。

## H 归位速度与 I/O 恢复（2026-10-09）

- v3 的首次关节归位和后续固定姿态 XYZ 归位统一采用 `HOME_SPEED_SCALE = 2.0`。
  相同起终点的规划时长为原来一半，曲线形状不变；规划速度为 2 倍、加速度为 4 倍。
  控制频率、`servo_t` 和柜体速度比例保持原设置，实测速度仍取决于柜体跟随。
  高度恢复后重建的归位轨迹继续使用该倍率。共享归位类默认仍为 1 倍。
- 按 H 时同步提交初始 I/O：左 DO00/DO01=LOW，右 DO00=HIGH。
  与启动共用同一函数生成电平，并更新主程序按键缓存；下一次右臂切换由 HIGH 到 LOW。
- H 的 I/O 请求通过独立事件队列提交，不增加等待或阻塞键盘退出处理。
  两臂运动完成时核对实际输出锁存，确认后才提示归位完成并允许下一次 E 遥操。

## 现场高度配置与本次提交记录（2026-10-09）

`dual.json` 的 −580/−535 mm 是本次现场配置快照，由现场调试继续确定。
无需修改控制器、标定文件或测试中的固定高度断言；修改所选 JSON 后重启即可。
`dual_no_smoothing.json` 仍是独立配置，不会自动继承 `dual.json` 的高度数值。
默认配置测试验证加载后实际使用该文件的下限，不再要求 −570/−550 mm。
其余测试中的数值是测试内显式构造的案例，不约束现场 JSON。

完整记录：[AUBO v3 现场配置与改动记录](../../../../运行命令及文档/AUBO双臂遥操作_v3_现场配置与改动记录_2026-10-09.md)。
