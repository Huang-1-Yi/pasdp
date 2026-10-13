实时显示控制键提示?

我想要创建一个可视化工具，用于可视化RealPushT数据集的RGB图像序列，并同步显示对应的状态和动作信息。
具体技术要求
1. 数据源与加载
从replay_buffer.zarr和videos文件夹中提取原始数据
通过real_data_to_replay_buffer函数加载数据
支持缓存机制加速重复加载
2. 显示内容要求
RGB图像：
支持显示1个或多个摄像头（camera_0, camera_1）
多个摄像头图像水平拼接显示
每个摄像头图像显示标签（camera_0, camera_1等）
状态和动作信息：
显示robot_eef_pose（6维，低维状态）
显示robot_gripper（1维，低维状态）
显示stage（1维，低维状态）
显示action（7维动作）
显示时间戳timestamp
显示episode索引和步数信息
所有数值显示到小数点后3位（通过np.set_printoptions(precision=3, suppress=True)）
3. 显示布局
上方：RGB图像区域（摄像头拼接图像）
下方：状态和动作信息面板
信息面板包含：
Episode编号、步数、总步数
时间信息
动作向量（7维）
所有状态向量（robot_eef_pose, robot_gripper, stage）
进度条显示播放进度
4. 控制参数
FPS控制：通过fps参数控制播放速度（帧率）,默认为10
缩放控制：通过scale参数控制图像缩放比例,默认为1,无缩放
可选参数：
episode_idx：选择要可视化的episode,默认为0
start_step：起始步数,默认为0
num_frames：要显示的总帧数,默认为所有
show_info：是否显示状态信息,默认为显示
save_video：是否保存为视频文件,默认为否
5. 交互控制功能
空格键：暂停/继续播放
'q'键：退出可视化
's'键：保存当前帧为图片
左右箭头：前进/后退1帧
'f'键：快进10帧
'b'键：后退10帧
'r'键：重置到开始
6. 数据处理要求
用于OpenCV显示
所有浮点数显示3位小数
错误处理：检查数据键是否存在，不存在则不略且print
7. 附加功能
视频保存功能（生成mp4文件）
显示总帧数和当前帧数
进度条显示当前播放位置,基于当前帧和总帧
8. 配置与数据格式
shape_meta:
  obs:
    camera_0: {shape: [3, 240, 320], type: rgb}
    camera_1: {shape: [3, 240, 320], type: rgb}
    robot_eef_pose: {shape: [6], type: low_dim}
    robot_gripper: {shape: [1], type: low_dim}
    stage: {shape: [1], type: low_dim}
  action: {shape: [7]}
9. 用户体验要求
支持从任意步数开始播放
自动适应episode长度
显示数据集基本信息（总步数、episode数等）