import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from diffusion_policy.envs.realword.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from diffusion_policy.envs.realword.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
import torch
from diffusion_policy.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2
    MOVE_TO_JOINT = 3
    #SCHEDULE_GRIPPER = 3

tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([-0.0336, 0, 0.247])

tx_flangerot45_flangerot90 = np.identity(4)
tx_flangerot45_flangerot90[:3, :3] = st.Rotation.from_euler('x', [np.pi / 2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
tx_flange_flangerot45[:3, :3] = st.Rotation.from_euler('z', [np.pi / 4]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot45_flangerot90 @ tx_flangerot90_tip
tx_tip_flange = np.linalg.inv(tx_flange_tip)

# Franka Hand gripper open/close speed (m/s). The server default is 0.02
# (a full 8 cm stroke takes ~4 s); the hand supports up to ~0.1 m/s.
GRIPPER_SPEED = 0.1


class FrankaInterface:
    """zerorpc client for one Franka arm.

    The two-arm server (launch_franka_interface_server_two_franka.py) exposes
    the left arm under the plain method names and the right arm under the same
    names with a ``1`` suffix (e.g. ``get_ee_pose1``, ``update_desired_ee_pose1``).
    Pass ``arm_suffix='1'`` to drive the right arm; the default ``''`` drives
    the left arm (and matches the single-arm server).
    """

    def __init__(self, ip='172.16.0.1', port=4242, arm_suffix=''):
        self.server = zerorpc.Client(heartbeat=20)
        self.server.connect(f"tcp://{ip}:{port}")#客户端
        self._suffix = arm_suffix
        self.gripper_flag = 0.08

    def _rpc(self, name):
        return getattr(self.server, name + self._suffix)

    def get_ee_pose(self):
        ee_pose = np.array(self._rpc('get_ee_pose')())
        return ee_pose

    def get_joint_positions(self):
        return np.array(self._rpc('get_joint_positions')())

    def get_joint_velocities(self):
        return np.array(self._rpc('get_joint_velocities')())

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float,
                                blocking: bool = True):
        # NOTE: zerorpc only forwards *positional* arguments over the wire —
        # keyword arguments are silently dropped. So pass `blocking` positionally.
        self._rpc('move_to_joint_positions')(
            positions.tolist(), time_to_go, blocking)

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        self._rpc('start_cartesian_impedance')(
            Kx.tolist(),
            Kxd.tolist()
        )

    def update_desired_ee_pose(self, pose: np.ndarray):
        self._rpc('update_desired_ee_pose')(pose.tolist())

    def terminate_current_policy(self):
        self._rpc('terminate_current_policy')()

    def open_Gripper(self, width):
        self._rpc('open_gripper')(width, GRIPPER_SPEED)

    def close_Gripper(self):
        # Server signature is close_gripper(speed=0.02, force=20.0) — it does
        # NOT take a width argument. Passing one here (the old 0.0 width) landed
        # in `speed`, commanding the gripper to close at 0 m/s (i.e. not move).
        self._rpc('close_gripper')(GRIPPER_SPEED)

    def get_Gripper_state(self):
        Gripper_state = self._rpc('get_gripper_state')()#这个函数去服务端处理，只拿宽度
        return np.array(Gripper_state)

    def close(self):
        self.server.close()



class FrankaInterpolationController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 robot_port=4242,
                 frequency=1000,
                 Kx_scale=1.0,
                 Kxd_scale=1.0,
                 launch_timeout=3,
                 joints_init=None,
                 joints_init_duration=None,
                 soft_real_time=False,
                 verbose=False,
                 get_max_k=None,
                 receive_latency=0.0,
                 arm_suffix=''
                 ):
        """
        robot_ip: the ip of the middle-layer controller (NUC)
        frequency: 1000 for franka
        Kx_scale: the scale of position gains
        Kxd: the scale of velocity gains
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (7,)

        super().__init__(name="FrankaPositionalController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([750.0, 750.0, 750.0, 15.0, 15.0, 15.0]) * Kx_scale
        self.Kxd = np.array([37.0, 37.0, 37.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency #0.0001
        self.arm_suffix = arm_suffix
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5) #5000

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((7,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # build ring buffer
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd', 'get_joint_velocities'),
            ('Actualgripper', 'get_Gripper_state')
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(7)
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)
            elif 'Gripper' in func_name:
                example[key] = np.zeros(1)
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        example['gripper_state'] = np.zeros(1)
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k, #5000
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue#用来放机器人执行的指令，如路径点、抓取指令，开始会尝试get，没有就不管，当get到了则执行
        self.ring_buffer = ring_buffer#用来存储机器人状态，只要.start了之后就会一直run()，然后会一直读取
        self.receive_keys = receive_keys

        self.i = 0
        self.t = 0
        self.max_timesteps = 400
        self.num_queries = 8
        self.state_dim = 7
        self.act_frame = np.zeros([self.max_timesteps, self.max_timesteps+self.num_queries, self.state_dim])
        self.single_act = np.zeros([9,7])

        # Non-blocking joint-space move state. When set, the Cartesian impedance
        # update is paused (so it doesn't fight the joint trajectory) while the
        # 1kHz loop keeps streaming fresh robot state to the ring buffer.
        self._joint_move_active = False
        self._joint_move_deadline = 0.0
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FrankaPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        # drop_oldest so the STOP always gets through even if the queue is
        # full of undrained waypoints (e.g. the robot stalled/died).
        self.input_queue.put(message, drop_oldest=True)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join(timeout=5.0)
        if self.is_alive():
            self.terminate()
            self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert (duration >= (1 / self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message, drop_oldest=True)

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (7,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value, #2
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message, drop_oldest=True) #循环八次 加权重处理

    def move_to_joint_positions(self, positions, time_to_go=5.0):
        """Joint-space move to absolute joint positions (blocking in the
        controller process). After the move the Cartesian impedance policy is
        restarted and the pose interpolator reset to the new pose."""
        positions = np.array(positions)
        assert positions.shape == (7,)

        message = {
            'cmd': Command.MOVE_TO_JOINT.value,
            'target_pose': positions,
            'duration': time_to_go
        }
        self.input_queue.put(message, drop_oldest=True)
        
    # def schedule_gripper(self, width):
    #     width = np.array(width)
    #     assert width.shape ==(1,)

    #     message = {
    #         'cmd': Command.SCHEDULE_GRIPPER.value,
    #         'target_pose':width
    #     }

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # enable soft real-time
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # start polymetis interface
        robot = FrankaInterface(self.robot_ip, self.robot_port, arm_suffix=self.arm_suffix) #作为客户端连接服务端launch_franka_interface
        commands = None
        traj_point = 0
        print('已初始化')
        try:
            if self.verbose:
                print(f"[FrankaPositionalController] Connect to robot: {self.robot_ip}")

            # init pose
            if self.joints_init is not None:
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=self.joints_init_duration
                )

            # main loop
            dt = 1. / self.frequency #1/1000 = 0.001
            curr_pose = robot.get_ee_pose() #当前末端执行器的位姿6D
            target_copy = curr_pose

            # use monotonic time to make sure the control loop never go backward  |  monotonic()单调时间
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )# 初始化轨迹插值器 当前位置和当前时刻

            # start franka cartesian impedance policy
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            
            while keep_running:
                # send command to robot
                t_now = time.monotonic()

                # When a non-blocking joint-space move has finished, restart the
                # Cartesian impedance policy and reset the pose interpolator to the
                # current EEF pose so the arm doesn't snap back to a stale target.
                if self._joint_move_active and t_now >= self._joint_move_deadline:
                    robot.start_cartesian_impedance(Kx=self.Kx, Kxd=self.Kxd)
                    curr_pose = robot.get_ee_pose()
                    pose_interp = PoseTrajectoryInterpolator(
                        times=[t_now], poses=[curr_pose])
                    last_waypoint_time = t_now
                    self._joint_move_active = False

                # diff = t_now - pose_interp.times[-1]
                # if diff > 0:
                #     print('extrapolate', diff)
                tip_pose = pose_interp(t_now) if not self._joint_move_active else None
                # send command to robot
                # self.single_act[self.i] = tip_pose
                # if i == 8 :
                #     self.act_frame[[self.t],self.t:self.t+9] = self.single_act
                #     self.t+=1
                #     i = 0

                # "测试直接给点"
                # if not commands:
                #     robot.update_desired_ee_pose(curr_pose)
                # else:
                #     robot.update_desired_ee_pose(target_copy)
                # "测试直接给点"

                #print(tip_pose)
                #print('-------------------------------------')
                if tip_pose is not None:
                    robot.update_desired_ee_pose(tip_pose)#驱使franka移动的函数
                #print(f'franka从队列取了{.40}个点')


                # update robot state
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()
                # receive_keys = [
                #     ('ActualTCPPose', 'get_ee_pose'),
                #     ('ActualQ', 'get_joint_positions'),
                #     ('ActualQd', 'get_joint_velocities'),
                #     ('Actualgripper', 'get_Gripper_state')
                # ]

                #state {'ActualTCPPose':记录末端位姿,'ActualQ':记录关节角,'ActualQd':记录关节速度}
                #state['Actualgripper'] = robot.get_Gripper_state()#存宽度

                # print(commands)

                #######测试  这个位置存夹爪状态最合适
                if not commands:
                    commands = {'target_pose':np.array([[1,2,3,4,5,6,robot.gripper_flag]])}
                gripper_state = np.array(commands['target_pose'][0][6])
                state['gripper_state'] = gripper_state

                ############


                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency #t-0.0001
                self.ring_buffer.put(state) #ring_buffer最多存5000条state字典 
                
                # state = {
                #     'ActualTCPPose': np.array([...]),  # 6 维
                #     'ActualQ': np.array([...]),        # 7 维
                #     'ActualQd': np.array([...]),       # 7 维
                #     'Actualgripper': np.array          #0维？ 就一个宽度
                #     'robot_receive_timestamp': t_recv, # 浮点数
                #     'robot_timestamp': t_recv - self.receive_latency # 浮点数
                # }



                # fetch command from queue
                try:
                    # commands = self.input_queue.get_all()
                    # n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1) #commands为{'cmd': array([2]), 'target_pose': array([[1., 2., 3., 4., 5., 6.]]), 'duration': array([0.]), 'target_time': array([1.])}
                    #print('not empty!-----')
                    n_cmd = len(commands['cmd'])#1

                    traj_point += 1

                except Empty:
                    n_cmd = 0 #如果n_cmd为0则下面的for range不会循环
                    #print('empty!---------------')
                    #commands = {'target_pose':np.array([[1,2,3,4,5,6,robot.gripper_flag]])}

                # gripper_state = np.array(commands['target_pose'][0][6])
                # state['gripper_state'] = gripper_state
                # self.ring_buffer.put(state)
                # message = {
                #             'cmd': Command.SCHEDULE_WAYPOINT.value,  #值为2
                #             'target_pose': pose,
                #             'target_time': target_time
                #            }
            

                #commands为{'cmd': array([2]), 'target_pose': array([[1., 2., 3., 4., 5., 6.]]), 'duration': array([0.]), 'target_time': array([1.])}

                for i in range(n_cmd): #0 只循环一次
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    #command {'cmd':2,'target_pose':np.array([......]),'','target_time':1}
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FrankaPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose'][:6] #目标点
                        target_time = float(command['target_time'])#目标点对应的时间
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time # monotonic+delta
                        curr_time = t_now + dt #while循环开始时记录的 t_now=time.monotonic(),dt=0.001
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time#pose_interp初始化时候的monotonic()
                        )
                        last_waypoint_time = target_time
                        
                        # "测试直接给点"
                        # target_copy = target_pose
                        # "测试直接给点"

                        gripper_width = command['target_pose'][6]
                        # print('gripper_width: ',gripper_width)
                        if gripper_width == robot.gripper_flag:
                            continue
                        else:
                            robot.gripper_flag = gripper_width
                            if gripper_width == 0.08:
                                robot.open_Gripper(gripper_width)
                            else:
                                robot.close_Gripper()
                        #self.i += 1

                    elif cmd == Command.MOVE_TO_JOINT.value:
                        target_joints = command['target_pose']
                        time_to_go = float(command['duration'])
                        # Non-blocking joint-space move. The joint trajectory runs
                        # on the robot while this 1kHz loop keeps streaming fresh
                        # state (the Cartesian update is paused until the deadline,
                        # then the impedance policy and pose interpolator are
                        # restarted on the current EEF pose, see main loop above).
                        robot.move_to_joint_positions(
                            target_joints, time_to_go, blocking=False)
                        self._joint_move_active = True
                        self._joint_move_deadline = time.monotonic() + time_to_go + 0.5

                    else:
                        keep_running = False
                        break

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                if self.verbose:
                    print(f"[FrankaPositionalController] Actual frequency {1 / (time.monotonic() - t_now)}")

        finally:
            # manditory cleanup
            # terminate
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            try:
                robot.terminate_current_policy()
            except Exception as e:
                # don't mask the original exception (if any) with a cleanup error
                print(f"[FrankaPositionalController] terminate_current_policy failed: {e}")
            finally:
                del robot
                self.ready_event.set()

            if self.verbose:
                print(f"[FrankaPositionalController] Disconnected from robot: {self.robot_ip}")
