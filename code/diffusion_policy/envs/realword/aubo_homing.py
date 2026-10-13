"""Hardware-free recorded joint home validation and bounded quintic trajectory."""
import json
import time
from pathlib import Path
import numpy as np


def joints(value):
    q = np.asarray(value, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError('Expected six finite joint angles in radians')
    return q.copy()


def load_home(path, ip, serial):
    data = json.loads(Path(path).read_text())
    matches = [arm for arm in data['arms'] if arm['ip'] == ip and arm['serial'] == serial]
    if len(matches) != 1:
        raise ValueError('Recorded home does not match this arm IP and camera serial')
    return joints(matches[0]['joint_positions_rad'])


class JointHome:
    def __init__(self, start, target, speed=.15, acceleration=.3, speed_scale=1.):
        self.start, self.target = joints(start), joints(target)
        if not all(np.isfinite(v) and v > 0 for v in (speed, acceleration)):
            raise ValueError('Positive finite home limits required')
        if not np.isfinite(speed_scale) or speed_scale <= 0:
            raise ValueError('Positive finite home speed scale required')
        self.speed_scale = float(speed_scale)
        distance = np.max(np.abs(self.target-self.start))
        self.duration = max(1., 1.875*distance/speed, np.sqrt(5.774*distance/acceleration))
        # 时长缩放保持五次曲线形状；2 倍速对应半时长、2 倍规划速度、4 倍规划加速度。
        # 默认 1 倍保留所有旧调用行为，v3 显式选择 2 倍。
        self.duration /= self.speed_scale
        self.elapsed = 0.
        self.deadline = time.monotonic() + self.duration * 2 + 10.
        self.settled = 0
        self.last_received = None

    def sample(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
        u = min(1., (self.elapsed+dt)/self.duration)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        return self.start + blend*(self.target-self.start)

    def accepted(self, dt):
        self.elapsed = min(self.duration, self.elapsed+dt)

    def check(self, actual, speed, received, command):
        actual, speed, command = joints(actual), joints(speed), joints(command)
        if time.monotonic() > self.deadline:
            raise RuntimeError('Home timeout; stopping both arms')
        if np.max(np.abs(actual-command)) > .2:
            raise RuntimeError('Home joint following error > 0.2 rad')
        if received != self.last_received:
            self.last_received = received
            arrived = (self.elapsed >= self.duration and
                       np.max(np.abs(actual-self.target)) < .01 and np.max(np.abs(speed)) < .02)
            self.settled = self.settled+1 if arrived else 0
        return self.settled >= 5


class LockedCartesianHome(JointHome):
    """严格锁定时只规划归位 XYZ，每帧使用归位记录姿态。

    关节归位途中可能旋转 TCP，因此改走固定姿态的 XYZ 曲线。
    关节记录仍用于 FK 确定终点、估计原有归位时长；最终关节解允许不同。
    """
    def __init__(self, start_joints, target_joints, start_pose, target_pose, speed_scale=1.):
        super().__init__(start_joints, target_joints, speed_scale=speed_scale)
        self.start_pose = np.asarray(start_pose, dtype=float).copy()
        self.target_pose = np.asarray(target_pose, dtype=float).copy()
        if any(pose.shape != (6,) or not np.isfinite(pose).all()
               for pose in (self.start_pose, self.target_pose)):
            raise ValueError('Cartesian home requires finite six-value poses')

    def sample(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
        u = min(1., (self.elapsed + dt) / self.duration)
        blend = 10*u**3 - 15*u**4 + 6*u**5
        # 只对位置使用五次进度；从第一帧起姿态固定，不插值起点旋转。
        pose = self.target_pose.copy()
        pose[:3] = self.start_pose[:3] + blend * (self.target_pose[:3] - self.start_pose[:3])
        return pose

    def check_pose(self, actual_pose, tcp_speed, received):
        from scipy.spatial.transform import Rotation
        actual_pose = np.asarray(actual_pose, dtype=float)
        tcp_speed = np.asarray(tcp_speed, dtype=float)
        if any(value.shape != (6,) or not np.isfinite(value).all()
               for value in (actual_pose, tcp_speed)):
            raise ValueError('Invalid Cartesian home feedback')
        if time.monotonic() > self.deadline:
            raise RuntimeError('Home timeout; stopping both arms')
        # 只计入新的 RTDE 样本；完成判定容差不放宽待发姿态锁。
        # IK 可能得到不同关节解，以 TCP 位置/姿态/速度判断连续五帧停稳。
        if received != self.last_received:
            relative = (Rotation.from_rotvec(actual_pose[3:])
                        * Rotation.from_rotvec(self.target_pose[3:]).inv())
            arrived = (self.elapsed >= self.duration
                       and np.linalg.norm(actual_pose[:3] - self.target_pose[:3]) < .005
                       and relative.magnitude() < np.deg2rad(1.)
                       and np.linalg.norm(tcp_speed[:3]) < .01
                       and np.linalg.norm(tcp_speed[3:]) < .05)
            self.settled = self.settled + 1 if arrived else 0
            self.last_received = received
        return self.settled >= 5

    def restart(self, joints, pose):
        # 高度恢复后从新鲜实测状态重启 XYZ，保留原终点和锁定姿态。
        return type(self)(joints, self.target, pose, self.target_pose, speed_scale=self.speed_scale)
