"""双臂 Aubo v3 环境：沿用观测/录制/动作接口，左右臂显式使用 v3 控制器。"""
import time
import numpy as np

from diffusion_policy.envs.realword.dual_aubo_real_env import DualAuboRealEnv as _DualAuboRealEnv
from diffusion_policy.envs.realword.aubo_interpolation_controller_v3 import AuboInterpolationControllerV3


class DualAuboRealEnvV3(_DualAuboRealEnv):
    def __init__(self, *args, manual_home=False, home_speed_scale=None, **kwargs):
        # v3 环境固定选择 v3 控制器，阻止调用方再传同名工厂造成版本混用。
        if 'controller_class' in kwargs:
            raise ValueError('The v3 environment selects its own v3 controller')
        if home_speed_scale is not None and (not np.isfinite(home_speed_scale) or home_speed_scale <= 0):
            raise ValueError('home_speed_scale must be finite and positive')
        # 父类构造左右臂时使用这个工厂，观测、I/O、相机和录制逻辑继续由父类提供。
        super().__init__(*args, controller_class=AuboInterpolationControllerV3, **kwargs)
        for side, robot in (('left', self.left_robot), ('right', self.right_robot)):
            robot.workspace = self.workspace
            robot.workspace_side = side
            robot.manual_home = bool(manual_home)
            # v4 可只覆盖本环境的两个实例；未提供时沿用控制器原有归位倍率。
            if home_speed_scale is not None:
                robot.HOME_SPEED_SCALE = float(home_speed_scale)

    def exec_actions(self, actions, timestamps, stages=None):
        # 策略 chunk 保持排程语义；只归一化副本，时间戳/stage/I/O 列不变。
        # 父类录制归一化后的动作，使记录姿态与实际待发姿态一致。
        array = np.asarray(actions, dtype=float).copy()
        if array.ndim == 1:
            array = array[None]
        if self.workspace and array.ndim == 2 and array.shape[1] == self.action_dim:
            right_start = 6 + self.gripper_dims['left']
            for action in array:
                for side, pose in (('left', action[:6]), ('right', action[right_start:right_start + 6])):
                    if self.workspace.orientation_locked(side):
                        pose[:] = self.workspace.limit_target(side, pose)
                    else:
                        self.workspace.validate_target(side, pose)
        return super().exec_actions(array, timestamps, stages)

    def set_gripper(self, side, values):
        """Send one reliable I/O event; no pose or future timestamp is attached."""
        if side not in ('left', 'right'):
            raise ValueError('Unknown arm selection')
        if not self.grippers_enabled:
            raise RuntimeError('Grippers are disabled')
        values = np.asarray(values, dtype=float)
        if values.shape != (self.gripper_dims[side],) or not np.isin(values, [0., 1.]).all():
            raise ValueError('Digital output values must have the configured shape and be 0 or 1')
        self.assert_healthy()
        robot = self.left_robot if side == 'left' else self.right_robot
        try:
            robot.set_gripper(self.encode_gripper(side, values))
        except Exception:
            self.stop()
            raise

    def set_teleop_target(self, action, stage=0):
        """Overwrite both realtime goals; output columns are recorded, not sent.

        Explicit set_gripper events own I/O. Policy chunks still use exec_actions.
        """
        action = np.asarray(action, dtype=float).copy()
        if action.shape != (self.action_dim,) or not np.isfinite(action).all():
            raise ValueError(f'Require one finite action with shape ({self.action_dim},)')
        if not np.isscalar(stage) or not np.isfinite(stage) or stage != int(stage):
            raise ValueError('Stage must be a finite integer')
        self.assert_healthy()
        if self.home_state == 'moving':
            raise RuntimeError('Cartesian targets are blocked during joint homing')
        right_start = 6 + self.gripper_dims['left']
        left, right = action[:6], action[right_start:right_start + 6]
        if self.grippers_enabled:
            if not all(np.isin(values, [0., 1.]).all() for values in
                       (action[6:right_start], action[right_start + 6:])):
                raise ValueError('Digital output actions must be 0 or 1')
        if self.workspace:
            left[:] = self.workspace.limit_target('left', left)
            right[:] = self.workspace.limit_target('right', right)
        submitted = time.time()
        try:
            self.left_robot.set_teleop_target(left)
            self.right_robot.set_teleop_target(right)
        except Exception:
            self.stop()
            raise
        if self._recording:
            self.action_accumulator.put(action[None], np.array([submitted]))
            self.stage_accumulator.put(np.array([stage], dtype=np.int64), np.array([submitted]))


# 保留与原环境模块相同的导出名，便于按模块版本替换导入。
DualAuboRealEnv = DualAuboRealEnvV3
