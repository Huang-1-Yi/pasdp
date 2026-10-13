"""TCP 高度与姿态的统一出口；输入输出均为各臂 BASE 下的 xyz+rotvec。"""
from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.spatial.transform import Rotation

from .calibration import load_calibration, rotation_matrix, valid_gravity


class DualWorkspace:
    def __init__(self, arms, settings):
        self.enabled = settings.get('enabled', False)
        if not isinstance(self.enabled, bool):
            raise ValueError('workspace.enabled must be boolean')
        self.reference = settings.get('reference', 'world')
        if self.reference not in ('world', 'gravity_base', 'calibrated_base'):
            raise ValueError('Unknown workspace reference')
        self.transforms, self.up = {}, {}
        self.locked_orientations = {}
        self.minimum_z = settings.get('minimum_world_z')
        self.minimum_heights = {}  # 左右高度下限独立保存，单位为米。
        self.lower, self.upper = settings.get('lower_world_xyz'), settings.get('upper_world_xyz')
        self.target_margin = settings.get('target_margin', 0.)
        if not np.isfinite(self.target_margin) or self.target_margin != 0:
            raise ValueError('workspace.target_margin must be 0; use one height boundary')
        if not self.enabled:
            return
        if self.reference in ('gravity_base', 'calibrated_base'):
            distance = settings.get('max_downward_distance')
            # 兼容旧的统一数值；新配置显式给出 left/right，缺项或拼写错误立即报错。
            distances = distance if isinstance(distance, dict) else dict(left=distance, right=distance)
            if set(distances) != {'left', 'right'}:
                raise ValueError('max_downward_distance must include exactly left and right')
            if any(isinstance(value, bool) or not isinstance(value, Real)
                   or not np.isfinite(value) or value <= 0 for value in distances.values()):
                raise ValueError('max_downward_distance values must be finite and positive')
            if self.lower is not None or self.upper is not None or self.minimum_z is not None:
                raise ValueError('Base-relative height cannot use world-coordinate bounds')
            self.minimum_heights = {side: -float(value) for side, value in distances.items()}
            # 保留旧标量属性供旧配置读取；所有控制逻辑通过 minimum_height(side) 查询。
            self.minimum_z = None if isinstance(distance, dict) else -float(distance)
            for arm in arms:
                calibration = load_calibration(arm['calibration'], arm['ip'], arm['serial'])
                if self.reference == 'calibrated_base':
                    # 标定矩阵各列是在 BASE 中表达的轴，第三列为各臂固定向上方向。
                    # 高度零点仍是各自 BASE 原点；只在初始化时读取，不逐帧重估。
                    self.up[arm['side']] = rotation_matrix(calibration['operator_basis_base'])[:, 2].copy()
                else:
                    self.update_gravity(arm['side'], calibration['gravity'])
            return
        if self.minimum_z is None and (self.lower is None or self.upper is None):
            raise ValueError('Enabled workspace requires measured world bounds')
        if self.minimum_z is not None and not np.isfinite(self.minimum_z):
            raise ValueError('Invalid minimum_world_z')
        if (self.lower is None) != (self.upper is None):
            raise ValueError('Provide both lower and upper world bounds')
        if self.lower is not None:
            self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
            if (self.lower.shape != (3,) or self.upper.shape != (3,)
                    or not np.isfinite([self.lower, self.upper]).all()
                    or np.any(self.lower >= self.upper)):
                raise ValueError('Invalid world bounds')
        for arm in arms:
            matrix = np.asarray(arm.get('world_from_base'), dtype=float)
            if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                    or not np.allclose(matrix[3], [0, 0, 0, 1])):
                raise ValueError('World checks require measured world_from_base for both arms')
            rotation_matrix(matrix[:3, :3])
            self.transforms[arm['side']] = matrix

    def update_gravity(self, side, gravity):
        """Use verified live installation gravity before starting the servo processes."""
        # 标定 Z 不允许被实时安装重力覆盖；重力仍可在入口用于安装核对。
        if self.reference == 'calibrated_base':
            raise RuntimeError('calibrated_base height axis is fixed by calibration')
        gravity = valid_gravity(gravity)
        self.up[side] = -gravity / np.linalg.norm(gravity)

    def _pose(self, pose):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError('TCP target must be finite xyz+rotvec')
        return pose

    def set_locked_orientation(self, side, rotvec):
        """保存各臂归位记录姿态；启动前设定，不随重力或相机反馈改变。"""
        rotvec = np.asarray(rotvec, dtype=float)
        if side not in ('left', 'right') or rotvec.shape != (3,) or not np.isfinite(rotvec).all():
            raise ValueError('Locked orientation requires an arm and a finite rotation vector')
        self.locked_orientations[side] = rotvec.copy()

    def orientation_locked(self, side):
        return side in self.locked_orientations

    def minimum_height(self, side):
        """各臂专属下限；两种 BASE 高度模式以各自基座原点为零，world 使用世界原点。"""
        if side not in ('left', 'right'):
            raise ValueError('Unknown arm selection')
        return self.minimum_heights.get(side, self.minimum_z)

    def height(self, side, pose):
        pose = self._pose(pose)
        if self.reference in ('gravity_base', 'calibrated_base'):
            return float(self.up[side] @ pose[:3])
        matrix = self.transforms[side]
        return float((matrix[:3, :3] @ pose[:3] + matrix[:3, 3])[2])

    def validate(self, side, pose, target=False):
        pose = self._pose(pose)
        if not self.enabled:
            return
        floor = self.minimum_height(side)
        height = self.height(side, pose)
        # Feedback and targets use the same inclusive floor. The tiny tolerance
        # only accounts for floating-point projection; it is not a motion margin.
        if floor is not None and height < floor - 1e-12:
            raise RuntimeError(f'{side} TCP below height limit ({self.reference}): '
                               f'{height * 1000:.2f} mm, limit {floor * 1000:.2f} mm')
        if self.lower is not None:
            matrix = self.transforms[side]
            world = matrix[:3, :3] @ pose[:3] + matrix[:3, 3]
            if np.any(world < self.lower) or np.any(world > self.upper):
                raise RuntimeError(f'{side} TCP outside measured world workspace')

    def validate_target(self, side, pose):
        self.validate(side, pose, target=True)
        if self.orientation_locked(side):
            # 比较旋转差而非 rotvec 分量；等价旋转表示不应被误判。
            relative = (Rotation.from_rotvec(self._pose(pose)[3:])
                        * Rotation.from_rotvec(self.locked_orientations[side]).inv())
            # 只容纳 IK→FK 数值误差；待发命令仍使用固定的记录姿态。
            if relative.magnitude() > 1e-4:
                raise RuntimeError(f'{side} target violates locked TCP orientation')

    def recovery_target(self, side, pose):
        """实测低高度只生成恢复目标，不修改反馈；姿态由最终锁定覆盖。"""
        pose = self._pose(pose)
        if (self.enabled and self.reference in ('gravity_base', 'calibrated_base')
                and self.height(side, pose) < self.minimum_height(side) - 1e-12):
            return self.limit_target(side, pose)
        return None

    def validate_feedback(self, side, pose):
        """允许 BASE 高度模式的低反馈进入抬升恢复，不修改原始 RTDE 数据。"""
        recovery = self.recovery_target(side, pose)
        self.validate(side, pose if recovery is None else recovery)

    def limit_target(self, side, pose):
        """目标出口：复制输入 → 沿所选高度轴修正 → 覆盖归位姿态 → 校验。"""
        pose = self._pose(pose).copy()
        floor = self.minimum_height(side)
        if self.enabled and floor is not None:
            correction = floor - self.height(side, pose)
            if correction > 0:
                up = self.up[side] if self.reference in ('gravity_base', 'calibrated_base') else self.transforms[side][2, :3]
                pose[:3] += correction * up
        # 倾斜安装时高度是 up·xyz，不能直接把 BASE z 写成高度下限。
        # 姿态最后覆盖；恢复/暂停/归位共用此出口，不沿用实测倾斜姿态。
        # 只调整待发目标，调用方数组与 RTDE 原始反馈保持原值。
        if self.orientation_locked(side):
            pose[3:] = self.locked_orientations[side]
        self.validate_target(side, pose)
        return pose
