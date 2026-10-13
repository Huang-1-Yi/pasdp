"""TCP 高度与姿态的统一出口 (v4 版)；输入输出均为各臂 BASE 下的 xyz+rotvec。

相比 v3 (DualWorkspace)，v4 针对末端触及下限位时的抖动现象增加了两项关键控制优化：
1. 回差死区与恢复容差 (Hysteresis Deadband & Recovery Margin)：
   - recovery_margin (默认 3 mm): 区分目标下限与物理保护下限。机械臂因动态跟随滞后产生微米级
     超调时，不触发底层 _height_recovery，消除目标截断与底层恢复的高频乒乓自激振荡。
   - 施密特迟滞回线 (Schmitt Trigger Hysteresis): 仅在深度跌破容差边界后触发强制抬升；
     且必须恢复到安全边界才退出恢复，彻底消除边界频繁跳变。
2. 软限位平滑阻尼与抗饱和 (Soft Boundary Damping & Anti-Windup)：
   - soft_zone (默认 15 mm): 在限位上方 15 mm 缓冲区内，对向下的速度分量执行 Smoothstep S-curve
     阻尼衰减，接触限位前平滑降速至 0，消除速度阶跃与加速度冲击 (Zero Jerk)。
   - 向上运动无阻尼：操作员向上抬手时保持 100% 原始灵敏度响应。
   - 映射器抗积分饱和 (Anti-Windup)：限位处下压时同步截断 mapper.pose，防止虚拟位姿深陷越界区，
     反向抬手时零延迟响应。
"""
from __future__ import annotations

import copy
from numbers import Real
import numpy as np
from scipy.spatial.transform import Rotation

from .workspace import DualWorkspace


class DualWorkspaceV4(DualWorkspace):
    def __init__(self, arms, settings):
        super().__init__(arms, settings)
        # 物理超调保护容差 (单位: 米，默认 3 mm)
        self.recovery_margin = float(settings.get('recovery_margin', 0.003))
        if not np.isfinite(self.recovery_margin) or self.recovery_margin < 0:
            raise ValueError('recovery_margin must be finite and nonnegative')

        # 软阻尼减速缓冲带厚度 (单位: 米，默认 15 mm)
        self.soft_zone = float(settings.get('soft_zone', 0.015))
        if not np.isfinite(self.soft_zone) or self.soft_zone < 0:
            raise ValueError('soft_zone must be finite and nonnegative')

        # 迟滞回线释放退出余量 (单位: 米，默认 0.1 mm)
        self.hysteresis_exit = float(settings.get('hysteresis_exit', 0.0001))

        # 左右臂独立维护强制恢复状态锁存
        self._recovering = {'left': False, 'right': False}

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        result._recovering = {'left': False, 'right': False}
        result.locked_orientations = dict(self.locked_orientations)
        return result

    def recovery_target(self, side, pose):
        """实测低高度恢复：引入 recovery_margin 死区与施密特迟滞，避免正常伺服超调诱发高频抢占振荡。"""
        pose = self._pose(pose)
        if not (self.enabled and self.reference in ('gravity_base', 'calibrated_base')):
            return None
        floor = self.minimum_height(side)
        if floor is None:
            return None

        current_h = self.height(side, pose)
        is_recovering = getattr(self, '_recovering', {}).get(side, False)

        if is_recovering:
            # 迟滞释放：已处于强制恢复中，必须抬升回到 floor - 0.1 mm 才退出
            if current_h < floor - self.hysteresis_exit:
                return self.limit_target(side, pose)
            else:
                self._recovering[side] = False
                return None
        else:
            # 迟滞触发：正常运行状态，只有实际深度跌破 (floor - recovery_margin) 才触发恢复
            if current_h < floor - self.recovery_margin:
                self._recovering[side] = True
                return self.limit_target(side, pose)
            return None

    def validate(self, side, pose, target=False):
        """区分待发目标与实测反馈：目标严格等于或高于 floor，实测反馈容纳 recovery_margin 跟踪公差。"""
        pose = self._pose(pose)
        if not self.enabled:
            return
        floor = self.minimum_height(side)
        height = self.height(side, pose)
        # target 严格不允许低于 floor；实测反馈允许在物理超调容差 recovery_margin 以内
        threshold = floor if target else (floor - self.recovery_margin if floor is not None else None)
        if threshold is not None and height < threshold - 1e-12:
            raise RuntimeError(f'{side} TCP below height limit ({self.reference}): '
                               f'{height * 1000:.2f} mm, limit {threshold * 1000:.2f} mm')
        if self.lower is not None:
            matrix = self.transforms[side]
            world = matrix[:3, :3] @ pose[:3] + matrix[:3, 3]
            if np.any(world < self.lower) or np.any(world > self.upper):
                raise RuntimeError(f'{side} TCP outside measured world workspace')

    def validate_feedback(self, side, pose):
        """实测反馈校验：容差范围内正常放行，深度越界则提交恢复目标进行校验。"""
        recovery = self.recovery_target(side, pose)
        self.validate(side, pose if recovery is None else recovery, target=False)

    def apply_soft_damping(self, side, raw_target, prev_target):
        """在接近限位边界时，对向下的速度分量进行 S-curve 平滑阻尼缓冲，消除加速度冲击与抖动。"""
        if not self.enabled:
            return raw_target
        floor = self.minimum_height(side)
        if floor is None:
            return raw_target

        up = self.up[side] if self.reference in ('gravity_base', 'calibrated_base') else self.transforms[side][2, :3]

        prev_h = float(up @ prev_target[:3])
        raw_h = float(up @ raw_target[:3])
        dh = raw_h - prev_h

        # 仅当向下运动且处于软阻尼区时执行平滑衰减；向上运动保持 100% 灵敏回抽
        if dh < 0:
            distance_to_floor = prev_h - floor
            if distance_to_floor <= self.soft_zone:
                # 归一化深度 rho in [0, 1]
                rho = max(0.0, min(1.0, distance_to_floor / max(1e-6, self.soft_zone)))
                # Smoothstep S-curve: 3*rho^2 - 2*rho^3
                damping = 3.0 * (rho ** 2) - 2.0 * (rho ** 3)
                damped_dh = dh * damping
                target_h = max(floor, prev_h + damped_dh)
                pose = raw_target.copy()
                pose[:3] += (target_h - raw_h) * up
                return self.limit_target(side, pose)

        return self.limit_target(side, raw_target)

    def sync_mapper_anti_windup(self, side, mapper, target):
        """同步映射器内部平移位姿，防止在限位处继续下压时发生积分饱和（Anti-Windup）。"""
        if not self.enabled:
            return
        up = self.up[side] if self.reference in ('gravity_base', 'calibrated_base') else self.transforms[side][2, :3]
        mapper_h = float(up @ mapper.pose[:3])
        target_h = float(up @ target[:3])
        if mapper_h < target_h:
            mapper.pose[:3] += (target_h - mapper_h) * up
