"""DualWorkspaceV4 回差死区与软阻尼限位的测试。"""
import copy
import unittest
import numpy as np

from diffusion_policy.envs.realword.aubo_t265.config import load_config
from diffusion_policy.envs.realword.aubo_t265.workspace_v4 import DualWorkspaceV4


class WorkspaceV4Tests(unittest.TestCase):
    def workspace(self, **kwargs):
        c = load_config()
        settings = dict(c['workspace'], reference='gravity_base',
                        max_downward_distance={'left': 0.58, 'right': 0.535},
                        recovery_margin=0.003, soft_zone=0.015)
        settings.update(kwargs)
        w = DualWorkspaceV4(c['arms'], settings)
        for side in ('left', 'right'):
            # 基座 -Y 为重力方向，向上轴为 +Y: up = [0, 1, 0]
            w.update_gravity(side, [0., -9.81, 0.])
        return w

    def test_initialization_defaults(self):
        w = self.workspace()
        self.assertEqual(w.recovery_margin, 0.003)
        self.assertEqual(w.soft_zone, 0.015)
        self.assertEqual(w.minimum_height('right'), -0.535)

    def test_hysteresis_deadband_ignores_small_overshoot(self):
        w = self.workspace()
        # 限位 -535mm。物理反馈处于 -536mm (仅超调 1mm < 3mm 容差)
        pose_overshoot_1mm = np.array([0.1, -0.536, 0.2, 0.1, 0.2, 0.3])
        # 1. 不应触发强制恢复，避免高频自激振荡
        self.assertIsNone(w.recovery_target('right', pose_overshoot_1mm))
        # 2. 反馈校验放行，不发生崩溃
        w.validate_feedback('right', pose_overshoot_1mm)
        # 3. 目标校验依然严格执行 (目标不允许越界)
        with self.assertRaises(RuntimeError):
            w.validate_target('right', pose_overshoot_1mm)

    def test_deep_intrusion_triggers_recovery_with_schmitt_trigger(self):
        w = self.workspace()
        # 深度越界到 -539mm (超调 4mm > 3mm 容差)
        pose_deep = np.array([0.1, -0.539, 0.2, 0.1, 0.2, 0.3])
        recovery = w.recovery_target('right', pose_deep)
        self.assertIsNotNone(recovery)
        self.assertAlmostEqual(w.height('right', recovery), -0.535)
        self.assertTrue(w._recovering['right'])

        # 正在抬升恢复中：即使回升到 -536mm (在3mm容差内)，由于处于恢复状态，必须彻底回到 floor 才退出
        pose_mid = np.array([0.1, -0.536, 0.2, 0.1, 0.2, 0.3])
        self.assertIsNotNone(w.recovery_target('right', pose_mid))
        self.assertTrue(w._recovering['right'])

        # 抬升回到 -0.5349m (已经回到安全区)，恢复解除
        pose_safe = np.array([0.1, -0.5349, 0.2, 0.1, 0.2, 0.3])
        self.assertIsNone(w.recovery_target('right', pose_safe))
        self.assertFalse(w._recovering['right'])

    def test_soft_boundary_damping(self):
        w = self.workspace(soft_zone=0.015)
        # floor = -0.535, soft_top = -0.520
        # 1. 缓冲区外自由下移：无衰减
        prev = np.array([0.1, -0.500, 0.2, 0.0, 0.0, 0.0])
        raw = np.array([0.1, -0.505, 0.2, 0.0, 0.0, 0.0])
        out = w.apply_soft_damping('right', raw, prev)
        self.assertAlmostEqual(out[1], -0.505)

        # 2. 向上回抽：即使在缓冲区内，也无衰减 (保持灵敏响应)
        prev_low = np.array([0.1, -0.530, 0.2, 0.0, 0.0, 0.0])
        raw_up = np.array([0.1, -0.525, 0.2, 0.0, 0.0, 0.0])
        out_up = w.apply_soft_damping('right', raw_up, prev_low)
        self.assertAlmostEqual(out_up[1], -0.525)

        # 3. 缓冲区内向下：S-curve 阻尼平滑衰减
        # prev at -0.530 (距离底面 5mm，rho = 5/15 = 1/3)
        # raw 尝试下压 3mm 到 -0.533
        raw_down = np.array([0.1, -0.533, 0.2, 0.0, 0.0, 0.0])
        out_damped = w.apply_soft_damping('right', raw_down, prev_low)
        # 衰减后位移必定小于 3mm，高度必定高于 -0.533
        self.assertTrue(out_damped[1] > -0.533)
        self.assertTrue(out_damped[1] < -0.530)

        # 4. 越界下压兜底：无论怎么下压，最终输出必定 >= floor
        raw_penetrate = np.array([0.1, -0.550, 0.2, 0.0, 0.0, 0.0])
        out_floor = w.apply_soft_damping('right', raw_penetrate, prev_low)
        self.assertAlmostEqual(out_floor[1], -0.535)

    def test_anti_windup_synchronization(self):
        w = self.workspace()
        # 模拟 mapper 对象
        class MockMapper:
            def __init__(self, pose):
                self.pose = pose.copy()

        # mapper 内部严重饱和下沉到 -0.600m，而实际 target 被截断在 -0.535m
        mapper = MockMapper(np.array([0.1, -0.600, 0.2, 0.0, 0.0, 0.0]))
        target = np.array([0.1, -0.535, 0.2, 0.0, 0.0, 0.0])

        w.sync_mapper_anti_windup('right', mapper, target)
        # 同步后 mapper.pose[1] 应被截断对齐至 -0.535，消除反向抬手迟滞
        self.assertAlmostEqual(mapper.pose[1], -0.535)


if __name__ == '__main__':
    unittest.main()
