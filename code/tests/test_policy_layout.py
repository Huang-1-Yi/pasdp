import unittest
from pathlib import Path

import torch


class PolicyLayoutTest(unittest.TestCase):
    def test_workspace_suite_neutral_layout(self):
        workspace_dir = (
            Path(__file__).parents[1] / "diffusion_policy" / "workspace"
        )
        self.assertTrue(
            (workspace_dir / "train_policy_workspace.py").is_file())
        self.assertTrue(
            (workspace_dir / "train_padp_workspace.py").is_file())
        self.assertTrue(
            (workspace_dir / "train_padp_unrolled_workspace.py").is_file())
        self.assertTrue(
            (workspace_dir / "common" /
             "base_training_workspace.py").is_file())
        self.assertTrue(
            (workspace_dir / "common" / "rollout.py").is_file())
        self.assertTrue(
            (workspace_dir / "common" / "action_horizon.py").is_file())
        self.assertFalse(
            (workspace_dir / "train_diffusion_workspace.py").exists())

        robomimic_dir = workspace_dir / "robomimic"
        self.assertTrue(
            (robomimic_dir / "train_bcrnn_workspace.py").is_file())
        self.assertFalse((robomimic_dir / "train_dp_workspace.py").exists())
        self.assertFalse(
            (robomimic_dir / "train_padp_workspace_v3.py").exists())
        self.assertFalse(
            (robomimic_dir / "train_padp_unrolled_workspace_v3.py").exists())

        from diffusion_policy.workspace.train_policy_workspace import (
            TrainPolicyWorkspace,
        )
        from diffusion_policy.workspace.train_padp_unrolled_workspace import (
            TrainPADPUnrolledWorkspace,
        )
        from diffusion_policy.workspace.train_padp_workspace import (
            TrainPADPWorkspace,
        )
        from diffusion_policy.workspace.robomimic.train_bcrnn_workspace import (
            TrainBCRNNWorkspace,
        )
        from diffusion_policy.workspace.base_workspace import BaseWorkspace

        for workspace_cls in (
                TrainPolicyWorkspace,
                TrainPADPWorkspace,
                TrainPADPUnrolledWorkspace,
                TrainBCRNNWorkspace):
            self.assertTrue(issubclass(workspace_cls, BaseWorkspace))

    def test_evaluation_entrypoints_are_unambiguous(self):
        root = Path(__file__).parents[1]
        self.assertTrue((root / "eval_sim_rollout.py").is_file())
        self.assertTrue((root / "eval_action_horizon.py").is_file())
        self.assertFalse((root / "eval_sim.py").exists())
        self.assertFalse((root / "eval_sim_2.py").exists())

    def test_model_legacy_layout_is_removed(self):
        model_dir = Path(__file__).parents[1] / "diffusion_policy" / "model"
        self.assertFalse((model_dir / "diffusion").exists())
        self.assertFalse((model_dir / "vision").exists())
        self.assertTrue(
            (model_dir / "diffusion_heads" /
             "conditional_unet1d_streaming.py").is_file())
        self.assertTrue(
            (model_dir / "transformer_heads" /
             "diffusion_transformer.py").is_file())
        self.assertTrue(
            (model_dir / "encoders" / "pointcloud" /
             "dp3_encoder.py").is_file())

    def test_common_base_exports(self):
        from diffusion_policy.policy.common import (
            BaseDiffusionPolicy,
            BasePolicy,
        )

        self.assertTrue(issubclass(BaseDiffusionPolicy, BasePolicy))

    def test_flow_schedule_keeps_upstream_scalar_rule(self):
        from diffusion_policy.policy.common.flow_matching import (
            ConsistencyFlowSchedule,
        )

        schedule = ConsistencyFlowSchedule(
            noise_scale=2.0,
            sigma_variance=0.3,
            sample_steps=4,
        )
        self.assertEqual(schedule.sample_steps, 4)
        self.assertEqual(schedule.noise_scale, 2.0)
        self.assertAlmostEqual(schedule.sigma(0.25), 0.225)
        with self.assertRaises(ValueError):
            ConsistencyFlowSchedule(sample_steps=0)

    def test_dp3_filters_rgb_without_mutating_input(self):
        from diffusion_policy.policy.dp3 import _pointcloud_observations

        point_cloud = torch.randn(2, 8, 3)
        state = torch.randn(2, 4)
        eye_in_hand = torch.randn(2, 3, 16, 16)
        agentview = torch.randn(2, 3, 16, 16)
        observations = {
            "point_cloud": point_cloud,
            "agent_pos": state,
            "robot0_eye_in_hand_image": eye_in_hand,
            "agentview_image": agentview,
        }

        filtered = _pointcloud_observations(observations)

        self.assertEqual(
            set(filtered), {"point_cloud", "agent_pos"})
        self.assertIs(filtered["point_cloud"], point_cloud)
        self.assertIs(filtered["agent_pos"], state)
        self.assertEqual(len(observations), 4)
        self.assertIs(
            observations["robot0_eye_in_hand_image"], eye_in_hand)
        self.assertIs(observations["agentview_image"], agentview)


if __name__ == "__main__":
    unittest.main()
