import pathlib
import unittest
import gc

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf


CONFIG_DIR = pathlib.Path(__file__).parents[1] / "diffusion_policy" / "config"
MAX_STEPS = {
    "stack_d1": 400,
}


class ComposableConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        OmegaConf.register_new_resolver("eval", eval, replace=True)
        OmegaConf.register_new_resolver(
            "get_max_steps", lambda name: MAX_STEPS[name], replace=True)

    def tearDown(self):
        GlobalHydra.instance().clear()

    def _compose(self, experiment, *overrides):
        with initialize_config_dir(
                version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(
                config_name="train",
                overrides=[f"experiment={experiment}", *overrides],
            )
            OmegaConf.resolve(cfg)
            return cfg

    def test_four_presets_resolve_to_compatible_workspace_contract(self):
        cases = {
            "dp_robomimic": (10, "rgb", 16, 2, 8),
            "padp_robomimic": (10, "rgb", 40, 1, 1),
            "padp_metaworld": (4, "point_cloud", 40, 1, 1),
            "padp_adroit": (28, "point_cloud", 40, 1, 1),
        }
        for experiment, expected in cases.items():
            with self.subTest(experiment=experiment):
                action_dim, modality, horizon, obs_steps, action_steps = expected
                cfg = self._compose(experiment)
                self.assertIn("_target_", cfg.task.dataset)
                self.assertIn("_target_", cfg.task.env_runner)
                self.assertIn("_target_", cfg.policy)
                self.assertEqual(cfg.shape_meta.action.shape[0], action_dim)
                self.assertEqual(cfg.horizon, horizon)
                self.assertEqual(cfg.n_obs_steps, obs_steps)
                self.assertEqual(cfg.n_action_steps, action_steps)
                obs_types = {
                    value.get("type", "low_dim")
                    for value in cfg.shape_meta.obs.values()
                }
                self.assertIn(modality, obs_types)

    def test_suite_neutral_workspace_targets(self):
        expected_targets = {
            "dp_robomimic": (
                "diffusion_policy.workspace.train_policy_workspace."
                "TrainPolicyWorkspace"
            ),
            "padp_robomimic": (
                "diffusion_policy.workspace.train_padp_workspace."
                "TrainPADPWorkspace"
            ),
            "padp_unrolled_robomimic": (
                "diffusion_policy.workspace.train_padp_unrolled_workspace."
                "TrainPADPUnrolledWorkspace"
            ),
            "bcrnn_robomimic": (
                "diffusion_policy.workspace.robomimic."
                "train_bcrnn_workspace.TrainBCRNNWorkspace"
            ),
        }
        for experiment, expected_target in expected_targets.items():
            with self.subTest(experiment=experiment):
                cfg = self._compose(experiment)
                self.assertEqual(cfg._target_, expected_target)

    def test_adroit_task_override_propagates_action_and_dataset(self):
        cfg = self._compose(
            "padp_adroit", "task=adroit/hammer_pointcloud")
        self.assertEqual(cfg.task_name, "hammer")
        self.assertEqual(cfg.shape_meta.action.shape[0], 26)
        self.assertEqual(cfg.task.dataset.task_name, "hammer")
        self.assertTrue(cfg.dataset_path.endswith("adroit_hammer_expert.zarr"))

    def test_encoder_override_is_a_single_axis(self):
        cfg = self._compose(
            "padp_robomimic", "encoder=rgb/timm_convnext_tiny")
        self.assertEqual(cfg.policy.rgb_model, "timm_convnext_tiny")
        self.assertEqual(cfg.task.name, "mimicgen_abs_rgb")
        self.assertEqual(cfg.shape_meta.action.shape[0], 10)

    def test_native_bcrnn_preset_uses_recurrent_dataset_sequence(self):
        cfg = self._compose("bcrnn_robomimic")
        self.assertEqual(
            cfg._target_,
            "diffusion_policy.workspace.robomimic.train_bcrnn_workspace."
            "TrainBCRNNWorkspace",
        )
        self.assertEqual(
            cfg.policy._target_,
            "diffusion_policy.policy.robomimic_policy."
            "RobomimicBCRNNPolicy",
        )
        self.assertEqual(
            (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps),
            (10, 1, 1),
        )
        self.assertEqual(cfg.dataset_obs_steps, cfg.horizon)
        self.assertEqual(cfg.output_group, "bcrnn")
        self.assertFalse(cfg.training.early_stopping.enabled)
        self.assertEqual(cfg.training.early_stopping.patience, 3)

    def test_migrated_policy_presets_keep_independent_axes(self):
        cases = {
            "streaming_diffusion_robomimic": (
                "streaming_diffusion_policy.StreamingDiffusionPolicy",
                "streaming_diffusion", (20, 1, 1),
                "hydra"),
            "streaming_diffusion_optimized_robomimic": (
                "streaming_diffusion_policy_optimized."
                "OptimizedStreamingDiffusionPolicy",
                "streaming_diffusion_optimized", (20, 1, 1), "hydra"),
            "act_robomimic": (
                "act.ACTPolicyWrapper", "act", (10, 1, 10),
                "model"),
            "flow_policy_robomimic": (
                "flow_policy.FlowPolicy",
                "flow_policy", (40, 2, 8), "hydra"),
            "dp3_metaworld": (
                "dp3.DP3", "dp3", (16, 2, 8), "hydra"),
            "diffusion_transformer_robomimic": (
                "diffusion_transformer."
                "DiffusionTransformerHybridImagePolicy",
                "diffusion_transformer", (16, 2, 8),
                "policy_method"),
        }
        for experiment, expected in cases.items():
            with self.subTest(experiment=experiment):
                target_suffix, output_group, temporal, optimizer_source = expected
                cfg = self._compose(experiment)
                self.assertTrue(cfg.policy._target_.endswith(target_suffix))
                self.assertEqual(cfg.output_group, output_group)
                self.assertEqual(
                    (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps),
                    temporal,
                )
                self.assertEqual(
                    cfg.training.optimizer_source, optimizer_source)
                self.assertEqual(cfg.policy.down_dims, [256, 512, 1024])
                if experiment in {
                        "streaming_diffusion_robomimic",
                        "streaming_diffusion_optimized_robomimic",
                        "dp3_metaworld",
                        "diffusion_transformer_robomimic"}:
                    self.assertEqual(
                        OmegaConf.to_container(
                            cfg.policy.scheduler_step_kwargs, resolve=True),
                        {},
                    )
                expected_batch = 128 if experiment == "dp3_metaworld" else 64
                self.assertEqual(cfg.dataloader.batch_size, expected_batch)

    def test_all_focal_metaworld_scenes_resolve_and_are_registered(self):
        import metaworld

        task_files = sorted(
            (CONFIG_DIR / "task" / "metaworld").glob("*_pointcloud.yaml")
        )
        self.assertEqual(len(task_files), 44)
        registered = (
            metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
        )
        for task_file in task_files:
            choice = task_file.stem
            task_name = choice.removesuffix("_pointcloud")
            with self.subTest(task=task_name):
                cfg = self._compose(
                    "padp_metaworld", f"task=metaworld/{choice}")
                self.assertEqual(cfg.task_name, task_name)
                self.assertEqual(cfg.task.task_name, task_name)
                self.assertEqual(cfg.shape_meta.obs.point_cloud.shape, [1024, 3])
                self.assertEqual(cfg.shape_meta.action.shape, [4])
                self.assertTrue(
                    cfg.dataset_path.endswith(
                        f"metaworld_{task_name}_expert.zarr"))
                self.assertIn(
                    f"{task_name}-v2-goal-observable", registered)

    def test_pointcloud_comparison_presets_use_declared_native_temporals(self):
        expected_temporal = {
            "dp": (12, 2, 4),
            "padp": (12, 2, 4),
            "focal": (12, 2, 4),
        }
        for suite in ("metaworld", "adroit"):
            configs = {
                algorithm: self._compose(
                    f"focal_{suite}" if algorithm == "focal"
                    else f"compare_{algorithm}_{suite}")
                for algorithm in ("dp", "padp", "focal")
            }
            reference = configs["focal"]
            for algorithm, cfg in configs.items():
                with self.subTest(suite=suite, algorithm=algorithm):
                    self.assertEqual(cfg.shape_meta, reference.shape_meta)
                    self.assertEqual(cfg.output_group, algorithm)
                    self.assertEqual(
                        (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps),
                        expected_temporal[algorithm],
                    )
                    self.assertEqual(
                        cfg.policy.pointcloud_model, "pointnet_focal_xyz")
                    self.assertEqual(
                        cfg.policy.down_dims, [256, 512, 1024])
                    self.assertEqual(cfg.policy.diffusion_step_embed_dim, 128)
                    self.assertEqual(cfg.dataloader.batch_size, 128)
                    self.assertEqual(cfg.val_dataloader.batch_size, 128)
                    self.assertEqual(cfg.training.num_epochs, 1000)
                    self.assertEqual(cfg.training.rollout_every, 100)
                    self.assertEqual(cfg.training.checkpoint_every, 100)

    def test_three_pointcloud_policies_instantiate_with_smoke_capacity(self):
        expected = {
            "dp": ("DiffusionUnetHybridImagePolicy", 12, 4),
            "padp": ("SlidingWindowDiffusionPolicy", 12, 4),
            "focal": ("FocalPolicy", 12, 4),
        }
        for algorithm, (expected_name, horizon, action_steps) in expected.items():
            with self.subTest(algorithm=algorithm):
                cfg = self._compose(
                    "focal_metaworld" if algorithm == "focal"
                    else f"compare_{algorithm}_metaworld",
                    "capacity=smoke")
                policy = hydra.utils.instantiate(cfg.policy)
                self.assertEqual(type(policy).__name__, expected_name)
                self.assertEqual(policy.horizon, horizon)
                self.assertEqual(policy.n_action_steps, action_steps)
                self.assertEqual(
                    type(policy.obs_encoder).__name__,
                    "FocalPointCloudObsEncoder",
                )
                del policy
                gc.collect()

    def test_focal_suite_presets_and_upstream_recipe_resolve(self):
        cases = {
            "focal_robomimic": (16, 2, 4, 64, 251),
            "focal_metaworld": (12, 2, 4, 128, 1000),
            "focal_adroit": (12, 2, 4, 128, 1000),
        }
        for experiment, expected in cases.items():
            with self.subTest(experiment=experiment):
                cfg = self._compose(experiment)
                h, obs, action, batch, epochs = expected
                self.assertTrue(cfg.policy._target_.endswith("FocalPolicy"))
                self.assertEqual(
                    (cfg.horizon, cfg.n_obs_steps, cfg.n_action_steps),
                    (h, obs, action))
                self.assertEqual(cfg.dataloader.batch_size, batch)
                self.assertEqual(cfg.training.num_epochs, epochs)
                self.assertFalse(cfg.training.early_stopping.enabled)

        cfg = self._compose(
            "focal_adroit", "capacity=large", "recipe=focal_upstream")
        self.assertEqual(cfg.policy.down_dims, [512, 1024, 2048])
        self.assertEqual(cfg.policy.diffusion_step_embed_dim, 128)
        self.assertEqual(cfg.policy.pointcloud_encoder_kwargs.point_feature_dim, 64)
        self.assertEqual(
            cfg.policy.pointcloud_encoder_kwargs.state_mlp_size, [64, 64])
        self.assertTrue(
            cfg.policy.pointcloud_encoder_kwargs.use_layer_norm)
        self.assertEqual(
            cfg.policy.pointcloud_encoder_kwargs.final_norm, "layernorm")
        self.assertEqual(cfg.policy.sample_cfg.tr_sampler, "v0")
        self.assertEqual(cfg.optimizer.betas, [0.95, 0.999])
        self.assertEqual(cfg.optimizer.weight_decay, 1.0e-6)
        self.assertEqual(cfg.ema.power, 0.75)
        self.assertEqual(cfg.training.num_epochs, 3000)
        self.assertEqual(cfg.training.rollout_every, 200)

    def test_all_default_policy_targets_instantiate(self):
        for experiment in (
                "dp_robomimic",
                "padp_robomimic",
                "padp_metaworld",
                "padp_adroit"):
            with self.subTest(experiment=experiment):
                cfg = self._compose(experiment)
                self.assertEqual(cfg.policy.down_dims, [256, 512, 1024])
                if experiment.endswith("robomimic"):
                    self.assertEqual(
                        cfg.output_group,
                        "dp" if experiment.startswith("dp_") else "padp",
                    )
                    self.assertEqual(cfg.dataloader.batch_size, 64)
                    self.assertEqual(cfg.val_dataloader.batch_size, 64)
                else:
                    self.assertEqual(cfg.output_group, "padp")
                    self.assertEqual(cfg.dataloader.batch_size, 128)
                    self.assertEqual(cfg.val_dataloader.batch_size, 128)
                    self.assertEqual(cfg.training.num_epochs, 1000)
                    self.assertEqual(cfg.training.rollout_every, 100)
                    self.assertEqual(cfg.training.checkpoint_every, 100)
                policy = hydra.utils.instantiate(cfg.policy)
                self.assertEqual(policy.horizon, cfg.horizon)
                self.assertEqual(policy.n_action_steps, cfg.n_action_steps)
                del policy
                gc.collect()


if __name__ == "__main__":
    unittest.main()
