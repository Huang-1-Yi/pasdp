import unittest
from unittest.mock import patch

import torch

from diffusion_policy.model.encoders import build_obs_encoder
from diffusion_policy.model.encoders.pointcloud.factory import (
    available_pointcloud_models,
    build_pointcloud_obs_encoder,
)
from diffusion_policy.model.encoders.pointcloud.focal_obs_encoder import (
    FocalPointCloudObsEncoder,
    FocalPointNetXYZ,
    FocalPointNetXYZRGB,
)
from diffusion_policy.model.encoders.shape_meta import obs_keys_by_type
from diffusion_policy.model.encoders.voxel.factory import build_voxel_obs_encoder


class ShapeMetaTest(unittest.TestCase):
    def test_aliases_and_ignored_keys(self):
        grouped = obs_keys_by_type({
            "obs": {
                "image": {"shape": [3, 8, 8], "type": "image"},
                "points": {"shape": [32, 3], "type": "pointcloud"},
                "volume": {"shape": [2, 8, 8, 8], "type": "voxels"},
                "state": {"shape": [4]},
                "ignored": {
                    "shape": [1],
                    "ignore_by_policy": True,
                },
            }
        })
        self.assertEqual(grouped["rgb"], ("image",))
        self.assertEqual(grouped["point_cloud"], ("points",))
        self.assertEqual(grouped["voxel"], ("volume",))
        self.assertEqual(grouped["low_dim"], ("state",))


class PointCloudEncoderTest(unittest.TestCase):
    def test_xyz_selection_and_low_dim_concat(self):
        shape_meta = {
            "obs": {
                "point_cloud": {
                    "shape": [32, 6],
                    "type": "point_cloud",
                },
                "state": {"shape": [9]},
            },
            "action": {"shape": [10]},
        }
        encoder = build_pointcloud_obs_encoder(
            "pointnet_xyz",
            shape_meta,
            encoder_kwargs={"feature_dim": 24},
        )
        output = encoder({
            "point_cloud": torch.randn(2, 32, 6),
            "state": torch.randn(2, 9),
        })
        self.assertEqual(encoder.output_shape(), (33,))
        self.assertEqual(tuple(output.shape), (2, 33))

    def test_channels_first(self):
        shape_meta = {
            "obs": {
                "points": {"shape": [3, 20], "type": "point_cloud"},
            },
            "action": {"shape": [2]},
        }
        encoder = build_pointcloud_obs_encoder(
            "pointnet",
            shape_meta,
            encoder_kwargs={
                "feature_dim": 12,
                "point_layout": "channels_first",
            },
        )
        self.assertEqual(
            tuple(encoder({"points": torch.randn(3, 3, 20)}).shape),
            (3, 12),
        )


class FocalPointCloudEncoderTest(unittest.TestCase):
    @staticmethod
    def _shape_meta(
            state_dim=9,
            point_channels=3,
            imagined=False,
            num_points=32):
        obs = {
            "point_cloud": {
                "shape": [num_points, point_channels],
                "type": "point_cloud",
            },
            "agent_pos": {"shape": [state_dim], "type": "low_dim"},
        }
        if imagined:
            obs["imagin_robot"] = {
                "shape": [7, point_channels],
                "type": "point_cloud",
            }
        return {"obs": obs, "action": {"shape": [4]}}

    def test_focal_factory_registration_and_active_output_shape(self):
        self.assertIn("pointnet_focal", available_pointcloud_models())
        for task_family, state_dim in (("metaworld", 9), ("adroit", 24)):
            with self.subTest(task_family=task_family):
                encoder = build_pointcloud_obs_encoder(
                    "pointnet_focal",
                    self._shape_meta(
                        state_dim=state_dim,
                        num_points=512,
                    ),
                )
                self.assertIsInstance(encoder, FocalPointCloudObsEncoder)
                output = encoder({
                    "point_cloud": torch.randn(2, 512, 3),
                    "agent_pos": torch.randn(2, state_dim),
                })
                self.assertEqual(encoder.output_shape(), (128,))
                self.assertEqual(tuple(output.shape), (2, 128))

    def test_xyz_and_xyzrgb_keep_distinct_upstream_topologies(self):
        xyz = FocalPointNetXYZ()
        xyzrgb = FocalPointNetXYZRGB()
        xyz_linears = [
            module for module in xyz.mlp if isinstance(module, torch.nn.Linear)
        ]
        xyzrgb_linears = [
            module for module in xyzrgb.mlp
            if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            [(m.in_features, m.out_features) for m in xyz_linears],
            [(3, 64), (64, 128), (128, 256)],
        )
        self.assertEqual(
            [(m.in_features, m.out_features) for m in xyzrgb_linears],
            [(6, 64), (64, 128), (128, 256), (256, 512)],
        )

    def test_xyz_preset_selects_leading_channels(self):
        encoder = build_pointcloud_obs_encoder(
            "pointnet_focal_xyz",
            self._shape_meta(point_channels=6),
        )
        encoder.eval()
        points = torch.randn(2, 32, 6)
        state = torch.randn(2, 9)
        output_a = encoder({"point_cloud": points, "agent_pos": state})
        points_with_other_color = points.clone()
        points_with_other_color[..., 3:] += 1000
        output_b = encoder({
            "point_cloud": points_with_other_color,
            "agent_pos": state,
        })
        torch.testing.assert_close(output_a, output_b)

    def test_imagined_points_are_pooled_with_observed_points(self):
        encoder = build_pointcloud_obs_encoder(
            "pointnet_focal",
            self._shape_meta(imagined=True),
        )
        encoder.eval()
        observed = torch.randn(2, 32, 3)
        imagined = torch.randn(2, 7, 3)
        state = torch.randn(2, 9)
        output = encoder({
            "point_cloud": observed,
            "imagin_robot": imagined,
            "agent_pos": state,
        })
        point_feature = encoder.extractor(
            torch.cat((observed, imagined), dim=1)
        )
        state_feature = encoder.state_mlp(state)
        expected = torch.cat((point_feature, state_feature), dim=-1)
        torch.testing.assert_close(output, expected)

    def test_xyzrgb_preset_requires_six_channels(self):
        with self.assertRaisesRegex(ValueError, "requires 6"):
            build_pointcloud_obs_encoder(
                "pointnet_focal_xyzrgb",
                self._shape_meta(point_channels=3),
            )


class VoxelEncoderTest(unittest.TestCase):
    def test_arbitrary_volume_and_crop(self):
        shape_meta = {
            "obs": {
                "voxels": {"shape": [4, 16, 20, 24], "type": "voxel"},
                "state": {"shape": [5]},
            },
            "action": {"shape": [2]},
        }
        encoder = build_voxel_obs_encoder(
            "voxel_cnn_small",
            shape_meta,
            encoder_kwargs={
                "feature_dim": 20,
                "crop_shape": [14, 18, 22],
            },
        )
        output = encoder({
            "voxels": torch.randn(2, 4, 16, 20, 24),
            "state": torch.randn(2, 5),
        })
        self.assertEqual(encoder.output_shape(), (25,))
        self.assertEqual(tuple(output.shape), (2, 25))


class MultiModalFactoryTest(unittest.TestCase):
    def test_rgb_only_factory_does_not_add_a_wrapper(self):
        shape_meta = {
            "obs": {
                "image": {"shape": [3, 84, 84], "type": "rgb"},
                "state": {"shape": [4]},
            },
            "action": {"shape": [2]},
        }
        sentinel = object()
        with patch(
                "diffusion_policy.model.encoders.rgb.factory."
                "build_rgb_obs_encoder",
                return_value=sentinel) as builder:
            result = build_obs_encoder(shape_meta, rgb_model="robomimic_raw")
        self.assertIs(result, sentinel)
        builder.assert_called_once_with(
            rgb_model="robomimic_raw",
            shape_meta=shape_meta,
            encoder_kwargs=None,
        )

    def test_pointcloud_voxel_and_low_dim_are_concatenated_once(self):
        shape_meta = {
            "obs": {
                "points": {"shape": [24, 3], "type": "point_cloud"},
                "volume": {"shape": [2, 12, 12, 12], "type": "voxel"},
                "state": {"shape": [4]},
            },
            "action": {"shape": [2]},
        }
        encoder = build_obs_encoder(
            shape_meta,
            pointcloud_encoder_kwargs={"feature_dim": 12},
            voxel_encoder_kwargs={
                "feature_dim": 16,
                "channels": [8, 16],
            },
        )
        output = encoder({
            "points": torch.randn(3, 24, 3),
            "volume": torch.randn(3, 2, 12, 12, 12),
            "state": torch.randn(3, 4),
        })
        self.assertEqual(tuple(encoder.encoders), ("point_cloud", "voxel"))
        self.assertEqual(encoder.output_shape(), (32,))
        self.assertEqual(tuple(output.shape), (3, 32))


if __name__ == "__main__":
    unittest.main()
