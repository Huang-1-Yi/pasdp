import tempfile
import unittest

import numpy as np
import zarr

from diffusion_policy.dataset.metaworld.metaworld_pointcloud_dataset import (
    MetaWorldPointCloudDataset,
)


class MetaWorldPointCloudDatasetTest(unittest.TestCase):
    def _make_dataset(self, root_path):
        root = zarr.open(root_path, mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        rng = np.random.default_rng(7)
        data.array("state", rng.normal(size=(10, 9)).astype(np.float32))
        data.array("action", rng.normal(size=(10, 4)).astype(np.float32))
        data.array(
            "point_cloud",
            rng.normal(size=(10, 4, 6)).astype(np.float32),
        )
        meta.array("episode_ends", np.asarray([5, 10], dtype=np.int64))

    def test_xyz_shapes_split_normalizer_and_loss_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp)
            dataset = MetaWorldPointCloudDataset(
                zarr_path=tmp,
                horizon=3,
                n_obs_steps=1,
                val_ratio=0.5,
                num_points=4,
                point_cloud_channels=3,
            )
            self.assertEqual(len(dataset), 3)
            self.assertEqual(len(dataset.get_validation_dataset()), 3)
            sample = dataset[0]
            self.assertEqual(tuple(sample["obs"]["point_cloud"].shape), (1, 4, 3))
            self.assertEqual(tuple(sample["obs"]["agent_pos"].shape), (1, 9))
            self.assertEqual(tuple(sample["action"].shape), (3, 4))
            self.assertEqual(
                set(dataset.get_normalizer().params_dict),
                {"action", "agent_pos", "point_cloud"},
            )

    def test_point_count_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp)
            with self.assertRaisesRegex(ValueError, "silently changing"):
                MetaWorldPointCloudDataset(zarr_path=tmp, num_points=512)


if __name__ == "__main__":
    unittest.main()
