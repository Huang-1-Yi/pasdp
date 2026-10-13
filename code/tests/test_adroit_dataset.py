import tempfile
import unittest

import numpy as np
import zarr

from diffusion_policy.dataset.adroit.adroit_pointcloud_dataset import (
    AdroitPointCloudDataset,
)


class AdroitPointCloudDatasetTest(unittest.TestCase):
    def _make_dataset(self, root_path, action_dim=28):
        root = zarr.open(root_path, mode="w")
        data = root.create_group("data")
        meta = root.create_group("meta")
        rng = np.random.default_rng(7)
        data.array("state", rng.normal(size=(12, 24)).astype(np.float32))
        data.array(
            "action", rng.normal(size=(12, action_dim)).astype(np.float32))
        data.array(
            "point_cloud",
            rng.normal(size=(12, 8, 6)).astype(np.float32),
        )
        meta.array("episode_ends", np.asarray([6, 12], dtype=np.int64))

    def test_xyz_shapes_split_and_normalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp)
            dataset = AdroitPointCloudDataset(
                zarr_path=tmp,
                horizon=4,
                n_obs_steps=2,
                val_ratio=0.5,
                num_points=8,
                task_name="door",
            )
            self.assertEqual(len(dataset), 3)
            self.assertEqual(len(dataset.get_validation_dataset()), 3)
            sample = dataset[0]
            self.assertEqual(tuple(sample["obs"]["point_cloud"].shape), (2, 8, 3))
            self.assertEqual(tuple(sample["obs"]["agent_pos"].shape), (2, 24))
            self.assertEqual(tuple(sample["action"].shape), (4, 28))
            self.assertEqual(
                set(dataset.get_normalizer().params_dict),
                {"action", "agent_pos", "point_cloud"},
            )

    def test_action_dimension_mismatch_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._make_dataset(tmp, action_dim=26)
            with self.assertRaisesRegex(
                    ValueError, "expects state/action dimensions 24/28"):
                AdroitPointCloudDataset(
                    zarr_path=tmp,
                    num_points=8,
                    task_name="door",
                )


if __name__ == "__main__":
    unittest.main()
