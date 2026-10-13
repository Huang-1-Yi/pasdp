import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from diffusion_policy.dataset.rtv8 import (
    RTV8CacheMessages,
    RTV8EpochStateMixin,
    build_balanced_epoch_pairs,
    load_or_build_rtv8_replay_buffer,
    resolve_mapping_index,
)


class RTV8CommonTest(unittest.TestCase):
    def setUp(self):
        self.episode_map = np.zeros((5, 6), dtype=np.int64)
        self.episode_map[:, 0] = np.arange(5)
        self.episode_map[:, 3] = [1, 101, 201, 301, 401]
        self.episode_map[:, 5] = [3, 1, 4, 2, 5]
        self.episode_ids = np.arange(5, dtype=np.int64)

    def test_mapping_matches_frozen_legacy_fixture(self):
        expected = {
            0: np.asarray([
                [1, 0, 0, 0, 0, 1, 0, 2, 3, 0, 3, 1],
                [4, 0, 4, 1, 4, 2, 4, 3, 4, 4, 1, 0],
                [2, 0, 2, 1, 2, 2, 2, 3, 1, 0, 4, 0],
            ], dtype=np.int32).reshape(3, 6, 2),
            1: np.asarray([
                [3, 0, 3, 1, 4, 0, 4, 1, 4, 2, 4, 3, 4, 4],
                [2, 0, 2, 1, 2, 2, 2, 3, 3, 0, 3, 1, 2, 0],
                [1, 0, 0, 0, 0, 1, 0, 2, 3, 0, 3, 1, 2, 0],
            ], dtype=np.int32).reshape(3, 7, 2),
        }
        for epoch_index, expected_pairs in expected.items():
            with self.subTest(epoch=epoch_index):
                pairs, max_cols = build_balanced_epoch_pairs(
                    episode_map=self.episode_map,
                    episode_ids=self.episode_ids,
                    batch_size=3,
                    seed=42,
                    epoch_index=epoch_index,
                )
                np.testing.assert_array_equal(pairs, expected_pairs)
                self.assertEqual(max_cols, expected_pairs.shape[1])
                self.assertEqual(pairs.dtype, np.int32)

    def test_all_three_adapters_write_identical_pairs(self):
        from diffusion_policy.dataset.adroit.adroit_pointcloud_dataset_padp import (
            AdroitPointCloudDatasetPadp,
        )
        from diffusion_policy.dataset.metaworld.metaworld_pointcloud_dataset_padp import (
            MetaWorldPointCloudDatasetPadp,
        )
        from diffusion_policy.dataset.robomimic.replay_image_dataset_padp import (
            RobomimicReplayImageDataset,
        )

        outputs = []
        for dataset_cls in (
                RobomimicReplayImageDataset,
                MetaWorldPointCloudDatasetPadp,
                AdroitPointCloudDatasetPadp):
            with self.subTest(dataset=dataset_cls.__name__), \
                    tempfile.TemporaryDirectory() as tmp:
                dataset = dataset_cls.__new__(dataset_cls)
                dataset.seed = 42
                dataset.batch_size = 3
                dataset.episode_map = self.episode_map
                dataset.train_episode_ids = self.episode_ids
                dataset.mapping_dir = tmp
                dataset._build_and_save_epoch_mapping(7, self.episode_ids)
                with np.load(dataset._mapping_file_path(7)) as data:
                    outputs.append(data["pairs"].copy())
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[0], outputs[2])

    def test_robomimic_manifest_collects_generation_time_batch_counts(self):
        from diffusion_policy.dataset.robomimic.replay_image_dataset_padp import (
            MAPPING_MANIFEST_SCHEMA_VERSION,
            RobomimicReplayImageDataset,
        )

        def make_dataset(mapping_dir):
            dataset = RobomimicReplayImageDataset.__new__(
                RobomimicReplayImageDataset)
            dataset.seed = 42
            dataset.batch_size = 3
            dataset.episode_map = self.episode_map.copy()
            dataset.train_episode_ids = self.episode_ids.copy()
            dataset.mapping_dir = mapping_dir
            dataset.num_epochs = 3
            dataset.epoch_batch_counts = None
            dataset.total_training_batches = None
            return dataset

        with tempfile.TemporaryDirectory() as tmp:
            dataset = make_dataset(tmp)
            counts = dataset._prepare_mapping_manifest(3)
            expected = np.asarray([
                build_balanced_epoch_pairs(
                    self.episode_map,
                    self.episode_ids,
                    3,
                    42,
                    epoch_index,
                )[1]
                for epoch_index in range(3)
            ], dtype=np.int64)
            np.testing.assert_array_equal(counts, expected)
            self.assertEqual(
                dataset.total_training_batches, int(expected.sum()))

            with np.load(dataset._mapping_manifest_path()) as manifest:
                self.assertEqual(
                    int(manifest['mapping_schema_version']),
                    MAPPING_MANIFEST_SCHEMA_VERSION,
                )
                np.testing.assert_array_equal(
                    manifest['epoch_batch_counts'], expected)
                self.assertEqual(
                    int(manifest['total_batches']), int(expected.sum()))

            # A later seed/run loads one aggregate manifest and does not
            # regenerate or rescan the per-epoch mapping metadata.
            reused = make_dataset(tmp)
            with mock.patch.object(
                    reused,
                    '_ensure_epoch_mapping',
                    side_effect=AssertionError('mapping should be reused')):
                reused_counts = reused._prepare_mapping_manifest(3)
            np.testing.assert_array_equal(reused_counts, expected)

    def test_robomimic_manifest_invalidates_when_mapping_inputs_change(self):
        from diffusion_policy.dataset.robomimic.replay_image_dataset_padp import (
            RobomimicReplayImageDataset,
        )

        def make_dataset(mapping_dir, episode_map):
            dataset = RobomimicReplayImageDataset.__new__(
                RobomimicReplayImageDataset)
            dataset.seed = 42
            dataset.batch_size = 3
            dataset.episode_map = episode_map
            dataset.train_episode_ids = self.episode_ids.copy()
            dataset.mapping_dir = mapping_dir
            dataset.num_epochs = 2
            dataset.epoch_batch_counts = None
            dataset.total_training_batches = None
            return dataset

        with tempfile.TemporaryDirectory() as tmp:
            original = make_dataset(tmp, self.episode_map.copy())
            original._prepare_mapping_manifest(2)
            original_fingerprint = original._mapping_fingerprint(
                original.train_episode_ids)

            changed_map = self.episode_map.copy()
            changed_map[0, 5] += 9
            changed = make_dataset(tmp, changed_map)
            changed_counts = changed._prepare_mapping_manifest(2)
            changed_fingerprint = changed._mapping_fingerprint(
                changed.train_episode_ids)

            self.assertNotEqual(original_fingerprint, changed_fingerprint)
            with np.load(changed._mapping_manifest_path()) as manifest:
                self.assertEqual(
                    str(manifest['dataset_fingerprint']),
                    changed_fingerprint,
                )
                np.testing.assert_array_equal(
                    manifest['epoch_batch_counts'], changed_counts)

    def test_index_resolution_matches_legacy_arithmetic(self):
        pairs, max_cols = build_balanced_epoch_pairs(
            self.episode_map, self.episode_ids, 3, 42, 0)
        for idx in (0, 1, 2, 3, 8, 17, 999):
            p = idx % 3
            q = min(idx // 3, max_cols - 1)
            expected_e = int(pairs[p, q, 0])
            expected_m = int(pairs[p, q, 1])
            expected_abs = (
                int(self.episode_map[expected_e, 3]) - 1 + expected_m)
            self.assertEqual(
                resolve_mapping_index(
                    pairs, self.episode_map, 3, max_cols, idx),
                (expected_e, expected_m, expected_abs),
            )

    def test_epoch_state_keeps_legacy_publish_and_reload_order(self):
        class DummyDataset(RTV8EpochStateMixin):
            def __init__(self, mapping_dir):
                self.mapping_dir = mapping_dir
                self.train_episode_ids = np.asarray([2, 4], dtype=np.int64)
                self.batch_size = 2
                self._max_cols = 3
                self.calls = []

            def _load_or_build_epoch_mapping(self, epoch_index, episode_ids):
                self.calls.append((int(epoch_index), episode_ids.copy()))

        with tempfile.TemporaryDirectory() as tmp:
            dataset = DummyDataset(tmp)
            dataset.set_epoch(5)
            self.assertEqual(dataset._epoch_index, 5)
            self.assertEqual(dataset._cached_epoch_value, 5)
            self.assertEqual(len(dataset), 6)
            self.assertEqual(dataset.calls[0][0], 5)
            np.testing.assert_array_equal(
                dataset.calls[0][1], dataset.train_episode_ids)
            with open(dataset._current_epoch_file_path(), "r") as f:
                self.assertEqual(f.read(), "5")

            dataset._cached_epoch_mtime = -1.0
            dataset._cached_epoch_value = 4
            self.assertEqual(dataset._read_current_epoch_fast(), 5)

    def test_cache_helper_preserves_memory_build_disk_build_and_load_paths(self):
        messages = RTV8CacheMessages(
            acquiring="lock",
            creating="create",
            created="created",
            loading="load",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = f"{tmp}/cache.zarr"
            stores = []

            def converter(store):
                import zarr
                zarr.group(store).attrs["built"] = True
                stores.append(store)
                return "built"

            result = load_or_build_rtv8_replay_buffer(
                cache_path, False, converter, messages)
            self.assertEqual(result, "built")
            self.assertEqual(type(stores[-1]).__name__, "MemoryStore")

            result = load_or_build_rtv8_replay_buffer(
                cache_path, True, converter, messages)
            self.assertEqual(result, "built")
            self.assertEqual(type(stores[-1]).__name__, "DirectoryStore")

            with mock.patch(
                    "diffusion_policy.dataset.rtv8.cache.ReplayBuffer",
                    return_value="loaded") as replay_buffer_cls:
                result = load_or_build_rtv8_replay_buffer(
                    cache_path, True, converter, messages)
            self.assertEqual(result, "loaded")
            replay_buffer_cls.assert_called_once()

    def test_cache_helper_removes_partial_directory_on_conversion_failure(self):
        messages = RTV8CacheMessages("lock", "create", "created", "load")
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = f"{tmp}/partial.zarr"

            def fail_converter(store):
                import zarr
                zarr.group(store).attrs["partial"] = True
                raise RuntimeError("conversion failed")

            with self.assertRaisesRegex(RuntimeError, "conversion failed"):
                load_or_build_rtv8_replay_buffer(
                    cache_path, True, fail_converter, messages)
            self.assertFalse(os.path.exists(cache_path))


if __name__ == "__main__":
    unittest.main()
