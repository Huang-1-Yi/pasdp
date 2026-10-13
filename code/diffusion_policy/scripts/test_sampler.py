import unittest

import numpy as np

from diffusion_policy.common.sampler import (
    BalancedEpisodeBatchSamplerV2,
    SequenceSampler,
    create_indices,
)


class FakeReplayBuffer:
    def __init__(self, episode_lengths, data=None):
        self.episode_ends = np.cumsum(episode_lengths, dtype=np.int64)
        self._data = {} if data is None else dict(data)

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class FailingArray:
    shape = (3, 2)
    dtype = np.dtype(np.uint8)

    def __getitem__(self, item):
        raise OSError("synthetic read failure")


def make_v2(episode_lengths, batch_size, **kwargs):
    replay_buffer = FakeReplayBuffer(episode_lengths)
    return BalancedEpisodeBatchSamplerV2(
        replay_buffer=replay_buffer,
        sequence_length=1,
        keys=[],
        batch_size=batch_size,
        **kwargs,
    )


def flatten_batches(batches):
    return [idx for batch in batches for idx in batch]


def episode_ids_for_batch(sampler, batch):
    starts = sampler.indices[np.asarray(batch, dtype=np.int64), 0]
    episode_ends = np.asarray(sampler.replay_buffer.episode_ends[:])
    return np.searchsorted(episode_ends, starts, side="right")


class SequenceSamplerRobustnessTest(unittest.TestCase):
    def test_create_indices_checks_episode_mask_shape(self):
        with self.assertRaises(AssertionError):
            create_indices(
                episode_ends=np.array([2], dtype=np.int64),
                sequence_length=1,
                episode_mask=np.array([True, False]),
            )

    def test_key_first_k_is_copied_per_instance(self):
        replay_buffer = FakeReplayBuffer(
            [1], data={"value": np.array([1], dtype=np.float32)}
        )
        source = {"value": 1}
        first = SequenceSampler(
            replay_buffer, sequence_length=1, keys=["value"], key_first_k=source
        )
        second = SequenceSampler(
            replay_buffer, sequence_length=1, keys=["value"]
        )

        source["value"] = 99
        first.key_first_k["extra"] = 2

        self.assertEqual(first.key_first_k["value"], 1)
        self.assertEqual(second.key_first_k, {})

    def test_key_first_k_fill_values_preserve_dtype(self):
        replay_buffer = FakeReplayBuffer(
            [3],
            data={
                "integer": np.array([1, 2, 3], dtype=np.uint8),
                "boolean": np.array([True, False, True], dtype=np.bool_),
                "floating": np.array([1.0, 2.0, 3.0], dtype=np.float32),
            },
        )
        sampler = SequenceSampler(
            replay_buffer,
            sequence_length=3,
            keys=["integer", "boolean", "floating"],
            key_first_k={"integer": 1, "boolean": 1, "floating": 1},
        )

        sample = sampler.sample_sequence(0)

        np.testing.assert_array_equal(
            sample["integer"], np.array([1, 0, 0], dtype=np.uint8)
        )
        np.testing.assert_array_equal(
            sample["boolean"], np.array([True, False, False], dtype=np.bool_)
        )
        self.assertEqual(sample["floating"].dtype, np.float32)
        self.assertEqual(sample["integer"].dtype, np.uint8)
        self.assertEqual(sample["boolean"].dtype, np.bool_)
        self.assertTrue(np.isnan(sample["floating"][1:]).all())

    def test_key_first_k_read_error_contains_context_and_cause(self):
        replay_buffer = FakeReplayBuffer([3], data={"broken": FailingArray()})
        sampler = SequenceSampler(
            replay_buffer,
            sequence_length=3,
            keys=["broken"],
            key_first_k={"broken": 1},
        )

        with self.assertRaises(RuntimeError) as caught:
            sampler.sample_sequence(0)

        message = str(caught.exception)
        for expected in (
                "key='broken'",
                "buffer_start_idx=0",
                "buffer_end_idx=3",
                "k_data=1",
                "input_shape=(3, 2)",
                "input_dtype=uint8"):
            self.assertIn(expected, message)
        self.assertIsInstance(caught.exception.__cause__, OSError)

    def test_empty_source_window_has_explicit_error(self):
        replay_buffer = FakeReplayBuffer(
            [1], data={"value": np.array([1], dtype=np.float32)}
        )
        sampler = SequenceSampler(
            replay_buffer, sequence_length=1, keys=["value"]
        )
        sampler.indices = np.array([[0, 0, 0, 0]], dtype=np.int64)

        with self.assertRaisesRegex(RuntimeError, "empty source window"):
            sampler.sample_sequence(0)

    def test_existing_padding_semantics_are_unchanged(self):
        replay_buffer = FakeReplayBuffer(
            [3], data={"value": np.array([10, 20, 30], dtype=np.int64)}
        )
        sampler = SequenceSampler(
            replay_buffer,
            sequence_length=4,
            pad_before=1,
            pad_after=2,
            keys=["value"],
        )

        np.testing.assert_array_equal(
            sampler.sample_sequence(0)["value"], [10, 10, 20, 30]
        )
        np.testing.assert_array_equal(
            sampler.sample_sequence(2)["value"], [20, 30, 30, 30]
        )


class BalancedEpisodeBatchSamplerV2Test(unittest.TestCase):
    def assert_complete_without_replacement(self, sampler, batches):
        indices = flatten_batches(batches)
        self.assertEqual(len(indices), sampler.total_samples)
        self.assertEqual(len(set(indices)), sampler.total_samples)
        self.assertEqual(set(indices), set(range(sampler.total_samples)))

    def test_all_indices_are_used_once(self):
        sampler = make_v2([5, 4, 3], batch_size=5)
        batches = list(sampler)

        self.assert_complete_without_replacement(sampler, batches)

    def test_short_final_batch_and_drop_last(self):
        keep_tail = make_v2([4, 3, 3], batch_size=4, drop_last=False)
        drop_tail = make_v2([4, 3, 3], batch_size=4, drop_last=True)

        keep_batches = list(keep_tail)
        drop_batches = list(drop_tail)

        self.assertEqual([len(batch) for batch in keep_batches], [4, 4, 2])
        self.assertEqual([len(batch) for batch in drop_batches], [4, 4])
        self.assert_complete_without_replacement(keep_tail, keep_batches)
        dropped_indices = flatten_batches(drop_batches)
        self.assertEqual(len(dropped_indices), 8)
        self.assertEqual(len(set(dropped_indices)), 8)

    def test_batch_maximises_distinct_episode_coverage(self):
        sampler = make_v2([2, 2, 2, 2, 2, 2], batch_size=4)
        first_batch = next(iter(sampler))
        episode_ids = episode_ids_for_batch(sampler, first_batch)

        self.assertEqual(len(np.unique(episode_ids)), 4)

    def test_fewer_active_episodes_are_balanced_within_batch(self):
        sampler = make_v2([10, 10, 10], batch_size=8)
        first_batch = next(iter(sampler))
        episode_ids = episode_ids_for_batch(sampler, first_batch)
        counts = np.bincount(episode_ids, minlength=3)

        self.assertLessEqual(int(counts.max() - counts.min()), 1)

    def test_unequal_episode_lengths_do_not_reset_short_pools(self):
        sampler = make_v2([10, 5, 2], batch_size=4)
        batches = list(sampler)

        self.assert_complete_without_replacement(sampler, batches)
        episode_batches = [episode_ids_for_batch(sampler, batch) for batch in batches]
        batches_with_short_episode = [
            i for i, episode_ids in enumerate(episode_batches)
            if 2 in episode_ids
        ]
        self.assertEqual(batches_with_short_episode, [0, 1])
        self.assertEqual(
            sum(int(np.sum(episode_ids == 2)) for episode_ids in episode_batches),
            2,
        )

    def test_epoch_order_is_reproducible_and_changes_with_epoch(self):
        sampler = make_v2([8, 7, 6, 5], batch_size=5, seed=123)

        sampler.set_epoch(0)
        epoch_zero_first = list(sampler)
        sampler.set_epoch(0)
        epoch_zero_second = list(sampler)
        sampler.set_epoch(1)
        epoch_one = list(sampler)

        self.assertEqual(epoch_zero_first, epoch_zero_second)
        self.assertNotEqual(epoch_zero_first, epoch_one)
        self.assert_complete_without_replacement(sampler, epoch_zero_first)
        self.assert_complete_without_replacement(sampler, epoch_one)

    def test_num_batches_per_epoch_must_be_natural_count(self):
        inferred = make_v2([10], batch_size=4)
        explicit = make_v2(
            [10], batch_size=4, num_batches_per_epoch=3
        )

        self.assertEqual(len(inferred), 3)
        self.assertEqual(len(explicit), 3)
        with self.assertRaisesRegex(ValueError, "natural number"):
            make_v2([10], batch_size=4, num_batches_per_epoch=2)
        with self.assertRaisesRegex(ValueError, "natural number"):
            make_v2([10], batch_size=4, num_batches_per_epoch=4)

    def test_len_reports_batches_and_repr_reports_contract(self):
        sampler = make_v2([5, 5], batch_size=4, drop_last=False)

        self.assertEqual(len(sampler), 3)
        self.assertEqual(sampler.n_episodes, 2)
        self.assertEqual(sampler.total_samples, 10)
        self.assertIn("drop_last=False", repr(sampler))

    def test_empty_and_all_false_episode_masks(self):
        empty = make_v2([], batch_size=4)
        replay_buffer = FakeReplayBuffer([3, 2])
        all_false = BalancedEpisodeBatchSamplerV2(
            replay_buffer=replay_buffer,
            sequence_length=1,
            keys=[],
            episode_mask=np.array([False, False]),
            batch_size=4,
        )

        for sampler in (empty, all_false):
            self.assertEqual(list(sampler), [])
            self.assertEqual(len(sampler), 0)
            self.assertEqual(sampler.total_samples, 0)


if __name__ == "__main__":
    unittest.main()
