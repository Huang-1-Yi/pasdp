import unittest

import numpy as np

from diffusion_policy.envs.gym_util.point_sampling import point_cloud_sampling
from diffusion_policy.envs.gym_util.scalar_recorder import LargestKRecorder


class PointCloudSamplingTest(unittest.TestCase):
    def test_torch_fps_preserves_rows_and_extra_channels(self):
        points = np.arange(60, dtype=np.float32).reshape(10, 6)
        sampled = point_cloud_sampling(
            points, 4, method="fps", fps_backend="torch")
        self.assertEqual(sampled.shape, (4, 6))
        self.assertEqual(sampled.dtype, np.float32)
        for row in sampled:
            self.assertTrue(any(np.array_equal(row, item) for item in points))

    def test_short_cloud_is_zero_padded(self):
        points = np.ones((2, 3), dtype=np.float32)
        sampled = point_cloud_sampling(points, 5)
        np.testing.assert_array_equal(sampled[:2], points)
        np.testing.assert_array_equal(sampled[2:], np.zeros((3, 3)))

    def test_empty_cloud_keeps_channel_count(self):
        sampled = point_cloud_sampling(
            np.empty((0, 6), dtype=np.float32), 3)
        self.assertEqual(sampled.shape, (3, 6))
        self.assertEqual(sampled.dtype, np.float32)

    def test_invalid_values_are_rejected(self):
        points = np.zeros((8, 3), dtype=np.float32)
        points[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            point_cloud_sampling(points, 4)


class LargestKRecorderTest(unittest.TestCase):
    def test_retains_largest_values(self):
        recorder = LargestKRecorder(k=3)
        for value in (0.2, 0.8, 0.4, 0.9, 0.1):
            recorder.record(value)
        self.assertAlmostEqual(recorder.average(), (0.4 + 0.8 + 0.9) / 3)


if __name__ == "__main__":
    unittest.main()
