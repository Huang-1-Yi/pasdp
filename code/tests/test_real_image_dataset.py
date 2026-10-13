"""Small recorded fixtures exercise video alignment, actions, and cache reuse."""

import tempfile
from pathlib import Path
import unittest

import av
import numpy as np
import torch
import zarr

from diffusion_policy.dataset.real.real_image_dataset import RealImageDataset
from diffusion_policy.envs.realword.real_image_runner import RealImageRunner


class RealImageDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.source = self.base / "recordings"
        self.source.mkdir()
        root = zarr.open_group(str(self.source / "replay_buffer.zarr"), mode="w")
        root.require_group("meta").array("episode_ends", np.array([8, 16]))
        data = root.require_group("data")
        self.actions = np.arange(16 * 15, dtype=np.float64).reshape(16, 15) / 100
        self.actions[:, [6, 7, 14]] = np.arange(16)[:, None] % 2
        data.array("action", self.actions)
        data.array("timestamp", np.r_[np.arange(8) / 10, 10 + np.arange(8) / 10])
        data.array("left_robot_eef_pose", self.actions[:, :6])
        data.array("left_gripper_state", self.actions[:, 6:8])
        data.array("right_robot_eef_pose", self.actions[:, 8:14])
        data.array("right_gripper_state", self.actions[:, 14:15])
        data.array("stage", np.arange(16))
        for episode in range(2):
            directory = self.source / "videos" / str(episode)
            directory.mkdir(parents=True)
            for camera in range(2):
                with av.open(str(directory / f"{camera}.mp4"), mode="w") as container:
                    stream = container.add_stream("mpeg4", rate=10)
                    stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
                    for index in range(8):
                        values = np.full((24, 32, 3), 20 + camera * 100 + index * 5, np.uint8)
                        frame = av.VideoFrame.from_ndarray(values, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    for packet in stream.encode():
                        container.mux(packet)
        self.meta = {"obs": {
            "camera_0": {"shape": [3, 24, 32], "type": "rgb"},
            "camera_1": {"shape": [3, 24, 32], "type": "rgb"},
            "left_robot_eef_pose": {"shape": [6]},
            "left_gripper_state": {"shape": [2]},
            "right_robot_eef_pose": {"shape": [6]},
            "right_gripper_state": {"shape": [1]},
            "stage": {"shape": [1]},
        }, "action": {"shape": [15]}}

    def tearDown(self):
        self.temp.cleanup()

    def make_dataset(self, **kwargs):
        return RealImageDataset(self.meta, str(self.source), horizon=4, n_obs_steps=1,
                                cache_dir=str(self.base / "cache"), **kwargs)

    def test_actions_video_shapes_and_cache_reuse(self):
        before = sorted(str(p.relative_to(self.source)) for p in self.source.rglob("*"))
        dataset = self.make_dataset()
        sample = dataset[0]
        np.testing.assert_array_equal(sample["action"].numpy(), self.actions[:4].astype(np.float32))
        self.assertEqual(sample["action"].shape, (4, 15))
        self.assertEqual(sample["obs"]["camera_0"].shape, (1, 3, 24, 32))
        self.assertEqual(sample["obs"]["left_gripper_state"].shape, (1, 2))
        self.assertEqual(sample["obs"]["stage"].shape, (1, 1))
        self.assertGreater(float(sample["obs"]["camera_1"].mean()), float(sample["obs"]["camera_0"].mean()))
        self.assertAlmostEqual(float(dataset[1]["obs"]["camera_0"].mean()) * 255, 25, delta=5)
        cached = self.make_dataset()
        self.assertEqual(dataset.cache_path, cached.cache_path)
        np.testing.assert_array_equal(dataset.replay_buffer["action"][:], self.actions)
        self.assertEqual(before, sorted(str(p.relative_to(self.source)) for p in self.source.rglob("*")))

    def test_split_normalizer_and_latency(self):
        dataset = self.make_dataset(val_ratio=0.5, n_latency_steps=1)
        self.assertEqual(len(dataset.get_validation_dataset()), 4)
        self.assertEqual(dataset[0]["action"].shape, (4, 15))
        normalizer = dataset.get_normalizer()
        action = dataset[0]["action"]
        restored = normalizer["action"].unnormalize(normalizer["action"].normalize(action))
        self.assertTrue(torch.allclose(restored, action, atol=1e-5))
        ends = dataset.replay_buffer.episode_ends[:]
        selected = np.flatnonzero(dataset.train_mask)[0]
        start = 0 if selected == 0 else ends[selected - 1]
        expected = self.actions[start:ends[selected]].min(axis=0)
        np.testing.assert_allclose(normalizer["action"].get_input_stats()["min"].numpy(), expected, atol=1e-6)

    def test_missing_video_and_shape_mismatch_fail(self):
        meta = dict(self.meta, action={"shape": [7]})
        with self.assertRaisesRegex(ValueError, "action: expected"):
            RealImageDataset(meta, str(self.source), horizon=4, use_cache=False)
        (self.source / "videos/1/1.mp4").unlink()
        with self.assertRaises(FileNotFoundError):
            self.make_dataset()

    def test_smoke_episode_limit_and_offline_runner(self):
        dataset = self.make_dataset(max_episodes=1)
        self.assertEqual(dataset.replay_buffer.n_episodes, 1)
        self.assertEqual(dataset.replay_buffer["action"].shape, (8, 15))
        self.assertEqual(RealImageRunner(str(self.base)).run(None), {})


if __name__ == "__main__":
    unittest.main()
