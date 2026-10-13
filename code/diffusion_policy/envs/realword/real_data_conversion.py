"""Convert recorded Zarr + MP4 data, adapted from task_padp/0518.

Uses this repository's ReplayBuffer, timestamp-based read_video and image codecs.
Only the supplied output store is written; the recordings are opened read-only.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import av
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.imagecodecs_numcodecs import Jpeg2k, register_codecs
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.envs.realword.video_recorder import read_video


def real_data_to_replay_buffer(
        dataset_path, out_store=None, out_resolutions=None,
        lowdim_keys=None, image_keys=None, lowdim_shapes=None,
        max_episodes=None, n_decoding_threads=2, n_encoding_threads=4):
    """Copy selected trajectories and align video frames to their time grids.

    max_episodes is an explicit first-N-episodes limit for smoke tests. Normal
    training uses all episodes. Output action columns are never rearranged.
    """
    register_codecs(verbose=False)
    source = Path(dataset_path).expanduser().resolve()
    root = zarr.open_group(str(source / "replay_buffer.zarr"), mode="r")
    ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    if not len(ends) or ends[0] <= 0 or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be nonempty and strictly increasing")
    if max_episodes is not None:
        if int(max_episodes) < 1:
            raise ValueError("max_episodes must be positive")
        ends = ends[:int(max_episodes)]
    total = int(ends[-1])
    data = root["data"]
    keys = list(lowdim_keys or data.array_keys())
    shapes = lowdim_shapes or {}
    output = zarr.group(store=out_store if out_store is not None else zarr.MemoryStore())
    out_data = output.require_group("data")
    output.require_group("meta").array("episode_ends", ends, overwrite=True)
    for key in keys:
        values = np.asarray(data[key][:total])
        if len(values) != total or not np.isfinite(values).all():
            raise ValueError(f"Invalid length or nonfinite data: {key}")
        expected = shapes.get(key)
        if expected is not None:
            expected = tuple(expected)
            if values.ndim == 1 and expected == (1,):
                values = values[:, None]
            if values.shape[1:] != expected:
                raise ValueError(f"{key}: expected {expected}, got {values.shape[1:]}")
        out_data.array(key, values, overwrite=True)

    times = np.asarray(data["timestamp"][:total], dtype=np.float64)
    if times.shape != (total,) or not np.isfinite(times).all():
        raise ValueError("Invalid timestamp array")
    if image_keys is None:
        image_keys = [f"camera_{p.stem}" for p in sorted((source / "videos/0").glob("*.mp4"))]
    if not image_keys:
        raise ValueError("No cameras selected")
    starts = np.r_[0, ends[:-1]]
    jobs = []
    for key in image_keys:
        camera = int(key.rsplit("_", 1)[1])
        first_video = source / "videos/0" / f"{camera}.mp4"
        with av.open(str(first_video)) as container:
            context = container.streams.video[0].codec_context
            resolution = (context.width, context.height)
        if isinstance(out_resolutions, dict):
            resolution = tuple(out_resolutions.get(key, resolution))
        elif out_resolutions is not None:
            resolution = tuple(out_resolutions)
        width, height = resolution
        out_data.create_dataset(key, shape=(total, height, width, 3),
                                chunks=(1, height, width, 3), dtype="uint8",
                                compressor=Jpeg2k(level=50), overwrite=True)
        for episode, (start, end) in enumerate(zip(starts, ends)):
            video = source / "videos" / str(episode) / f"{camera}.mp4"
            if not video.is_file():
                raise FileNotFoundError(video)
            grid = times[start:end]
            if len(grid) < 2:
                raise ValueError(f"Episode {episode} needs at least two timestamps")
            delta = np.diff(grid)
            dt = float(np.median(delta))
            if dt <= 0 or not np.allclose(delta, dt, rtol=1e-3, atol=1e-5):
                raise ValueError(f"Episode {episode}: timestamps are not a uniform grid")
            jobs.append((key, video, int(start), int(end), dt, resolution))

    def convert_video(job):
        key, video, start, end, dt, resolution = job
        with av.open(str(video)) as container:
            context = container.streams.video[0].codec_context
            input_res = (context.width, context.height)
        transform = get_image_transform(input_res=input_res, output_res=resolution,
                                        bgr_to_rgb=False)
        count = 0
        frames = read_video(str(video), dt=dt, img_transform=transform,
                            thread_type="FRAME", thread_count=int(n_decoding_threads))
        try:
            for count, frame in enumerate(frames, start=1):
                out_data[key][start + count - 1] = frame
                if count == end - start:
                    break
        finally:
            frames.close()
        if count != end - start:
            raise ValueError(f"{video}: {count} frames for {end - start} control steps")
        return count

    with ThreadPoolExecutor(max_workers=max(1, int(n_encoding_threads))) as executor:
        with tqdm(total=total * len(image_keys), desc="Caching real images") as progress:
            for count in executor.map(convert_video, jobs):
                progress.update(count)
    return ReplayBuffer(output)
