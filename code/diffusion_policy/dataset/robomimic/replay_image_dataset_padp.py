from typing import Dict, List, Optional
import os
import copy
import json
import numpy as np
import h5py
import zarr
from tqdm import tqdm
import multiprocessing

import torch
from omegaconf import OmegaConf

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.dataset.rtv8 import (
    RTV8CacheMessages,
    RTV8EpochStateMixin,
    build_balanced_epoch_pairs,
    load_or_build_rtv8_replay_buffer,
    resolve_mapping_index,
)
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
import hashlib
import tempfile
import time

register_codecs()


MAPPING_MANIFEST_SCHEMA_VERSION = 1


class RobomimicReplayImageDataset(RTV8EpochStateMixin, BaseImageDataset):
    """
    RTV8: 四步法实现（转换期物理化 + 每集映射 + 每 epoch 全局映射表文件化 + 运行期按表取数）

    关键点：
    - 在数据转换函数中，针对每个 episode 执行“首尾各复制 horizon-1 帧”的物理化填充；
      将填充后的序列直接写入 zarr（obs/action 与 rgb/low_dim），并并行生成 pos_info（与 obs 并列）
      和 episode_map（每集统计 real_len, real_len_new, buffer_pos_id_start/end, window_num 等）。
    - window_num_1 = real_len + horizon - 1；window_num_2 = real_len_new - horizon + 1；二者应恒等。
    - 每个 epoch 生成一个“全局二维映射总表”并保存到文件：形状为 [batch_size, max_cols, 2]，
      单元存 (episode_id, episode_get_id)。按列组织 batch：p=id%bs, q=id//bs。
    - 运行期按 epoch 读取映射表文件，__getitem__ 根据 (e, m) 计算绝对下标，直接切片 zarr 数组，返回
      obs(n_obs_steps)、action(horizon) 与 pos_info(horizon, 5)。
    """

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon: int = 1,
        n_obs_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        abs_action: bool = False,
        rotation_rep: str = 'rotation_6d',  # ignored when abs_action=False
        use_legacy_normalizer: bool = False,
        use_cache: bool = False,
        seed: int = 42,
        val_ratio: float = 0.0,
        n_demo: int = 100,
        batch_size: int = 1,
        mapping_dir: Optional[str] = None,
        build_epoch_mappings: int = 0,
        epoch_index: int = 0,
        num_epochs: Optional[int] = None,
        # 【v4改动·B Step1】图像归一化位置开关。False（默认）=CPU 内 astype(float32)/255（原行为）；
        # True=仅返回 uint8 张量，归一化交给 workspace 在 GPU 上做 float().div_(255.)，
        # 以降低 CPU 内存分配与 PCIe 带宽。json/yaml 通过 hydra 传入。
        uint8_to_gpu: bool = False,
    ):
        start_time = time.time()
        print(f"数据化初始化开始于: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.n_demo = n_demo
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 1
        self.uint8_to_gpu = bool(uint8_to_gpu)
        self.n_action_steps = int(n_action_steps) if n_action_steps is not None else self.horizon
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.batch_size = int(batch_size) if batch_size is not None else 1
        if self.batch_size <= 0:
            self.batch_size = 1

        rotation_transformer = RotationTransformer(from_rep='axis_angle', to_rep=rotation_rep)
        # 记录训练总 epoch 数（可用于预构建映射表）
        self.num_epochs = int(num_epochs) if num_epochs is not None else None

# 1+2 步：转换阶段完成“物理化填充 + pos_info + episode_map”
        rotation_tag = str(rotation_rep).replace('/', '_')
        shape_tag = self._shape_cache_tag(shape_meta)
        cache_zarr_path = (
            f'{dataset_path}'
            + f'.n{n_demo}.h{self.horizon}.obs{self.n_obs_steps}.act{self.n_action_steps}'
            + f'.abs{int(abs_action)}.rot{rotation_tag}.shape{shape_tag}'
            + f'.bs{self.batch_size}'
            + f'.v8.zarr'
        )

        def convert(store):
            return _convert_robomimic_to_replay_v8(
                store=store,
                shape_meta=shape_meta,
                dataset_path=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                n_demo=n_demo,
                horizon=self.horizon,
            )

        replay_buffer = load_or_build_rtv8_replay_buffer(
            cache_zarr_path=cache_zarr_path,
            use_cache=use_cache,
            converter=convert,
            messages=RTV8CacheMessages(
                acquiring='Acquiring lock on cache (RTV8).',
                creating='Cache folder does not exist. Creating (RTV8)!',
                created='Successfully created RTv8 DirectoryStore.',
                loading='Loading RTV8 ReplayBuffer directly from Disk folder.',
            ),
        )

        # 解析 obs 键（rgb / low_dim）
        rgb_keys: List[str] = []
        lowdim_keys: List[str] = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            t = attr.get('type', 'low_dim')
            if t == 'rgb':
                rgb_keys.append(key)
            elif t == 'low_dim':
                lowdim_keys.append(key)

        # 划分训练/验证（按 episode 级）
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask

        # 保存属性
        self.replay_buffer = replay_buffer
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.train_mask = train_mask
        self.seed = seed

        # episode 映射表（存放于 meta/episode_map）
        # episode_map: [n_episodes, 6] = [episode_id, real_len, real_len_new, buf_start(1-based), buf_end(1-based), window_num]
        self.episode_map = np.asarray(self.replay_buffer.meta['episode_map'][:], dtype=np.int64)
        assert self.episode_map.shape[0] == self.replay_buffer.n_episodes

        # 提取训练集 episodes
        self.train_episode_ids: np.ndarray = np.nonzero(self.train_mask)[0].astype(np.int64)
        if self.train_episode_ids.size == 0:
            raise RuntimeError('No training episodes available!')

        # 校验 window_num_1 == window_num_2（转换时已等价，这里再次稳妥检查）
        # window_num 已存为列 5
        # 若需要详细核对，可按 real_len_new - h + 1 的公式重新验证。
        self._validate_window_num(self.train_episode_ids)

        # 全局映射目录
        if mapping_dir is None:
            mapping_dir = (
                os.path.splitext(dataset_path)[0]
                + f'.rtdp_v8_maps.n{self.n_demo}.h{self.horizon}.obs{self.n_obs_steps}.act{self.n_action_steps}'
                + f'.b{self.batch_size}'
            )
        os.makedirs(mapping_dir, exist_ok=True)
        self.mapping_dir = mapping_dir

        # 映射表在生成时已经得到每个 epoch 的 max_cols。将这些数量一次性
        # 汇总到 manifest，供 workspace 构建与当前 task 真实映射长度一致的
        # cosine scheduler。后续 seed 只需读取 manifest，无需重新计算。
        planned_mapping_epochs = 0
        if build_epoch_mappings and build_epoch_mappings > 0:
            planned_mapping_epochs = int(build_epoch_mappings)
        elif self.num_epochs is not None and self.num_epochs > 0:
            planned_mapping_epochs = int(self.num_epochs)

        self.epoch_batch_counts = None
        self.total_training_batches = None
        if planned_mapping_epochs > 0:
            self._prepare_mapping_manifest(planned_mapping_epochs)

        # 加载当前 epoch 的映射表（若不存在则生成后加载）
        self._epoch_index = int(epoch_index)
        self._load_or_build_epoch_mapping(self._epoch_index, self.train_episode_ids)
        self._write_current_epoch_file(self._epoch_index)
        # 【v4改动·A.4】mtime 缓存初值，供 __getitem__ 跳过 open/read 噪声
        try:
            self._cached_epoch_mtime = os.stat(self._current_epoch_file_path()).st_mtime
        except Exception:
            self._cached_epoch_mtime = 0.0
        self._cached_epoch_value = int(self._epoch_index)
        print(f"数据化初始化结束在: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.train_mask = ~self.train_mask
        val_set.train_episode_ids = np.nonzero(val_set.train_mask)[0].astype(np.int64)
        if val_set.train_episode_ids.size == 0:
            return val_set

        val_set._validate_window_num(val_set.train_episode_ids)

        # 使用独立的映射目录
        val_set.mapping_dir = self.mapping_dir + '.val'
        os.makedirs(val_set.mapping_dir, exist_ok=True)

        # 默认加载或构建第 0 个 epoch 的映射表
        val_set._load_or_build_epoch_mapping(0, val_set.train_episode_ids)
        return val_set

    # ----------------- Normalizer -----------------
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs lowdim
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # obs image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    # ----------------- Dataloader 接口 -----------------
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 【v4改动·A.3】threadpool_limits(1) 已移到 worker_init_fn，此处删除
        # 【v4改动·A.4】原每样本 open/read/close current_epoch.txt 改为 os.stat mtime 比较
        self._sync_epoch_mapping_from_file()
        _, _, abs_start = resolve_mapping_index(
            self._global_pairs,
            self.episode_map,
            self.batch_size,
            self._max_cols,
            idx,
        )

        S = self.n_obs_steps
        h = self.horizon

        # 采样 obs/action/pos_info
        obs_dict = {}
        # 【v4改动·B Step4】rgb 路径已固化为默认走 uint8 路径（uint8_to_gpu=true 为 yaml 默认），
        # 由 workspace 在 GPU 上做 float().div_(255.) 归一化，降低 CPU/PCIe 负担、释放 worker 给 Zarr 解压。
        # low_dim / action / pos_info 不受影响，始终 CPU float32。
        # 如需回退对照（调试用），命令行 override training.uint8_to_gpu=false 即可走原 CPU /255 路径。
        if self.uint8_to_gpu:
            for key in self.rgb_keys:
                seq = self.replay_buffer[key][abs_start: abs_start + S]  # (S, H, W, C) uint8
                seq = np.moveaxis(seq, -1, 1)  # -> (S, C, H, W) uint8，不 astype、不 /255
                obs_dict[key] = seq
        else:
            for key in self.rgb_keys:
                seq = self.replay_buffer[key][abs_start: abs_start + S]  # (S, H, W, C)
                seq = np.moveaxis(seq, -1, 1).astype(np.float32) / 255.0  # -> (S, C, H, W) float32 [0,1]
                obs_dict[key] = seq
        for key in self.lowdim_keys:
            seq = self.replay_buffer[key][abs_start: abs_start + S].astype(np.float32)
            obs_dict[key] = seq

        actions = self.replay_buffer['action'][abs_start: abs_start + h].astype(np.float32)
        pos_info = self.replay_buffer['pos_info'][abs_start: abs_start + h].astype(np.int64)

        out = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(actions),
            'window_info': torch.from_numpy(pos_info),  # [h, 5]
        }
        return out

    # ######## 映射身份、manifest 路径与原子写入 ########
    def _mapping_fingerprint(self, episode_ids: np.ndarray) -> str:
        """Fingerprint every input that changes deterministic row mappings."""
        digest = hashlib.sha256()
        digest.update(np.asarray(
            [MAPPING_MANIFEST_SCHEMA_VERSION, self.batch_size, self.seed],
            dtype=np.int64,
        ).tobytes())
        digest.update(np.asarray(episode_ids, dtype=np.int64).tobytes())
        digest.update(np.asarray(self.episode_map, dtype=np.int64).tobytes())
        return digest.hexdigest()

    def _mapping_manifest_path(self) -> str:
        return os.path.join(self.mapping_dir, 'mapping_manifest.npz')

    @staticmethod
    def _npz_scalar(data, key):
        if key not in data.files:
            raise KeyError(key)
        return np.asarray(data[key]).item()

    @staticmethod
    def _atomic_savez(path: str, **arrays) -> None:
        """Atomically publish an npz so interrupted generation is recoverable."""
        fd, tmp_path = tempfile.mkstemp(
            prefix='.rtv8_tmp_', suffix='.npz',
            dir=os.path.dirname(path),
        )
        os.close(fd)
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    # ######## 单 epoch 映射有效性检查与缺失重建 ########
    def _read_mapping_metadata(
            self,
            epoch_index: int,
            episode_ids: np.ndarray,
    ) -> Optional[int]:
        """Return max_cols only when an archive matches the current dataset."""
        path = self._mapping_file_path(epoch_index)
        if not os.path.exists(path):
            return None
        expected_fingerprint = self._mapping_fingerprint(episode_ids)
        try:
            with np.load(path, allow_pickle=False) as data:
                if 'pairs' not in data.files:
                    return None
                schema_version = int(self._npz_scalar(
                    data, 'mapping_schema_version'))
                fingerprint = str(self._npz_scalar(
                    data, 'dataset_fingerprint'))
                batch_size = int(self._npz_scalar(data, 'batch_size'))
                mapping_seed = int(self._npz_scalar(data, 'mapping_seed'))
                saved_epoch = int(self._npz_scalar(data, 'epoch_index'))
                max_cols = int(self._npz_scalar(data, 'max_cols'))
        except (OSError, ValueError, KeyError, EOFError):
            return None

        if (
                schema_version != MAPPING_MANIFEST_SCHEMA_VERSION
                or fingerprint != expected_fingerprint
                or batch_size != self.batch_size
                or mapping_seed != self.seed
                or saved_epoch != int(epoch_index)
                or max_cols <= 0):
            return None
        return max_cols

    def _ensure_epoch_mapping(
            self,
            epoch_index: int,
            episode_ids: np.ndarray,
    ) -> int:
        max_cols = self._read_mapping_metadata(epoch_index, episode_ids)
        if max_cols is None:
            print(
                f'Mapping for epoch {epoch_index} is missing or stale. '
                'Building...'
            )
            self._build_and_save_epoch_mapping(epoch_index, episode_ids)
            max_cols = self._read_mapping_metadata(epoch_index, episode_ids)
        if max_cols is None:
            raise RuntimeError(
                f'Failed to build a valid RTV8 mapping for epoch {epoch_index}'
            )
        return max_cols

    # ######## 全训练周期 manifest 校验、生成与对外读取 ########
    def _load_valid_mapping_manifest(
            self,
            num_epochs: int,
            episode_ids: np.ndarray,
    ) -> Optional[np.ndarray]:
        path = self._mapping_manifest_path()
        if not os.path.exists(path):
            return None
        expected_fingerprint = self._mapping_fingerprint(episode_ids)
        try:
            with np.load(path, allow_pickle=False) as data:
                schema_version = int(self._npz_scalar(
                    data, 'mapping_schema_version'))
                fingerprint = str(self._npz_scalar(
                    data, 'dataset_fingerprint'))
                batch_size = int(self._npz_scalar(data, 'batch_size'))
                mapping_seed = int(self._npz_scalar(data, 'mapping_seed'))
                saved_num_epochs = int(self._npz_scalar(
                    data, 'num_epochs'))
                counts = np.asarray(
                    data['epoch_batch_counts'], dtype=np.int64).copy()
                total_batches = int(self._npz_scalar(
                    data, 'total_batches'))
        except (OSError, ValueError, KeyError, EOFError):
            return None

        if (
                schema_version != MAPPING_MANIFEST_SCHEMA_VERSION
                or fingerprint != expected_fingerprint
                or batch_size != self.batch_size
                or mapping_seed != self.seed
                or saved_num_epochs != int(num_epochs)
                or counts.shape != (int(num_epochs),)
                or np.any(counts <= 0)
                or int(counts.sum()) != total_batches):
            return None

        # A valid manifest is only reusable when all archives it summarizes
        # still exist. Archive content is checked lazily by set_epoch(); a
        # deterministic rebuild has the same max_cols recorded here.
        if any(
                not os.path.exists(self._mapping_file_path(epoch_index))
                for epoch_index in range(int(num_epochs))):
            return None
        return counts

    def _prepare_mapping_manifest(self, num_epochs: int) -> np.ndarray:
        num_epochs = int(num_epochs)
        if num_epochs <= 0:
            raise ValueError('num_epochs must be positive')

        counts = self._load_valid_mapping_manifest(
            num_epochs, self.train_episode_ids)
        if counts is None:
            counts = np.asarray([
                self._ensure_epoch_mapping(
                    epoch_index, self.train_episode_ids)
                for epoch_index in range(num_epochs)
            ], dtype=np.int64)
            fingerprint = self._mapping_fingerprint(self.train_episode_ids)
            self._atomic_savez(
                self._mapping_manifest_path(),
                mapping_schema_version=np.asarray(
                    MAPPING_MANIFEST_SCHEMA_VERSION, dtype=np.int64),
                dataset_fingerprint=np.asarray(fingerprint),
                mapping_seed=np.asarray(self.seed, dtype=np.int64),
                batch_size=np.asarray(self.batch_size, dtype=np.int64),
                num_epochs=np.asarray(num_epochs, dtype=np.int64),
                epoch_batch_counts=counts,
                total_batches=np.asarray(counts.sum(), dtype=np.int64),
            )

        self.epoch_batch_counts = counts
        self.total_training_batches = int(counts.sum())
        return counts.copy()

    def get_epoch_batch_counts(self, num_epochs: Optional[int] = None) -> np.ndarray:
        if num_epochs is None:
            num_epochs = self.num_epochs
        if num_epochs is None:
            raise ValueError('num_epochs is required for RTV8 batch planning')
        num_epochs = int(num_epochs)
        if (
                self.epoch_batch_counts is None
                or len(self.epoch_batch_counts) != num_epochs):
            return self._prepare_mapping_manifest(num_epochs)
        return self.epoch_batch_counts.copy()

    # ######## 运行期加载当前 epoch 映射 ########
    def _load_or_build_epoch_mapping(self, epoch_index: int, episode_ids: np.ndarray):
        expected_max_cols = self._ensure_epoch_mapping(
            epoch_index, episode_ids)
        path = self._mapping_file_path(epoch_index)
        with np.load(path, allow_pickle=False) as data:
            pairs = np.asarray(data['pairs'], dtype=np.int32)
            max_cols = int(self._npz_scalar(data, 'max_cols'))

        if pairs.shape != (self.batch_size, max_cols, 2):
            raise RuntimeError(
                f'RTV8 mapping epoch {epoch_index} has shape {pairs.shape}; '
                f'expected {(self.batch_size, max_cols, 2)}'
            )
        if max_cols != expected_max_cols:
            raise RuntimeError(
                f'RTV8 mapping epoch {epoch_index} metadata changed while loading'
            )
        unique_ep_ids = np.unique(pairs[..., 0])
        if not np.all(np.isin(unique_ep_ids, episode_ids)):
            raise RuntimeError(
                f'RTV8 mapping epoch {epoch_index} contains episodes '
                'outside the current split'
            )

        self._global_pairs = pairs.astype(np.int32)
        self._max_cols = max_cols

    # ######## 生成并保存单 epoch 映射 ########
    def _build_and_save_epoch_mapping(self, epoch_index: int, episode_ids: np.ndarray):
        start_time = time.time()
        print(f"Epoch mapping 创建开始于: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        dataset_fingerprint = self._mapping_fingerprint(episode_ids)
        pairs, max_cols = build_balanced_epoch_pairs(
            episode_map=self.episode_map,
            episode_ids=episode_ids,
            batch_size=self.batch_size,
            seed=self.seed,
            epoch_index=epoch_index,
        )

        path = self._mapping_file_path(epoch_index)
        self._atomic_savez(
            path,
            pairs=pairs, 
            mapping_schema_version=np.asarray(
                MAPPING_MANIFEST_SCHEMA_VERSION, dtype=np.int64),
            dataset_fingerprint=np.asarray(dataset_fingerprint),
            mapping_seed=np.asarray(self.seed, dtype=np.int64),
            epoch_index=np.asarray(epoch_index, dtype=np.int64),
            batch_size=np.asarray(self.batch_size, dtype=np.int64),
            max_cols=np.asarray(max_cols, dtype=np.int64),
        )
        print(f"Epoch mapping 创建结束在 {time.time() - start_time:.2f} seconds")
    
    def _read_current_epoch_file(self) -> Optional[int]:
        try:
            with open(self._current_epoch_file_path(), 'r') as f:
                txt = f.read().strip()
            if txt:
                return int(txt)
        except Exception:
            return None
        return None

    def _validate_window_num(self, episode_ids: np.ndarray):
        # 验证 window_num_1 == window_num_2
        # window_num_1 = real_len + h - 1；window_num_2 = real_len_new - h + 1
        h = self.horizon
        bad: List[int] = []
        for e in episode_ids:
            real_len = int(self.episode_map[e, 1])
            real_len_new = int(self.episode_map[e, 2])
            win1 = real_len + h - 1
            win2 = real_len_new - h + 1
            if win1 != win2 or win1 != int(self.episode_map[e, 5]):
                bad.append(int(e))
        if len(bad) > 0:
            print('Window num mismatch for episode ids:', bad)
            raise RuntimeError(
                f'window_num_1 != window_num_2 for some episodes! '
                f'Current horizon={self.horizon}. '
                f'If you changed horizon / abs_action / rotation_rep, delete the old RTV8 cache and rebuild it.'
            )

    def _shape_cache_tag(self, shape_meta: dict) -> str:
        plain_shape_meta = OmegaConf.to_container(shape_meta, resolve=True)
        payload = {
            'action': plain_shape_meta.get('action', {}),
            'obs': {
                key: {
                    'type': value.get('type', 'low_dim'),
                    'shape': value.get('shape', None),
                }
                for key, value in sorted(plain_shape_meta.get('obs', {}).items())
            },
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.md5(raw.encode()).hexdigest()[:10]


# ----------------- 转换：物理化填充 + pos_info + episode_map -----------------
def _convert_robomimic_to_replay_v8(
    store,
    shape_meta,
    dataset_path,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    n_demo=100,
    horizon=1,
):
    start_time = time.time()
    print(f"数据化转换开始于：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    h = int(horizon)
    pad_b = max(h - 1, 0)
    pad_a = max(h - 1, 0)

    # 解析 obs 键
    rgb_keys: List[str] = []
    lowdim_keys: List[str] = []
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        t = attr.get('type', 'low_dim')
        if t == 'rgb':
            rgb_keys.append(key)
        elif t == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    # 读取一次，统计每集长度与填充后长度
    with h5py.File(dataset_path, 'r') as file:
        demos = file['data']
        real_lens = []
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            real_lens.append(int(demo['actions'].shape[0]))
        real_lens = np.asarray(real_lens, dtype=np.int64)
        real_len_news = real_lens + 2 * (h - 1)
        total_padded = int(real_len_news.sum())

        # 创建 episode_map 数组（注意：应存放在 meta 组，避免违反 ReplayBuffer 对 data 组长度一致性的断言）
        episode_map = np.zeros((n_demo, 6), dtype=np.int64)
        episode_map_arr = meta_group.array(
            name='episode_map',
            data=np.zeros((n_demo, 6), dtype=np.int64),
            shape=(n_demo, 6),
            chunks=(n_demo, 6), # 一次性写入
            dtype=np.int64,
            compressor=None,
            overwrite=True
        )

        # 确保填充后的总长度与累计和一致
        assert total_padded == real_len_news.sum(), "Total padded length mismatch"
        episode_ends_new = np.cumsum(real_len_news)
        # 确保最后一个累计和等于总长度
        assert episode_ends_new[-1] == total_padded, "Episode ends mismatch with total padded length"
        _ = meta_group.array('episode_ends', episode_ends_new, dtype=np.int64, compressor=None, overwrite=True)

        # 分配 action / lowdim 数组
        # 注：chunks=full shape 仅为简单起见，action/lowdim 为整体写入；图像用 (1,H,W,C) chunk。
        # 动作
        action_dim = int(np.prod(shape_meta['action']['shape']))
        action_arr = data_group.array(
            name='action',
            data=np.empty((total_padded, action_dim), dtype=np.float32),
            shape=(total_padded, action_dim),
            chunks=(min(total_padded, max(1, 1024)), action_dim),
            compressor=None,
            dtype=np.float32,
            overwrite=True,
        )

        # 低维
        lowdim_arrs: Dict[str, zarr.Array] = {}
        for key in lowdim_keys:
            this_shape = (total_padded,) + tuple(shape_meta['obs'][key]['shape'])
            lowdim_arrs[key] = data_group.array(
                name=key,
                data=np.empty(this_shape, dtype=np.float32),
                shape=this_shape,
                chunks=(min(total_padded, max(1, 1024)),) + tuple(shape_meta['obs'][key]['shape']),
                compressor=None,
                dtype=np.float32,
                overwrite=True,
            )

        # 图像
        from numcodecs import Blosc
        img_arrs: Dict[str, zarr.Array] = {}
        for key in rgb_keys:
            c, h_img, w_img = tuple(shape_meta['obs'][key]['shape'])
            # 使用业界高效的 Blosc-LZ4 压缩，显著降低 CPU 解压负担
            this_compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.NOSHUFFLE)
            img_arrs[key] = data_group.require_dataset(
                name=key,
                shape=(total_padded, h_img, w_img, c),
                # 增大 chunks 到 16，提升 NVMe 硬盘的随机读取效能
                chunks=(16, h_img, w_img, c),
                compressor=this_compressor,
                dtype=np.uint8,
            )

        # pos_info: [total_padded, 5]
        pos_info_arr = data_group.array(
            name='pos_info',
            data=np.empty((total_padded, 5), dtype=np.int64),
            shape=(total_padded, 5),
            chunks=(min(total_padded, max(1, 1024)), 5),
            compressor=None,
            dtype=np.int64,
            overwrite=True,
        )

        # episode_map: [n_demo, 6] = [episode_id, real_len, real_len_new, buf_start(1b), buf_end(1b), window_num]
        episode_map = np.zeros((n_demo, 6), dtype=np.int64)
        write_ptr = 0  # 全局 0-based 写指针

        # 遍历每集，生成填充写入
        for e in tqdm(range(n_demo), desc='RTV8: materialize padded episodes'):
            demo = demos[f'demo_{e}']
            real_len = real_lens[e]
            real_len_new = real_len_news[e]
            win_num = real_len + h - 1

            # 计算该集写入区间 [ofs, ofs+real_len_new)
            ofs = write_ptr
            ofs_end = ofs + real_len_new

            # 读取本集原始数据
            act_raw = demo['actions'][:].astype(np.float32)
            act_raw = _convert_actions_v8(act_raw, abs_action=abs_action, rotation_transformer=rotation_transformer)

            # 填充后的 action 写入
            # 前 pad_b 段
            if pad_b > 0:
                action_arr[ofs: ofs + pad_b] = act_raw[0:1].repeat(pad_b, axis=0)
            # 中间原始段
            action_arr[ofs + pad_b: ofs + pad_b + real_len] = act_raw
            # 尾 pad_a 段
            if pad_a > 0:
                action_arr[ofs + pad_b + real_len: ofs_end] = act_raw[-1: ].repeat(pad_a, axis=0)

            # 低维 obs 填充
            for key in lowdim_keys:
                arr = demo['obs'][key][:].astype(np.float32)
                if pad_b > 0:
                    lowdim_arrs[key][ofs: ofs + pad_b] = arr[0:1].repeat(pad_b, axis=0)
                lowdim_arrs[key][ofs + pad_b: ofs + pad_b + real_len] = arr
                if pad_a > 0:
                    lowdim_arrs[key][ofs + pad_b + real_len: ofs_end] = arr[-1: ].repeat(pad_a, axis=0)

            # # 图像 obs 填充（逐帧复制）
            # for key in rgb_keys:
            #     arr = demo['obs'][key]
            #     # 前 pad_b
            #     for t in range(pad_b):
            #         img_arrs[key][ofs + t] = arr[0]
            #     # 中段原始
            #     for t in range(real_len):
            #         img_arrs[key][ofs + pad_b + t] = arr[t]
            #     # 尾 pad_a
            #     for t in range(pad_a):
            #         img_arrs[key][ofs + pad_b + real_len + t] = arr[-1]

            # 图像 obs 填充（向量化操作）
            for key in rgb_keys:
                arr = demo['obs'][key]
                # 创建填充后的图像数组
                padded_arr = np.empty((real_len_new, *arr.shape[1:]), dtype=arr.dtype)
                
                # 前 pad_b
                if pad_b > 0:
                    padded_arr[:pad_b] = arr[0]
                
                # 中段原始
                padded_arr[pad_b:pad_b+real_len] = arr[:real_len]
                
                # 尾 pad_a
                if pad_a > 0:
                    padded_arr[pad_b+real_len:] = arr[-1]
                
                # 批量写入
                img_arrs[key][ofs:ofs+real_len_new] = padded_arr

            # pos_info 构建
            # 列说明：0=is_warmup, 1=episode_index, 2=episode_new_pos_id, 3=episode_id, 4=buffer_pos_id(1-based)
            pos = np.zeros((real_len_new, 5), dtype=np.int64)
            # is_warmup: 前 pad_b 段为 1
            if pad_b > 0:
                pos[:pad_b, 0] = 1
            # episode_index：前段=0，中段=0..real_len-1，尾段=real_len-1
            if real_len > 0:
                pos[pad_b: pad_b + real_len, 1] = np.arange(real_len, dtype=np.int64)
                if pad_b > 0:
                    pos[:pad_b, 1] = 0
                if pad_a > 0:
                    pos[pad_b + real_len:, 1] = real_len - 1
            # episode_new_pos_id：0..real_len_new-1
            pos[:, 2] = np.arange(real_len_new, dtype=np.int64)
            # episode_id
            pos[:, 3] = e
            # buffer_pos_id（1-based，全局）
            buf_start_1b = ofs + 1
            pos[:, 4] = buf_start_1b + pos[:, 2]
            # 写入 pos_info
            pos_info_arr[ofs: ofs_end] = pos

            # 更新episode_map数组（而不是最后一次性写入）
            episode_map[e, 0] = e
            episode_map[e, 1] = real_len
            episode_map[e, 2] = real_len_new
            episode_map[e, 3] = buf_start_1b
            episode_map[e, 4] = ofs_end  # 最后一个 1-based = ofs_end
            episode_map[e, 5] = win_num

            # 立即写入当前episode_map行（确保数据持久化）
            episode_map_arr[e] = episode_map[e]

            # 前进写指针
            write_ptr = ofs_end

        # 将最终 episode_map 写入 meta 组（data 组内仅包含时间维度一致的序列数据）
        episode_map_arr[:, :] = episode_map

    # 在转换函数的最后，创建 ReplayBuffer 之前添加：
    assert write_ptr == total_padded, f"Write pointer ({write_ptr}) does not match total padded length ({total_padded})"
    
    # 确保文件夹元数据刷新到磁盘
    if hasattr(store, 'flush'):
        store.flush()
        
    print(f"数据化转换结束在： {time.time() - start_time:.2f} seconds")
    # 创建ReplayBuffer（直接关联到刚才写入的磁盘 root 组）
    replay_buffer = ReplayBuffer(root)
    print(f"数据化转换后保存结束在： {time.time() - start_time:.2f} seconds")
    return replay_buffer


# ----------------- 工具：动作转换 -----------------
def _convert_actions_v8(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([pos, rot, gripper], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1 / max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat,
    )
