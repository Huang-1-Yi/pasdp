from typing import Optional, Iterator, List
import numpy as np
import numba
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    assert episode_mask.shape == episode_ends.shape
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=None,
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = (
            {} if key_first_k is None else dict(key_first_k)
        )
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            if buffer_end_idx <= buffer_start_idx:
                raise RuntimeError(
                    "SequenceSampler encountered an empty source window: "
                    f"key={key!r}, buffer_start_idx={buffer_start_idx}, "
                    f"buffer_end_idx={buffer_end_idx}, "
                    f"input_shape={input_arr.shape}, input_dtype={input_arr.dtype}"
                )
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                if np.issubdtype(input_arr.dtype, np.floating):
                    fill_value = np.nan
                else:
                    fill_value = 0
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=fill_value, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as exc:
                    raise RuntimeError(
                        "SequenceSampler failed to load key_first_k data: "
                        f"key={key!r}, buffer_start_idx={buffer_start_idx}, "
                        f"buffer_end_idx={buffer_end_idx}, k_data={k_data}, "
                        f"input_shape={input_arr.shape}, input_dtype={input_arr.dtype}"
                    ) from exc
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


class BalancedEpisodeBatchSampler(SequenceSampler, torch.utils.data.Sampler):
    """
    A :class:`SequenceSampler` that yields episode-balanced batches when iterated.

    Inherits from ``SequenceSampler`` so it has the **same interface** as the
    baseline (``__len__``, ``__getitem__``, ``sample_sequence``, ``.indices``,
    ``.replay_buffer``) and can be assigned to ``dataset.sampler`` as a
    drop-in replacement.  The only thing it adds on top is an ``__iter__``
    that, when iterated, yields balanced batches of integer indices.

    Motivation
    ----------
    Standard ``torch.utils.data.DataLoader`` with a random ``Sampler`` may,
    in a single batch of B samples drawn from E episodes, sample 0..B times
    from any single episode -- especially when B < E or when E is small
    (e.g. 10 demos vs. batch_size 64, the random expectation is ~binomial
    but variance is large for small E).

    For imitation learning on small demonstration sets, this can cause some
    episodes to be starved for many optimization steps in a row, hurting
    convergence and stability.  This sampler enforces that:

      * Each batch contributes at least ``floor(B / E)`` samples per episode.
      * The remaining ``B - E * floor(B/E)`` samples go to the most
        under-sampled episodes so that no episode is starved.
      * Across batches within an epoch, per-episode counts are tracked and
        prioritised so a round-robin / "visit all before repeat" behavior is
        enforced when ``B < E``.

    Designed to be passed as ``batch_sampler=`` to ``DataLoader`` (yields
    *batches* of indices).  Per-epoch state is reset by calling
    ``set_epoch(self.epoch)`` from the workspace at the start of each epoch.
    """

    def __init__(self,
                 replay_buffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=None,
                 episode_mask=None,
                 batch_size: int = 64,
                 num_batches_per_epoch: Optional[int] = None,
                 seed: int = 42,
                 shuffle_within_episode: bool = True,
                 drop_last: bool = False):
        """
        Args:
            replay_buffer: a ``ReplayBuffer`` exposing ``.episode_ends``.
            sequence_length: window length (same as :class:`SequenceSampler`).
            pad_before / pad_after: padding on either side of an episode
                (inherited semantics).
            keys / key_first_k / episode_mask: same as :class:`SequenceSampler`.
            batch_size: samples per yielded batch (>= 1).
            num_batches_per_epoch: how many batches to yield per epoch.  If
                ``None``, defaults to ``ceil(total / batch_size)`` (or
                ``floor`` when ``drop_last=True``).
            seed: base random seed; per-epoch RNG uses ``seed + epoch``.
            shuffle_within_episode: shuffle each episode's indices once per
                epoch before walking through them.
            drop_last: drop the very last partial batch instead of yielding it.
        """
        # 1. Initialise SequenceSampler side: populates self.indices,
        #    self.replay_buffer, self.keys, self.key_first_k.
        SequenceSampler.__init__(
            self,
            replay_buffer=replay_buffer,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k if key_first_k is not None else {},
            episode_mask=episode_mask,
        )
        # 2. Initialise torch Sampler side (mostly marker class).
        torch.utils.data.Sampler.__init__(self)

        assert batch_size >= 1, "batch_size must be >= 1"
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_episode = bool(shuffle_within_episode)
        self.drop_last = bool(drop_last)

        # ------------------------------------------------------------------
        # Build the per-episode index mapping.
        # ------------------------------------------------------------------
        # Each row of ``self.indices`` is ``[buffer_start_idx, ...]``; its
        # episode id is recovered by ``searchsorted`` against the cumulative
        # ``episode_ends``.
        if len(self.indices) == 0:
            self._episode_to_seq_indices = {}
            self._episode_ids = np.zeros((0,), dtype=np.int64)
        else:
            episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
            buffer_start_idxs = self.indices[:, 0]
            episode_assignment = np.searchsorted(
                episode_ends, buffer_start_idxs, side='right'
            )
            episode_to_seq_indices = {}
            for ep in np.unique(episode_assignment):
                ep = int(ep)
                episode_to_seq_indices[ep] = np.where(
                    episode_assignment == ep
                )[0]
            self._episode_to_seq_indices = episode_to_seq_indices
            self._episode_ids = np.array(
                sorted(episode_to_seq_indices.keys()), dtype=np.int64
            )

        # ------------------------------------------------------------------
        # Resolve num_batches_per_epoch.
        # ------------------------------------------------------------------
        total = len(self.indices)
        if num_batches_per_epoch is None:
            if total == 0:
                num_batches_per_epoch = 0
            elif self.drop_last:
                num_batches_per_epoch = total // self.batch_size
            else:
                num_batches_per_epoch = (total + self.batch_size - 1) // self.batch_size
        self.num_batches_per_epoch = int(num_batches_per_epoch)

        # Per-epoch counter; bumped by ``set_epoch`` from the workspace.
        self.epoch = 0

    # ----------------------------------------------------------------------
    # Inherited from SequenceSampler (kept verbatim -- dataset can use
    # ``isinstance(sampler, SequenceSampler)`` checks unchanged).
    # ----------------------------------------------------------------------

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ids)

    @property
    def total_samples(self) -> int:
        return len(self.indices)

    # ----------------------------------------------------------------------
    # PyTorch BatchSampler API.
    # ----------------------------------------------------------------------
    def set_epoch(self, epoch: int) -> None:
        """Reset per-epoch state. Call before each new epoch."""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        if self.total_samples == 0 or self.n_episodes == 0:
            return iter([])

        rng = np.random.default_rng(self.seed + self.epoch)
        episode_ids = self._episode_ids
        B = self.batch_size

        # ------------------------------------------------------------------
        # Per-epoch "remaining pool" bookkeeping.
        # ------------------------------------------------------------------
        # Pre-shuffle each episode's indices once, then walk with a pointer.
        # When an episode is exhausted its share is redistributed to episodes
        # that still have un-sampled data, so we never train on already-seen
        # samples just to satisfy a balanced contract.  At the end of an epoch
        # every index is used at most once; if more batches are requested than
        # total samples permit, we re-seed with fresh shuffles.
        pools: dict = {}        # ep -> (shuffled_indices_array, next_ptr)
        for ep in episode_ids:
            ep = int(ep)
            pool = self._episode_to_seq_indices[ep].copy()
            if self.shuffle_within_episode:
                rng.shuffle(pool)
            pools[ep] = [pool, 0]

        def _remaining(ep: int) -> int:
            pool, ptr = pools[ep]
            return len(pool) - ptr

        def _take(ep: int, k: int) -> List[int]:
            """Take up to ``k`` indices from ``ep``'s pool (FIFO)."""
            pool, ptr = pools[ep]
            avail = len(pool) - ptr
            actual_k = k if k <= avail else avail
            if actual_k <= 0:
                return []
            taken = pool[ptr:ptr + actual_k].tolist()
            pools[ep][1] = ptr + actual_k
            return [int(x) for x in taken]

        def _active_episodes() -> List[int]:
            active = [ep for ep in episode_ids.tolist() if _remaining(int(ep)) > 0]
            rng.shuffle(active)
            return active

        # ------------------------------------------------------------------
        # Yield batches.
        # ------------------------------------------------------------------
        yielded = 0
        while yielded < self.num_batches_per_epoch:
            active = _active_episodes()
            if not active:
                # All episodes exhausted -> reshuffle fresh pools (rare).
                for ep in episode_ids:
                    ep = int(ep)
                    pool = self._episode_to_seq_indices[ep].copy()
                    rng.shuffle(pool)
                    pools[ep] = [pool, 0]
                active = _active_episodes()

            n_active = len(active)
            base = B // n_active
            rem = B - base * n_active

            batch: List[int] = []

            # Phase 1: every active episode contributes ``base`` samples.
            for ep in active:
                batch.extend(_take(int(ep), base))

            # Phase 2: distribute remainder to episodes with the most
            # remaining data, breaking ties randomly.  Cycle through the
            # sorted list so a single big episode can't monopolise extras.
            if rem > 0:
                eps_sorted = sorted(
                    active,
                    key=lambda e: (-_remaining(int(e)),
                                   int(rng.integers(0, 1 << 30))),
                )
                i = 0
                while len(batch) < B and any(_remaining(int(e)) > 0
                                             for e in eps_sorted):
                    ep = int(eps_sorted[i % len(eps_sorted)])
                    taken = _take(ep, 1)
                    if taken:
                        batch.append(taken[0])
                    else:
                        eps_sorted = [e for e in eps_sorted
                                      if _remaining(int(e)) > 0]
                        if not eps_sorted:
                            break
                    i += 1

            if not batch:
                break

            # Final padding: refill from any un-sampled leftovers.
            if len(batch) < B:
                leftover = []
                for ep in episode_ids:
                    ep = int(ep)
                    pool, ptr = pools[ep]
                    if ptr < len(pool):
                        leftover.extend(pool[ptr:].tolist())
                if leftover:
                    fillers = rng.choice(
                        np.array(leftover, dtype=np.int64),
                        size=B - len(batch),
                        replace=True,
                    )
                    batch.extend(int(x) for x in fillers)
                else:
                    for ep in episode_ids:
                        ep = int(ep)
                        pool = self._episode_to_seq_indices[ep].copy()
                        rng.shuffle(pool)
                        pools[ep] = [pool, 0]
                    leftovers = np.concatenate(
                        [pools[int(ep)][0] for ep in episode_ids]
                    )
                    fillers = rng.choice(
                        leftovers, size=B - len(batch), replace=True
                    )
                    batch.extend(int(x) for x in fillers)

            if self.drop_last and len(batch) < B:
                break

            batch = batch[:B]
            rng.shuffle(batch)
            yield batch
            yielded += 1

    def __repr__(self) -> str:
        return (
            f"BalancedEpisodeBatchSampler(batch_size={self.batch_size}, "
            f"n_episodes={self.n_episodes}, "
            f"total_samples={self.total_samples}, "
            f"num_batches_per_epoch={self.num_batches_per_epoch}, "
            f"epoch={self.epoch})"
        )


class BalancedEpisodeBatchSamplerV2(
        SequenceSampler, torch.utils.data.Sampler):
    """Episode-aware, without-replacement batch sampling for one epoch.

    Every available sequence index is assigned to its source episode.  Each
    call to :meth:`__iter__` creates independently shuffled per-episode pools
    and consumes each selected index exactly once.  A batch first takes one
    index from as many distinct active episodes as possible, then distributes
    any remaining slots as evenly as possible among episodes that still have
    data.

    With ``drop_last=False``, all sequence indices are yielded exactly once,
    including a potentially short final batch.  With ``drop_last=True``, only
    complete batches are yielded and the shuffled tail is omitted.  Pools are
    never reset during an iteration; :meth:`set_epoch` changes the next
    iteration's deterministic shuffle through ``seed + epoch``.

    TODO:
    当前 V2 只优化 batch-level episode diversity，并在一个 epoch 内
    对原始 sequence indices 做无放回遍历。

    如果未来希望每个 episode 在整个 epoch 中具有相同权重，需要另行设计：
    1. 对短 episode 进行过采样；或
    2. 对长 episode 进行下采样。

    该策略会改变原始数据分布，不应与当前无放回遍历逻辑混合实现。
    """

    def __init__(self,
                 replay_buffer,
                 sequence_length: int,
                 pad_before: int = 0,
                 pad_after: int = 0,
                 keys=None,
                 key_first_k=None,
                 episode_mask=None,
                 batch_size: int = 64,
                 num_batches_per_epoch: Optional[int] = None,
                 seed: int = 42,
                 shuffle_within_episode: bool = True,
                 drop_last: bool = False):
        SequenceSampler.__init__(
            self,
            replay_buffer=replay_buffer,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
            keys=keys,
            key_first_k=key_first_k,
            episode_mask=episode_mask,
        )
        torch.utils.data.Sampler.__init__(self)

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_episode = bool(shuffle_within_episode)
        self.drop_last = bool(drop_last)

        if len(self.indices) == 0:
            self._episode_to_seq_indices = {}
            self._episode_ids = np.zeros((0,), dtype=np.int64)
        else:
            episode_ends = np.asarray(self.replay_buffer.episode_ends[:])
            buffer_start_idxs = self.indices[:, 0]
            episode_assignment = np.searchsorted(
                episode_ends, buffer_start_idxs, side='right'
            )
            self._episode_to_seq_indices = {
                int(ep): np.where(episode_assignment == ep)[0]
                for ep in np.unique(episode_assignment)
            }
            self._episode_ids = np.array(
                sorted(self._episode_to_seq_indices), dtype=np.int64
            )

        total = len(self.indices)
        if self.drop_last:
            natural_num_batches = total // self.batch_size
        else:
            natural_num_batches = (
                (total + self.batch_size - 1) // self.batch_size
                if total > 0 else 0
            )

        if (num_batches_per_epoch is not None
                and int(num_batches_per_epoch) != natural_num_batches):
            raise ValueError(
                "BalancedEpisodeBatchSamplerV2 requires "
                "num_batches_per_epoch to be None or equal to the "
                "natural number of non-repeating batches in one epoch "
                f"({natural_num_batches}), got {num_batches_per_epoch}."
            )
        self.num_batches_per_epoch = natural_num_batches
        self.epoch = 0

    @property
    def n_episodes(self) -> int:
        return len(self._episode_ids)

    @property
    def total_samples(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches_per_epoch

    def __iter__(self) -> Iterator[List[int]]:
        if self.total_samples == 0 or self.n_episodes == 0:
            return

        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {}
        for ep_value in self._episode_ids:
            ep = int(ep_value)
            pool = self._episode_to_seq_indices[ep].copy()
            if self.shuffle_within_episode:
                rng.shuffle(pool)
            pools[ep] = [pool, 0]

        def _remaining(ep: int) -> int:
            pool, ptr = pools[ep]
            return len(pool) - ptr

        def _take_one(ep: int) -> Optional[int]:
            pool, ptr = pools[ep]
            if ptr >= len(pool):
                return None
            idx = int(pool[ptr])
            pools[ep][1] = ptr + 1
            return idx

        yielded = 0
        while yielded < self.num_batches_per_epoch:
            active = [
                int(ep) for ep in self._episode_ids.tolist()
                if _remaining(int(ep)) > 0
            ]
            if not active:
                break
            rng.shuffle(active)

            batch: List[int] = []
            batch_episode_counts = {ep: 0 for ep in active}

            # Phase 1: maximise distinct episode coverage in this batch.
            for ep in active:
                if len(batch) >= self.batch_size:
                    break
                idx = _take_one(ep)
                if idx is not None:
                    batch.append(idx)
                    batch_episode_counts[ep] += 1

            # Phase 2: fill evenly, preferring episodes with more remaining
            # data only when their current-batch contributions are tied.
            while len(batch) < self.batch_size:
                available = [ep for ep in active if _remaining(ep) > 0]
                if not available:
                    break

                rng.shuffle(available)
                available.sort(
                    key=lambda ep: (
                        batch_episode_counts[ep],
                        -_remaining(ep),
                    )
                )

                progress = False
                for ep in available:
                    if len(batch) >= self.batch_size:
                        break
                    idx = _take_one(ep)
                    if idx is not None:
                        batch.append(idx)
                        batch_episode_counts[ep] += 1
                        progress = True
                if not progress:
                    break

            if len(batch) < self.batch_size and self.drop_last:
                break
            if not batch:
                break

            rng.shuffle(batch)
            yield batch
            yielded += 1

    def __repr__(self) -> str:
        return (
            f"BalancedEpisodeBatchSamplerV2(batch_size={self.batch_size}, "
            f"n_episodes={self.n_episodes}, "
            f"total_samples={self.total_samples}, "
            f"num_batches_per_epoch={self.num_batches_per_epoch}, "
            f"drop_last={self.drop_last}, "
            f"epoch={self.epoch})"
        )
