"""Deterministic RTV8 epoch mapping and flat-index resolution."""

from typing import Tuple

import numpy as np


def build_balanced_epoch_pairs(
        episode_map: np.ndarray,
        episode_ids: np.ndarray,
        batch_size: int,
        seed: int,
        epoch_index: int,
) -> Tuple[np.ndarray, int]:
    """Build the legacy balanced ``[B, max_cols, 2]`` mapping exactly.

    The operation order intentionally matches the three former in-class
    implementations: epoch-seeded shuffle, greedy shortest-row assignment,
    then deterministic prefix filling for shorter rows.
    """

    rng = np.random.RandomState(seed + epoch_index)
    ep_ids = episode_ids.copy()
    rng.shuffle(ep_ids)

    B = batch_size
    rows = [[] for _ in range(B)]
    row_lens = np.zeros(B, dtype=np.int64)

    for e in ep_ids:
        win = int(episode_map[e, 5])
        r = int(np.argmin(row_lens))
        rows[r].extend([(int(e), m) for m in range(win)])
        row_lens[r] += win

    max_cols = int(row_lens.max()) if B > 0 else 0
    if max_cols == 0:
        raise RuntimeError("No columns in epoch mapping!")

    for r in range(B):
        deficit = max_cols - len(rows[r])
        if deficit <= 0:
            continue
        fill_ptr = 0
        while len(rows[r]) < max_cols:
            e = int(ep_ids[fill_ptr % len(ep_ids)])
            win = int(episode_map[e, 5])
            need = max_cols - len(rows[r])
            take = min(win, need)
            rows[r].extend([(e, m) for m in range(take)])
            fill_ptr += 1

    pairs = np.zeros((B, max_cols, 2), dtype=np.int32)
    for r in range(B):
        for c, (e, m) in enumerate(rows[r]):
            pairs[r, c, 0] = e
            pairs[r, c, 1] = m

    return pairs, max_cols


def resolve_mapping_index(
        global_pairs: np.ndarray,
        episode_map: np.ndarray,
        batch_size: int,
        max_cols: int,
        idx: int,
) -> Tuple[int, int, int]:
    """Resolve a flattened dataset index using the legacy arithmetic."""

    p = idx % batch_size
    q = idx // batch_size
    if q >= max_cols:
        q = max_cols - 1

    episode_id = int(global_pairs[p, q, 0])
    episode_window_index = int(global_pairs[p, q, 1])
    buf_start_1b = int(episode_map[episode_id, 3])
    abs_start = (buf_start_1b - 1) + episode_window_index
    return episode_id, episode_window_index, abs_start


__all__ = ["build_balanced_epoch_pairs", "resolve_mapping_index"]
