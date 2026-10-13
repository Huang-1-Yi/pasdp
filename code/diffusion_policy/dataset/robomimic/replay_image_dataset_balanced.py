"""Dataset variant that uses BalancedEpisodeBatchSampler for training.

Inherits from RobomimicReplayImageDataset.  The only thing that differs is
*what* sampler is constructed in __init__:

* Baseline: ``SequenceSampler`` (PyTorch DataLoader uses random shuffling).
* Balanced: ``BalancedEpisodeBatchSampler`` (a SequenceSampler subclass that
  also implements ``__iter__`` to yield episode-balanced batches when used
  as ``batch_sampler=`` in a DataLoader).

The new dataset class is therefore a drop-in replacement for the base class:
it has the same interface (``__len__``, ``__getitem__``, ``get_normalizer``,
``get_validation_dataset``, ...) and exposes a ``get_train_dataloader``
helper that the workspace calls to build a balanced DataLoader.

Usage
-----
Switch the dataset field in your yaml:

    dataset: diffusion_policy.dataset.robomimic.replay_image_dataset_balanced.RobomimicReplayImageBalancedDataset

Optionally override the sampler seed via:

    task:
      dataset:
        sampler_seed: 42
"""

from torch.utils.data import DataLoader

from diffusion_policy.common.sampler import BalancedEpisodeBatchSampler
from diffusion_policy.dataset.robomimic.replay_image_dataset import (
    RobomimicReplayImageDataset,
)


class RobomimicReplayImageBalancedDataset(RobomimicReplayImageDataset):
    """RobomimicReplayImageDataset trained with BalancedEpisodeBatchSampler.

    All other behavior (normalization, validation split, sample sequence,
    padding) is inherited unchanged.  Only the ``self.sampler`` attribute is
    swapped: instead of a plain ``SequenceSampler`` we hold a
    ``BalancedEpisodeBatchSampler`` (which IS a SequenceSampler -- same
    interface -- but additionally implements ``__iter__`` to produce
    balanced batches).
    """

    def __init__(self,
                 *args,
                 sampler_seed: int = 42,
                 balanced_shuffle_within_episode: bool = True,
                 balanced_drop_last: bool = False,
                 **kwargs):
        # Set attributes before super().__init__ would have a chance to
        # read them -- but actually the parent class doesn't look at these
        # names, so we just store them on ``self`` after super() runs.
        super().__init__(*args, **kwargs)
        self.sampler_seed = int(sampler_seed)
        self.balanced_shuffle_within_episode = bool(balanced_shuffle_within_episode)
        self.balanced_drop_last = bool(balanced_drop_last)

        # ------------------------------------------------------------------
        # Swap the parent's SequenceSampler for a BalancedEpisodeBatchSampler
        # that shares the same underlying indices / replay_buffer.  The
        # balanced sampler inherits from SequenceSampler so dataset.__len__
        # and dataset.__getitem__ still work as before.
        # ------------------------------------------------------------------
        balanced_sampler = BalancedEpisodeBatchSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sampler.keys,
            key_first_k=self.sampler.key_first_k,
            episode_mask=self.train_mask,
            batch_size=kwargs.get('batch_size', 64),
            seed=self.sampler_seed,
            shuffle_within_episode=self.balanced_shuffle_within_episode,
            drop_last=self.balanced_drop_last,
        )
        self.sampler = balanced_sampler

    # ------------------------------------------------------------------
    # DataLoader construction
    # ------------------------------------------------------------------
    def get_train_dataloader(self,
                             batch_size: int = 64,
                             num_workers: int = 32,
                             pin_memory: bool = True,
                             persistent_workers: bool = True,
                             **kwargs):
        """Build a training DataLoader using the balanced batch sampler.

        NOTE: ``shuffle`` is NOT honoured -- balanced sampling replaces
        random shuffling.  Any extra kwargs are passed through to
        ``DataLoader``.
        """
        # ``shuffle`` is incompatible with ``batch_sampler`` so we drop it.
        kwargs.pop('shuffle', None)
        kwargs.pop('drop_last', None)
        kwargs.pop('batch_size', None)

        # Re-build the balanced sampler with the supplied batch_size (it
        # may differ from what was passed to __init__).
        balanced_sampler = BalancedEpisodeBatchSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            keys=self.sampler.keys,
            key_first_k=self.sampler.key_first_k,
            episode_mask=self.train_mask,
            batch_size=batch_size,
            seed=self.sampler_seed,
            shuffle_within_episode=self.balanced_shuffle_within_episode,
            drop_last=self.balanced_drop_last,
        )
        # ``self.sampler`` is a SequenceSampler (for __len__/__getitem__).
        # The balanced sampler is what we use as batch_sampler.
        self.sampler = balanced_sampler

        return DataLoader(
            self,
            batch_sampler=balanced_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            **kwargs,
        )