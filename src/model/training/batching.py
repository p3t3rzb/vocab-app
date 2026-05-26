"""Bucket-style batching to keep padding minimal."""
import random

import torch


def bucket_batches(lengths: torch.Tensor, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    """Sort sequence indices by length, slice into fixed-size batches, optionally shuffle order.

    Sorting by length keeps padding minimal within each batch; shuffling the
    *order* of the batches preserves SGD's stochasticity during training
    while still benefiting from the bucketing.
    """
    sorted_idx = torch.argsort(lengths)
    batches = [sorted_idx[i : i + batch_size] for i in range(0, len(lengths), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches
