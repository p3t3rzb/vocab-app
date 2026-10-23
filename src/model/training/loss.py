"""Masked binary cross-entropy for variable-length sequences."""

import torch
import torch.nn as nn


def masked_bce(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Summed binary cross-entropy over the steps ``mask`` selects.

    ``mask`` drops padding and, for validation, the history steps before the
    time cutoff (see :func:`src.model.dataset.split_by_time`).

    Returns the *sum* over real steps, not the mean: batches are length-bucketed,
    so a per-batch mean would weight each rep by roughly 1/L of its sequence, and
    L grows with failures — discounting exactly the hard words. The caller
    normalises by a constant so every rep counts the same.
    """
    loss = nn.functional.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum()
