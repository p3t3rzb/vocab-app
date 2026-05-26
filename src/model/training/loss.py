"""Masked binary cross-entropy for variable-length sequences."""
import torch
import torch.nn as nn


def masked_bce(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy that ignores padded positions.

    The mask is built from ``lengths`` so padded steps contribute nothing to
    either the numerator or the denominator of the mean.
    """
    seq_len = pred.shape[1]
    mask = torch.arange(seq_len, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    loss = nn.functional.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()
