"""Batched final-timestep forward, shared by the inference paths.

Both the param scheduler (every word's stored time constant) and the expected-gain
scoring need the same thing: run a pile of variable-length histories through the
model and read off each one's *final* timestep. The read-off is indexed by each
sequence's own length, so padded steps never leak into the result.

Sequences are bucketed by length before batching — the same trick
:func:`src.model.training.batching.bucket_batches` uses — so each chunk pads to
roughly its own longest member instead of to the longest in the whole call.
Histories here span 1 to ~240 repetitions, so feeding them in database order
pads most of every batch with zeros *and* hands the backend a differently-shaped
tensor per chunk; bucketing turns both into non-issues. Results are returned in
the caller's original order.
"""
from __future__ import annotations

import torch

from ..curve import time_constant
from ..lstm import RecallLSTM


def final_step_time_constants(
    model: RecallLSTM,
    sequences: list[list[list[float]]],
    chunk_size: int | None = None,
) -> list[float]:
    """Forward every sequence and return its final timestep's time constant.

    Args:
        model: Trained network, already on its device and in ``eval`` mode.
        sequences: One list of input rows per task (see
            :mod:`src.model.features`). May be empty.
        chunk_size: Sequences per batched forward. ``None`` runs them all in one
            batch — only safe when the caller has already chunked.

    Returns:
        One time constant in seconds per input sequence, in the caller's original
        order.
    """
    if not sequences:
        return []

    device = next(model.parameters()).device
    step = chunk_size or len(sequences)
    n_features = len(sequences[0][0])
    out: list[float | None] = [None] * len(sequences)

    # Length buckets: neighbours in this order pad to nearly the same length.
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))

    for start in range(0, len(order), step):
        picks = order[start : start + step]
        lengths = [len(sequences[i]) for i in picks]
        max_len = max(lengths)
        batch = torch.zeros(
            len(picks), max_len, n_features, dtype=torch.float32, device=device
        )
        for pos, i in enumerate(picks):
            batch[pos, : lengths[pos]] = torch.tensor(
                sequences[i], dtype=torch.float32
            )

        with torch.inference_mode():
            raw = model(batch)  # (B, max_len, 1)

        idx = torch.tensor([n - 1 for n in lengths], device=device)
        raw_last = raw[torch.arange(len(picks), device=device), idx]  # (B, 1)
        tau = time_constant(raw_last)  # (B,)
        for pos, i in enumerate(picks):
            out[i] = tau[pos].item()

    return out  # type: ignore[return-value]
