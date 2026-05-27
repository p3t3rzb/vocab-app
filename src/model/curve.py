"""Forgetting-curve math for the :class:`~src.model.lstm.RecallLSTM` head.

The network predicts, per timestep, the *parameters* of a forgetting curve
``R(Δt)`` rather than ``P(remembered)`` directly. This module turns the
network's raw 3-channel output into a recall probability at a given gap, and
inverts the curve to find the next-review time analytically (no bisection).

The curve is a **scaled power-law**::

    R(Δt) = p0 · (1 + Δt / S) ** (−d)

where the three parameters are derived from the head's raw outputs:

* ``p0 = sigmoid(raw0)``        — recall ceiling at ``Δt = 0`` (in ``(0, 1)``)
* ``S  = softplus(raw1) + eps`` — time-scale, in seconds
* ``d  = softplus(raw2) + eps`` — decay exponent

``R`` starts below 1 (matching the observation that recall right after a rep is
not certain) and decays monotonically toward 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Floor added to the strictly-positive params so they never collapse to 0.
PARAM_EPS = 1e-6


def split_params(raw_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a raw ``(..., 3)`` head output to ``(p0, S, d)`` in their valid ranges.

    Args:
        raw_params: The network's unactivated output, last dim of size 3.

    Returns:
        ``(p0, S, d)``, each shaped like ``raw_params[..., 0]``.
    """
    p0 = torch.sigmoid(raw_params[..., 0])
    s = F.softplus(raw_params[..., 1]) + PARAM_EPS
    d = F.softplus(raw_params[..., 2]) + PARAM_EPS
    return p0, s, d


def curve_recall(
    deltas: torch.Tensor, raw_params: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Evaluate ``R(Δt) = p0·(1 + Δt/S)**(−d)`` and clamp for safe BCE.

    Args:
        deltas: Query gaps in seconds, shape ``(B, L)`` (must be ``≥ 0``).
        raw_params: Raw head output, shape ``(B, L, 3)``.
        eps: Probabilities are clamped to ``[eps, 1 − eps]`` so the downstream
            ``log`` in BCE never sees 0 or 1.

    Returns:
        Recall probabilities of shape ``(B, L)``.
    """
    p0, s, d = split_params(raw_params)
    recall = p0 * torch.pow(1.0 + deltas / s, -d)
    return recall.clamp(eps, 1.0 - eps)


def next_delta(
    raw_params_last: torch.Tensor,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Analytically invert the curve: seconds until ``R(Δt)`` hits ``threshold``.

    Solving ``threshold = p0·(1 + Δt/S)**(−d)`` for ``Δt`` gives
    ``Δt = S·((p0 / threshold)**(1/d) − 1)``. If ``p0 ≤ threshold`` the word is
    already below the threshold the instant after review, so it is due now.

    Args:
        raw_params_last: Raw head output for a single timestep, shape ``(3,)``.
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the returned interval (default 2 years),
            mirroring :class:`~src.model.config.PredictConfig`. A small decay
            ``d`` makes the closed form explode, so the result is clamped.

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    p0_t, s_t, d_t = split_params(raw_params_last)
    p0, s, d = float(p0_t), float(s_t), float(d_t)
    if p0 <= threshold:
        return 0.0
    try:
        delta = s * (math.pow(p0 / threshold, 1.0 / d) - 1.0)
    except OverflowError:
        return max_delta_seconds
    return min(delta, max_delta_seconds)
