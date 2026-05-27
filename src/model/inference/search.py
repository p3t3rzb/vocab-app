"""Single-task threshold-crossing search.

Given a probe function that returns ``P(recall)`` at a hypothetical delta,
:class:`ThresholdSearch` finds the delta at which the probability first drops
below the configured threshold. The same control flow is used by both the
per-word :class:`Predictor` and the batched scheduler — only the *probe*
differs (single-sample LSTM forward vs. batched LSTM forward).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.model.config import PredictConfig


@dataclass
class SearchTask:
    """Mutable state for one threshold-search in progress.

    Used by the batched scheduler to track each (word, direction) task as it
    moves through the doubling and bisect phases independently.

    Attributes:
        word_id: Database id of the word being scheduled.
        direction: ``int(Direction.FORWARD)`` or ``int(Direction.REVERSE)``.
        base_seq: Pre-built LSTM input rows for the rep history, **excluding**
            the probe timestep (which is appended per probe).
        last_remembered: ``float(reps[-1].remembered)`` — the previous-step
            ``remembered`` feature for the probe timestep.
        last_ts: ``reps[-1].practiced_at`` — Unix time of the most recent rep.
        lo: Current lower bracket (Δt where P ≥ threshold).
        hi: Current upper bracket (Δt to probe / where P < threshold).
    """

    word_id: int
    direction: int
    base_seq: list[list[float]]
    last_remembered: float
    last_ts: int
    lo: float = 0.0
    hi: float = 0.0


class ThresholdSearch:
    """Find the Δt at which ``P(recall)`` first drops below ``recall_threshold``.

    The algorithm has three stages:

    1. **Initial check**: if ``P(Δ = 1s) < threshold`` the word is already due.
    2. **Doubling**: start with ``hi = initial_delta_seconds`` and double until
       ``P(hi) < threshold``. Returns ``max_delta_seconds`` if the threshold
       is never crossed within :attr:`PredictConfig.max_doubling_iters`.
    3. **Bisect**: ``bisect_steps`` of binary search between ``lo`` and ``hi``;
       the final ``hi`` is returned as the threshold-crossing estimate.
    """

    def __init__(self, cfg: PredictConfig):
        self._cfg = cfg

    def find_delta(self, probe: Callable[[float], float]) -> float:
        """Run the full search using the supplied single-sample probe."""
        cfg = self._cfg
        threshold = cfg.recall_threshold

        # Stage 1: already below threshold at Δ≈0 → study now
        if probe(1.0) < threshold:
            return 0.0

        lo = 0.0
        hi = cfg.initial_delta_seconds

        # Stage 2: doubling
        for _ in range(cfg.max_doubling_iters):
            if probe(hi) < threshold:
                break
            lo = hi
            hi *= 2
            if hi > cfg.max_delta_seconds:
                return cfg.max_delta_seconds
        else:
            return cfg.max_delta_seconds

        # Stage 3: bisect
        for _ in range(cfg.bisect_steps):
            mid = (lo + hi) / 2
            if probe(mid) >= threshold:
                lo = mid
            else:
                hi = mid

        return hi
