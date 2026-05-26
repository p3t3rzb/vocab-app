"""Single-task threshold-crossing search.

Given a probe function that returns ``P(recall)`` at a hypothetical delta,
:class:`ThresholdSearch` finds the delta at which the probability first drops
below the configured threshold. The same control flow is used by both the
per-word :class:`Predictor` and the batched scheduler — only the *probe*
differs (single-sample LSTM forward vs. batched LSTM forward).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from src.model.config import PredictConfig
from src.model.inference.polynomial import PolynomialRefiner


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
        points: All ``(delta, P)`` pairs probed so far, used for polynomial
            refinement at the end.
    """

    word_id: int
    direction: int
    base_seq: list[list[float]]
    last_remembered: float
    last_ts: int
    lo: float = 0.0
    hi: float = 0.0
    points: list[tuple[float, float]] = field(default_factory=list)


class ThresholdSearch:
    """Find the Δt at which ``P(recall)`` first drops below ``recall_threshold``.

    The algorithm has four stages:

    1. **Initial check**: if ``P(Δ = 1s) < threshold`` the word is already due.
    2. **Doubling**: start with ``hi = initial_delta_seconds`` and double until
       ``P(hi) < threshold``. Returns ``max_delta_seconds`` if the threshold
       is never crossed within :attr:`PredictConfig.max_doubling_iters`.
    3. **Bisect**: ``bisect_steps`` of binary search between ``lo`` and ``hi``.
    4. **Polynomial refinement**: fit a polynomial to all probed points and
       return the geometric mean of the bisect result and the polynomial root
       (falling back to the bisect result alone if the root is invalid).
    """

    def __init__(self, cfg: PredictConfig):
        self._cfg = cfg
        self._refiner = PolynomialRefiner(cfg.poly_degree)

    def find_delta(self, probe: Callable[[float], float]) -> float:
        """Run the full search using the supplied single-sample probe."""
        cfg = self._cfg
        threshold = cfg.recall_threshold

        points: list[tuple[float, float]] = []

        def record(delta: float) -> float:
            p = probe(delta)
            points.append((delta, p))
            return p

        # Stage 1: already below threshold at Δ≈0 → study now
        if record(1.0) < threshold:
            return 0.0

        lo = 0.0
        hi = cfg.initial_delta_seconds

        # Stage 2: doubling
        for _ in range(cfg.max_doubling_iters):
            if record(hi) < threshold:
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
            if record(mid) >= threshold:
                lo = mid
            else:
                hi = mid

        binary_result = hi

        # Stage 4: polynomial refinement
        poly_result = self._refiner.crossing(
            points, threshold, reference_log_delta=math.log(binary_result + 1)
        )
        if poly_result is None or poly_result <= 0 or not math.isfinite(poly_result):
            return binary_result

        poly_result = max(1.0, min(poly_result, cfg.max_delta_seconds))
        return math.sqrt(binary_result * poly_result)
