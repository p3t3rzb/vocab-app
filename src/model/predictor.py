"""Wrap a trained :class:`RecallLSTM` for prediction-time use.

The interesting work happens in :meth:`Predictor.next_repetition_delta`,
which combines a binary search with a polynomial refinement to estimate the
moment when ``P(recall)`` falls below the configured threshold.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from src.database.models import Repetition
from src.model.config import PredictConfig
from src.model.lstm import RecallLSTM


def poly_threshold_crossing(
    points: list[tuple[float, float]],
    threshold: float,
    degree: int,
    reference_log_delta: float,
) -> float | None:
    """Fit a polynomial to probed ``(delta, P(recall))`` points and solve for the threshold crossing.

    The fit is done in ``log(delta + 1)`` space — the same feature the model
    consumes — which keeps the polynomial well-conditioned across many orders
    of magnitude.

    A root is accepted only if it is strictly positive (in log-space) and the
    polynomial is **decreasing** at the root (derivative < 0). When multiple
    valid roots exist, the one nearest ``reference_log_delta`` is returned.

    Returns:
        The crossing delta in seconds, or ``None`` if no valid root is found
        or the fit fails.
    """
    xs = np.array([math.log(d + 1) for d, _ in points])
    ys = np.array([p for _, p in points])

    try:
        coeffs = np.polyfit(xs, ys, deg=degree)
        deriv_coeffs = np.polyder(coeffs)

        shifted = coeffs.copy()
        shifted[-1] -= threshold
        roots = np.roots(shifted)

        real_roots = [
            r.real for r in roots
            if abs(r.imag) < 1e-6
            and r.real > 0
            and float(np.polyval(deriv_coeffs, r.real)) < 0  # function is decreasing
        ]
        if not real_roots:
            return None

        x_root = min(real_roots, key=lambda r: abs(r - reference_log_delta))
        return math.exp(x_root) - 1
    except Exception:
        return None


class Predictor:
    """Inference helper for a trained :class:`RecallLSTM`.

    Exposes :meth:`recall_probability` (point query) and
    :meth:`next_repetition_delta` (when should the word be reviewed next).
    """

    def __init__(self, model: RecallLSTM, config: PredictConfig | None = None) -> None:
        """Wrap an already-trained, already-on-device model.

        Args:
            model: The trained network. Caller is responsible for placing it
                on the desired device and calling ``.eval()``.
            config: Override :class:`PredictConfig` defaults. ``None`` uses
                the dataclass defaults.
        """
        self.model = model
        self.config = config or PredictConfig()
        self.device = next(model.parameters()).device

    def recall_probability(self, reps: list[Repetition], delta_seconds: float) -> float:
        """Return P(remembered) if the word is tested ``delta_seconds`` after its last rep.

        Args:
            reps: Repetition history, oldest first. Must be non-empty.
            delta_seconds: Hypothetical gap (in seconds) after the most recent
                rep at which to probe the model.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        inputs: list[list[float]] = []

        for i in range(1, len(reps)):
            dt = reps[i].practiced_at - reps[i - 1].practiced_at
            inputs.append([math.log(dt + 1), float(reps[i - 1].remembered)])

        inputs.append([math.log(delta_seconds + 1), float(reps[-1].remembered)])

        x = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(x)
        return probs[0, -1].item()

    def next_repetition_delta(self, reps: list[Repetition]) -> float:
        """Estimate seconds-until-next-review.

        Algorithm:

        1. If ``P(Δt = 1s) < threshold`` the word is already due — return ``0``.
        2. **Doubling**: start with ``hi = initial_delta_seconds`` and double
           until ``P(hi) < threshold``. Return ``max_delta_seconds`` if the
           threshold is never crossed.
        3. **Bisect**: run ``bisect_steps`` of binary search between ``lo``
           and ``hi``, collecting more ``(delta, prob)`` points.
        4. **Polynomial refinement**: fit a polynomial to every probed point
           and solve for the threshold crossing analytically. Use it only
           when it is positive, finite, and the polynomial is decreasing at
           the root.
        5. Return the **geometric mean** of the binary-search result and the
           polynomial result (falling back to the binary-search result alone
           when the polynomial root is rejected).
        """
        cfg = self.config
        threshold = cfg.recall_threshold

        # If already below threshold at delta≈0, the word needs study now
        if self.recall_probability(reps, 1.0) < threshold:
            return 0.0

        points: list[tuple[float, float]] = []

        def probe(delta: float) -> float:
            p = self.recall_probability(reps, delta)
            points.append((delta, p))
            return p

        lo = 0.0
        hi = cfg.initial_delta_seconds

        # Phase 1: double hi until P(hi) < threshold
        while probe(hi) >= threshold:
            lo = hi
            hi *= 2
            if hi > cfg.max_delta_seconds:
                return cfg.max_delta_seconds

        # Phase 2: bisect
        for _ in range(cfg.bisect_steps):
            mid = (lo + hi) / 2
            if probe(mid) >= threshold:
                lo = mid
            else:
                hi = mid

        binary_result = hi

        # Fit polynomial in log(delta+1) space and solve for threshold crossing
        poly_result = poly_threshold_crossing(
            points,
            threshold,
            cfg.poly_degree,
            reference_log_delta=math.log(binary_result + 1),
        )

        if poly_result is None or poly_result <= 0 or not math.isfinite(poly_result):
            return binary_result

        poly_result = max(1.0, min(poly_result, cfg.max_delta_seconds))

        # Geometric mean of the two estimates
        return math.sqrt(binary_result * poly_result)
