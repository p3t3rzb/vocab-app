from __future__ import annotations

import math

import numpy as np
import torch

from src.database.models import Repetition
from src.model.config import PredictConfig
from src.model.lstm import RecallLSTM


def _poly_threshold_crossing(
    points: list[tuple[float, float]],
    threshold: float,
    degree: int,
    reference_log_delta: float,
) -> float | None:
    """
    Fit a degree-`degree` polynomial to (log(delta+1), prob) points and return
    the delta (in seconds) where the polynomial crosses `threshold`, or None if
    no valid crossing is found.

    A crossing is valid only if:
    - the root is positive (in log-space), and
    - the polynomial is strictly decreasing at the root (derivative < 0).

    `reference_log_delta` selects among multiple roots: we pick the real root
    closest to it.
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
    def __init__(self, model: RecallLSTM, config: PredictConfig | None = None) -> None:
        self.model = model
        self.config = config or PredictConfig()
        self.device = next(model.parameters()).device

    def recall_probability(self, reps: list[Repetition], delta_seconds: float) -> float:
        """P(remembered) if tested `delta_seconds` after the last rep in `reps`."""
        if not reps:
            raise ValueError("Need at least one historical repetition")

        inputs: list[list[float]] = []

        for i in range(1, len(reps)):
            dt = reps[i].practiced_at - reps[i - 1].practiced_at
            inputs.append([math.log(max(dt, 0) + 1), float(reps[i - 1].remembered)])

        inputs.append([math.log(delta_seconds + 1), float(reps[-1].remembered)])

        x = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(x)
        return probs[0, -1].item()

    def next_repetition_delta(self, reps: list[Repetition]) -> float:
        """
        Seconds after the last rep when recall probability crosses below the threshold.

        Runs a binary search to bracket the crossing, fits a polynomial to all
        probed points, solves for the polynomial crossing, and returns the
        geometric mean of the two estimates.
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
        poly_result = _poly_threshold_crossing(
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
