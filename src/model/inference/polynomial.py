"""Polynomial refinement of binary-search probe points.

The single-sample :class:`Predictor` and the batched scheduler both probe a
handful of ``(delta, P(recall))`` points and want a sharper estimate of the
threshold crossing than the bisect midpoint alone gives. Fitting a low-degree
polynomial in ``log(Δ + 1)`` space — the same feature the LSTM consumes —
and solving for its threshold crossing analytically does this cheaply.
"""
from __future__ import annotations

import math

import numpy as np


class PolynomialRefiner:
    """Fit a polynomial through probed points and return the threshold-crossing Δt."""

    def __init__(self, degree: int):
        """Store the polynomial degree to fit (typically 2)."""
        self._degree = degree

    def crossing(
        self,
        points: list[tuple[float, float]],
        threshold: float,
        reference_log_delta: float,
    ) -> float | None:
        """Return the Δt (seconds) where the fitted polynomial crosses ``threshold``.

        A root is accepted only if it is strictly positive in log-space and
        the polynomial is **decreasing** at the root (derivative < 0). When
        multiple valid roots exist, the one nearest ``reference_log_delta`` is
        returned. Returns ``None`` if no valid root is found or the fit fails.
        """
        xs = np.array([math.log(d + 1) for d, _ in points])
        ys = np.array([p for _, p in points])

        try:
            coeffs = np.polyfit(xs, ys, deg=self._degree)
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
