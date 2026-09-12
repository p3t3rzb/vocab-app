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


def recall_at(p0: float, s: float, d: float, delta_seconds: float) -> float:
    """Evaluate the recall curve ``R(Δt) = p0·(1 + Δt/S)**(−d)`` on Python floats.

    The pure-``math`` counterpart of :func:`curve_recall`, for live recall scoring
    from stored params (no tensor / no model forward). ``Δt`` is clamped to ``≥ 0``
    and the result to ``(0, 1)``.
    """
    delta = max(0.0, delta_seconds)
    try:
        recall = p0 * math.pow(1.0 + delta / s, -d)
    except (OverflowError, ValueError):
        return PARAM_EPS
    return min(max(recall, PARAM_EPS), 1.0 - PARAM_EPS)


def invert_curve(
    p0: float,
    s: float,
    d: float,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Seconds until ``R(Δt)`` falls to ``threshold``, from stored Python floats.

    Solving ``threshold = p0·(1 + Δt/S)**(−d)`` for ``Δt`` gives
    ``Δt = S·((p0 / threshold)**(1/d) − 1)``. If ``p0 ≤ threshold`` the word is
    already below the threshold the instant after review, so it is due now.

    Args:
        p0, s, d: Forgetting-curve params (already activated, ``s``/``d`` > 0).
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).
            A small decay ``d`` makes the closed form explode, so it is clamped.

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    if p0 <= threshold:
        return 0.0
    try:
        delta = s * (math.pow(p0 / threshold, 1.0 / d) - 1.0)
    except OverflowError:
        return max_delta_seconds
    return min(delta, max_delta_seconds)


def next_delta(
    raw_params_last: torch.Tensor,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Analytically invert the curve from a raw ``(3,)`` head output.

    Thin wrapper over :func:`invert_curve` that first activates the raw params
    via :func:`split_params`.

    Args:
        raw_params_last: Raw head output for a single timestep, shape ``(3,)``.
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    p0_t, s_t, d_t = split_params(raw_params_last)
    return invert_curve(
        float(p0_t), float(s_t), float(d_t), threshold, max_delta_seconds
    )


def retained_seconds(
    p0: float,
    s: float,
    d: float,
    horizon_seconds: float,
    offset_seconds: float = 0.0,
) -> float:
    """Area under ``R(Δt)`` over a window of ``horizon_seconds``, in seconds.

    ``R`` is the probability that a test at ``Δt`` succeeds, so its integral is the
    expected amount of time the word stays recallable over the window — the
    continuous counterpart of :func:`invert_curve`'s "time until recall falls to
    ``threshold``". Unlike that crossing time, it is finite and strictly positive
    for *every* curve, which is what lets cards be compared whether or not a
    single review lifts them over the user's threshold.

    The window starts at ``offset_seconds``, which is what makes the function
    usable for a curve that is already part-way through its decay: a card's
    *current* curve is anchored at its last review, so scoring it over the window
    starting *now* means integrating from ``now − last_practiced``. The curves a
    review would produce are anchored at the review itself, so they keep the
    default offset of 0.

    Integrating the power law gives, for ``d ≠ 1``::

        ∫ p0·(1 + t/S)**(−d) dt = p0·S/(1 − d)·(1 + t/S)**(1 − d)

    and ``p0·S·ln(1 + t/S)`` at ``d = 1``, where that exponent vanishes; the area
    is that antiderivative's value at the end of the window minus its value at the
    start.

    Args:
        p0, s, d: Forgetting-curve params (already activated, ``s``/``d`` > 0).
        horizon_seconds: Width of the window. Clamped to ``≥ 0``.
        offset_seconds: ``Δt`` the window starts at. Clamped to ``≥ 0``.

    Returns:
        Expected recallable seconds, in ``[0, horizon_seconds]``.
    """
    start = max(0.0, offset_seconds)
    end = start + max(0.0, horizon_seconds)
    try:
        if abs(d - 1.0) < PARAM_EPS:
            area = p0 * s * (math.log1p(end / s) - math.log1p(start / s))
        else:
            exponent = 1.0 - d
            area = (
                p0
                * s
                / exponent
                * (
                    math.pow(1.0 + end / s, exponent)
                    - math.pow(1.0 + start / s, exponent)
                )
            )
    except (OverflowError, ValueError, ZeroDivisionError):
        return 0.0
    return min(max(area, 0.0), end - start)


def expected_retained(
    recall_now: float,
    success: tuple[float, float, float],
    failure: tuple[float, float, float],
    horizon_seconds: float,
) -> float:
    """Expected recallable seconds a card *would have* after practising it now.

    Practising replaces the card's curve with a freshly-fitted one — ``success``
    if the answer is right, ``failure`` if it is wrong — and ``recall_now`` is
    itself the probability of getting it right, so the expected area under the
    card's curve afterwards is::

        V = p·∫R_success + (1 − p)·∫R_failure

    Because it integrates the curve instead of solving for the moment it crosses
    the recall threshold, it stays informative for the cards a single review cannot
    lift over that threshold — the ones :func:`invert_curve` collapses to a flat
    ``0``.

    Note this is a *level*, not a gain: a card already holding its recall scores
    high here whether or not a review adds anything. :func:`expected_gain` is what
    the practice queue sorts on; this is its post-review term.

    Args:
        recall_now: Current ``R(Δt)`` for the card, in ``(0, 1)``. For a word
            never practised in this direction there is no curve to evaluate, so
            the deck's empirical first-attempt success rate stands in for it.
        success: ``(p0, S, d)`` of the curve fitted after a successful rep.
        failure: ``(p0, S, d)`` of the curve fitted after a failed rep.
        horizon_seconds: Horizon for both integrals.

    Returns:
        Expected recallable seconds, in ``[0, horizon_seconds]``.
    """
    return (
        recall_now * retained_seconds(*success, horizon_seconds)
        + (1.0 - recall_now) * retained_seconds(*failure, horizon_seconds)
    )


def expected_gain(
    recall_now: float,
    current: tuple[float, float, float] | None,
    elapsed_seconds: float,
    success: tuple[float, float, float],
    failure: tuple[float, float, float],
    horizon_seconds: float,
) -> float:
    """Expected recallable seconds practising a card *right now* would **add**.

    The practice queue's ordering key, largest first. Practising is worth the
    difference between what the card will retain if it is reviewed and what it
    would have retained if it were left alone, both measured over the same
    wall-clock window ``[now, now + horizon]``::

        gain = E[∫R_after] − ∫R_current

    The second term is where ``elapsed_seconds`` comes in: the card's current
    curve is anchored at its last review, so the do-nothing area has to be
    integrated from ``now − last_practiced`` rather than from 0.

    Ordering on the increase rather than on the level is what greedily maximises
    the total recall held across the whole vocabulary, ``∫ Σ_cells R_cell(t) dt``:
    a card that already holds its recall contributes that recall whether or not it
    is practised, so only the part a review *adds* is worth spending the session on.

    Two consequences worth knowing:

    * **The gain moves as the clock moves**, and not in one direction: waiting
      lowers the do-nothing baseline, which raises the gain, but it also lowers
      ``recall_now``, which shifts weight from the success curve onto the worse
      failure curve. Which effect wins depends on the card. Either way a score
      goes stale as soon as the clock does — unlike the level, which barely shifts
      over a session — so a queued card is re-scored when it is served rather than
      keeping the score it was built with.
    * **The gain can be negative.** A review risks a wrong answer, and the failure
      curve weighted by ``1 − recall_now`` can drag the expectation below the
      undisturbed current curve. That is a real signal — the card is better left
      alone — so it is kept rather than clamped.

    Args:
        recall_now: Current ``R(Δt)`` for the card, in ``(0, 1)`` — also the
            probability the answer is right. For a word never practised in this
            direction, the deck's first-attempt success rate stands in.
        current: ``(p0, S, d)`` of the card's present curve, or ``None`` when the
            card has never been practised in this direction. A never-practised
            card retains nothing, so its baseline is 0 and its gain is the whole
            post-review area.
        elapsed_seconds: Seconds since the card's last review, i.e. the ``Δt`` the
            baseline integral starts at. Ignored when ``current`` is ``None``.
        success: ``(p0, S, d)`` of the curve fitted after a successful rep.
        failure: ``(p0, S, d)`` of the curve fitted after a failed rep.
        horizon_seconds: Horizon for every integral.

    Returns:
        Expected added recallable seconds, in ``[−horizon_seconds, horizon_seconds]``.
    """
    after = expected_retained(recall_now, success, failure, horizon_seconds)
    if current is None:
        return after
    return after - retained_seconds(*current, horizon_seconds, elapsed_seconds)
