"""Forgetting-curve math for the :class:`~src.model.lstm.RecallLSTM` head.

The network predicts, per timestep, the *parameter* of a forgetting curve
``R(Δt)`` rather than ``P(remembered)`` directly. This module turns the
network's raw single-channel output into a recall probability at a given gap,
and inverts the curve to find the next-review time analytically (no bisection).

The curve is a plain **exponential decay**::

    R(Δt) = exp(−Δt / τ)

with a single parameter derived from the head's raw output:

* ``τ = exp(raw)`` — the **time constant**, in seconds: the gap at which recall
  has fallen to ``1/e``, and equally the total area under the curve.

The head emits the time constant's **natural logarithm**, not the time constant
itself. Time constants span minutes to years — five orders of magnitude — and a
linear head would have to emit ``10**6`` to schedule a word a fortnight out,
which a linear layer over a bounded hidden state does not reach. In log space
the same fortnight is ``raw = 14.0``, and one unit of raw output is one
e-folding of the interval, so the head's natural output range covers the whole
schedule.

Base ``e`` rather than base 2 is what keeps every formula downstream free of a
stray ``ln 2``: the interval to a threshold is ``τ·ln(1/θ)``, and the area under
the curve is ``τ`` itself.

``R`` starts at 1 and decays monotonically toward 0, so a word's whole
forgetting behaviour — how hard it is, how well it is known — is carried by how
long its time constant is. There is no separate recall ceiling: recall in the
instant after a review is taken to be certain, and everything the model has to
say is said through ``τ``.
"""
from __future__ import annotations

import math

import torch

# Probabilities are held this far away from 0 and 1 so a downstream ``log`` is safe.
PARAM_EPS = 1e-6

# Bounds on the raw ``ln(τ)`` output, i.e. a time constant in [~1 µs, ~4e11 years].
# They exist only to keep ``exp(raw)`` finite and strictly positive; both ends are
# far outside any schedule, so the clamp never shapes a real prediction.
LOG_TAU_MIN = -14.0
LOG_TAU_MAX = 44.0

# Starting time constant for an untrained head, as ``ln(seconds)``: three days. The
# head's bias is initialised here so the first forward pass already lands in the
# range real gaps live in, where the curve has a usable gradient.
LOG_TAU_INIT = math.log(3 * 86_400.0)


def time_constant(raw_params: torch.Tensor) -> torch.Tensor:
    """Map a raw ``(..., 1)`` head output to a positive time constant in seconds.

    The raw output *is* ``ln(τ)``, so this is an exponential, clamped at both
    ends only to keep the result finite and non-zero.

    Args:
        raw_params: The network's unactivated output, last dim of size 1.

    Returns:
        ``τ`` in seconds, shaped like ``raw_params[..., 0]``.
    """
    log_tau = raw_params[..., 0].clamp(LOG_TAU_MIN, LOG_TAU_MAX)
    return torch.exp(log_tau)


def curve_recall(
    deltas: torch.Tensor, raw_params: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Evaluate ``R(Δt) = exp(−Δt/τ)`` and clamp for safe BCE.

    Args:
        deltas: Query gaps in seconds, shape ``(B, L)`` (must be ``≥ 0``).
        raw_params: Raw head output, shape ``(B, L, 1)``.
        eps: Probabilities are clamped to ``[eps, 1 − eps]`` so the downstream
            ``log`` in BCE never sees 0 or 1.

    Returns:
        Recall probabilities of shape ``(B, L)``.
    """
    tau = time_constant(raw_params)
    return torch.exp(-deltas / tau).clamp(eps, 1.0 - eps)


def recall_at(tau: float, delta_seconds: float) -> float:
    """Evaluate the recall curve ``R(Δt) = exp(−Δt/τ)`` on Python floats.

    The pure-``math`` counterpart of :func:`curve_recall`, for live recall scoring
    from the stored time constant (no tensor / no model forward). ``Δt`` is clamped
    to ``≥ 0`` and the result to ``(0, 1)``.
    """
    delta = max(0.0, delta_seconds)
    recall = math.exp(-delta / tau)
    return min(max(recall, PARAM_EPS), 1.0 - PARAM_EPS)


def invert_curve(
    tau: float,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Seconds until ``R(Δt)`` falls to ``threshold``, from the stored time constant.

    Solving ``threshold = exp(−Δt/τ)`` for ``Δt`` gives ``Δt = τ·ln(1/threshold)``
    — the interval is simply the time constant scaled by a constant the threshold
    picks, so raising the threshold shortens every word's interval by the same
    factor.

    Args:
        tau: The curve's time constant in seconds (already activated, ``> 0``).
        threshold: Recall level below which the word is considered due. A
            ``threshold ≥ 1`` is unreachable — recall is below it the instant
            after a review — so the word is due now.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).
            ``math.inf`` leaves the interval uncapped — the user's "No maximum"
            setting — in which case it is the time constant alone that bounds it.

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    if threshold >= 1.0:
        return 0.0
    if threshold <= 0.0:
        return max_delta_seconds
    return min(tau * math.log(1.0 / threshold), max_delta_seconds)


def next_delta(
    raw_params_last: torch.Tensor,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Analytically invert the curve from a raw ``(1,)`` head output.

    Thin wrapper over :func:`invert_curve` that first activates the raw param
    via :func:`time_constant`.

    Args:
        raw_params_last: Raw head output for a single timestep, shape ``(1,)``.
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the predicted interval (default 2 years).

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    return invert_curve(
        float(time_constant(raw_params_last)), threshold, max_delta_seconds
    )


def retained_seconds(tau: float) -> float:
    """Total area under ``R(Δt)``, in seconds — the whole curve, out to infinity.

    ``R`` is the probability that a test at ``Δt`` succeeds, so its integral is the
    expected amount of time the word stays recallable — the continuous counterpart
    of :func:`invert_curve`'s "time until recall falls to ``threshold``". Unlike
    that crossing time, it is finite and strictly positive for *every* curve, which
    is what lets cards be compared whether or not a single review lifts them over
    the user's threshold.

    Integrating the exponential over the whole half-line is what the time constant
    *is*::

        ∫₀^∞ exp(−t/τ) dt = τ

    so a curve is worth recallable seconds in direct proportion to its time
    constant, and comparing two curves is comparing two time constants.

    A curve already part-way through its decay needs no separate formula. The
    exponential is memoryless — ``R(s + t) = R(s)·R(t)`` — so the area still ahead
    of a card last reviewed ``s`` ago is just the total scaled by the recall it has
    left::

        ∫ₛ^∞ exp(−t/τ) dt = R(s) · τ

    which is why :func:`expected_gain` needs only the card's current recall, not
    how long it has been sitting.

    Args:
        tau: The curve's time constant in seconds (already activated, ``> 0``).

    Returns:
        Expected recallable seconds, ``> 0``.
    """
    return tau


def expected_retained(
    recall_now: float,
    success: float,
    failure: float,
) -> float:
    """Expected recallable seconds a card *would have* after practising it now.

    Practising replaces the card's curve with a freshly-fitted one — ``success``
    if the answer is right, ``failure`` if it is wrong — and ``recall_now`` is
    itself the probability of getting it right, so the expected area under the
    card's curve afterwards is::

        V = p·∫R_success + (1 − p)·∫R_failure = p·τ_success + (1 − p)·τ_failure

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
        success: Time constant of the curve fitted after a successful rep.
        failure: Time constant of the curve fitted after a failed rep.

    Returns:
        Expected recallable seconds, ``> 0``.
    """
    return (
        recall_now * retained_seconds(success)
        + (1.0 - recall_now) * retained_seconds(failure)
    )


def expected_gain(
    recall_now: float,
    current: float | None,
    success: float,
    failure: float,
) -> float:
    """Expected recallable seconds practising a card *right now* would **add**.

    The practice queue's ordering key, largest first. Practising is worth the
    difference between what the card will retain if it is reviewed and what it
    would have retained if it were left alone::

        gain = E[∫R_after] − ∫R_current

    Both integrals run to infinity, where :func:`retained_seconds` is the time
    constant itself and the do-nothing baseline — the area still ahead of a curve
    already part-way through its decay — is that time constant scaled by the recall
    the card has left. So the whole key is three time constants weighted by one
    probability::

        gain = p·τ_success + (1 − p)·τ_failure − p·τ_current

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
      alone — so it is kept rather than clamped. About half the learned deck sits
      here at any moment.

    Args:
        recall_now: Current ``R(Δt)`` for the card, in ``(0, 1)`` — the probability
            the answer is right, and equally the fraction of its curve the card has
            left. For a word never practised in this direction, the deck's
            first-attempt success rate stands in.
        current: Time constant of the card's present curve, or ``None`` when the
            card has never been practised in this direction. A never-practised card
            retains nothing, so its baseline is 0 and its gain is the whole
            post-review area.
        success: Time constant of the curve fitted after a successful rep.
        failure: Time constant of the curve fitted after a failed rep.

    Returns:
        Expected added recallable seconds. Negative when the card is better left
        alone than reviewed.
    """
    after = expected_retained(recall_now, success, failure)
    if current is None:
        return after
    return after - recall_now * retained_seconds(current)
