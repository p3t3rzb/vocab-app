"""Forgetting-curve math for the :class:`~src.model.lstm.RecallLSTM` head.

The network predicts, per timestep, the *parameter* of a forgetting curve
``R(Δt)`` rather than ``P(remembered)`` directly. This module turns the
network's raw single-channel output into a recall probability at a given gap,
and inverts the curve to find the next-review time analytically (no bisection).

The curve is a plain **exponential decay**::

    R(Δt) = 2 ** (−Δt / H)

with a single parameter derived from the head's raw output:

* ``H = 2 ** raw`` — the **half-life**, in seconds: the gap at which recall has
  fallen to one half.

The head emits the half-life's **base-2 logarithm**, not the half-life itself.
Half-lives span minutes to years — five orders of magnitude — and a linear head
would have to emit ``10**6`` to schedule a word a fortnight out, which a linear
layer over a bounded hidden state does not reach. In log space the same
fortnight is ``raw = 20.1``, and one unit of raw output is one doubling of the
interval, so the head's natural output range covers the whole schedule.

``R`` starts at 1 and decays monotonically toward 0, so a word's whole
forgetting behaviour — how hard it is, how well it is known — is carried by how
long its half-life is. There is no separate recall ceiling: recall in the
instant after a review is taken to be certain, and everything the model has to
say is said through ``H``.
"""
from __future__ import annotations

import math

import torch

# Probabilities are held this far away from 0 and 1 so a downstream ``log`` is safe.
PARAM_EPS = 1e-6

# Bounds on the raw ``log2(H)`` output, i.e. a half-life in [~1 µs, ~6e11 years].
# They exist only to keep ``2 ** raw`` finite and strictly positive; both ends are
# far outside any schedule, so the clamp never shapes a real prediction.
LOG2_HALF_LIFE_MIN = -20.0
LOG2_HALF_LIFE_MAX = 64.0

# Starting half-life for an untrained head, as ``log2(seconds)``: three days. The
# head's bias is initialised here so the first forward pass already lands in the
# range real gaps live in, where the curve has a usable gradient.
LOG2_HALF_LIFE_INIT = math.log2(3 * 86_400.0)

# ``2 ** (−Δt/H) = exp(−ln2·Δt/H)`` — the conversion factor between the two forms,
# needed whenever the curve is integrated.
_LN2 = math.log(2.0)


def half_life(raw_params: torch.Tensor) -> torch.Tensor:
    """Map a raw ``(..., 1)`` head output to a positive half-life in seconds.

    The raw output *is* ``log2(H)``, so this is an exponential, clamped at both
    ends only to keep the result finite and non-zero.

    Args:
        raw_params: The network's unactivated output, last dim of size 1.

    Returns:
        ``H`` in seconds, shaped like ``raw_params[..., 0]``.
    """
    log2_h = raw_params[..., 0].clamp(LOG2_HALF_LIFE_MIN, LOG2_HALF_LIFE_MAX)
    return torch.exp2(log2_h)


def curve_recall(
    deltas: torch.Tensor, raw_params: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Evaluate ``R(Δt) = 2**(−Δt/H)`` and clamp for safe BCE.

    Args:
        deltas: Query gaps in seconds, shape ``(B, L)`` (must be ``≥ 0``).
        raw_params: Raw head output, shape ``(B, L, 1)``.
        eps: Probabilities are clamped to ``[eps, 1 − eps]`` so the downstream
            ``log`` in BCE never sees 0 or 1.

    Returns:
        Recall probabilities of shape ``(B, L)``.
    """
    h = half_life(raw_params)
    return torch.exp2(-deltas / h).clamp(eps, 1.0 - eps)


def recall_at(h: float, delta_seconds: float) -> float:
    """Evaluate the recall curve ``R(Δt) = 2**(−Δt/H)`` on Python floats.

    The pure-``math`` counterpart of :func:`curve_recall`, for live recall scoring
    from the stored half-life (no tensor / no model forward). ``Δt`` is clamped to
    ``≥ 0`` and the result to ``(0, 1)``.
    """
    delta = max(0.0, delta_seconds)
    recall = 2.0 ** (-delta / h)
    return min(max(recall, PARAM_EPS), 1.0 - PARAM_EPS)


def invert_curve(
    h: float,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Seconds until ``R(Δt)`` falls to ``threshold``, from the stored half-life.

    Solving ``threshold = 2**(−Δt/H)`` for ``Δt`` gives ``Δt = H·log2(1/threshold)``
    — the interval is simply the half-life scaled by a constant the threshold
    picks, so raising the threshold shortens every word's interval by the same
    factor.

    Args:
        h: The curve's half-life in seconds (already activated, ``> 0``).
        threshold: Recall level below which the word is considered due. A
            ``threshold ≥ 1`` is unreachable — recall is below it the instant
            after a review — so the word is due now.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).
            ``math.inf`` leaves the interval uncapped — the user's "No maximum"
            setting — in which case it is the half-life alone that bounds it.

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    if threshold >= 1.0:
        return 0.0
    if threshold <= 0.0:
        return max_delta_seconds
    return min(h * math.log2(1.0 / threshold), max_delta_seconds)


def next_delta(
    raw_params_last: torch.Tensor,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Analytically invert the curve from a raw ``(1,)`` head output.

    Thin wrapper over :func:`invert_curve` that first activates the raw param
    via :func:`half_life`.

    Args:
        raw_params_last: Raw head output for a single timestep, shape ``(1,)``.
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    return invert_curve(float(half_life(raw_params_last)), threshold, max_delta_seconds)


def retained_seconds(h: float) -> float:
    """Total area under ``R(Δt)``, in seconds — the whole curve, out to infinity.

    ``R`` is the probability that a test at ``Δt`` succeeds, so its integral is the
    expected amount of time the word stays recallable — the continuous counterpart
    of :func:`invert_curve`'s "time until recall falls to ``threshold``". Unlike
    that crossing time, it is finite and strictly positive for *every* curve, which
    is what lets cards be compared whether or not a single review lifts them over
    the user's threshold.

    Integrating the exponential over the whole half-line collapses to a constant
    times the half-life::

        ∫₀^∞ 2**(−t/H) dt = H/ln2

    so a curve is worth recallable seconds in direct proportion to its half-life,
    and comparing two curves is comparing two half-lives.

    A curve already part-way through its decay needs no separate formula. The
    exponential is memoryless — ``R(s + t) = R(s)·R(t)`` — so the area still ahead
    of a card last reviewed ``s`` ago is just the total scaled by the recall it has
    left::

        ∫ₛ^∞ 2**(−t/H) dt = R(s) · H/ln2

    which is why :func:`expected_gain` needs only the card's current recall, not
    how long it has been sitting.

    Args:
        h: The curve's half-life in seconds (already activated, ``> 0``).

    Returns:
        Expected recallable seconds, ``> 0``.
    """
    return h / _LN2


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
        success: Half-life of the curve fitted after a successful rep.
        failure: Half-life of the curve fitted after a failed rep.

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

    Both integrals run to infinity, where :func:`retained_seconds` reduces to
    ``H/ln2`` and the do-nothing baseline — the area still ahead of a curve
    already part-way through its decay — is the same constant scaled by the recall
    the card has left. So the whole key is three half-lives weighted by one
    probability::

        gain · ln2 = p·H_success + (1 − p)·H_failure − p·H_current

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
        current: Half-life of the card's present curve, or ``None`` when the card
            has never been practised in this direction. A never-practised card
            retains nothing, so its baseline is 0 and its gain is the whole
            post-review area.
        success: Half-life of the curve fitted after a successful rep.
        failure: Half-life of the curve fitted after a failed rep.

    Returns:
        Expected added recallable seconds. Negative when the card is better left
        alone than reviewed.
    """
    after = expected_retained(recall_now, success, failure)
    if current is None:
        return after
    return after - recall_now * retained_seconds(current)
