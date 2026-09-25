"""Forgetting-curve math for the :class:`~src.model.lstm.RecallLSTM` head.

The network predicts, per timestep, the *parameters of a forgetting curve*
``R(Δt)`` rather than ``P(remembered)`` directly. This module turns the
network's raw two-channel output into a recall probability at a given gap,
and inverts the curve to find the next-review time analytically (no bisection).

The curve is an exponential decay under a **ceiling**::

    R(Δt) = p0 * exp(−Δt / τ)

with both parameters derived from the head's raw output, per cell and per
timestep:

* ``τ = exp(raw)`` — the **time constant**, in seconds: the gap at which recall
  has fallen to ``1/e`` of the ceiling. It is *not* the half-life, which is the
  shorter ``τ·ln2``.
* ``p0`` — the **ceiling**, recall in the instant after a review. The head's
  *second* channel, emitted per timestep exactly like ``τ`` and passed through a
  sigmoid so it can never leave ``(0, 1)``.

A curve is therefore a **pair**, and the pair is what this module passes around
(:class:`Curve`). A time constant on its own does not name a curve, and putting
one next to the wrong ceiling is the single easiest mistake to make here, so the
two travel together wherever a caller holds a whole curve.

**Why base e rather than base 2.** The two are the same family — ``2**(−Δt/H)``
is this curve at ``τ = H/ln2`` — so the choice is one of bookkeeping, and base e
puts the constant where it does no work. The time constant *is* the area under
the curve (:func:`retained_seconds`), so every integral below is a plain
product: a cell is worth ``p0·τ`` recallable seconds, and the practice queue's
ordering key is three time constants weighted by one probability with no
conversion factor anywhere. Under base 2 each of those carried a ``1/ln2`` that
cancelled out of every comparison but had to be written, and got in the way of
reading the parameter as the quantity being maximised. What is lost is that one
unit of raw output used to mean exactly one doubling of the interval; it now
means one factor of e. The half-life is still one multiplication away when a
human wants to read it.

``R`` starts at ``p0`` and decays monotonically toward 0. A word's forgetting
behaviour is carried by both numbers: ``τ`` says how long it holds, ``p0`` how
likely the answer is to be there at all the moment after a review.

**Why the ceiling is predicted and not a constant.** ``τ`` sets the *rate* a
curve decays at; ``p0`` sets the *level* it starts from. With a fixed ceiling
those collapse into a single knob, because the only way to predict a low recall
is to decay down to it — and decay does not stop there. Measured on the French
deck's held-out words, the best single curve fitted to the reps that follow a
*lapse* scores 1.228 nats with ``p0`` pinned at a deck-wide 0.900 and 0.671 with
``p0`` free, which it puts at 0.403. That 0.557 is bought with no new information
at all: the network already receives the previous outcome on every input row. It
simply had no degree of freedom to act on it, and was spending ``τ`` to
approximate a level. Post-lapse recall is close to a flat 0.40 and post-success
recall close to a flat 0.77 — two states differing almost entirely in level — and
a per-cell head can express more than that split: how many times *this* word has
lapsed, and how recently, are in the history the LSTM already carries.

**Why a ceiling at all.** With ``p0 = 1`` recall in the instant after a review is
certain, and a card re-shown forty seconds after a lapse scores ``exp(−40/τ) ≈ 1``
for any time constant worth scheduling — the family cannot express a same-session
relapse without collapsing ``τ`` to under a minute, which then has to serve that
cell's next review a day out as well.

``ceiling=1.0`` — :data:`NO_CEILING`, the default on the float-taking functions
here — reproduces the single-parameter family exactly. That is what
:class:`~src.model.inference.HeuristicPredictor` emits (SM-2 fits no ceiling) and
what a database or checkpoint written before the ceiling existed carries.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import torch

# Probabilities are held this far away from 0 and 1 so a downstream ``log`` is safe.
PARAM_EPS = 1e-6

# Bounds on the raw ``ln τ`` output, i.e. a time constant in [~1 µs, ~4e11 years].
# They exist only to keep ``exp(raw)`` finite and strictly positive; both ends are
# far outside any schedule, so the clamp never shapes a real prediction.
LOG_TAU_MIN = -14.0
LOG_TAU_MAX = 44.0

# Starting time constant for an untrained head, as ``ln(seconds)``: three days. The
# head's bias is initialised here so the first forward pass already lands in the
# range real gaps live in, where the curve has a usable gradient.
LOG_TAU_INIT = math.log(3 * 86_400.0)

# Starting ceiling for an untrained head, as a logit (the channel runs through a
# sigmoid so it can never leave ``(0, 1)``). 0.98 sits just below the
# single-parameter family's implicit 1.0, so an untrained model behaves as the
# old one did while still having a usable gradient: at a near-zero gap
# ``∂R/∂p0 = 1``, which is where a lapse rate is actually observed. The bias
# starts flat here rather than anywhere state-specific so nothing about the
# post-lapse curve is baked in before the data speaks.
CEILING_INIT = 0.98
CEILING_LOGIT_INIT = math.log(CEILING_INIT / (1.0 - CEILING_INIT))

# Bounds on the raw ceiling logit. ``sigmoid(±18)`` is within 2e-8 of the ends,
# so like the ``τ`` clamp this only keeps the value strictly inside ``(0, 1)``;
# it is far outside anything a fitted ceiling reaches.
CEILING_LOGIT_MIN = -18.0
CEILING_LOGIT_MAX = 18.0

# The ceiling a caller assumes when it has none to pass: recall right after a
# review is certain, i.e. the single-parameter curve this family generalises.
NO_CEILING = 1.0

#: Channels the head emits per timestep: ``[ln τ, ceiling logit]``.
HEAD_OUTPUTS = 2


class Curve(NamedTuple):
    """One forgetting curve: a time constant and the ceiling it starts from.

    Both numbers are needed to evaluate, invert or integrate ``R``, and a ``τ``
    beside the wrong ``p0`` is silently wrong rather than loudly so. Keeping them
    in one object is the cheapest guard available, and it is why the functions
    below that take a *whole* curve take this rather than two floats.

    The exception is the do-nothing baseline in :func:`expected_gain`, which takes
    a bare ``τ``: there the ceiling is already inside the card's current recall,
    and passing it would apply it twice. See :func:`remaining_seconds`.

    Attributes:
        tau: Time constant in seconds, ``> 0``.
        ceiling: ``p0`` in ``(0, 1]``, recall in the instant after the review that
            started this curve.
    """

    tau: float
    ceiling: float


def time_constant(raw_params: torch.Tensor) -> torch.Tensor:
    """Map channel 0 of a raw ``(..., 2)`` head output to a time constant in seconds.

    That channel *is* ``ln τ``, so this is an exponential, clamped at both
    ends only to keep the result finite and non-zero.

    Args:
        raw_params: The network's unactivated output, last dim of size
            :data:`HEAD_OUTPUTS`.

    Returns:
        ``τ`` in seconds, shaped like ``raw_params[..., 0]``.
    """
    log_tau = raw_params[..., 0].clamp(LOG_TAU_MIN, LOG_TAU_MAX)
    return torch.exp(log_tau)


def ceiling(raw_params: torch.Tensor) -> torch.Tensor:
    """Map channel 1 of a raw ``(..., 2)`` head output to a ceiling in ``(0, 1)``.

    That channel is the ceiling's **logit**, so this is a sigmoid. Emitting a
    logit rather than the probability is what lets an unconstrained linear head
    drive the ceiling anywhere in the open interval without a projection step,
    exactly as ``ln τ`` does for the time constant.

    Args:
        raw_params: The network's unactivated output, last dim of size
            :data:`HEAD_OUTPUTS`.

    Returns:
        ``p0``, shaped like ``raw_params[..., 1]``.
    """
    logit = raw_params[..., 1].clamp(CEILING_LOGIT_MIN, CEILING_LOGIT_MAX)
    return torch.sigmoid(logit)


def curve_from(raw_params: torch.Tensor) -> Curve:
    """Read a single timestep's raw ``(2,)`` output into a plain-float :class:`Curve`.

    The bridge from a model forward to everything stored and derived downstream.

    Args:
        raw_params: One timestep's unactivated output, shape ``(HEAD_OUTPUTS,)``.
    """
    raw = raw_params.detach()
    return Curve(float(time_constant(raw)), float(ceiling(raw)))


def curve_recall(
    deltas: torch.Tensor,
    raw_params: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Evaluate ``R(Δt) = p0·exp(−Δt/τ)`` and clamp for safe BCE.

    Both parameters come out of ``raw_params``, so training needs no second
    input: one backward pass fits the time constant and the ceiling together,
    each through the curve. At short gaps ``∂R/∂p0 → 1`` while the ``τ``
    gradient is damped, and at long gaps the reverse — which is what lets the two
    channels separate at all.

    Args:
        deltas: Query gaps in seconds, shape ``(B, L)`` (must be ``≥ 0``).
        raw_params: Raw head output, shape ``(B, L, HEAD_OUTPUTS)``.
        eps: Probabilities are clamped to ``[eps, 1 − eps]`` so the downstream
            ``log`` in BCE never sees 0 or 1.

    Returns:
        Recall probabilities of shape ``(B, L)``.
    """
    tau = time_constant(raw_params)
    return (ceiling(raw_params) * torch.exp(-deltas / tau)).clamp(eps, 1.0 - eps)


def recall_at(tau: float, delta_seconds: float, p0: float = NO_CEILING) -> float:
    """Evaluate the recall curve ``R(Δt) = p0·exp(−Δt/τ)`` on Python floats.

    The pure-``math`` counterpart of :func:`curve_recall`, for live recall scoring
    from the stored time constant (no tensor / no model forward). ``Δt`` is clamped
    to ``≥ 0`` and the result to ``(0, 1)``.

    Args:
        tau: The curve's time constant in seconds (already activated, ``> 0``).
        delta_seconds: Gap since the last review.
        p0: The ceiling ``tau`` was emitted beside — the other half of the same
            :class:`Curve`. A time constant means a different curve under a
            different ceiling, so these are stored together per (word, direction)
            and must be read together.
    """
    delta = max(0.0, delta_seconds)
    recall = p0 * math.exp(-delta / tau)
    return min(max(recall, PARAM_EPS), 1.0 - PARAM_EPS)


def invert_curve(
    tau: float,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
    p0: float = NO_CEILING,
) -> float:
    """Seconds until ``R(Δt)`` falls to ``threshold``, from the stored time constant.

    Solving ``threshold = p0·exp(−Δt/τ)`` for ``Δt`` gives
    ``Δt = τ·ln(p0/threshold)`` — the interval is the time constant scaled by a
    constant the threshold picks, so raising the threshold shortens every word's
    interval by the same factor. The ceiling shifts that constant: under
    ``p0 < 1`` every interval is shorter than the ceiling-free family gave for
    the same ``τ``.

    Args:
        tau: The curve's time constant in seconds (already activated, ``> 0``).
        threshold: Recall level below which the word is considered due. A
            ``threshold ≥ p0`` is unreachable — recall is below it from the
            instant after a review — so the word is due now. With the ceiling
            predicted per cell this is no longer a rare edge case: any card whose
            fitted ``p0`` sits under the user's threshold is *permanently* due,
            which is the correct reading of "this answer is only 40% likely to be
            there" against a threshold of 0.8, and what the relearning step in the
            practice queue then handles.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).
            ``math.inf`` leaves the interval uncapped — the user's "No maximum"
            setting — in which case it is the time constant alone that bounds it.
        p0: The ceiling ``tau`` was emitted beside.

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    if threshold >= p0:
        return 0.0
    if threshold <= 0.0:
        return max_delta_seconds
    return min(tau * math.log(p0 / threshold), max_delta_seconds)


def next_delta(
    raw_params_last: torch.Tensor,
    threshold: float,
    max_delta_seconds: float = 63_072_000.0,
) -> float:
    """Analytically invert the curve from a raw ``(HEAD_OUTPUTS,)`` head output.

    Thin wrapper over :func:`invert_curve` that first activates both channels via
    :func:`curve_from`, so the ceiling is the one this very timestep emitted.

    Args:
        raw_params_last: Raw head output for a single timestep, shape
            ``(HEAD_OUTPUTS,)``.
        threshold: Recall level below which the word is considered due.
        max_delta_seconds: Hard cap on the returned interval (default 2 years).

    Returns:
        Seconds until the next review, in ``[0, max_delta_seconds]``.
    """
    curve = curve_from(raw_params_last)
    return invert_curve(curve.tau, threshold, max_delta_seconds, curve.ceiling)


def retained_seconds(curve: Curve) -> float:
    """Total area under a *freshly reviewed* ``R(Δt)``, in seconds.

    ``R`` is the probability that a test at ``Δt`` succeeds, so its integral is the
    expected amount of time the word stays recallable — the continuous counterpart
    of :func:`invert_curve`'s "time until recall falls to ``threshold``". Unlike
    that crossing time, it is finite and strictly positive for *every* curve, which
    is what lets cards be compared whether or not a single review lifts them over
    the user's threshold.

    In base e the integral is the parameter itself, scaled by the ceiling::

        ∫₀^∞ p0·exp(−t/τ) dt = p0·τ

    so a fresh curve is worth the plain product of its two parameters. With the
    ceiling predicted per cell it no longer cancels out of a comparison: two
    cards are ranked on ``p0·τ``, not on ``τ`` alone, and a card the model
    expects to blank on is discounted for that on top of its shorter ``τ``.

    This is the area of a curve starting *at* its ceiling, which is what a review
    resets a card to. For the area still ahead of a card part-way through its
    decay use :func:`remaining_seconds`, which takes the recall it has left
    instead — the ceiling is already inside that number, and applying it twice is
    the one mistake this split exists to prevent.

    Args:
        curve: The whole curve — both parameters, as emitted together.

    Returns:
        Expected recallable seconds, ``> 0``.
    """
    return curve.ceiling * curve.tau


def remaining_seconds(recall_now: float, tau: float) -> float:
    """Area still ahead of a curve already ``recall_now`` of the way down.

    The exponential is memoryless — ``R(s + t) = R(s)·R(t)/p0`` — so the area left
    to a card last reviewed ``s`` ago is the ceiling-free total scaled by the
    recall it has left::

        ∫ₛ^∞ p0·exp(−t/τ) dt = R(s) · τ

    which is why :func:`expected_gain` needs only the card's current recall, not
    how long it has been sitting.

    Note there is **no ceiling factor here**: ``p0`` is already inside
    ``recall_now``. That asymmetry with :func:`retained_seconds` is real, not an
    oversight — one measures a curve from its start, the other from part-way
    down — and it is why the two are separate functions.

    Args:
        recall_now: Current ``R(s)`` for the card, in ``(0, 1)``.
        tau: The card's time constant in seconds (already activated, ``> 0``).

    Returns:
        Expected remaining recallable seconds, ``> 0``.
    """
    return recall_now * tau


def expected_retained(
    recall_now: float,
    success: Curve,
    failure: Curve,
) -> float:
    """Expected recallable seconds a card *would have* after practising it now.

    Practising replaces the card's curve with a freshly-fitted one — ``success``
    if the answer is right, ``failure`` if it is wrong — and ``recall_now`` is
    itself the probability of getting it right, so the expected area under the
    card's curve afterwards is::

        V = p·∫R_success + (1 − p)·∫R_failure

    Both post-review curves start at their own ceiling, so both areas come from
    :func:`retained_seconds` — and the two ceilings are generally different, which
    is what makes an expected lapse cost something here rather than merely
    shortening an interval.

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
        success: The whole curve fitted after a successful rep.
        failure: The whole curve fitted after a failed rep.

    Returns:
        Expected recallable seconds, ``> 0``.
    """
    return (
        recall_now * retained_seconds(success)
        + (1.0 - recall_now) * retained_seconds(failure)
    )


def expected_gain(
    recall_now: float,
    current_tau: float | None,
    success: Curve,
    failure: Curve,
) -> float:
    """Expected recallable seconds practising a card *right now* would **add**.

    The practice queue's ordering key, largest first. Practising is worth the
    difference between what the card will retain if it is reviewed and what it
    would have retained if it were left alone::

        gain = E[∫R_after] − ∫R_current

    Both integrals run to infinity. The post-review curves start fresh at their
    ceilings, while the do-nothing baseline is the area still ahead of a curve
    already part-way through its decay, which carries its ceiling inside
    ``recall_now`` instead (see :func:`remaining_seconds`). In base e both
    collapse to plain products, so the whole key is three time constants weighted
    by one probability, each post-review term scaled by its *own* ceiling, and no
    conversion factor at all::

        gain = p·p0_success·τ_success + (1 − p)·p0_failure·τ_failure − p·τ_current

    Note the last term carries no ceiling. ``current_tau`` is therefore a bare
    ``float`` rather than a :class:`Curve`: the current curve's ``p0`` is already
    inside ``recall_now``, and multiplying by it again is the one real trap in
    this file.

    Ordering on the increase rather than on the level is what greedily maximises
    the total recall held across the whole vocabulary, ``∫ Σ_cells R_cell(t) dt``:
    a card that already holds its recall contributes that recall whether or not it
    is practised, so only the part a review *adds* is worth spending the session on.

    Three consequences worth knowing:

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
    * **A ceiling below 1 lowers every gain**, because a review can only restore a
      card to ``p0``, not to certainty, while the baseline it is measured against
      is unscaled. Cards whose recall is already near their ceiling therefore drop
      out of contention sooner than they did under the single-parameter family.
      Where the failure curve's ceiling is well below the success curve's, that
      branch is discounted twice over — a shorter time constant *and* a lower
      ceiling — so a card the model expects to blank on is worth less of a session
      than its time constants alone would suggest. Because the ceilings are now
      per cell they no longer cancel between two cards either: ordering compares
      products, which is more expressive and correspondingly more sensitive to a
      badly-fitted ``p0``.

    Args:
        recall_now: Current ``R(Δt)`` for the card, in ``(0, 1)`` — the probability
            the answer is right, and equally the fraction of its curve the card has
            left. For a word never practised in this direction, the deck's
            first-attempt success rate stands in.
        current_tau: Time constant of the card's present curve — *without* its
            ceiling, which is already inside ``recall_now``. ``None`` when the card
            has never been practised in this direction: such a card retains
            nothing, so its baseline is 0 and its gain is the whole post-review
            area.
        success: The whole curve fitted after a successful rep.
        failure: The whole curve fitted after a failed rep.

    Returns:
        Expected added recallable seconds. Negative when the card is better left
        alone than reviewed.
    """
    after = expected_retained(recall_now, success, failure)
    if current_tau is None:
        return after
    return after - remaining_seconds(recall_now, current_tau)
