"""Model-free fallback estimator for language pairs with no trained checkpoint.

A brand-new database is a chicken-and-egg problem: the LSTM needs repetition
history to train on, but without a model nothing schedules the repetitions that
would produce that history. :class:`HeuristicPredictor` breaks the cycle with an
SM-2-style rule that needs no training at all.

The trick is that it emits the **same** forgetting-curve time constant a trained
:class:`~src.model.inference.predictor.Predictor` does, so it is a drop-in for
it: the practice queue, the word list's due cache and the predict CLI all keep
reading the stored time constant and never learn which produced it. Once the pair has
enough history to train on, the real model takes over and overwrites it word by
word.

**How the interval is chosen.** Classic SM-2, adapted to this app's binary
remembered / not-remembered grades:

* an ease factor starts at 2.5 and moves ±0.1 / ∓0.2 per success / failure,
* a successful rep schedules the next one at ``elapsed_since_previous × ease``
  (so practising late and still recalling stretches the interval, and the
  short gaps of an in-session relearn ramp back up gradually),
* the first success after a lapse is capped at one day — SM-2 restarts the
  ladder rather than resuming it,
* a failed rep is put on a ten-minute relearning step, so the practice loop
  brings the card back later in the same session.

**How that becomes a curve.** With the target interval ``I`` chosen above, the
time constant is solved so the curve crosses the reference threshold ``θ`` exactly
at ``I``::

    θ = exp(−I/τ)   ⇒   τ = I / ln(1/θ)

which is :func:`~src.model.curve.invert_curve` run backwards. The stored time
constant therefore stays threshold-independent — the user's own recall threshold is applied
live on top, as it is for a trained model.
"""
from __future__ import annotations

import math

from src.database import Direction
from src.database.models import Repetition

from ..config import HeuristicConfig, PredictConfig
from ..curve import invert_curve, recall_at


class HeuristicPredictor:
    """SM-2-style stand-in for :class:`Predictor`, usable with no trained model.

    Exposes the same ``config`` / ``time_constant`` / ``recall_probability`` /
    ``next_repetition_delta`` surface, so callers can hold either one.
    """

    def __init__(
        self,
        config: PredictConfig | None = None,
        heuristic_config: HeuristicConfig | None = None,
    ) -> None:
        """Build the estimator. ``None`` configs use their dataclass defaults."""
        self._config = config or PredictConfig()
        self._h = heuristic_config or HeuristicConfig()

    @property
    def config(self) -> PredictConfig:
        """Expose the active prediction config (mirrors :class:`Predictor`)."""
        return self._config

    def _ease(self, reps: list[Repetition]) -> float:
        """Ease factor implied by the whole history's successes and failures."""
        h = self._h
        successes = sum(1 for rep in reps if rep.remembered)
        lapses = len(reps) - successes
        ease = h.ease_start + h.ease_bonus * successes - h.ease_penalty * lapses
        return min(max(ease, h.ease_min), h.ease_max)

    @staticmethod
    def _last_success_run(reps: list[Repetition]) -> int:
        """Number of consecutive successes at the end of the history."""
        streak = 0
        for rep in reversed(reps):
            if not rep.remembered:
                break
            streak += 1
        return streak

    def time_constant(
        self, reps: list[Repetition], direction: Direction | None = None
    ) -> float:
        """Return the curve time constant, in seconds, implied by this history.

        Args:
            reps: Repetition history for one (word, direction), oldest first.
                Must be non-empty.
            direction: Accepted for signature-compatibility with
                :meth:`Predictor.time_constant` and ignored — the history is already
                direction-specific, and the heuristic has no direction-dependent
                behaviour to condition on.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        h = self._h

        if not reps[-1].remembered:
            # Just failed: back in ten minutes, i.e. later in the same session.
            interval = h.relearn_interval
        else:
            prev_gap = (
                reps[-1].practiced_at - reps[-2].practiced_at if len(reps) >= 2 else 0
            )
            if self._last_success_run(reps) == 1 or prev_gap <= 0:
                # First success ever, or the first after a lapse — SM-2 restarts
                # the ladder at one day rather than resuming where it left off.
                interval = h.first_interval
            else:
                # Floored at one day so a card relearned mid-session graduates
                # instead of ramping up from the seconds-scale gap that a
                # same-session re-queue leaves behind.
                interval = max(prev_gap * self._ease(reps), h.first_interval)
            interval = min(interval, h.max_interval)

        # Solve τ so the curve crosses the reference threshold exactly at `interval`.
        return interval / math.log(1.0 / h.reference_threshold)

    def recall_probability(
        self,
        reps: list[Repetition],
        delta_seconds: float,
        direction: Direction | None = None,
    ) -> float:
        """Return P(remembered) if tested ``delta_seconds`` after the last rep."""
        return recall_at(self.time_constant(reps, direction), delta_seconds)

    def next_repetition_delta(
        self, reps: list[Repetition], direction: Direction | None = None
    ) -> float:
        """Seconds until the next review, under the user's own recall threshold."""
        return invert_curve(
            self.time_constant(reps, direction),
            self._config.recall_threshold,
            self._config.max_delta_seconds,
        )

    def post_rep_time_constants(
        self,
        histories: list[tuple[list[Repetition], Direction]],
        practiced_at: int,
        chunk_size: int | None = None,
    ) -> list[tuple[float, float]]:
        """Time constant each card's curve would take if it were answered now.

        The model-free counterpart of :meth:`Predictor.post_rep_time_constants`: it
        appends a hypothetical repetition at ``practiced_at`` to each history —
        once remembered, once forgotten — and re-runs the SM-2 rule.

        Note that the heuristic's time constant follows the ease ladder and the gap
        the card was actually left for, never anything word-specific, so the
        ordering it produces is close to plain worst-recalled-first. The scoring
        only really bites once a trained model is fitting the curves per word.

        Args:
            histories: One ``(reps, direction)`` per card, each oldest-first. An
                **empty** history is allowed and means a never-practised card: the
                hypothetical rep is then the whole history.
            practiced_at: Unix timestamp of the hypothetical repetition.
            chunk_size: Accepted for signature-compatibility with
                :meth:`Predictor.post_rep_time_constants` and ignored — there is no
                batching to do without a model.

        Returns:
            One ``(success_tau, failure_tau)`` per card, in the input order.
        """
        out: list[tuple[float, float]] = []
        for reps, direction in histories:
            word_id = reps[-1].word_id if reps else 0
            taus = tuple(
                self.time_constant(
                    reps
                    + [
                        Repetition(
                            word_id=word_id,
                            direction=int(direction),
                            practiced_at=practiced_at,
                            remembered=remembered,
                        )
                    ],
                    direction,
                )
                for remembered in (True, False)
            )
            out.append(taus)  # type: ignore[arg-type]
        return out
