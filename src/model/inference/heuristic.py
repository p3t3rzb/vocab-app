"""Model-free fallback estimator for language pairs with no trained checkpoint.

A brand-new database is a chicken-and-egg problem: the LSTM needs repetition
history to train on, but without a model nothing schedules the repetitions that
would produce that history. :class:`HeuristicPredictor` breaks the cycle with an
SM-2-style rule that needs no training at all.

The trick is that it emits the **same** ``(p0, S, d)`` forgetting-curve params a
trained :class:`~src.model.inference.predictor.Predictor` does, so it is a
drop-in for it: the practice queue, the word list's due cache and the predict CLI
all keep reading stored params and never learn which produced them. Once the pair
has enough history to train on, the real model takes over and overwrites the
params word by word.

**How the interval is chosen.** Classic SM-2, adapted to this app's binary
remembered / not-remembered grades:

* an ease factor starts at 2.5 and moves ±0.1 / ∓0.2 per success / failure,
* a successful rep schedules the next one at ``elapsed_since_previous × ease``
  (so practising late and still recalling stretches the interval, and the
  short gaps of an in-session relearn ramp back up gradually),
* the first success after a lapse is capped at one day — SM-2 restarts the
  ladder rather than resuming it,
* a failed rep is assigned a recall ceiling *below* the reference threshold, so
  inverting the curve returns "due now" and the practice loop re-queues the card
  in the same session.

**How that becomes a curve.** With the target interval ``I`` and ceiling ``p0``
chosen above and ``d`` fixed, ``S`` is solved so the curve crosses the reference
threshold ``θ`` exactly at ``I``::

    θ = p0·(1 + I/S)**(−d)   ⇒   S = I / ((p0/θ)**(1/d) − 1)

which is :func:`~src.model.curve.invert_curve` run backwards. The stored params
therefore stay threshold-independent — the user's own recall threshold is applied
live on top, as it is for a trained model.
"""
from __future__ import annotations

from src.database import Direction
from src.database.models import Repetition

from ..config import HeuristicConfig, PredictConfig
from ..curve import invert_curve, recall_at


class HeuristicPredictor:
    """SM-2-style stand-in for :class:`Predictor`, usable with no trained model.

    Exposes the same ``config`` / ``curve_params`` / ``recall_probability`` /
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
    def _streak(reps: list[Repetition]) -> int:
        """Number of consecutive successes at the end of the history."""
        streak = 0
        for rep in reversed(reps):
            if not rep.remembered:
                break
            streak += 1
        return streak

    def curve_params(
        self, reps: list[Repetition], direction: Direction | None = None
    ) -> tuple[float, float, float]:
        """Return the ``(p0, S, d)`` curve params implied by this history.

        Args:
            reps: Repetition history for one (word, direction), oldest first.
                Must be non-empty.
            direction: Accepted for signature-compatibility with
                :meth:`Predictor.curve_params` and ignored — the history is
                already direction-specific, and the heuristic has no
                direction-dependent behaviour to condition on.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        h = self._h

        if not reps[-1].remembered:
            # Just failed: ceiling below the reference threshold ⇒ due now.
            p0, interval = h.lapse_p0, h.relearn_interval
        else:
            streak = self._streak(reps)
            p0 = min(h.p0_max, h.p0_base + h.p0_streak_bonus * (streak - 1))

            prev_gap = (
                reps[-1].practiced_at - reps[-2].practiced_at if len(reps) >= 2 else 0
            )
            if streak == 1 or prev_gap <= 0:
                # First success ever, or the first after a lapse — SM-2 restarts
                # the ladder at one day rather than resuming where it left off.
                interval = h.first_interval
            else:
                # Floored at one day so a card relearned mid-session graduates
                # instead of ramping up from the seconds-scale gap that a
                # same-session re-queue leaves behind.
                interval = max(prev_gap * self._ease(reps), h.first_interval)
            interval = min(interval, h.max_interval)

        # Solve S so the curve crosses the reference threshold exactly at `interval`.
        ratio = p0 / h.reference_threshold
        denominator = ratio ** (1.0 / h.decay) - 1.0 if ratio > 1.0 else 0.0
        s = interval / denominator if denominator > 0.0 else interval

        return p0, s, h.decay

    def recall_probability(
        self,
        reps: list[Repetition],
        delta_seconds: float,
        direction: Direction | None = None,
    ) -> float:
        """Return P(remembered) if tested ``delta_seconds`` after the last rep."""
        p0, s, d = self.curve_params(reps, direction)
        return recall_at(p0, s, d, delta_seconds)

    def next_repetition_delta(
        self, reps: list[Repetition], direction: Direction | None = None
    ) -> float:
        """Seconds until the next review, under the user's own recall threshold."""
        p0, s, d = self.curve_params(reps, direction)
        return invert_curve(
            p0, s, d, self._config.recall_threshold, self._config.max_delta_seconds
        )
