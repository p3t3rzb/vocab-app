"""Per-word inference helper for a trained :class:`RecallLSTM`.

The model predicts the parameters of a forgetting curve from a word's
repetition history; this helper turns those parameters into a point recall
probability at a given gap and into an analytically-derived next-review time
(no bisection — see :func:`src.model.curve.next_delta`).
"""
from __future__ import annotations

import torch

from src.database import Direction
from src.database.models import Repetition

from ..config import PredictConfig, ScheduleConfig
from ..curve import curve_recall, next_delta, split_params
from ..features import history_rows, rep_row
from ..lstm import RecallLSTM
from .batch import Params, final_step_params


class Predictor:
    """Inference helper exposing point probabilities and next-review estimates.

    Wraps an already-trained, already-on-device model. ``eval()`` is
    enforced on the wrapped model so dropout can't leak into inference
    even if the caller forgot to switch modes.
    """

    def __init__(self, model: RecallLSTM, config: PredictConfig | None = None) -> None:
        """Wrap a trained model. ``None`` config uses :class:`PredictConfig` defaults."""
        self._model = model
        self._model.eval()
        self._config = config or PredictConfig()
        self._device = next(model.parameters()).device

    @property
    def config(self) -> PredictConfig:
        """Expose the active prediction config (used by the predict CLI)."""
        return self._config

    @property
    def model(self) -> RecallLSTM:
        """The wrapped network, for callers running their own batched forwards."""
        return self._model

    def _curve_params(self, reps: list[Repetition], direction: Direction) -> torch.Tensor:
        """Forward the history and return the curve params for the next test.

        Builds the history-only input (one row per rep: ``[log(gap-before-this-
        rep + 1), remembered, not_remembered, is_forward, is_reverse]``, first
        row's gap is 0) and returns the raw ``(3,)`` param vector from the final
        timestep.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        rows = history_rows(reps, direction)
        x = torch.tensor(rows, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            raw = self._model(x)  # (1, L, 3)
        return raw[0, -1]  # (3,)

    def curve_params(
        self, reps: list[Repetition], direction: Direction
    ) -> tuple[float, float, float]:
        """Return the activated ``(p0, S, d)`` curve params for the next test.

        These are the values persisted per (word, direction) so recall and
        next-review time can be derived live without another model forward.

        Args:
            reps: Repetition history, oldest first. Must be non-empty.
            direction: Practice direction the history belongs to.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        raw_last = self._curve_params(reps, direction)
        p0, s, d = split_params(raw_last)
        return float(p0), float(s), float(d)

    def recall_probability(
        self, reps: list[Repetition], delta_seconds: float, direction: Direction
    ) -> float:
        """Return P(remembered) if the word is tested ``delta_seconds`` after its last rep.

        Args:
            reps: Repetition history, oldest first. Must be non-empty.
            delta_seconds: Hypothetical gap (in seconds) after the most recent
                rep at which to probe the curve.
            direction: Practice direction the history belongs to.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        raw_last = self._curve_params(reps, direction)
        delta = torch.tensor([[delta_seconds]], dtype=torch.float32, device=self._device)
        return curve_recall(delta, raw_last.view(1, 1, 3)).item()

    def next_repetition_delta(
        self, reps: list[Repetition], direction: Direction
    ) -> float:
        """Seconds-until-next-review, by analytically inverting the forgetting curve."""
        raw_last = self._curve_params(reps, direction)
        return next_delta(
            raw_last, self._config.recall_threshold, self._config.max_delta_seconds
        )

    def post_rep_params(
        self,
        histories: list[tuple[list[Repetition], Direction]],
        practiced_at: int,
        chunk_size: int | None = None,
    ) -> list[tuple[Params, Params]]:
        """Curve params each card would take if it were answered now.

        Appends a *hypothetical* repetition at ``practiced_at`` to each history —
        once remembered, once forgotten — and reads off the curve the model fits
        afterwards. Feeding both outcomes through the same batched forward keeps
        the cost at one pass per chunk, which is what makes precomputing these for
        every card in the deck affordable (see
        :class:`~src.model.inference.scheduler.ParamScheduler`).

        Args:
            histories: One ``(reps, direction)`` per card, each oldest-first. An
                **empty** history is allowed and means a never-practised card: the
                hypothetical rep is then the whole sequence, with a zero gap.
            practiced_at: Unix timestamp of the hypothetical repetition; its gap
                from each history's last rep becomes the appended row's input.
            chunk_size: Sequences per batched forward — two per card. Defaults to
                :attr:`ScheduleConfig.batch_sequences`.

        Returns:
            One ``(success_params, failure_params)`` per card, in the input order.
        """
        sequences: list[list[list[float]]] = []
        for reps, direction in histories:
            rows = history_rows(reps, direction)
            gap = practiced_at - reps[-1].practiced_at if reps else 0.0
            sequences.append(rows + [rep_row(gap, True, direction)])
            sequences.append(rows + [rep_row(gap, False, direction)])

        params = final_step_params(
            self._model,
            sequences,
            chunk_size=chunk_size or ScheduleConfig().batch_sequences,
        )
        return [
            (params[2 * i], params[2 * i + 1]) for i in range(len(histories))
        ]
