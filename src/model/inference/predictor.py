"""Per-word inference helper for a trained :class:`RecallLSTM`.

The model predicts the parameters of a forgetting curve from a word's
repetition history; this helper turns those parameters into a point recall
probability at a given gap and into an analytically-derived next-review time
(no bisection — see :func:`src.model.curve.next_delta`).
"""
from __future__ import annotations

import math

import torch

from src.database.models import Repetition
from src.model.config import PredictConfig
from src.model.curve import curve_recall, next_delta
from src.model.lstm import RecallLSTM


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

    def _curve_params(self, reps: list[Repetition]) -> torch.Tensor:
        """Forward the history and return the curve params for the next test.

        Builds the history-only input (one row per rep: ``[log(gap-before-this-
        rep + 1), remembered]``, first row's gap is 0) and returns the raw
        ``(3,)`` parameter vector from the final timestep.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        rows: list[list[float]] = []
        for i, rep in enumerate(reps):
            log_gap = 0.0 if i == 0 else math.log(rep.practiced_at - reps[i - 1].practiced_at + 1)
            rows.append([log_gap, float(rep.remembered)])

        x = torch.tensor(rows, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            raw = self._model(x)  # (1, L, 3)
        return raw[0, -1]  # (3,)

    def recall_probability(self, reps: list[Repetition], delta_seconds: float) -> float:
        """Return P(remembered) if the word is tested ``delta_seconds`` after its last rep.

        Args:
            reps: Repetition history, oldest first. Must be non-empty.
            delta_seconds: Hypothetical gap (in seconds) after the most recent
                rep at which to probe the curve.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        raw_last = self._curve_params(reps)
        delta = torch.tensor([[delta_seconds]], dtype=torch.float32, device=self._device)
        return curve_recall(delta, raw_last.view(1, 1, 3)).item()

    def next_repetition_delta(self, reps: list[Repetition]) -> float:
        """Seconds-until-next-review, by analytically inverting the forgetting curve."""
        raw_last = self._curve_params(reps)
        return next_delta(
            raw_last, self._config.recall_threshold, self._config.max_delta_seconds
        )
