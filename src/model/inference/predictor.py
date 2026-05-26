"""Per-word inference helper for a trained :class:`RecallLSTM`."""
from __future__ import annotations

import math

import torch

from src.database.models import Repetition
from src.model.config import PredictConfig
from src.model.inference.search import ThresholdSearch
from src.model.lstm import RecallLSTM


class Predictor:
    """Inference helper exposing point probabilities and next-review estimates.

    Wraps an already-trained, already-on-device model. Caller is responsible
    for placing the model on its desired device and calling ``.eval()``
    (which :func:`src.model.load_model` does).
    """

    def __init__(self, model: RecallLSTM, config: PredictConfig | None = None) -> None:
        """Wrap a trained model. ``None`` config uses :class:`PredictConfig` defaults."""
        self._model = model
        self._config = config or PredictConfig()
        self._device = next(model.parameters()).device
        self._search = ThresholdSearch(self._config)

    @property
    def config(self) -> PredictConfig:
        """Expose the active prediction config (used by the predict CLI)."""
        return self._config

    def recall_probability(self, reps: list[Repetition], delta_seconds: float) -> float:
        """Return P(remembered) if the word is tested ``delta_seconds`` after its last rep.

        Args:
            reps: Repetition history, oldest first. Must be non-empty.
            delta_seconds: Hypothetical gap (in seconds) after the most recent
                rep at which to probe the model.

        Raises:
            ValueError: if ``reps`` is empty.
        """
        if not reps:
            raise ValueError("Need at least one historical repetition")

        inputs: list[list[float]] = []
        for i in range(1, len(reps)):
            dt = reps[i].practiced_at - reps[i - 1].practiced_at
            inputs.append([math.log(dt + 1), float(reps[i - 1].remembered)])
        inputs.append([math.log(delta_seconds + 1), float(reps[-1].remembered)])

        x = torch.tensor(inputs, dtype=torch.float32, device=self._device).unsqueeze(0)
        with torch.no_grad():
            probs = self._model(x)
        return probs[0, -1].item()

    def next_repetition_delta(self, reps: list[Repetition]) -> float:
        """Estimate seconds-until-next-review for this word.

        Delegates to :class:`ThresholdSearch` with a probe closure over
        :meth:`recall_probability`.
        """
        return self._search.find_delta(lambda delta: self.recall_probability(reps, delta))
