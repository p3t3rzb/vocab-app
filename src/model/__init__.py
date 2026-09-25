"""LSTM-based recall predictor and spaced-repetition scheduler.

Public surface:

* :class:`TrainConfig`, :class:`PredictConfig`, :class:`ScheduleConfig` —
  hyperparameter dataclasses.
* :class:`RecallLSTM` — the network.
* :func:`load_model` — load a saved checkpoint.
* :class:`Predictor` — per-word recall probability and next-review estimates.
* :class:`HeuristicPredictor` — SM-2-style stand-in used when a language pair
  has no trained checkpoint yet (same :class:`Curve` output, no model needed).
* :class:`Trainer`, :func:`train` — training entry points.
* :class:`Curve` — a forgetting curve as the pair it is, ``(tau, ceiling)``. Both
  are predicted per cell and stored together; a time constant beside the wrong
  ceiling names a different curve.
* :func:`compute_all_params` — batched recomputation of every word's curves.
* :func:`backfill_heuristic_params` — the same, model-free, for untrained pairs.
* :func:`retained_seconds`, :func:`remaining_seconds`, :func:`expected_retained`,
  :func:`expected_gain` — the area under a freshly reviewed forgetting curve, the
  area still ahead of one part-way through its decay, the expected area a card
  would hold after a review, and the increase that review buys over leaving the
  card alone. The last is the practice queue's ordering key.
* :data:`NO_CEILING` — the curve ceiling ``p0 = 1`` assumed by a caller that has
  none to pass, i.e. the single-parameter family this one generalises.
"""
from .checkpoint import load_model
from .config import (
    HeuristicConfig,
    PredictConfig,
    ScheduleConfig,
    TrainConfig,
)
from .curve import (
    NO_CEILING,
    Curve,
    curve_recall,
    expected_gain,
    expected_retained,
    invert_curve,
    next_delta,
    recall_at,
    remaining_seconds,
    retained_seconds,
)
from .inference import (
    HeuristicPredictor,
    Predictor,
    RecallEstimator,
    backfill_heuristic_params,
    compute_all_params,
)
from .lstm import RecallLSTM
from .training import Trainer, train

__all__ = [
    "HeuristicConfig",
    "PredictConfig",
    "ScheduleConfig",
    "TrainConfig",
    "RecallLSTM",
    "Curve",
    "curve_recall",
    "recall_at",
    "NO_CEILING",
    "retained_seconds",
    "remaining_seconds",
    "expected_retained",
    "expected_gain",
    "invert_curve",
    "next_delta",
    "load_model",
    "Predictor",
    "HeuristicPredictor",
    "RecallEstimator",
    "Trainer",
    "train",
    "compute_all_params",
    "backfill_heuristic_params",
]
