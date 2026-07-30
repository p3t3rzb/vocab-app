"""LSTM-based recall predictor and spaced-repetition scheduler.

Public surface:

* :class:`TrainConfig`, :class:`PredictConfig`, :class:`ScheduleConfig` —
  hyperparameter dataclasses.
* :class:`RecallLSTM` — the network.
* :func:`load_model` — load a saved checkpoint.
* :class:`Predictor` — per-word recall probability and next-review estimates.
* :class:`HeuristicPredictor` — SM-2-style stand-in used when a language pair
  has no trained checkpoint yet (same ``(p0, S, d)`` output, no model needed).
* :class:`Trainer`, :func:`train` — training entry points.
* :func:`compute_all_params` — batched recomputation of every word's curve params.
* :func:`backfill_heuristic_params` — the same, model-free, for untrained pairs.
"""
from .checkpoint import load_model
from .config import (
    HeuristicConfig,
    PredictConfig,
    ScheduleConfig,
    TrainConfig,
)
from .curve import curve_recall, invert_curve, next_delta, recall_at
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
    "curve_recall",
    "recall_at",
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
