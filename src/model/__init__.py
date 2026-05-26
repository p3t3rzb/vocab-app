"""LSTM-based recall predictor and spaced-repetition scheduler.

Public surface:

* :class:`TrainConfig`, :class:`PredictConfig`, :class:`ScheduleConfig` —
  hyperparameter dataclasses.
* :class:`RecallLSTM` — the network.
* :func:`load_model` — load a saved checkpoint.
* :class:`Predictor` — per-word recall probability and next-review estimates.
* :class:`Trainer`, :func:`train` — training entry points.
* :func:`compute_all_schedules` — batched re-scheduling of every word.
"""
from src.model.checkpoint import load_model
from src.model.config import PredictConfig, ScheduleConfig, TrainConfig
from src.model.inference import Predictor, compute_all_schedules
from src.model.lstm import RecallLSTM
from src.model.training import Trainer, train

__all__ = [
    "PredictConfig",
    "ScheduleConfig",
    "TrainConfig",
    "RecallLSTM",
    "load_model",
    "Predictor",
    "Trainer",
    "train",
    "compute_all_schedules",
]
