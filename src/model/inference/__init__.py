"""Inference: per-word predictions and batched param computation."""
from src.model.inference.heuristic import HeuristicPredictor
from src.model.inference.predictor import Predictor
from src.model.inference.scheduler import (
    ParamScheduler,
    backfill_heuristic_params,
    compute_all_params,
)

#: Either estimator — both emit the same ``(p0, S, d)`` curve params, so callers
#: that just read params can hold one without caring which produced it.
RecallEstimator = Predictor | HeuristicPredictor

__all__ = [
    "Predictor",
    "HeuristicPredictor",
    "RecallEstimator",
    "ParamScheduler",
    "compute_all_params",
    "backfill_heuristic_params",
]
