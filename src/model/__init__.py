"""LSTM-based recall predictor and spaced-repetition scheduler.

Re-exports the main entry points from the submodules:

* :class:`TrainConfig`, :class:`PredictConfig` — hyperparameter dataclasses.
* :class:`Sequence`, :func:`build_sequences`, :func:`split_sequences` — data
  pipeline that turns repetition history into LSTM training sequences.
* :class:`RecallLSTM` — the network itself.
* :class:`Predictor` — wraps a trained model to expose
  ``recall_probability`` and ``next_repetition_delta``.
* :func:`train`, :func:`load_model` — training entry point and checkpoint
  loader.
"""
from src.model.config import PredictConfig, TrainConfig
from src.model.dataset import Sequence, build_sequences, split_sequences
from src.model.lstm import RecallLSTM
from src.model.predictor import Predictor
from src.model.train import load_model, train
