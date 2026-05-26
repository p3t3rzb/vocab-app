"""Hyperparameter dataclasses for training and prediction.

Keeping these in one place makes it easy to share defaults between the CLI
trainer, the GUI training screen, and the prediction code paths.
"""
from dataclasses import dataclass


@dataclass
class TrainConfig:
    """LSTM training hyperparameters.

    Attributes:
        epochs: Number of full passes over the training set.
        hidden_size: LSTM hidden state size per layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between LSTM layers and before
            the output head.
        lr: Adam learning rate.
        batch_size: Number of sequences per mini-batch.
        val_split: Fraction of words (not sequences) held out for validation,
            ensuring no word leaks across the train/val split.
        seed: RNG seed for reproducible splits and weight initialisation.
    """

    epochs: int = 100
    hidden_size: int = 16
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    val_split: float = 0.2
    seed: int = 42


@dataclass
class PredictConfig:
    """Prediction-time hyperparameters for :class:`Predictor`.

    Attributes:
        recall_threshold: P(recall) level below which a word is due for review.
            Lower thresholds → longer intervals between repetitions.
        bisect_steps: Number of binary-search iterations performed after the
            initial bracketing phase.
        initial_delta_seconds: Starting upper-bound guess for the doubling
            phase (default 1 day).
        max_delta_seconds: Hard cap on the predicted interval (default 1 year).
        poly_degree: Degree of the polynomial fitted to probed points to
            refine the threshold-crossing estimate.
    """

    recall_threshold: float = 0.8
    bisect_steps: int = 16
    initial_delta_seconds: float = 86_400.0  # 1 day starting upper-bound guess
    max_delta_seconds: float = 31_536_000.0  # 1-year cap
    poly_degree: int = 2
