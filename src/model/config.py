"""Hyperparameter dataclasses for training, prediction, and scheduling.

Keeping these in one place makes it easy to share defaults between the CLI
trainer, the GUI training screen, the prediction code paths, and the batched
scheduler.
"""

from dataclasses import dataclass, field
from pathlib import Path


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
        checkpoint_dir: Directory where ``<src>_<tgt>.pt`` checkpoints are
            written. Created automatically if missing.
        lr_patience: Epochs of stagnant val loss before ``ReduceLROnPlateau``
            cuts the learning rate.
        lr_factor: Multiplicative factor applied to the learning rate when
            ``ReduceLROnPlateau`` fires.
        grad_clip_max_norm: Global gradient-norm cap applied before each
            optimizer step (``torch.nn.utils.clip_grad_norm_``).
    """

    epochs: int = 100
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch_size: int = 256
    val_split: float = 0.2
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("storage") / "models")
    lr_patience: int = 5
    lr_factor: float = 0.5
    grad_clip_max_norm: float = 1.0


@dataclass
class PredictConfig:
    """Prediction-time hyperparameters for the threshold-crossing search.

    Attributes:
        recall_threshold: P(recall) level below which a word is due for review.
            Lower thresholds → longer intervals between repetitions.
        bisect_steps: Number of binary-search iterations performed after the
            initial bracketing phase.
        initial_delta_seconds: Starting upper-bound guess for the doubling
            phase (default 1 day).
        max_delta_seconds: Hard cap on the predicted interval (default 1 year).
        max_doubling_iters: Safety bound on the doubling phase — caps how many
            times ``hi`` may double before we conclude the threshold is
            unreachable within ``max_delta_seconds``.
    """

    recall_threshold: float = 0.8
    bisect_steps: int = 16
    initial_delta_seconds: float = 86_400.0  # 1 day starting upper-bound guess
    max_delta_seconds: float = 63_072_000.0  # 2-year cap
    max_doubling_iters: int = 30


@dataclass
class ScheduleConfig:
    """Batched-scheduling knobs for :class:`BatchScheduler`.

    Attributes:
        chunk_size: Number of words processed per batched LSTM forward.
            Trades GPU dispatch overhead against memory footprint.
    """

    chunk_size: int = 256
