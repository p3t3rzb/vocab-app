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
        warm_start: When ``True`` (default), resume from the existing
            ``<src>_<tgt>.pt`` checkpoint if one is present and its
            architecture matches this config. On a mismatch (or no
            checkpoint), training falls back to a fresh random init.
    """

    epochs: int = 100
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 3e-03
    batch_size: int = 128
    val_split: float = 0.2
    seed: int = 42
    checkpoint_dir: Path = field(default_factory=lambda: Path("storage") / "models")
    lr_patience: int = 5
    lr_factor: float = 0.5
    grad_clip_max_norm: float = 1.0
    warm_start: bool = True


@dataclass
class PredictConfig:
    """Prediction-time hyperparameters for the forgetting-curve scheduler.

    The next-review time is found by inverting the predicted forgetting curve
    analytically (see :func:`src.model.curve.next_delta`), so only the recall
    threshold and the hard interval cap are needed.

    Attributes:
        recall_threshold: P(recall) level below which a word is due for review.
            Lower thresholds → longer intervals between repetitions.
        max_delta_seconds: Hard cap on the predicted interval (default 2 years).
    """

    recall_threshold: float = 0.8
    max_delta_seconds: float = 63_072_000.0  # 2-year cap


@dataclass
class ScheduleConfig:
    """Batched-scheduling knobs for :class:`BatchScheduler`.

    Attributes:
        chunk_size: Number of words processed per batched LSTM forward.
            Trades GPU dispatch overhead against memory footprint.
    """

    chunk_size: int = 256
