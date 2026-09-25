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
        weight_decay: L2 regularization strength passed to the Adam
            optimizer (``0.0`` disables it).
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
    weight_decay: float = 0.0


@dataclass
class PredictConfig:
    """Prediction-time hyperparameters for the forgetting-curve scheduler.

    The next-review time is found by inverting the predicted forgetting curve
    analytically (see :func:`src.model.curve.next_delta`), so only the recall
    threshold and the hard interval cap are needed. Inverting an exponential is
    linear in the time constant, so the threshold scales every word's interval by
    the same factor: under a ceiling of 1, a threshold of ``0.8`` reviews every
    word after ``ln(1/0.8) = 0.22 × τ``. A lower ceiling shortens that, again by
    one factor shared across the deck.

    Attributes:
        recall_threshold: P(recall) level below which a word is due for review.
            Lower thresholds → longer intervals between repetitions.
        max_delta_seconds: Hard cap on the predicted interval (default 2 years).
    """

    recall_threshold: float = 0.8
    max_delta_seconds: float = 63_072_000.0  # 2-year cap


@dataclass
class HeuristicConfig:
    """Calibration constants for the model-free :class:`HeuristicPredictor`.

    These describe an SM-2-style schedule expressed in the same forgetting-curve
    parameterisation the LSTM emits, so the heuristic's output is stored and
    consumed exactly like a trained model's.

    ``reference_threshold`` is deliberately a constant here rather than the
    user's :class:`PredictConfig` setting: the stored time constant must stay
    threshold-independent (the user's threshold is applied live, downstream).
    It is the recall level at which the intervals below are the ones actually
    produced; raising the user's threshold shortens them from there.

    Attributes:
        reference_threshold: Recall level the intervals below are calibrated at.
        first_interval: Interval after the first successful rep, and after the
            first success following a lapse (SM-2 restarts the ladder rather
            than resuming it). Doubles as the floor under every success, so a
            card relearned mid-session graduates out of the session instead of
            ramping up from the seconds-scale gap a re-queue leaves behind.
        relearn_interval: Interval assigned right after a failed rep, so the
            practice loop brings the card back later in the same session — the
            10-minute relearning step SM-2 descendants use.
        max_interval: Hard cap on the derived interval.
        ease_start: Ease factor for a word with no history yet.
        ease_bonus: Ease gained per successful rep.
        ease_penalty: Ease lost per failed rep.
        ease_min / ease_max: Bounds on the ease factor.
    """

    reference_threshold: float = 0.8
    first_interval: float = 86_400.0     # 1 day
    relearn_interval: float = 600.0      # 10 minutes
    max_interval: float = 63_072_000.0   # 2 years
    ease_start: float = 2.5
    ease_bonus: float = 0.1
    ease_penalty: float = 0.2
    ease_min: float = 1.3
    ease_max: float = 3.0


@dataclass
class ScheduleConfig:
    """Batched param-computation knobs for :class:`ParamScheduler`.

    Attributes:
        chunk_size: Number of words per commit-and-report chunk. Bounds how much
            work a cancellation can discard and how often progress is reported;
            it does *not* set the tensor shape.
        batch_sequences: Number of sequences per batched LSTM forward. Each word
            contributes up to six (two directions × the current curve plus the two
            a review would produce), so this is the knob that actually sets memory
            footprint. Keep it at a few hundred: throughput is flat from 64 to 512
            but degrades sharply above that — measured on MPS, 14,190 sequences
            take ~14 s at 256, ~107 s at 1024 and ~375 s at 2048.
    """

    chunk_size: int = 256
    batch_sequences: int = 256
