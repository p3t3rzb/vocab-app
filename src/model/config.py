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
    threshold and the hard interval cap are needed.

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
    user's :class:`PredictConfig` setting: the stored ``(p0, S, d)`` must stay
    threshold-independent (the user's threshold is applied live, downstream).
    It is the recall level at which the intervals below are the ones actually
    produced; raising the user's threshold shortens them from there.

    Attributes:
        reference_threshold: Recall level the intervals below are calibrated at.
        decay: The curve's decay exponent ``d``, held fixed — with ``S`` solved
            per word, ``d`` only sets the curve's shape, not its due time.
        first_interval: Interval after the first successful rep, and after the
            first success following a lapse (SM-2 restarts the ladder rather
            than resuming it). Doubles as the floor under every success, so a
            card relearned mid-session graduates out of the session instead of
            ramping up from the seconds-scale gap a re-queue leaves behind.
        relearn_interval: Time-scale used right after a failed rep.
        max_interval: Hard cap on the derived interval.
        ease_start: Ease factor for a word with no history yet.
        ease_bonus: Ease gained per successful rep.
        ease_penalty: Ease lost per failed rep.
        ease_min / ease_max: Bounds on the ease factor.
        lapse_p0: Recall ceiling assigned right after a failure. Below
            ``reference_threshold`` on purpose, so a just-failed card inverts to
            "due now" and the practice loop re-queues it in the same session.
        p0_base: Recall ceiling after one successful rep.
        p0_streak_bonus: Ceiling gained per additional consecutive success.
        p0_max: Upper bound on the recall ceiling.
    """

    reference_threshold: float = 0.8
    decay: float = 0.5
    first_interval: float = 86_400.0     # 1 day
    relearn_interval: float = 600.0      # 10 minutes
    max_interval: float = 63_072_000.0   # 2 years
    ease_start: float = 2.5
    ease_bonus: float = 0.1
    ease_penalty: float = 0.2
    ease_min: float = 1.3
    ease_max: float = 3.0
    lapse_p0: float = 0.7
    p0_base: float = 0.9
    p0_streak_bonus: float = 0.01
    p0_max: float = 0.98


@dataclass
class ScheduleConfig:
    """Batched param-computation knobs for :class:`ParamScheduler`.

    Attributes:
        chunk_size: Number of words processed per batched LSTM forward.
            Trades GPU dispatch overhead against memory footprint.
    """

    chunk_size: int = 256
