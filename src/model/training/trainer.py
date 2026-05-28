"""Train a :class:`RecallLSTM` on a database's repetition history."""
from __future__ import annotations

import math
import random
import threading
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from src.database import LanguagePairRepository, get_session
from src.model.checkpoint import save_checkpoint
from src.model.config import TrainConfig
from src.model.curve import curve_recall
from src.model.dataset import Sequence, build_sequences, split_sequences
from src.model.device import get_device
from src.model.lstm import RecallLSTM
from src.model.training.batching import bucket_batches
from src.model.training.loss import masked_bce

EpochCallback = Callable[[int, float, float], None]


def _predict(model: RecallLSTM, bx: torch.Tensor) -> torch.Tensor:
    """Map a batch of dataset inputs to per-step recall probabilities.

    The model consumes the repetition *history only*, so the queried gap is
    shifted out of its input: ``bx[..., 0] = log(Δt + 1)`` is rolled one step
    right (first step → 0) to form the LSTM input, while the real gap (recovered
    as ``expm1(bx[..., 0])``) is the argument the predicted curve is evaluated
    at. ``bx[..., 1]`` (``prev_remembered``) is already history-aligned.
    """
    deltas = torch.expm1(bx[..., 0])  # (B, L) raw seconds since previous rep

    x_hist = torch.empty_like(bx)
    x_hist[:, 0, 0] = 0.0
    x_hist[:, 1:, 0] = bx[:, :-1, 0]
    x_hist[..., 1] = bx[..., 1]

    return curve_recall(deltas, model(x_hist))


class Trainer:
    """Encapsulates one training run: data prep, model, optimizer, epoch loop."""

    def __init__(self, config: TrainConfig) -> None:
        self._cfg = config
        self._device = get_device()

    def run(
        self,
        on_epoch: EpochCallback | None = None,
        stop_event: threading.Event | None = None,
    ) -> Path:
        """Train and return the path of the best checkpoint.

        Callers must call :func:`src.database.init_db` before invoking this
        method — the trainer reads sequences via :func:`get_session` and
        does not initialise the engine itself.

        The best checkpoint (lowest validation loss seen so far) is saved
        every time validation loss improves, so cancelling mid-training still
        leaves a usable model on disk.
        """
        cfg = self._cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)

        print(f"Device: {self._device}")

        print("Loading sequences from database…")
        all_seqs = build_sequences()
        print(f"  {len(all_seqs)} (word, direction) sequences found")

        pair_name = self._pair_name()
        train_seqs, val_seqs = split_sequences(all_seqs, cfg.val_split, cfg.seed)
        print(f"  Train: {len(train_seqs)}  |  Val: {len(val_seqs)}")

        print("Pre-loading data to device…")
        tr_inputs, tr_targets, tr_lengths = self._preload(train_seqs)
        vl_inputs, vl_targets, vl_lengths = self._preload(val_seqs)

        model = RecallLSTM(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(self._device)
        ckpt_path = cfg.checkpoint_dir / f"{pair_name}.pt"
        best_val = math.inf
        best_epoch = 0
        if cfg.warm_start:
            best_val = self._maybe_warm_start(model, ckpt_path)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=cfg.lr_patience, factor=cfg.lr_factor,
        )

        print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}")
        print("-" * 42)

        for epoch in range(cfg.epochs):
            if stop_event is not None and stop_event.is_set():
                break

            tr_loss = self._run_epoch(model, tr_inputs, tr_targets, tr_lengths, optimizer)
            vl_loss = self._run_epoch(model, vl_inputs, vl_targets, vl_lengths, None)
            scheduler.step(vl_loss)

            lr_now = optimizer.param_groups[0]["lr"]
            marker = " *" if vl_loss < best_val else ""
            print(f"{epoch + 1:>6}  {tr_loss:>10.5f}  {vl_loss:>10.5f}  {lr_now:>10.2e}{marker}")

            if on_epoch is not None:
                on_epoch(epoch + 1, tr_loss, vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                best_epoch = epoch
                save_checkpoint(model, ckpt_path, best_val, best_epoch)

        stopped_early = stop_event is not None and stop_event.is_set()
        if stopped_early:
            print(f"\nTraining stopped early. Best val loss {best_val:.5f} at epoch {best_epoch + 1}")
        else:
            print(f"\nBest val loss {best_val:.5f} at epoch {best_epoch + 1}")
        print(f"Checkpoint saved → {ckpt_path}")

        return ckpt_path

    def _maybe_warm_start(self, model: RecallLSTM, ckpt_path: Path) -> float:
        """Resume training from an existing checkpoint if one is compatible.

        Loads ``ckpt_path`` (if present), and — only when its saved
        ``hyperparams`` match this run's architecture — copies its weights into
        ``model`` in place. Returns the checkpoint's recorded ``val_loss`` so
        the epoch loop only overwrites the file on a genuine improvement;
        returns ``math.inf`` when no compatible checkpoint was loaded (so the
        run behaves like training from scratch).
        """
        if not ckpt_path.exists():
            print("Warm start: no existing checkpoint — training from scratch.")
            return math.inf

        ckpt = torch.load(ckpt_path, map_location=self._device, weights_only=True)
        if ckpt.get("hyperparams") != model.hyperparams():
            print(
                "Warm start: checkpoint architecture "
                f"{ckpt.get('hyperparams')} != config {model.hyperparams()} "
                "— training from scratch."
            )
            return math.inf

        model.load_state_dict(ckpt["state_dict"])
        prev_val = float(ckpt.get("val_loss", math.inf))
        print(
            f"Warm start: resumed from {ckpt_path} "
            f"(val loss {prev_val:.5f} @ epoch {ckpt.get('epoch', 0) + 1})."
        )
        return prev_val

    def _pair_name(self) -> str:
        """Resolve the ``<src>_<tgt>`` slug used in the checkpoint filename."""
        with get_session() as session:
            lp = LanguagePairRepository(session).get()
            if lp is None:
                return "model"
            return f"{lp.source_language}_{lp.target_language}".lower()

    def _preload(
        self, sequences: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad every sequence to the global max length and move to the active device.

        Doing this once up front avoids per-batch host→device copies during
        training. The padded dataset is small (tens of MB) so memory is not a concern.
        """
        lengths = [len(s.inputs) for s in sequences]
        max_len = max(lengths)
        inputs = torch.zeros(len(sequences), max_len, 2)
        targets = torch.zeros(len(sequences), max_len)
        for i, s in enumerate(sequences):
            seq_len = lengths[i]
            inputs[i, :seq_len] = s.inputs
            targets[i, :seq_len] = s.targets
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        return inputs.to(self._device), targets.to(self._device), lengths_t.to(self._device)

    def _run_epoch(
        self,
        model: RecallLSTM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        """Run one full pass over the dataset; ``None`` optimizer ⇒ validation."""
        training = optimizer is not None
        model.train(training)

        total_loss = 0.0
        n_batches = 0

        with torch.set_grad_enabled(training):
            for bi in bucket_batches(lengths, self._cfg.batch_size, shuffle=training):
                bx = inputs[bi]
                bt = targets[bi]
                bl = lengths[bi]

                max_l = int(bl.max().item())
                bx = bx[:, :max_l, :]
                bt = bt[:, :max_l]

                pred = _predict(model, bx)
                loss = masked_bce(pred, bt, bl)

                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self._cfg.grad_clip_max_norm,
                    )
                    optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)


def train(
    config: TrainConfig | None = None,
    on_epoch: EpochCallback | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Module-level convenience wrapper around :class:`Trainer`.

    The active database (set via :func:`src.database.init_db`) is read by
    the trainer; this function does not take a ``db_url`` itself.
    """
    return Trainer(config or TrainConfig()).run(on_epoch=on_epoch, stop_event=stop_event)
