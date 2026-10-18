"""Train a :class:`RecallLSTM` on a database's repetition history."""
from __future__ import annotations

import math
import random
import threading
from datetime import datetime
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn

from src.database import LanguagePairRepository, get_session

from ..checkpoint import load_model, save_checkpoint
from ..config import TrainConfig
from ..curve import ceiling, curve_recall
from ..dataset import Sequence, build_sequences, split_by_time
from ..lstm import RecallLSTM
from .batching import bucket_batches
from .loss import masked_bce

EpochCallback = Callable[[int, float, float], None]
#: ``(epoch, total_epochs, train_loss)`` for each epoch of the final refit.
RefitCallback = Callable[[int, int, float], None]


def _predict(model: RecallLSTM, bx: torch.Tensor) -> torch.Tensor:
    """Map a batch of dataset inputs to per-step recall probabilities.

    The model consumes the repetition *history only*, so the queried gap is
    shifted out of its input: ``bx[..., 0] = log(Δt + 1)`` is rolled one step
    right (first step → 0) to form the LSTM input, while the real gap (recovered
    as ``expm1(bx[..., 0])``) is the argument the predicted curve is evaluated
    at. ``bx[..., 1:]`` (the ``[prev_remembered, prev_not_remembered]`` one-hot
    plus the constant ``[is_forward, is_reverse]`` direction one-hot) is already
    history-aligned and copied through unshifted.

    Both curve parameters come out of the same forward, so nothing extra is
    passed here: one backward pass fits the time constant and the ceiling
    together, each through the curve.
    """
    deltas = torch.expm1(bx[..., 0])  # (B, L) raw seconds since previous rep

    x_hist = torch.empty_like(bx)
    x_hist[:, 0, 0] = 0.0
    x_hist[:, 1:, 0] = bx[:, :-1, 0]
    x_hist[..., 1:] = bx[..., 1:]

    return curve_recall(deltas, model(x_hist))


def _step_mask(lengths: torch.Tensor, starts: torch.Tensor, max_l: int) -> torch.Tensor:
    """``(B, max_l)`` mask of the scored steps: from each ``start`` up to its length."""
    steps = torch.arange(max_l, device=lengths.device).unsqueeze(0)
    return (steps < lengths.unsqueeze(1)) & (steps >= starts.unsqueeze(1))


class Trainer:
    """Encapsulates one training run: data prep, model, optimizer, epoch loop."""

    def __init__(self, config: TrainConfig) -> None:
        self._cfg = config
        # CPU, not get_device(): the MPS LSTM backward intermittently returns
        # garbage gradients (NaN / ~1e8) on the full deck, which poisons the
        # weights. Inference is forward-only and stays on MPS.
        self._device = torch.device("cpu")

    def run(
        self,
        on_epoch: EpochCallback | None = None,
        stop_event: threading.Event | None = None,
        on_refit_epoch: RefitCallback | None = None,
    ) -> Path:
        """Train and return the path of the checkpoint.

        Callers must call :func:`src.database.init_db` before invoking this
        method — the trainer reads sequences via :func:`get_session` and
        does not initialise the engine itself.

        Every run starts from a fresh random init — an existing checkpoint for
        this language pair is never resumed from, only overwritten.

        Two phases. The **validation run** trains on the reps before the time
        cutoff and scores each epoch on the ones after it, to find the best epoch
        and the learning rates that led there. The **refit** (:meth:`_refit`)
        then trains a fresh model on *every* rep for that many epochs, replaying
        those learning rates, and overwrites the checkpoint — so the shipped
        model has seen the most recent reps and newest words, which the
        validation run had to hold out.

        The best validation-run checkpoint is saved every time validation loss
        improves, so cancelling at any point — the refit included — still leaves
        a usable model on disk.
        """
        cfg = self._cfg
        print(f"Device: {self._device}")

        print("Loading sequences from database…")
        all_seqs = build_sequences()
        print(f"  {len(all_seqs)} (word, direction) sequences found")

        pair_name = self._pair_name()
        train_seqs, val_seqs = split_by_time(all_seqs, cfg.val_split)
        n_val = sum(len(s.targets) - s.scored_from for s in val_seqs)
        cutoff = min(int(s.practiced_at[s.scored_from]) for s in val_seqs)
        latest = max(int(s.practiced_at[-1]) for s in val_seqs)
        print(
            f"  Train: {len(train_seqs)}  |  Val: {len(val_seqs)} sequences, "
            f"{n_val} reps since {datetime.fromtimestamp(cutoff):%Y-%m-%d} "
            f"(last {(latest - cutoff) / 86_400:.1f} days)"
        )

        print("Pre-loading data to device…")
        tr_inputs, tr_targets, tr_lengths, tr_starts = self._preload(train_seqs)
        vl_inputs, vl_targets, vl_lengths, vl_starts = self._preload(val_seqs)

        model, optimizer = self._fresh_model()
        ckpt_path = cfg.checkpoint_dir / f"{pair_name}.pt"
        best_val = math.inf
        best_epoch = 0
        lrs: list[float] = []  # learning rate each epoch trained at, for the refit

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=cfg.lr_patience, factor=cfg.lr_factor,
        )

        print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}")
        print("-" * 42)

        for epoch in range(cfg.epochs):
            if stop_event is not None and stop_event.is_set():
                break

            lrs.append(optimizer.param_groups[0]["lr"])
            tr_loss = self._run_epoch(
                model, tr_inputs, tr_targets, tr_lengths, tr_starts, optimizer,
            )
            vl_loss = self._run_epoch(
                model, vl_inputs, vl_targets, vl_lengths, vl_starts, None,
            )
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
        self._report_ceilings(ckpt_path, vl_inputs, vl_lengths, vl_starts)

        if not stopped_early and lrs:
            # Free the split's device copies before loading the whole dataset.
            del tr_inputs, tr_targets, tr_lengths, tr_starts
            del vl_inputs, vl_targets, vl_lengths, vl_starts
            self._refit(
                all_seqs, lrs[: best_epoch + 1], ckpt_path, best_val, best_epoch,
                on_refit_epoch, stop_event,
            )
        print(f"Checkpoint saved → {ckpt_path}")

        return ckpt_path

    def _fresh_model(self) -> tuple[RecallLSTM, torch.optim.Optimizer]:
        """A newly initialised model and its optimizer, seeded from the config.

        Re-seeding here makes the refit start from the same weights and batch
        order the validation run started from, so the two differ only in data.
        """
        cfg = self._cfg
        torch.manual_seed(cfg.seed)
        random.seed(cfg.seed)
        model = RecallLSTM(
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            fixed_ceiling=cfg.fixed_ceiling,
        ).to(self._device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        return model, optimizer

    def _refit(
        self,
        sequences: list[Sequence],
        lrs: list[float],
        ckpt_path: Path,
        val_loss: float,
        best_epoch: int,
        on_epoch: RefitCallback | None,
        stop_event: threading.Event | None,
    ) -> None:
        """Retrain from scratch on every rep and overwrite the checkpoint.

        Runs one epoch per entry of ``lrs`` — the learning rates the validation
        run trained its best epochs at. ``ReduceLROnPlateau`` cannot run here, as
        there is no held-out loss to react to, so its decisions are replayed.

        The checkpoint keeps the validation run's ``val_loss`` and ``best_epoch``:
        that is the only honest estimate of how this recipe does on unseen reps.
        A cancelled refit leaves the validation run's checkpoint in place.
        """
        n_reps = sum(len(s.targets) for s in sequences)
        print(f"\nRefitting on all {n_reps} reps for {len(lrs)} epochs…")
        inputs, targets, lengths, starts = self._preload(sequences)
        model, optimizer = self._fresh_model()

        for epoch, lr in enumerate(lrs):
            if stop_event is not None and stop_event.is_set():
                print("Refit stopped — keeping the validation run's checkpoint.")
                return
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss = self._run_epoch(model, inputs, targets, lengths, starts, optimizer)
            print(f"{epoch + 1:>6}  {loss:>10.5f}  {'':>10}  {lr:>10.2e}")
            if on_epoch is not None:
                on_epoch(epoch + 1, len(lrs), loss)

        save_checkpoint(model, ckpt_path, val_loss, best_epoch)

    def _report_ceilings(
        self,
        ckpt_path: Path,
        inputs: torch.Tensor,
        lengths: torch.Tensor,
        starts: torch.Tensor,
    ) -> None:
        """Print the saved model's fitted ceilings over the validation reps.

        The number worth seeing is whether the head actually learned the split it
        was added for: mean ``p0`` after a remembered rep against after a lapse.
        A run where those two come out equal has not used the channel, whatever
        the loss did.

        Reloads the *saved* checkpoint rather than reading the live model, since
        the best epoch is generally not the last one. Runs in the same
        length-bucketed batches training uses — the validation set as one tensor
        is several GiB of LSTM activations.
        """
        model = load_model(ckpt_path, self._device)
        totals = {True: (0.0, 0), False: (0.0, 0)}
        lo, hi = 1.0, 0.0

        with torch.inference_mode():
            for bi in bucket_batches(lengths, self._cfg.batch_size, shuffle=False):
                bl = lengths[bi]
                max_l = int(bl.max().item())
                bx = inputs[bi][:, :max_l, :]
                mask = _step_mask(bl, starts[bi], max_l)

                # Same one-step roll as `_predict`: row i of the history carries
                # the outcome of the rep that starts the curve emitted at step i.
                x_hist = torch.empty_like(bx)
                x_hist[:, 0, 0] = 0.0
                x_hist[:, 1:, 0] = bx[:, :-1, 0]
                x_hist[..., 1:] = bx[..., 1:]

                p0 = ceiling(model(x_hist))
                if not bool(mask.any()):
                    continue
                lo = min(lo, float(p0[mask].min()))
                hi = max(hi, float(p0[mask].max()))
                for prev_ok in (True, False):
                    sel = mask & (
                        (x_hist[..., 1] > 0.5) if prev_ok else (x_hist[..., 1] <= 0.5)
                    )
                    n = int(sel.sum())
                    if n:
                        total, count = totals[prev_ok]
                        totals[prev_ok] = (total + float(p0[sel].sum()), count + n)

        def mean(prev_ok: bool) -> float:
            total, count = totals[prev_ok]
            return total / count if count else float("nan")

        print(
            f"Fitted ceiling p0 on val: after a remembered answer {mean(True):.4f}, "
            f"after a forgotten one {mean(False):.4f} (range {lo:.4f}–{hi:.4f})"
        )

    def _pair_name(self) -> str:
        """Resolve the ``<src>_<tgt>`` slug used in the checkpoint filename."""
        with get_session() as session:
            lp = LanguagePairRepository(session).get()
            if lp is None:
                return "model"
            return f"{lp.source_language}_{lp.target_language}".lower()

    def _preload(
        self, sequences: list[Sequence]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pad every sequence to the global max length and move to the active device.

        Returns ``(inputs, targets, lengths, starts)``, where ``starts`` holds each
        sequence's :attr:`~src.model.dataset.Sequence.scored_from`.

        Doing this once up front avoids per-batch host→device copies during
        training. The padded dataset is small (tens of MB) so memory is not a concern.
        """
        lengths = [len(s.inputs) for s in sequences]
        max_len = max(lengths)
        n_features = sequences[0].inputs.shape[1] if sequences else 5
        inputs = torch.zeros(len(sequences), max_len, n_features)
        targets = torch.zeros(len(sequences), max_len)
        for i, s in enumerate(sequences):
            seq_len = lengths[i]
            inputs[i, :seq_len] = s.inputs
            targets[i, :seq_len] = s.targets
        lengths_t = torch.tensor(lengths, dtype=torch.long)
        starts = torch.tensor([s.scored_from for s in sequences], dtype=torch.long)
        return (
            inputs.to(self._device),
            targets.to(self._device),
            lengths_t.to(self._device),
            starts.to(self._device),
        )

    def _run_epoch(
        self,
        model: RecallLSTM,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        lengths: torch.Tensor,
        starts: torch.Tensor,
        optimizer: torch.optim.Optimizer | None,
    ) -> float:
        """Run one full pass over the dataset; ``None`` optimizer ⇒ validation.

        Every rep weighs the same: each batch's summed loss is divided by one
        constant — the dataset's mean reps per batch — rather than by its own rep
        count, which keeps the gradient scale of the old per-batch mean. The
        returned loss is the mean per scored rep over the whole pass.
        """
        training = optimizer is not None
        model.train(training)

        batches = bucket_batches(lengths, self._cfg.batch_size, shuffle=training)
        n_reps = int((lengths - starts).sum().item())
        reps_per_batch = n_reps / max(len(batches), 1)
        total_loss = 0.0

        with torch.set_grad_enabled(training):
            for bi in batches:
                bx = inputs[bi]
                bt = targets[bi]
                bl = lengths[bi]

                max_l = int(bl.max().item())
                bx = bx[:, :max_l, :]
                bt = bt[:, :max_l]

                pred = _predict(model, bx)
                loss_sum = masked_bce(pred, bt, _step_mask(bl, starts[bi], max_l))

                if training:
                    optimizer.zero_grad()
                    (loss_sum / reps_per_batch).backward()
                    nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self._cfg.grad_clip_max_norm,
                    )
                    optimizer.step()

                total_loss += loss_sum.item()

        return total_loss / max(n_reps, 1)


def train(
    config: TrainConfig | None = None,
    on_epoch: EpochCallback | None = None,
    stop_event: threading.Event | None = None,
    on_refit_epoch: RefitCallback | None = None,
) -> Path:
    """Module-level convenience wrapper around :class:`Trainer`.

    The active database (set via :func:`src.database.init_db`) is read by
    the trainer; this function does not take a ``db_url`` itself.
    """
    return Trainer(config or TrainConfig()).run(
        on_epoch=on_epoch, stop_event=stop_event, on_refit_epoch=on_refit_epoch,
    )
