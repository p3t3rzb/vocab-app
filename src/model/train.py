"""Train the :class:`RecallLSTM` on a database's repetition history.

Usable both as a CLI and programmatically. Example CLI usage::

    uv run python -m src.model.train --db storage/french.db
    uv run python -m src.model.train --db storage/french.db --epochs 5 --hidden-size 32

Each run saves a checkpoint at ``storage/models/<source>_<target>.pt``
containing ``{state_dict, hyperparams, val_loss, epoch}``; subsequent runs
overwrite that file.
"""
from __future__ import annotations

import argparse
import math
import random
import threading
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.database import LanguagePairRepository, get_session
from src.model.config import TrainConfig
from src.model.dataset import Sequence, build_sequences, split_sequences
from src.model.lstm import RecallLSTM


def _get_device() -> torch.device:
    """Return the best available torch device — MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _preload(sequences: list[Sequence], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad every sequence to the global max length and move the whole set to ``device``.

    Doing this once up front avoids per-batch host→device copies during
    training. The padded dataset is small (tens of MB) so memory is not a concern.
    """
    lengths = [len(s.inputs) for s in sequences]
    max_L = max(lengths)
    N = len(sequences)
    inputs = torch.zeros(N, max_L, 2)
    targets = torch.zeros(N, max_L)
    for i, s in enumerate(sequences):
        L = lengths[i]
        inputs[i, :L] = s.inputs
        targets[i, :L] = s.targets
    lengths_t = torch.tensor(lengths, dtype=torch.long)
    return inputs.to(device), targets.to(device), lengths_t.to(device)


def _masked_bce(pred: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy that ignores padded positions.

    The mask is built from ``lengths`` so padded steps contribute nothing to
    either the numerator or the denominator of the mean.
    """
    L = pred.shape[1]
    mask = torch.arange(L, device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    loss = nn.functional.binary_cross_entropy(pred, target, reduction="none")
    return (loss * mask).sum() / mask.sum()


def _make_batches(lengths: torch.Tensor, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    """Bucket sort by length, then optionally shuffle batch order.

    Sorting by length keeps padding minimal within each batch; shuffling the
    *order* of the batches preserves SGD's stochasticity during training
    while still benefiting from the bucketing.
    """
    sorted_idx = torch.argsort(lengths)
    batches = [sorted_idx[i : i + batch_size] for i in range(0, len(lengths), batch_size)]
    if shuffle:
        random.shuffle(batches)
    return batches


def _run_epoch(
    model: RecallLSTM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one full pass over the dataset and return the mean per-batch loss.

    If ``optimizer`` is ``None`` the pass runs under ``torch.no_grad`` (used
    for validation). Otherwise gradients are computed, clipped, and applied.
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    n_batches = 0

    with torch.set_grad_enabled(training):
        for bi in _make_batches(lengths, batch_size, shuffle=training):
            bx = inputs[bi]
            bt = targets[bi]
            bl = lengths[bi]

            max_l = int(bl.max().item())
            bx = bx[:, :max_l, :]
            bt = bt[:, :max_l]

            pred = model(bx)
            loss = _masked_bce(pred, bt, bl)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def train(
    db_url: str,
    config: TrainConfig | None = None,
    on_epoch: Callable[[int, float, float], None] | None = None,
    stop_event: threading.Event | None = None,
) -> Path:
    """Train a :class:`RecallLSTM` and return the path of the best checkpoint.

    The best checkpoint (lowest validation loss seen so far) is saved every
    time validation loss improves, so cancelling mid-training still leaves a
    usable model on disk.

    Args:
        db_url: SQLAlchemy URL for the source database (e.g.
            ``sqlite:///storage/french.db``).
        config: Hyperparameter overrides. ``None`` uses :class:`TrainConfig`'s
            defaults.
        on_epoch: Optional callback invoked after each epoch with
            ``(epoch_num, train_loss, val_loss)``. Used by the GUI to update
            the live training plot.
        stop_event: Optional :class:`threading.Event` for cancellation.
            Checked between epochs.

    Returns:
        Path to the saved checkpoint (``storage/models/<src>_<tgt>.pt``).
    """
    if config is None:
        config = TrainConfig()

    torch.manual_seed(config.seed)
    random.seed(config.seed)

    device = _get_device()
    print(f"Device: {device}")

    print("Loading sequences from database…")
    all_seqs = build_sequences(db_url)
    print(f"  {len(all_seqs)} (word, direction) sequences found")

    with get_session() as session:
        lp = LanguagePairRepository(session).get()
        pair_name = f"{lp.source_language}_{lp.target_language}".lower() if lp else "model"

    train_seqs, val_seqs = split_sequences(all_seqs, val_split=config.val_split, seed=config.seed)
    print(f"  Train: {len(train_seqs)}  |  Val: {len(val_seqs)}")

    print("Pre-loading data to device…")
    tr_inputs, tr_targets, tr_lengths = _preload(train_seqs, device)
    vl_inputs, vl_targets, vl_lengths = _preload(val_seqs, device)

    model = RecallLSTM(
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    storage_dir = Path("storage") / "models"
    storage_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = storage_dir / f"{pair_name}.pt"

    best_val = math.inf
    best_epoch = 0

    print(f"\n{'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'LR':>10}")
    print("-" * 42)

    for epoch in range(config.epochs):
        if stop_event is not None and stop_event.is_set():
            break

        tr_loss = _run_epoch(model, tr_inputs, tr_targets, tr_lengths, config.batch_size, optimizer)
        vl_loss = _run_epoch(model, vl_inputs, vl_targets, vl_lengths, config.batch_size, None)
        scheduler.step(vl_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        marker = " *" if vl_loss < best_val else ""
        print(f"{epoch + 1:>6}  {tr_loss:>10.5f}  {vl_loss:>10.5f}  {lr_now:>10.2e}{marker}")

        if on_epoch is not None:
            on_epoch(epoch + 1, tr_loss, vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hyperparams": model.hyperparams(),
                    "val_loss": best_val,
                    "epoch": best_epoch,
                },
                ckpt_path,
            )

    stopped_early = stop_event is not None and stop_event.is_set()
    if stopped_early:
        print(f"\nTraining stopped early. Best val loss {best_val:.5f} at epoch {best_epoch + 1}")
    else:
        print(f"\nBest val loss {best_val:.5f} at epoch {best_epoch + 1}")
    print(f"Checkpoint saved → {ckpt_path}")

    return ckpt_path


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> RecallLSTM:
    """Load a saved :class:`RecallLSTM` checkpoint.

    The model is moved to ``device`` (auto-detected if not supplied) and put
    into ``eval`` mode, ready for inference.
    """
    if device is None:
        device = _get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RecallLSTM(**ckpt["hyperparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags, falling back to :class:`TrainConfig` defaults."""
    _d = TrainConfig()
    p = argparse.ArgumentParser(description="Train RecallLSTM on repetition history")
    p.add_argument("--db", required=True, help="Path to the SQLite database, e.g. storage/french.db")
    p.add_argument("--epochs", type=int, default=_d.epochs)
    p.add_argument("--hidden-size", type=int, default=_d.hidden_size)
    p.add_argument("--num-layers", type=int, default=_d.num_layers)
    p.add_argument("--dropout", type=float, default=_d.dropout)
    p.add_argument("--lr", type=float, default=_d.lr)
    p.add_argument("--batch-size", type=int, default=_d.batch_size)
    p.add_argument("--val-split", type=float, default=_d.val_split)
    p.add_argument("--seed", type=int, default=_d.seed)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    db_url = f"sqlite:///{args.db}" if not args.db.startswith("sqlite") else args.db
    cfg = TrainConfig(
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
    )
    train(db_url=db_url, config=cfg)
