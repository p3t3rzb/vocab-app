"""Save and load :class:`RecallLSTM` checkpoints.

The on-disk format is a torch-saved dict ``{state_dict, hyperparams,
val_loss, epoch}``. ``hyperparams`` is the kwargs needed to rebuild the
network, so :func:`load_model` reconstructs the right architecture before
loading the state dict — older checkpoints with different
``hidden_size``/``num_layers`` still load cleanly.
"""
from __future__ import annotations

from pathlib import Path

import torch

from src.model.device import get_device
from src.model.lstm import RecallLSTM


def save_checkpoint(model: RecallLSTM, path: Path, val_loss: float, epoch: int) -> None:
    """Persist ``model`` to ``path`` along with its hyperparams and val metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "hyperparams": model.hyperparams(),
            "val_loss": val_loss,
            "epoch": epoch,
        },
        path,
    )


def load_model(checkpoint_path: str | Path, device: torch.device | None = None) -> RecallLSTM:
    """Load a saved :class:`RecallLSTM` checkpoint.

    The model is moved to ``device`` (auto-detected if not supplied) and put
    into ``eval`` mode, ready for inference.
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RecallLSTM(**ckpt["hyperparams"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model
