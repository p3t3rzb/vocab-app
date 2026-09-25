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

from .curve import CEILING_LOGIT_MAX
from .device import get_device
from .lstm import RecallLSTM


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

    **Ceiling-free checkpoints are widened, not rejected.** Before the ceiling
    the head emitted a *single* channel, ``ln τ``, and the curve started at 1.
    That is exactly the ``p0 = 1`` corner of the family the head now emits, so
    such a checkpoint is widened rather than retrained: the new weight row is
    **zeroed** and the new bias row set to the logit of a ceiling of 1, which
    makes the ceiling channel emit that constant for every input — precisely the
    behaviour it is replacing. Predictions are unchanged to within the clamp's
    ``2e-8``.

    The keys are supplied explicitly rather than by loading non-strictly, so
    every *other* mismatch still raises.
    """
    if device is None:
        device = get_device()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = RecallLSTM(**ckpt["hyperparams"]).to(device)
    state = dict(ckpt["state_dict"])
    if state["head.weight"].shape[0] == 1:
        state["head.weight"] = torch.cat(
            [state["head.weight"], torch.zeros_like(state["head.weight"])]
        )
        state["head.bias"] = torch.cat(
            [
                state["head.bias"],
                torch.tensor(
                    [CEILING_LOGIT_MAX], device=state["head.bias"].device
                ),
            ]
        )
    model.load_state_dict(state)
    model.eval()
    return model
