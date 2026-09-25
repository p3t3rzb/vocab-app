"""Background workers for the training screen.

``training_worker`` runs :func:`src.model.train`, pushing per-epoch
events and a final ``"done"`` (or ``"cancelled"`` / ``"error"``).
``schedule_worker`` runs :func:`src.model.compute_all_params`
on a single database, pushing chunk-progress events.
``delete_model_worker`` removes the checkpoint and re-derives every
word's params with the model-free heuristic.
"""
from __future__ import annotations

import queue as queue_module
import threading
from pathlib import Path

from src.database import init_db
from src.model import compute_all_params
from src.model import train as run_training
from src.model.config import TrainConfig

from ..db_context import DbContext
from ..model_sync import force_heuristic_params


def training_worker(
    db_url: str,
    src_lang: str,
    tgt_lang: str,
    cfg: TrainConfig,
    out_queue: queue_module.Queue,
    stop_event: threading.Event,
) -> None:
    """Run :func:`train` and push outcome events onto ``out_queue``."""
    try:
        init_db(db_url, src_lang, tgt_lang)
        result_path = run_training(
            config=cfg,
            on_epoch=lambda e, tr, vl: out_queue.put(("epoch", e, tr, vl)),
            stop_event=stop_event,
        )
        if stop_event.is_set():
            out_queue.put(("cancelled",))
        else:
            out_queue.put(("done", str(result_path)))
    except Exception as exc:
        out_queue.put(("error", str(exc)))


def schedule_worker(
    model_path: Path,
    out_queue: queue_module.Queue,
    stop_event: threading.Event,
) -> None:
    """Run :func:`compute_all_params`, pushing progress and outcome events."""
    try:
        compute_all_params(
            model_path=model_path,
            on_progress=lambda done, total: out_queue.put(
                ("schedule_progress", done, total)
            ),
            stop_event=stop_event,
        )
        if stop_event.is_set():
            out_queue.put(("schedules_cancelled",))
        else:
            out_queue.put(("schedules_done",))
    except Exception as exc:
        out_queue.put(("schedules_error", str(exc)))


def delete_model_worker(
    ctx: DbContext,
    out_queue: queue_module.Queue,
) -> None:
    """Delete the checkpoint, then re-derive every word's params heuristically.

    Removing the file alone would leave the stored time constants as the deleted
    model computed them, so the app would keep scheduling from a model it no
    longer has. :func:`force_heuristic_params` rewrites the params of every
    word with history from :class:`HeuristicPredictor`, which is exactly what
    practice falls back to once the checkpoint is gone.
    """
    try:
        ctx.model_path.unlink(missing_ok=True)
        count = force_heuristic_params(ctx)
        out_queue.put(("delete_done", count))
    except Exception as exc:
        out_queue.put(("delete_error", str(exc)))
