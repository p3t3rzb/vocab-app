"""Background workers for the training screen.

``training_worker`` runs :func:`src.model.train.train`, pushing per-epoch
events and a final ``"done"`` (or ``"cancelled"`` / ``"error"``).
``schedule_worker`` runs :func:`src.model.schedule.compute_all_schedules`
on a single database, pushing chunk-progress events.
"""
from __future__ import annotations

import queue as queue_module
import threading
from pathlib import Path

from src.model.config import TrainConfig
from src.model.schedule import compute_all_schedules
from src.model.train import train as run_training
from src.settings import load_settings


def training_worker(
    db_url: str,
    cfg: TrainConfig,
    out_queue: queue_module.Queue,
    stop_event: threading.Event,
) -> None:
    """Run :func:`train` and push outcome events onto ``out_queue``."""
    try:
        result_path = run_training(
            db_url=db_url,
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
    """Run :func:`compute_all_schedules`, pushing progress and outcome events."""
    try:
        compute_all_schedules(
            model_path=model_path,
            on_progress=lambda done, total: out_queue.put(
                ("schedule_progress", done, total)
            ),
            stop_event=stop_event,
            cfg=load_settings().to_predict_config(),
        )
        if stop_event.is_set():
            out_queue.put(("schedules_cancelled",))
        else:
            out_queue.put(("schedules_done",))
    except Exception as exc:
        out_queue.put(("schedules_error", str(exc)))
