"""Background worker and target enumeration for the settings screen."""
from __future__ import annotations

import queue as queue_module
import threading
from dataclasses import dataclass
from pathlib import Path

from src.database import count_words, init_db, read_language_pair
from src.model import compute_all_schedules
from src.settings import AppSettings

from ..theme import Paths


@dataclass(frozen=True, slots=True)
class RecalcTarget:
    """One database that has a trained model and therefore needs recalculation."""

    db_path: Path
    ckpt_path: Path
    src_lang: str
    tgt_lang: str
    word_count: int

    @property
    def pair_label(self) -> str:
        """``"French↔Polish"`` style label for status messages."""
        return f"{self.src_lang}↔{self.tgt_lang}"


def iter_recalc_targets() -> list[RecalcTarget]:
    """Return one :class:`RecalcTarget` per DB with a matching checkpoint.

    Databases without a checkpoint are skipped — the next training run on
    those will pick up the new settings automatically.
    """
    storage = Paths.STORAGE_DIR
    if not storage.exists():
        return []

    targets: list[RecalcTarget] = []
    for db_path in sorted(storage.glob("*.db")):
        pair = read_language_pair(db_path)
        if pair is None:
            continue
        src, tgt = pair
        ckpt = Paths.model_path(src, tgt)
        if not ckpt.exists():
            continue
        targets.append(
            RecalcTarget(
                db_path=db_path,
                ckpt_path=ckpt,
                src_lang=src,
                tgt_lang=tgt,
                word_count=count_words(db_path),
            )
        )
    return targets


def recalc_worker(
    new_settings: AppSettings,
    targets: list[RecalcTarget],
    out_queue: queue_module.Queue,
    stop_event: threading.Event,
) -> None:
    """Iterate over targets and run :func:`compute_all_schedules` for each."""
    try:
        cfg = new_settings.to_predict_config()
        total = len(targets)

        for index, target in enumerate(targets, start=1):
            if stop_event.is_set():
                break
            init_db(f"sqlite:///{target.db_path}", target.src_lang, target.tgt_lang)

            def on_progress(
                done: int, total_in_db: int,
                _label=target.pair_label, _i=index, _n=total,
            ) -> None:
                out_queue.put(("recalc_progress", _label, done, total_in_db, _i, _n))

            compute_all_schedules(
                model_path=target.ckpt_path,
                on_progress=on_progress,
                stop_event=stop_event,
                cfg=cfg,
            )

        if stop_event.is_set():
            out_queue.put(("recalc_cancelled",))
        else:
            out_queue.put(("recalc_done",))
    except Exception as exc:
        out_queue.put(("recalc_error", str(exc)))
