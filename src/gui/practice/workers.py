"""Background workers for the practice screen.

These are plain free functions: no Tk imports, no screen references.
They communicate with the main thread through a :class:`queue.Queue` that
the :class:`~src.gui.background.BackgroundJob` polls.
"""
from __future__ import annotations

import queue as queue_module
import time

from src.database import (
    Direction,
    Repetition,
    RepetitionRepository,
    WordRepository,
    get_session,
)
from src.model import load_model
from src.model.predictor import Predictor
from src.settings import load_settings

from ..db_context import DbContext
from .queue_model import Card, build_queue


def init_worker(ctx: DbContext, out_queue: queue_module.Queue) -> None:
    """Load the predictor (if a checkpoint exists) and build the practice queue."""
    try:
        predictor: Predictor | None = None
        if ctx.model_path.exists():
            model = load_model(str(ctx.model_path))
            predictor = Predictor(model, load_settings().to_predict_config())

        cards = build_queue(now=int(time.time()))
        out_queue.put(("ready", predictor, cards))
    except Exception as exc:
        out_queue.put(("error", str(exc)))


def answer_worker(
    card: Card,
    remembered: bool,
    predictor: Predictor | None,
    out_queue: queue_module.Queue,
) -> None:
    """Record one repetition and recompute the per-direction next-due timestamp."""
    try:
        practiced_at = int(time.time())
        next_ts: int | None = None

        with get_session() as session:
            reps_repo = RepetitionRepository(session)
            reps_repo.add(
                Repetition(
                    word_id=card.word_id,
                    direction=int(card.direction),
                    practiced_at=practiced_at,
                    remembered=remembered,
                )
            )

            if predictor is not None:
                session.flush()
                all_reps = reps_repo.get_for_word(card.word_id, card.direction)
                try:
                    delta = predictor.next_repetition_delta(all_reps)
                    next_ts = practiced_at + int(delta)
                except Exception:
                    next_ts = 0
                word = WordRepository(session).get_by_id(card.word_id)
                if word is not None:
                    if card.direction is Direction.FORWARD:
                        word.next_rep_fwd_at = next_ts
                    else:
                        word.next_rep_rev_at = next_ts

        out_queue.put(("answered", card, practiced_at, next_ts))
    except Exception as exc:
        out_queue.put(("answer_error", str(exc)))
