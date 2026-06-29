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
from src.model import Predictor, load_model
from src.model.curve import invert_curve
from src.settings import load_settings

from ..db_context import DbContext
from .queue_model import Card, build_queue


def init_worker(ctx: DbContext, out_queue: queue_module.Queue) -> None:
    """Load the predictor (if a checkpoint exists) and build the practice queue."""
    try:
        cfg = load_settings().to_predict_config()
        predictor: Predictor | None = None
        if ctx.model_path.exists():
            model = load_model(str(ctx.model_path))
            predictor = Predictor(model, cfg)

        queue, waiting = build_queue(now=int(time.time()), cfg=cfg)
        out_queue.put(("ready", predictor, queue, waiting))
    except Exception as exc:
        out_queue.put(("error", str(exc)))


def answer_worker(
    card: Card,
    remembered: bool,
    predictor: Predictor | None,
    out_queue: queue_module.Queue,
) -> None:
    """Record one repetition, store its recomputed curve params, derive due/recall.

    Returns ``("answered", card, practiced_at, next_ts, recall_now)`` where
    ``next_ts`` is the live next-review timestamp (``None`` if no model) and
    ``recall_now`` is the recall ceiling ``p0`` right after this rep — the
    priority used if the card has to be re-queued.
    """
    try:
        practiced_at = int(time.time())
        next_ts: int | None = None
        recall_now: float | None = None

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
                cfg = predictor.config
                try:
                    p0, s, d = predictor.curve_params(all_reps, card.direction)
                    delta = invert_curve(
                        p0, s, d, cfg.recall_threshold, cfg.max_delta_seconds
                    )
                    next_ts = practiced_at + int(delta)
                    recall_now = p0
                except Exception:
                    p0 = s = d = None
                    next_ts = 0
                    recall_now = 0.0

                word = WordRepository(session).get_by_id(card.word_id)
                if word is not None:
                    if card.direction is Direction.FORWARD:
                        word.fwd_p0, word.fwd_s, word.fwd_d = p0, s, d
                    else:
                        word.rev_p0, word.rev_s, word.rev_d = p0, s, d

        out_queue.put(("answered", card, practiced_at, next_ts, recall_now))
    except Exception as exc:
        out_queue.put(("answer_error", str(exc)))
