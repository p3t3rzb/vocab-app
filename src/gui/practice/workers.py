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
from src.model import (
    HeuristicPredictor,
    Predictor,
    RecallEstimator,
    load_model,
)
from src.model.curve import invert_curve
from src.settings import load_settings

from ..db_context import DbContext
from ..formatting import day_start
from ..model_sync import ensure_heuristic_params
from .queue_model import Card, build_queue


def init_worker(ctx: DbContext, out_queue: queue_module.Queue) -> None:
    """Load the estimator and build the practice queue.

    Uses the pair's trained checkpoint when one exists, and the model-free
    :class:`HeuristicPredictor` otherwise — so a brand-new language pair still
    gets real due times from its very first answer, which is what eventually
    produces the history a model can be trained on.

    In the untrained case the stored half-lives are reconciled with the heuristic
    first (see :mod:`src.gui.model_sync`), so history recorded while the pair
    had no estimator — or scheduled by a checkpoint that has since been
    deleted — enters the queue correctly scored instead of sitting in the
    unscored bucket or on a date no live estimator would produce. It only
    touches words that have repetitions, and only once per app run.
    """
    try:
        cfg = load_settings().to_predict_config()
        predictor: RecallEstimator
        if ctx.model_path.exists():
            predictor = Predictor(load_model(str(ctx.model_path)), cfg)
        else:
            predictor = HeuristicPredictor(cfg)
            ensure_heuristic_params(ctx)

        queue, waiting = build_queue(now=int(time.time()), cfg=cfg)

        # Repetitions already recorded today, before this session started — the
        # screen adds its own answers on top of this baseline.
        with get_session() as session:
            today_count = RepetitionRepository(session).count_since(day_start())

        out_queue.put(("ready", predictor, queue, waiting, today_count))
    except Exception as exc:
        out_queue.put(("error", str(exc)))


def answer_worker(
    card: Card,
    remembered: bool,
    predictor: RecallEstimator | None,
    out_queue: queue_module.Queue,
) -> None:
    """Record one repetition, store its recomputed time constants, and derive the due time.

    Recomputes all three of the direction's curves from the history including the
    answer just given — the current one plus the two a further review would
    produce — so the word stays scoreable without waiting for the next full param
    pass.

    Returns ``("answered", card, practiced_at, next_ts, curves)`` where ``next_ts``
    is the live next-review timestamp (``None`` if no estimator) and ``curves`` is
    the ``(current, success, failure)`` time-constant triple just stored, each ``None``
    if they could not be computed. Scoring is left to the caller, which owns the moment the
    card is actually served — a gain depends on that moment, so computing one here
    would only date it to the answer instead.
    """
    try:
        practiced_at = int(time.time())
        next_ts: int | None = None
        current: float | None = None
        success: float | None = None
        failure: float | None = None

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
                    current = predictor.time_constant(all_reps, card.direction)
                    success, failure = predictor.post_rep_time_constants(
                        [(all_reps, card.direction)], practiced_at
                    )[0]
                    delta = invert_curve(
                        current, cfg.recall_threshold, cfg.max_delta_seconds
                    )
                    next_ts = practiced_at + int(delta)
                except Exception:
                    # Leave the time constants NULL so the next param pass recomputes
                    # them; the caller sorts an unscoreable card to the back.
                    current = success = failure = None
                    next_ts = 0

                word = WordRepository(session).get_by_id(card.word_id)
                if word is not None:
                    prefix = "fwd" if card.direction is Direction.FORWARD else "rev"
                    for suffix, tau in zip(
                        ("", "_ok", "_no"), (current, success, failure)
                    ):
                        setattr(word, f"{prefix}{suffix}_tau", tau)

        out_queue.put(
            ("answered", card, practiced_at, next_ts, (current, success, failure))
        )
    except Exception as exc:
        out_queue.put(("answer_error", str(exc)))
