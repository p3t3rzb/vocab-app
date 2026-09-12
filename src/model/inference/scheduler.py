"""Batched param computation: persist each word's forgetting-curve params.

Words are processed in chunks of :attr:`ScheduleConfig.chunk_size`. Within each
chunk every (word, direction) is forwarded through the model in a single batched
call to get **three** curves' worth of ``(p0, S, d)``, all stored on the word:

* the *current* curve, from the history as it stands — the one recall and the
  next-review time are derived from,
* the curve a *remembered* answer right now would produce,
* the curve a *forgotten* answer right now would produce.

The practice queue needs all three to rank cards by how much recall a review
*adds* over leaving the card alone (:func:`src.model.curve.expected_gain`) —
the current curve is the do-nothing baseline it is measured against. Three
sequences per card is far too much work to do at session start, so it happens
here instead. They are
computed for directions with no history too, since a never-practised card still
has to be ranked — only the *current* curve stays ``NULL`` there, which is what
marks a card as new.

Everything derived from these params — recall score, next-review time, queue
score — is computed *live* (see :mod:`src.model.curve`), so the recall threshold
and horizon can change without recomputing anything here.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import select, update as sa_update

from src.database import Direction, get_session
from src.database.models import Repetition, Word

from ..checkpoint import load_model
from ..config import HeuristicConfig, ScheduleConfig
from ..features import history_rows, rep_row
from ..lstm import RecallLSTM
from .batch import Params, final_step_params
from .heuristic import HeuristicPredictor

ProgressFn = Callable[[int, int], None]

#: One direction's three curves: ``(current, after_remembered, after_forgotten)``.
#: ``current`` is ``None`` when that direction has no history yet.
CardCurves = tuple[Params | None, Params, Params]


class ParamScheduler:
    """Compute every word's forgetting-curve params from the model in batched form."""

    def __init__(
        self,
        model: RecallLSTM,
        schedule_cfg: ScheduleConfig,
    ) -> None:
        self._model = model
        self._model.eval()
        self._schedule_cfg = schedule_cfg
        self._device = next(model.parameters()).device

    def run(
        self,
        on_progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Compute and persist the per-direction curve params for every word."""
        with get_session() as session:
            all_words = list(session.scalars(select(Word).order_by(Word.id)))
            all_reps = list(
                session.scalars(
                    select(Repetition).order_by(
                        Repetition.word_id, Repetition.direction, Repetition.practiced_at
                    )
                )
            )

        reps_map: dict[tuple[int, int], list[Repetition]] = defaultdict(list)
        for rep in all_reps:
            reps_map[(rep.word_id, rep.direction)].append(rep)

        total = len(all_words)
        chunk_size = self._schedule_cfg.chunk_size
        # One timestamp for the whole pass: the hypothetical "practised now" rep
        # is dated from it, so every card is scored against the same moment.
        now = int(time.time())

        for chunk_start in range(0, total, chunk_size):
            if stop_event is not None and stop_event.is_set():
                return

            chunk = all_words[chunk_start : chunk_start + chunk_size]
            chunk_updates = self._process_chunk(chunk, reps_map, now)

            if chunk_updates:
                _persist_params(chunk_updates)

            if on_progress is not None:
                on_progress(min(chunk_start + chunk_size, total), total)

    def _process_chunk(
        self,
        words: list[Word],
        reps_map: dict[tuple[int, int], list[Repetition]],
        now: int,
    ) -> dict[int, tuple[CardCurves, CardCurves]]:
        """Forward every (word, direction) and read off its three curves.

        Each direction contributes up to three sequences: its history as-is
        (omitted when there is none), and that history with a hypothetical
        remembered / forgotten rep dated ``now`` appended. They are forwarded in
        batches of :attr:`ScheduleConfig.batch_sequences` — a chunk of words is
        far too many sequences to shape into one tensor.

        Returns ``{word_id: (fwd_curves, rev_curves)}`` for every word in the
        chunk.
        """
        slots: list[tuple[tuple[int, int], int]] = []
        sequences: list[list[list[float]]] = []

        for word in words:
            for direction in Direction:
                key = (word.id, int(direction))
                reps = reps_map.get(key, [])
                rows = history_rows(reps, direction)
                gap = float(now - reps[-1].practiced_at) if reps else 0.0
                if reps:
                    slots.append((key, 0))
                    sequences.append(rows)
                slots.append((key, 1))
                sequences.append(rows + [rep_row(gap, True, direction)])
                slots.append((key, 2))
                sequences.append(rows + [rep_row(gap, False, direction)])

        by_key: dict[tuple[int, int], list[Params | None]] = defaultdict(
            lambda: [None, None, None]
        )
        for (key, which), params in zip(
            slots,
            final_step_params(
                self._model,
                sequences,
                chunk_size=self._schedule_cfg.batch_sequences,
            ),
        ):
            by_key[key][which] = params

        def curves(key: tuple[int, int]) -> CardCurves:
            current, ok, no = by_key[key]
            # `ok` / `no` are always produced above; the cast documents that.
            return current, ok, no  # type: ignore[return-value]

        return {
            word.id: (
                curves((word.id, int(Direction.FORWARD))),
                curves((word.id, int(Direction.REVERSE))),
            )
            for word in words
        }


def _column_values(prefix: str, curves: CardCurves) -> dict[str, float | None]:
    """Flatten one direction's three curves into its nine ``words`` columns.

    ``prefix`` is ``"fwd"`` or ``"rev"``; the current curve takes the bare
    ``<prefix>_p0`` / ``_s`` / ``_d`` names and the two hypothetical ones the
    ``_ok`` / ``_no`` variants. A ``None`` curve clears its three columns to SQL
    ``NULL``.
    """
    values: dict[str, float | None] = {}
    for suffix, params in zip(("", "_ok", "_no"), curves):
        p0, s, d = params if params is not None else (None, None, None)
        values[f"{prefix}{suffix}_p0"] = p0
        values[f"{prefix}{suffix}_s"] = s
        values[f"{prefix}{suffix}_d"] = d
    return values


def _persist_params(
    chunk_updates: dict[int, tuple[CardCurves, CardCurves]],
) -> None:
    """Write both directions' curve params for a chunk of words in one session.

    A direction with no history has a ``None`` *current* curve, so its three
    ``<dir>_p0/_s/_d`` columns go to SQL ``NULL`` — the word list renders "–" and
    the practice queue treats the card as new. Its two post-review curves are
    still written, because a new card has to be ranked like any other.
    """
    with get_session() as session:
        for word_id, (fwd, rev) in chunk_updates.items():
            session.execute(
                sa_update(Word)
                .where(Word.id == word_id)
                .values(**_column_values("fwd", fwd), **_column_values("rev", rev))
            )


def backfill_heuristic_params(
    heuristic_cfg: HeuristicConfig | None = None,
) -> int:
    """Fill in curve params from repetition history *without* a trained model.

    The model-free counterpart of :func:`compute_all_params`, for a language pair
    that has no checkpoint yet: it runs :class:`HeuristicPredictor` over every
    (word, direction) and persists the resulting current and post-review
    ``(p0, S, d)``. Without it, repetitions recorded before a pair had any
    estimator would sit at ``NULL`` — practised, but unscheduled and rendered "–"
    in the word list — until each was answered once more.

    Every word is touched, not only the ones with history: the post-review curves
    are what the practice queue ranks on, and a never-practised card needs them
    too. A direction with no history still gets a ``NULL`` *current* curve, which
    is what keeps it marked as new.

    Args:
        heuristic_cfg: Override the default :class:`HeuristicConfig`.

    Returns:
        The number of words whose params were written.
    """
    predictor = HeuristicPredictor(heuristic_config=heuristic_cfg)

    with get_session() as session:
        word_ids = list(session.scalars(select(Word.id).order_by(Word.id)))
        all_reps = list(
            session.scalars(
                select(Repetition).order_by(
                    Repetition.word_id, Repetition.direction, Repetition.practiced_at
                )
            )
        )

    reps_map: dict[tuple[int, int], list[Repetition]] = defaultdict(list)
    for rep in all_reps:
        reps_map[(rep.word_id, rep.direction)].append(rep)

    now = int(time.time())
    updates: dict[int, tuple[CardCurves, CardCurves]] = {}
    for word_id in word_ids:
        per_direction: list[CardCurves] = []
        for direction in Direction:
            reps = reps_map.get((word_id, int(direction)), [])
            current = predictor.curve_params(reps, direction) if reps else None
            ok, no = predictor.post_rep_params([(reps, direction)], now)[0]
            per_direction.append((current, ok, no))
        updates[word_id] = (per_direction[0], per_direction[1])

    if updates:
        _persist_params(updates)
    return len(updates)


def compute_all_params(
    model_path: str | Path,
    on_progress: ProgressFn | None = None,
    stop_event: threading.Event | None = None,
    schedule_cfg: ScheduleConfig | None = None,
) -> None:
    """Load the checkpoint and recompute every word's forgetting-curve params.

    The recall threshold / max interval are *not* applied here — they are honoured
    live when recall and due times are derived from the stored params.

    Args:
        model_path: Path to the saved ``.pt`` checkpoint.
        on_progress: Optional callback invoked as ``on_progress(words_done,
            total_words)`` after each chunk completes.
        stop_event: Optional :class:`threading.Event` for cancellation.
            Checked between chunks; already-committed chunks are preserved.
        schedule_cfg: Override the default :class:`ScheduleConfig`.
    """
    model = load_model(str(model_path))
    ParamScheduler(
        model=model,
        schedule_cfg=schedule_cfg or ScheduleConfig(),
    ).run(on_progress=on_progress, stop_event=stop_event)
