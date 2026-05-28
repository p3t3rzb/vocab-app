"""Batched scheduler: persist ``next_repetition_at`` for every word.

Words are processed in chunks of :attr:`ScheduleConfig.chunk_size`. Within each
chunk every (word, direction) history is forwarded through the model in a single
batched call to get its forgetting-curve parameters, and the next-review time is
then computed analytically by inverting the curve (no bisection).
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import torch
from sqlalchemy import select, update as sa_update

from src.database import Direction, get_session
from src.database.models import Repetition, Word
from src.model.checkpoint import load_model
from src.model.config import PredictConfig, ScheduleConfig
from src.model.curve import split_params
from src.model.lstm import RecallLSTM

ProgressFn = Callable[[int, int], None]


class BatchScheduler:
    """Compute every word's next-review time from the model in batched form."""

    def __init__(
        self,
        model: RecallLSTM,
        predict_cfg: PredictConfig,
        schedule_cfg: ScheduleConfig,
    ) -> None:
        self._model = model
        self._model.eval()
        self._predict_cfg = predict_cfg
        self._schedule_cfg = schedule_cfg
        self._device = next(model.parameters()).device

    def run(
        self,
        on_progress: ProgressFn | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Compute ``next_rep_fwd_at`` / ``next_rep_rev_at`` for every word and persist."""
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

        for chunk_start in range(0, total, chunk_size):
            if stop_event is not None and stop_event.is_set():
                return

            chunk = all_words[chunk_start : chunk_start + chunk_size]
            chunk_updates = self._process_chunk(chunk, reps_map)

            if chunk_updates:
                self._persist(chunk_updates)

            if on_progress is not None:
                on_progress(min(chunk_start + chunk_size, total), total)

    def _process_chunk(
        self,
        words: list[Word],
        reps_map: dict[tuple[int, int], list[Repetition]],
    ) -> dict[int, tuple[int | None, int | None]]:
        """Forward every (word, direction) history once and invert the curve analytically.

        Returns ``{word_id: (fwd_ts, rev_ts)}`` for every word in the chunk.
        Directions with no history yield ``None``, which the persistence step
        writes as SQL ``NULL`` so the list shows "–" until the word is practiced.
        """
        keys: list[tuple[int, int]] = []
        sequences: list[list[list[float]]] = []
        last_ts: list[int] = []

        for word in words:
            for direction in Direction:
                reps = reps_map.get((word.id, int(direction)), [])
                if not reps:
                    continue
                keys.append((word.id, int(direction)))
                sequences.append(self._history_rows(reps))
                last_ts.append(reps[-1].practiced_at)

        results: dict[tuple[int, int], int] = {}
        if keys:
            deltas = self._next_deltas(sequences)
            for (word_id, direction), ts, delta in zip(keys, last_ts, deltas):
                results[(word_id, direction)] = ts + int(delta)

        return {
            word.id: (
                results.get((word.id, int(Direction.FORWARD))),
                results.get((word.id, int(Direction.REVERSE))),
            )
            for word in words
        }

    @staticmethod
    def _history_rows(reps: list[Repetition]) -> list[list[float]]:
        """History-only LSTM input: ``[log(gap-before-this-rep + 1), remembered]`` per rep."""
        rows: list[list[float]] = []
        for i, rep in enumerate(reps):
            log_gap = 0.0 if i == 0 else math.log(rep.practiced_at - reps[i - 1].practiced_at + 1)
            rows.append([log_gap, float(rep.remembered)])
        return rows

    def _next_deltas(self, sequences: list[list[list[float]]]) -> list[float]:
        """Batched forward + vectorised analytic curve inversion → seconds per task."""
        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        batch = torch.zeros(len(sequences), max_len, 2, dtype=torch.float32, device=self._device)
        for i, seq in enumerate(sequences):
            batch[i, : lengths[i]] = torch.tensor(seq, dtype=torch.float32)

        with torch.inference_mode():
            raw = self._model(batch)  # (B, max_len, 3)

        idx = torch.tensor([n - 1 for n in lengths], device=self._device)
        raw_last = raw[torch.arange(len(sequences), device=self._device), idx]  # (B, 3)

        cfg = self._predict_cfg
        p0, s, d = split_params(raw_last)
        delta = s * (torch.pow(p0 / cfg.recall_threshold, 1.0 / d) - 1.0)
        delta = torch.where(p0 <= cfg.recall_threshold, torch.zeros_like(delta), delta)
        delta = delta.clamp(0.0, cfg.max_delta_seconds)
        delta = torch.nan_to_num(delta, nan=cfg.max_delta_seconds, posinf=cfg.max_delta_seconds)
        return delta.tolist()

    @staticmethod
    def _persist(chunk_updates: dict[int, tuple[int | None, int | None]]) -> None:
        """Write per-direction due-times for a chunk of words in one short session.

        ``None`` is written as SQL ``NULL`` so directions with no repetition
        history get cleared (and rendered as "–" in the word list).
        """
        with get_session() as session:
            for word_id, (fwd_ts, rev_ts) in chunk_updates.items():
                session.execute(
                    sa_update(Word)
                    .where(Word.id == word_id)
                    .values(next_rep_fwd_at=fwd_ts, next_rep_rev_at=rev_ts)
                )


def compute_all_schedules(
    model_path: str | Path,
    on_progress: ProgressFn | None = None,
    stop_event: threading.Event | None = None,
    cfg: PredictConfig | None = None,
    schedule_cfg: ScheduleConfig | None = None,
) -> None:
    """Load the checkpoint and update every word's ``next_repetition_at``.

    Args:
        model_path: Path to the saved ``.pt`` checkpoint.
        on_progress: Optional callback invoked as ``on_progress(words_done,
            total_words)`` after each chunk completes.
        stop_event: Optional :class:`threading.Event` for cancellation.
            Checked between chunks; already-committed chunks are preserved.
        cfg: Override the default :class:`PredictConfig` (e.g. with the
            user-configured recall threshold).
        schedule_cfg: Override the default :class:`ScheduleConfig`.
    """
    model = load_model(str(model_path))
    BatchScheduler(
        model=model,
        predict_cfg=cfg or PredictConfig(),
        schedule_cfg=schedule_cfg or ScheduleConfig(),
    ).run(on_progress=on_progress, stop_event=stop_event)
