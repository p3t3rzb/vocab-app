"""Batched param computation: persist each word's forgetting-curve params.

Words are processed in chunks of :attr:`ScheduleConfig.chunk_size`. Within each
chunk every (word, direction) history is forwarded through the model in a single
batched call to get its forgetting-curve parameters ``(p0, S, d)``, which are
stored on the word. Recall score and the next-review time are derived from these
params *live* (see :mod:`src.model.curve`), so the recall threshold can change
without recomputing anything here.
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
from src.model.config import ScheduleConfig
from src.model.curve import split_params
from src.model.lstm import RecallLSTM

ProgressFn = Callable[[int, int], None]

# A direction's curve params, or None when that direction has no history.
Params = tuple[float, float, float]


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
    ) -> dict[int, tuple[Params | None, Params | None]]:
        """Forward every (word, direction) history once and read off the curve params.

        Returns ``{word_id: (fwd_params, rev_params)}`` for every word in the
        chunk. Directions with no history yield ``None``, which the persistence
        step writes as SQL ``NULL`` so the word stays "not yet computed".
        """
        keys: list[tuple[int, int]] = []
        sequences: list[list[list[float]]] = []

        for word in words:
            for direction in Direction:
                reps = reps_map.get((word.id, int(direction)), [])
                if not reps:
                    continue
                keys.append((word.id, int(direction)))
                sequences.append(self._history_rows(reps, direction))

        results: dict[tuple[int, int], Params] = {}
        if keys:
            params = self._curve_params(sequences)
            for key, p in zip(keys, params):
                results[key] = p

        return {
            word.id: (
                results.get((word.id, int(Direction.FORWARD))),
                results.get((word.id, int(Direction.REVERSE))),
            )
            for word in words
        }

    @staticmethod
    def _history_rows(reps: list[Repetition], direction: Direction) -> list[list[float]]:
        """History-only LSTM input: ``[log(gap-before-this-rep + 1), remembered, not_remembered, is_forward, is_reverse]`` per rep."""
        is_rev = float(int(direction))
        is_fwd = 1.0 - is_rev
        rows: list[list[float]] = []
        for i, rep in enumerate(reps):
            log_gap = 0.0 if i == 0 else math.log(rep.practiced_at - reps[i - 1].practiced_at + 1)
            rem = float(rep.remembered)
            rows.append([log_gap, rem, 1.0 - rem, is_fwd, is_rev])
        return rows

    def _curve_params(self, sequences: list[list[list[float]]]) -> list[Params]:
        """Batched forward → final-timestep ``(p0, S, d)`` per task."""
        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        n_features = len(sequences[0][0])
        batch = torch.zeros(len(sequences), max_len, n_features, dtype=torch.float32, device=self._device)
        for i, seq in enumerate(sequences):
            batch[i, : lengths[i]] = torch.tensor(seq, dtype=torch.float32)

        with torch.inference_mode():
            raw = self._model(batch)  # (B, max_len, 3)

        idx = torch.tensor([n - 1 for n in lengths], device=self._device)
        raw_last = raw[torch.arange(len(sequences), device=self._device), idx]  # (B, 3)

        p0, s, d = split_params(raw_last)  # each (B,)
        return list(zip(p0.tolist(), s.tolist(), d.tolist()))

    @staticmethod
    def _persist(chunk_updates: dict[int, tuple[Params | None, Params | None]]) -> None:
        """Write per-direction curve params for a chunk of words in one short session.

        A ``None`` direction clears its three columns to SQL ``NULL`` (no history
        yet), so the word list renders "–" until that direction is practiced.
        """
        with get_session() as session:
            for word_id, (fwd, rev) in chunk_updates.items():
                fwd_p0, fwd_s, fwd_d = fwd if fwd is not None else (None, None, None)
                rev_p0, rev_s, rev_d = rev if rev is not None else (None, None, None)
                session.execute(
                    sa_update(Word)
                    .where(Word.id == word_id)
                    .values(
                        fwd_p0=fwd_p0, fwd_s=fwd_s, fwd_d=fwd_d,
                        rev_p0=rev_p0, rev_s=rev_s, rev_d=rev_d,
                    )
                )


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
