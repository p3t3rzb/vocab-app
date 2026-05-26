"""Batched scheduler: persist ``next_repetition_at`` for every word.

Words are processed in chunks of :attr:`ScheduleConfig.chunk_size`. Within
each chunk the full threshold search (initial check → doubling → bisect →
polynomial refinement) is vectorised via batched LSTM calls — roughly
``20`` model forwards per chunk instead of one per word.
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
from src.model.inference.polynomial import PolynomialRefiner
from src.model.inference.search import SearchTask
from src.model.lstm import RecallLSTM

ProgressFn = Callable[[int, int], None]


class BatchScheduler:
    """Run the threshold search across every word in the active database in batched form."""

    def __init__(
        self,
        model: RecallLSTM,
        predict_cfg: PredictConfig,
        schedule_cfg: ScheduleConfig,
    ) -> None:
        self._model = model
        self._predict_cfg = predict_cfg
        self._schedule_cfg = schedule_cfg
        self._device = next(model.parameters()).device
        self._refiner = PolynomialRefiner(predict_cfg.poly_degree)

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
        """Run the full search vectorised over every (word, direction) task in the chunk.

        Returns ``{word_id: (fwd_ts, rev_ts)}``. Words with no history in
        either direction are omitted entirely so existing DB values stay untouched.
        """
        cfg = self._predict_cfg
        threshold = cfg.recall_threshold

        results: dict[tuple[int, int], int] = {}
        tasks: list[SearchTask] = []

        for word in words:
            for direction in Direction:
                reps = reps_map.get((word.id, int(direction)), [])
                if not reps:
                    continue
                tasks.append(
                    SearchTask(
                        word_id=word.id,
                        direction=int(direction),
                        base_seq=self._make_base_seq(reps),
                        last_remembered=float(reps[-1].remembered),
                        last_ts=reps[-1].practiced_at,
                        hi=cfg.initial_delta_seconds,
                    )
                )

        if not tasks:
            return {}

        # Stage 1: due-now check at Δ=1s
        probs = self._batch_probe(tasks, [1.0] * len(tasks))
        doubling: list[SearchTask] = []
        bisect: list[SearchTask] = []

        for task, p in zip(tasks, probs):
            task.points.append((1.0, p))
            if p < threshold:
                results[(task.word_id, task.direction)] = task.last_ts
            else:
                doubling.append(task)

        # Stage 2: doubling — find an upper bracket for each remaining task
        for _ in range(cfg.max_doubling_iters):
            if not doubling:
                break
            deltas = [t.hi for t in doubling]
            probs = self._batch_probe(doubling, deltas)

            next_doubling: list[SearchTask] = []
            for task, p in zip(doubling, probs):
                task.points.append((task.hi, p))
                if p >= threshold:
                    task.lo = task.hi
                    task.hi *= 2
                    if task.hi > cfg.max_delta_seconds:
                        results[(task.word_id, task.direction)] = (
                            task.last_ts + int(cfg.max_delta_seconds)
                        )
                    else:
                        next_doubling.append(task)
                else:
                    bisect.append(task)
            doubling = next_doubling

        # Stage 3: bisect — N binary-search steps across all bracketed tasks
        for _ in range(cfg.bisect_steps):
            if not bisect:
                break
            deltas = [(t.lo + t.hi) / 2 for t in bisect]
            probs = self._batch_probe(bisect, deltas)
            for task, delta, p in zip(bisect, deltas, probs):
                task.points.append((delta, p))
                if p >= threshold:
                    task.lo = delta
                else:
                    task.hi = delta

        # Stage 4: polynomial refinement
        for task in bisect:
            binary_result = task.hi
            poly_result = self._refiner.crossing(
                task.points, threshold, reference_log_delta=math.log(binary_result + 1)
            )
            if poly_result is None or poly_result <= 0 or not math.isfinite(poly_result):
                delta = binary_result
            else:
                poly_result = max(1.0, min(poly_result, cfg.max_delta_seconds))
                delta = math.sqrt(binary_result * poly_result)
            results[(task.word_id, task.direction)] = task.last_ts + int(delta)

        chunk_results: dict[int, tuple[int | None, int | None]] = {}
        for word in words:
            fwd_ts = results.get((word.id, int(Direction.FORWARD)))
            rev_ts = results.get((word.id, int(Direction.REVERSE)))
            if fwd_ts is None and rev_ts is None:
                continue
            chunk_results[word.id] = (fwd_ts, rev_ts)
        return chunk_results

    @staticmethod
    def _make_base_seq(reps: list[Repetition]) -> list[list[float]]:
        """Build LSTM input rows for a rep history, excluding the probe timestep."""
        base: list[list[float]] = []
        for i in range(1, len(reps)):
            dt = reps[i].practiced_at - reps[i - 1].practiced_at
            base.append([math.log(dt + 1), float(reps[i - 1].remembered)])
        return base

    def _batch_probe(self, tasks: list[SearchTask], deltas: list[float]) -> list[float]:
        """Single batched LSTM forward over all tasks at their respective probe deltas."""
        sequences: list[list[list[float]]] = []
        lengths: list[int] = []

        for task, delta in zip(tasks, deltas):
            seq = task.base_seq + [[math.log(delta + 1), task.last_remembered]]
            sequences.append(seq)
            lengths.append(len(seq))

        max_len = max(lengths)
        batch = torch.zeros(len(sequences), max_len, 2, dtype=torch.float32, device=self._device)
        for i, seq in enumerate(sequences):
            batch[i, : lengths[i]] = torch.tensor(seq, dtype=torch.float32)

        with torch.inference_mode():
            probs = self._model(batch)  # (B, max_len)

        return [probs[i, lengths[i] - 1].item() for i in range(len(sequences))]

    @staticmethod
    def _persist(chunk_updates: dict[int, tuple[int | None, int | None]]) -> None:
        """Write per-direction due-times for a chunk of words in one short session."""
        with get_session() as session:
            for word_id, (fwd_ts, rev_ts) in chunk_updates.items():
                vals: dict[str, int] = {}
                if fwd_ts is not None:
                    vals["next_rep_fwd_at"] = fwd_ts
                if rev_ts is not None:
                    vals["next_rep_rev_at"] = rev_ts
                if vals:
                    session.execute(
                        sa_update(Word).where(Word.id == word_id).values(**vals)
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
