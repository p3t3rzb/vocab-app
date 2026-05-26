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
from src.model.config import PredictConfig
from src.model.lstm import RecallLSTM
from src.model.predictor import Predictor, _poly_threshold_crossing
from src.model.train import load_model

# Number of words processed in each batched chunk.
# Controls the trade-off between memory footprint and GPU dispatch overhead.
# At 256 words → up to 512 LSTM tasks, padded to max_seq_len:
#   input  ≈ 512 × 238 × 2 × 4 B ≈ 1 MB
#   output ≈ 512 × 238 × 64 × 4 B ≈ 31 MB  — well within any device budget.
_CHUNK_SIZE = 256


def compute_next_repetition_at(
    reps_fwd: list[Repetition],
    reps_rev: list[Repetition],
    predictor: Predictor,
) -> int:
    """Return Unix timestamp when this word next needs practice (single-word helper).

    Returns 0 when neither direction has any history (immediately due).
    Only directions with ≥1 rep contribute to the minimum.
    """
    if not reps_fwd and not reps_rev:
        return 0
    candidates: list[int] = []
    for reps in (reps_fwd, reps_rev):
        if not reps:
            continue
        try:
            delta = predictor.next_repetition_delta(reps)
            candidates.append(reps[-1].practiced_at + int(delta))
        except Exception:
            pass
    return min(candidates) if candidates else 0


# ---------------------------------------------------------------------------
# Batched inference helpers
# ---------------------------------------------------------------------------

def _make_base_seq(reps: list[Repetition]) -> list[list[float]]:
    """Build LSTM input rows for a rep history, excluding the probe timestep."""
    base: list[list[float]] = []
    for i in range(1, len(reps)):
        dt = reps[i].practiced_at - reps[i - 1].practiced_at
        base.append([math.log(max(dt, 0) + 1), float(reps[i - 1].remembered)])
    return base


def _batch_probe(
    tasks: list[dict],
    deltas: list[float],
    model: RecallLSTM,
    device: torch.device,
) -> list[float]:
    """
    Single batched LSTM forward pass for all tasks at their respective probe deltas.

    Appends the probe timestep [log(delta+1), last_remembered] to each base sequence,
    pads the batch to equal length, and returns the last-step probability per task.
    """
    sequences: list[list[list[float]]] = []
    lengths: list[int] = []

    for task, delta in zip(tasks, deltas):
        seq = task["base"] + [[math.log(delta + 1), task["last_remembered"]]]
        sequences.append(seq)
        lengths.append(len(seq))

    max_len = max(lengths)
    B = len(sequences)
    batch = torch.zeros(B, max_len, 2, dtype=torch.float32, device=device)
    for i, seq in enumerate(sequences):
        batch[i, : lengths[i]] = torch.tensor(seq, dtype=torch.float32)

    with torch.inference_mode():
        probs = model(batch)  # (B, max_len)

    return [probs[i, lengths[i] - 1].item() for i in range(B)]


def _process_chunk(
    words: list[Word],
    reps_map: dict[tuple[int, int], list[Repetition]],
    model: RecallLSTM,
    cfg: PredictConfig,
    device: torch.device,
) -> dict[int, tuple[int | None, int | None]]:
    """
    Run the full binary search for a chunk of words.

    Returns {word_id: (fwd_ts, rev_ts)} where each timestamp is the Unix time
    when that direction is next due, or None if that direction has no history.
    Words with no history in either direction are omitted from the result.

    The search (initial check → doubling → bisect → polynomial refinement) is
    vectorised across all (word, direction) tasks in the chunk via batched LSTM calls.
    """
    threshold = cfg.recall_threshold
    wd_results: dict[tuple[int, int], int] = {}
    tasks: list[dict] = []

    for word in words:
        fwd = reps_map.get((word.id, int(Direction.FORWARD)), [])
        rev = reps_map.get((word.id, int(Direction.REVERSE)), [])
        for direction_int, reps in ((int(Direction.FORWARD), fwd), (int(Direction.REVERSE), rev)):
            if not reps:
                continue
            tasks.append({
                "word_id": word.id,
                "direction": direction_int,
                "base": _make_base_seq(reps),
                "last_remembered": float(reps[-1].remembered),
                "last_ts": reps[-1].practiced_at,
                "lo": 0.0,
                "hi": cfg.initial_delta_seconds,
                "points": [],
            })

    if not tasks:
        return {}  # no history in any direction for this chunk

    # Step 0: find tasks already below threshold at Δ=1s (due immediately)
    probs = _batch_probe(tasks, [1.0] * len(tasks), model, device)
    doubling: list[dict] = []
    bisect: list[dict] = []

    for task, p in zip(tasks, probs):
        task["points"].append((1.0, p))
        if p < threshold:
            wd_results[(task["word_id"], task["direction"])] = task["last_ts"]
        else:
            doubling.append(task)

    # Step 1: doubling — find an upper bracket for each remaining task
    for _ in range(30):
        if not doubling:
            break
        deltas = [t["hi"] for t in doubling]
        probs = _batch_probe(doubling, deltas, model, device)

        next_doubling: list[dict] = []
        for task, p in zip(doubling, probs):
            task["points"].append((task["hi"], p))
            if p >= threshold:
                task["lo"] = task["hi"]
                task["hi"] *= 2
                if task["hi"] > cfg.max_delta_seconds:
                    wd_results[(task["word_id"], task["direction"])] = (
                        task["last_ts"] + int(cfg.max_delta_seconds)
                    )
                else:
                    next_doubling.append(task)
            else:
                bisect.append(task)
        doubling = next_doubling

    # Step 2: bisect — 16 binary search steps across all bracketed tasks
    for _ in range(cfg.bisect_steps):
        if not bisect:
            break
        deltas = [(t["lo"] + t["hi"]) / 2 for t in bisect]
        probs = _batch_probe(bisect, deltas, model, device)
        for task, delta, p in zip(bisect, deltas, probs):
            task["points"].append((delta, p))
            if p >= threshold:
                task["lo"] = delta
            else:
                task["hi"] = delta

    # Step 3: polynomial refinement for bisect tasks
    for task in bisect:
        binary_result = task["hi"]
        poly_result = _poly_threshold_crossing(
            task["points"], threshold, cfg.poly_degree, math.log(binary_result + 1)
        )
        if poly_result is None or poly_result <= 0 or not math.isfinite(poly_result):
            delta = binary_result
        else:
            poly_result = max(1.0, min(poly_result, cfg.max_delta_seconds))
            delta = math.sqrt(binary_result * poly_result)
        wd_results[(task["word_id"], task["direction"])] = task["last_ts"] + int(delta)

    # Build per-direction results; skip words with no history in either direction
    chunk_results: dict[int, tuple[int | None, int | None]] = {}
    for word in words:
        fwd_ts = wd_results.get((word.id, int(Direction.FORWARD)))
        rev_ts = wd_results.get((word.id, int(Direction.REVERSE)))
        if fwd_ts is None and rev_ts is None:
            continue  # no history — leave DB values unchanged
        chunk_results[word.id] = (fwd_ts, rev_ts)

    return chunk_results


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_all_schedules(
    model_path: str | Path,
    on_progress: Callable[[int, int], None] | None = None,
    stop_event: threading.Event | None = None,
    cfg: PredictConfig | None = None,
) -> None:
    """
    Compute next_repetition_at for every word and write results to the database.

    Words are processed in chunks of _CHUNK_SIZE. Within each chunk the full binary
    search (initial check, doubling, bisect, polynomial refinement) is vectorised via
    batched LSTM calls — ~20 forward passes per chunk instead of ~20 per word.

    on_progress(words_done, total_words) is called after each chunk completes.
    Respects stop_event: returns without writing if cancelled.
    """
    model = load_model(str(model_path))
    if cfg is None:
        cfg = PredictConfig()
    device = next(model.parameters()).device

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
    word_updates: dict[int, tuple[int | None, int | None]] = {}

    for chunk_start in range(0, total, _CHUNK_SIZE):
        if stop_event and stop_event.is_set():
            return

        chunk = all_words[chunk_start : chunk_start + _CHUNK_SIZE]
        word_updates.update(_process_chunk(chunk, reps_map, model, cfg, device))

        if on_progress:
            on_progress(min(chunk_start + _CHUNK_SIZE, total), total)

    if stop_event and stop_event.is_set():
        return

    with get_session() as session:
        for word_id, (fwd_ts, rev_ts) in word_updates.items():
            vals: dict[str, int] = {}
            if fwd_ts is not None:
                vals["next_rep_fwd_at"] = fwd_ts
            if rev_ts is not None:
                vals["next_rep_rev_at"] = rev_ts
            if vals:
                session.execute(
                    sa_update(Word).where(Word.id == word_id).values(**vals)
                )
