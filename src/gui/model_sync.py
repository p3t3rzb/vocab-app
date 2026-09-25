"""Keep stored time constants consistent with the checkpoint that exists on disk.

The per-direction curve columns are written either by the trained model
(:func:`~src.model.compute_all_params`) or by the model-free
:func:`~src.model.backfill_heuristic_params`. Nothing in the schema records
*which*, so if a checkpoint disappears — deleted from the train screen, or
removed outside the app — the stored params silently keep scheduling reviews
from a model that is no longer there.

This module closes that gap: whenever a language pair has no checkpoint, every
word is re-derived from :class:`~src.model.HeuristicPredictor`, which is exactly
the estimator practice falls back to in that state.

The recompute reads the pair's whole repetition table and rewrites every word's
curves (~7 s on the French database), so it is done at most once per database
per app run. ``_source`` tracks what last wrote the params so that a checkpoint
deleted *after* a training run still triggers a fresh recompute.
"""
from __future__ import annotations

import queue as queue_module
import threading

from src.model import backfill_heuristic_params

from .db_context import DbContext

# {db path: "model" | "heuristic"} — what last wrote this database's params
# during this app run. Absent means "unknown" (e.g. written by an earlier run).
_source: dict[str, str] = {}

# Serialises the recompute: the word-list screen and practice entry can both
# reach for it from their own worker threads, and two concurrent passes would
# write the same values twice while contending for the SQLite write lock.
_lock = threading.Lock()


def _key(ctx: DbContext) -> str:
    return str(ctx.db_path)


def needs_heuristic_sync(ctx: DbContext) -> bool:
    """``True`` if ``ctx`` has no checkpoint and its params aren't heuristic yet."""
    return not ctx.model_path.exists() and _source.get(_key(ctx)) != "heuristic"


def _recompute(ctx: DbContext) -> int:
    count = backfill_heuristic_params()
    _source[_key(ctx)] = "heuristic"
    return count


def force_heuristic_params(ctx: DbContext) -> int:
    """Re-derive every word-with-history's params from the heuristic, unconditionally.

    Used right after deleting a checkpoint, where the params are known to be
    the removed model's. Returns the number of words rewritten.
    """
    with _lock:
        return _recompute(ctx)


def ensure_heuristic_params(ctx: DbContext) -> int:
    """Re-derive every word's params heuristically if the checkpoint is missing.

    Returns the number of words rewritten — ``0`` when a checkpoint exists or
    the recompute already ran for this database. Blocks if another thread is
    mid-recompute, so the caller can rely on the params being current on return.
    """
    with _lock:
        if not needs_heuristic_sync(ctx):
            return 0
        return _recompute(ctx)


def note_model_params(ctx: DbContext) -> None:
    """Record that ``ctx``'s params now come from a trained model.

    Called after a training run so that deleting the checkpoint later in the
    same session is still recognised as "these params outlived their model".
    """
    _source[_key(ctx)] = "model"


def sync_worker(ctx: DbContext, out_queue: queue_module.Queue) -> None:
    """Background-thread wrapper around :func:`ensure_heuristic_params`."""
    try:
        out_queue.put(("synced", ensure_heuristic_params(ctx)))
    except Exception as exc:
        out_queue.put(("sync_error", str(exc)))
