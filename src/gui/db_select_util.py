"""Pure helpers and value objects used by the database-selection screen.

Kept separate from :mod:`src.gui.db_select` so the screen file is just the
two screen / dialog classes. None of these helpers touch Tk widgets.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .theme import Paths

_NON_ALPHANUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class DbEntry:
    """One row in the database-selection treeview.

    ``word_count`` and ``trained_mtime`` are carried alongside the identity
    fields so the row can be sorted on its underlying values (not the
    formatted cell text). ``trained_mtime`` is ``0.0`` when no checkpoint exists.
    """

    db_path: Path
    src_lang: str
    tgt_lang: str
    word_count: int = 0
    trained_mtime: float = 0.0


def slugify(value: str) -> str:
    """Lowercase, collapse runs of non-alphanumeric chars to ``_``, trim."""
    return _NON_ALPHANUM_RE.sub("_", value.lower()).strip("_")


def safe_db_basename(src: str, tgt: str) -> str:
    """Build a filesystem-safe ``<src>_<tgt>`` basename (no extension).

    Returns the empty string if either side sanitises to nothing — this is
    also a safety check, since path separators and ``..`` cannot survive
    :func:`slugify` and so the result cannot escape ``storage/``.
    """
    src_c, tgt_c = slugify(src), slugify(tgt)
    if not src_c or not tgt_c:
        return ""
    return f"{src_c}_{tgt_c}"


def last_trained_mtime(src: str, tgt: str) -> float:
    """Return the checkpoint's mtime as a Unix timestamp, or ``0.0`` if not trained."""
    ckpt = Paths.model_path(src, tgt)
    if not ckpt.exists():
        return 0.0
    return ckpt.stat().st_mtime


def last_trained_label(src: str, tgt: str) -> str:
    """Return the checkpoint mtime as a display string, or ``"—"`` if not trained."""
    mtime = last_trained_mtime(src, tgt)
    if mtime == 0.0:
        return "—"
    return datetime.fromtimestamp(mtime).strftime("%b %d, %Y  %H:%M")
