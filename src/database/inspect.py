"""Read-only sqlite3 peek helpers for browsing many databases.

The home screen and the settings recalc worker both need to glance at every
``*.db`` in ``storage/`` without going through the global SQLAlchemy engine
(which would have to be swapped per file). These helpers open a short-lived
raw :mod:`sqlite3` connection, run one query, and return ``None`` / ``0`` on
any failure so callers can skip unreadable or unrelated files.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def read_language_pair(db_path: Path) -> tuple[str, str] | None:
    """Return ``(source_language, target_language)`` for ``db_path``, or ``None``.

    ``None`` is returned for any failure (file unreadable, table missing,
    empty ``language_pair``) so callers can simply skip the database.
    """
    try:
        con = sqlite3.connect(str(db_path))
    except Exception:
        return None
    try:
        row = con.execute(
            "SELECT source_language, target_language FROM language_pair LIMIT 1"
        ).fetchone()
    except Exception:
        return None
    finally:
        con.close()
    return (row[0], row[1]) if row else None


def count_words(db_path: Path) -> int:
    """Return ``COUNT(*)`` from the ``words`` table, or ``0`` on any error."""
    try:
        con = sqlite3.connect(str(db_path))
    except Exception:
        return 0
    try:
        row = con.execute("SELECT COUNT(*) FROM words").fetchone()
    except Exception:
        return 0
    finally:
        con.close()
    return row[0] if row else 0
