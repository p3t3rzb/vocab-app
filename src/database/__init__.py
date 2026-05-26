"""SQLAlchemy ORM layer for the vocab-repetition app.

Public surface:

* :class:`Direction` — FORWARD (source→target) vs REVERSE (target→source).
* :class:`LanguagePair`, :class:`Word`, :class:`Repetition` — ORM models.
* :class:`LanguagePairRepository`, :class:`WordRepository`,
  :class:`RepetitionRepository` — query/update helpers.
* :func:`init_db` — set up the engine and create tables; call once at startup.
* :func:`get_session` — context-manager yielding a Session that commits on
  clean exit and rolls back on exception.
* :func:`read_language_pair`, :func:`count_words` — read-only ``sqlite3``
  peek helpers for browsing many databases without touching the global engine.
"""
from .inspect import count_words, read_language_pair
from .models import Direction, LanguagePair, Repetition, Word
from .repositories import (
    LanguagePairRepository,
    RepetitionRepository,
    WordRepository,
)
from .session import get_session, init_db

__all__ = [
    "Direction",
    "LanguagePair",
    "Repetition",
    "Word",
    "LanguagePairRepository",
    "RepetitionRepository",
    "WordRepository",
    "count_words",
    "get_session",
    "init_db",
    "read_language_pair",
]
