"""SQLAlchemy ORM layer for the vocab-repetition app.

Public surface:

* :class:`Direction` — FORWARD (source→target) vs REVERSE (target→source).
* :class:`LanguagePair`, :class:`Word`, :class:`Repetition` — ORM models.
* :class:`LanguagePairRepository`, :class:`WordRepository`,
  :class:`RepetitionRepository` — query/update helpers.
* :func:`init_db` — set up the engine and create tables; call once at startup.
* :func:`get_session` — context-manager yielding a Session that commits on
  clean exit and rolls back on exception.
"""
from .models import Direction, LanguagePair, Repetition, Word
from .repository import LanguagePairRepository, RepetitionRepository, WordRepository
from .session import get_session, init_db

__all__ = [
    "Direction",
    "LanguagePair",
    "Repetition",
    "Word",
    "LanguagePairRepository",
    "RepetitionRepository",
    "WordRepository",
    "get_session",
    "init_db",
]
