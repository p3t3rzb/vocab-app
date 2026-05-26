"""Repository classes — typed query and mutation helpers for the ORM models.

Each repository wraps a SQLAlchemy :class:`Session` and exposes the small set
of operations the rest of the app needs. All repositories share
:class:`BaseRepository` so the session-binding boilerplate lives in one place.
"""
from .base import BaseRepository
from .language_pair import LanguagePairRepository
from .repetition import RepetitionRepository
from .word import WordRepository

__all__ = [
    "BaseRepository",
    "LanguagePairRepository",
    "RepetitionRepository",
    "WordRepository",
]
