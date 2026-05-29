"""ORM model definitions for the vocab-repetition database.

The schema is fully described in ``CLAUDE.md``. Briefly:

* One :class:`LanguagePair` row per database (id always ``LanguagePair.SINGLETON_ID``).
* One :class:`Word` row per vocabulary entry.
* One :class:`Repetition` row per practice event, scoped to a single
  :class:`Direction` (source→target or target→source).
"""
from .base import BaseORM
from .direction import Direction
from .language_pair import LanguagePair
from .repetition import Repetition
from .word import Word

__all__ = ["BaseORM", "Direction", "LanguagePair", "Repetition", "Word"]
