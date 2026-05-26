"""CRUD helpers for :class:`Word`."""
from sqlalchemy import func, select

from ..models import Word
from .base import BaseRepository


class WordRepository(BaseRepository):
    """CRUD helpers for :class:`Word`."""

    def get_by_id(self, word_id: int) -> Word | None:
        """Return the word with the given id, or ``None`` if no such word exists."""
        return self._session.get(Word, word_id)

    def get_all(self) -> list[Word]:
        """Return every word, ordered by id ascending."""
        return list(self._session.scalars(select(Word).order_by(Word.id)))

    def find_by_source_text(self, source_text: str) -> Word | None:
        """Return the word with the given ``source_text``, or ``None`` if absent."""
        return self._session.scalars(
            select(Word).where(Word.source_text == source_text).limit(1)
        ).first()

    def add(self, word: Word) -> None:
        """Stage ``word`` for insertion on the next commit."""
        self._session.add(word)

    def delete(self, word: Word) -> None:
        """Stage ``word`` for deletion. Repetitions cascade automatically."""
        self._session.delete(word)

    def get_next_id(self) -> int:
        """Return ``max(id) + 1``, or ``0`` if the table is empty.

        Used to assign an id to a newly created word — the schema uses
        explicit ids (no autoincrement) so the migration row indices remain
        stable.
        """
        max_id = self._session.scalar(select(func.max(Word.id)))
        return (max_id + 1) if max_id is not None else 0
