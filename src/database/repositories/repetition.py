"""Query and insert helpers for :class:`Repetition`."""
from sqlalchemy import select

from ..models import Direction, Repetition
from .base import BaseRepository


class RepetitionRepository(BaseRepository):
    """Query and insert helpers for :class:`Repetition`."""

    def get_for_word(self, word_id: int, direction: Direction) -> list[Repetition]:
        """Return all repetitions for one (word, direction) pair, oldest first."""
        stmt = (
            select(Repetition)
            .where(
                Repetition.word_id == word_id,
                Repetition.direction == int(direction),
            )
            .order_by(Repetition.practiced_at)
        )
        return list(self._session.scalars(stmt))

    def get_latest_for_word(self, word_id: int, direction: Direction) -> Repetition | None:
        """Return the most recent repetition for a (word, direction) pair, or ``None``."""
        stmt = (
            select(Repetition)
            .where(
                Repetition.word_id == word_id,
                Repetition.direction == int(direction),
            )
            .order_by(Repetition.practiced_at.desc())
            .limit(1)
        )
        return self._session.scalars(stmt).first()

    def add(self, repetition: Repetition) -> None:
        """Stage ``repetition`` for insertion on the next commit."""
        self._session.add(repetition)
