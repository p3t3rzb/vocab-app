"""Query and insert helpers for :class:`Repetition`."""
from sqlalchemy import func, select

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

    def latest_practiced_at_by_word(self) -> dict[int, int]:
        """Map every word_id to its most recent ``practiced_at`` across all directions.

        One aggregate query for the whole table — suitable for rendering the
        word list's "Last revised" column without a per-word round-trip.
        """
        stmt = (
            select(Repetition.word_id, func.max(Repetition.practiced_at))
            .group_by(Repetition.word_id)
        )
        return {word_id: latest for word_id, latest in self._session.execute(stmt)}

    def latest_practiced_at_by_word_direction(self) -> dict[tuple[int, int], int]:
        """Map every (word_id, direction) to its most recent ``practiced_at``.

        One aggregate query for the whole table — powers the practice queue
        builder and the word list's live due-time cache without a per-pair
        round-trip. Keys are ``(word_id, direction_int)``.
        """
        stmt = (
            select(
                Repetition.word_id,
                Repetition.direction,
                func.max(Repetition.practiced_at),
            )
            .group_by(Repetition.word_id, Repetition.direction)
        )
        return {
            (word_id, direction): latest
            for word_id, direction, latest in self._session.execute(stmt)
        }

    def add(self, repetition: Repetition) -> None:
        """Stage ``repetition`` for insertion on the next commit."""
        self._session.add(repetition)
