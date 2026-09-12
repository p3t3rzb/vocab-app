"""Query and insert helpers for :class:`Repetition`."""
from collections import defaultdict

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

    def history_by_word_direction(self) -> dict[tuple[int, int], list[Repetition]]:
        """Group the whole table into per-(word, direction) histories, oldest first.

        One query for every repetition — the bulk counterpart of
        :meth:`get_for_word`, for callers that need many histories at once (the
        param scheduler forwards every card's history in batched passes).
        Keys are ``(word_id, direction_int)``; pairs with no history are absent.
        """
        stmt = select(Repetition).order_by(
            Repetition.word_id, Repetition.direction, Repetition.practiced_at
        )
        histories: dict[tuple[int, int], list[Repetition]] = defaultdict(list)
        for rep in self._session.scalars(stmt):
            histories[(rep.word_id, rep.direction)].append(rep)
        return histories

    def first_attempt_success_rate(self, default: float = 0.5) -> float:
        """Fraction of (word, direction) pairs remembered on their *first* attempt.

        Stands in for "current recall" when the practice queue scores a
        never-practised card: there is no curve to evaluate yet, but the deck's
        own history says how often a first exposure goes well, and that is the
        probability the expected-retention score needs to weight the remembered
        and forgotten branches by.

        Args:
            default: Returned when the table is empty (a brand-new database).

        Returns:
            The mean outcome of every pair's earliest repetition, in ``[0, 1]``.
        """
        earliest = (
            select(
                Repetition.remembered.label("remembered"),
                func.row_number()
                .over(
                    partition_by=(Repetition.word_id, Repetition.direction),
                    order_by=(Repetition.practiced_at, Repetition.id),
                )
                .label("rn"),
            )
            .subquery()
        )
        stmt = select(func.avg(earliest.c.remembered)).where(earliest.c.rn == 1)
        rate = self._session.scalar(stmt)
        return default if rate is None else float(rate)

    def count_since(self, since: int) -> int:
        """Count repetition events recorded at or after ``since`` (Unix seconds).

        Counts *events*, not distinct pairs — the same (word, direction) answered
        twice counts twice. Powers the practice screen's "Today" tally.
        """
        stmt = (
            select(func.count())
            .select_from(Repetition)
            .where(Repetition.practiced_at >= since)
        )
        return int(self._session.scalar(stmt) or 0)

    def add(self, repetition: Repetition) -> None:
        """Stage ``repetition`` for insertion on the next commit."""
        self._session.add(repetition)
