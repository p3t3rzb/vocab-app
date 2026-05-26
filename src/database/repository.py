"""Repository classes — typed query and mutation helpers for the ORM models.

Each repository wraps a SQLAlchemy :class:`Session` and exposes the small set
of operations the rest of the app needs. Repositories never commit on their
own; callers obtain a Session via :func:`get_session`, which commits on clean
exit and rolls back on exception.
"""
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Direction, LanguagePair, Repetition, Word


class LanguagePairRepository:
    """Read-only access to the single-row ``language_pair`` table."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self) -> LanguagePair | None:
        """Return the language pair stored in this database, or ``None`` if absent."""
        return self._session.scalars(select(LanguagePair)).first()


class WordRepository:
    """CRUD helpers for :class:`Word`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, word_id: int) -> Word | None:
        """Return the word with the given id, or ``None`` if no such word exists."""
        return self._session.get(Word, word_id)

    def get_all(self) -> list[Word]:
        """Return every word, ordered by id ascending."""
        return list(self._session.scalars(select(Word).order_by(Word.id)))

    def get_count(self) -> int:
        """Return the total number of words in the database."""
        return self._session.scalar(select(func.count()).select_from(Word)) or 0

    def find_by_source_text(self, source_text: str) -> Word | None:
        """Return the word with the given ``source_text``, or ``None`` if absent."""
        return self._session.scalars(
            select(Word).where(Word.source_text == source_text).limit(1)
        ).first()

    def add(self, word: Word) -> None:
        """Stage ``word`` for insertion on the next commit."""
        self._session.add(word)

    def add_many(self, words: list[Word]) -> None:
        """Stage every word in ``words`` for insertion on the next commit."""
        self._session.add_all(words)

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


class RepetitionRepository:
    """Query and insert helpers for :class:`Repetition`."""

    def __init__(self, session: Session) -> None:
        self._session = session

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

    def get_count_for_word(self, word_id: int, direction: Direction) -> int:
        """Return the number of repetitions for a (word, direction) pair."""
        stmt = (
            select(func.count())
            .select_from(Repetition)
            .where(
                Repetition.word_id == word_id,
                Repetition.direction == int(direction),
            )
        )
        return self._session.scalar(stmt) or 0

    def add(self, repetition: Repetition) -> None:
        """Stage ``repetition`` for insertion on the next commit."""
        self._session.add(repetition)

    def add_many(self, repetitions: list[Repetition]) -> None:
        """Stage every repetition in the list for insertion on the next commit."""
        self._session.add_all(repetitions)
