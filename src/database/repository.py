from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import Direction, LanguagePair, Repetition, Word


class LanguagePairRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self) -> LanguagePair | None:
        return self._session.scalars(select(LanguagePair)).first()


class WordRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_by_id(self, word_id: int) -> Word | None:
        return self._session.get(Word, word_id)

    def get_all(self) -> list[Word]:
        return list(self._session.scalars(select(Word).order_by(Word.id)))

    def get_count(self) -> int:
        return self._session.scalar(select(func.count()).select_from(Word)) or 0

    def add(self, word: Word) -> None:
        self._session.add(word)

    def add_many(self, words: list[Word]) -> None:
        self._session.add_all(words)

    def delete(self, word: Word) -> None:
        self._session.delete(word)

    def get_next_id(self) -> int:
        max_id = self._session.scalar(select(func.max(Word.id)))
        return (max_id + 1) if max_id is not None else 0


class RepetitionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def get_for_word(self, word_id: int, direction: Direction) -> list[Repetition]:
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
        self._session.add(repetition)

    def add_many(self, repetitions: list[Repetition]) -> None:
        self._session.add_all(repetitions)
