"""Repetition model — one practice event for a (word, direction) pair."""

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, ForeignKey, Index, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseORM

if TYPE_CHECKING:
    from .word import Word


class Repetition(BaseORM):
    """A single practice event for a (word, direction) pair.

    Composite index on ``(word_id, direction)`` keeps the
    "all reps for one word in one direction" lookup fast.

    Attributes:
        direction: Integer-encoded :class:`Direction`.
        practiced_at: Unix timestamp when the user attempted the word.
        remembered: Whether the user recalled the translation correctly.
    """

    __tablename__ = "repetitions"
    __table_args__ = (Index("ix_repetitions_word_direction", "word_id", "direction"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    word_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    direction: Mapped[int] = mapped_column(Integer, nullable=False)
    practiced_at: Mapped[int] = mapped_column(Integer, nullable=False)
    remembered: Mapped[bool] = mapped_column(Boolean, nullable=False)

    word: Mapped["Word"] = relationship(back_populates="repetitions")

    def __repr__(self) -> str:
        return (
            f"Repetition(id={self.id!r}, word_id={self.word_id!r}, "
            f"direction={self.direction!r}, practiced_at={self.practiced_at!r}, "
            f"remembered={self.remembered!r})"
        )
