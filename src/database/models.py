"""ORM model definitions.

The schema is fully described in ``CLAUDE.md``. Briefly:

* One :class:`LanguagePair` row per database (id always 1).
* One :class:`Word` row per vocabulary entry.
* One :class:`Repetition` row per practice event, scoped to a single
  :class:`Direction` (source→target or target→source).
"""
import enum

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Direction(enum.IntEnum):
    """Which way a word is being practiced.

    Stored as an integer in the database so that SQL queries can compare
    directly against :class:`Repetition.direction` without an enum cast.
    """

    FORWARD = 0  # source → target
    REVERSE = 1  # target → source


class LanguagePair(Base):
    """Single-row table identifying the languages stored in this database.

    The id is hard-coded to 1 so every database has exactly one language pair.
    """

    __tablename__ = "language_pair"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_language: Mapped[str] = mapped_column(Text, nullable=False)
    target_language: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return (
            f"LanguagePair(id={self.id!r}, "
            f"source_language={self.source_language!r}, "
            f"target_language={self.target_language!r})"
        )


class Word(Base):
    """One vocabulary entry — a source/target text pair plus due timestamps.

    The id is 0-based to match the source spreadsheet row index used by the
    initial migration.

    Attributes:
        next_rep_fwd_at: Unix timestamp at which the FORWARD direction is next
            due. ``None`` means no model has been trained yet; ``0`` means due
            immediately.
        next_rep_rev_at: Same, but for the REVERSE direction.
        repetitions: All practice events for this word, in any direction.
            Cascades on delete so removing a Word also removes its history.
    """

    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    next_rep_fwd_at: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    next_rep_rev_at: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    repetitions: Mapped[list["Repetition"]] = relationship(
        back_populates="word",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return (
            f"Word(id={self.id!r}, "
            f"source_text={self.source_text!r}, "
            f"target_text={self.target_text!r})"
        )


class Repetition(Base):
    """A single practice event for a (word, direction) pair.

    Composite index on ``(word_id, direction)`` keeps the
    "all reps for one word in one direction" lookup fast.

    Attributes:
        direction: Integer-encoded :class:`Direction`.
        practiced_at: Unix timestamp when the user attempted the word.
        remembered: Whether the user recalled the translation correctly.
    """

    __tablename__ = "repetitions"
    __table_args__ = (
        Index("ix_repetitions_word_direction", "word_id", "direction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    word_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("words.id", ondelete="CASCADE"), nullable=False
    )
    direction: Mapped[int] = mapped_column(Integer, nullable=False)
    practiced_at: Mapped[int] = mapped_column(Integer, nullable=False)
    remembered: Mapped[bool] = mapped_column(Boolean, nullable=False)

    word: Mapped["Word"] = relationship(back_populates="repetitions")

    @property
    def direction_enum(self) -> Direction:
        """Return ``direction`` typed as :class:`Direction` rather than ``int``."""
        return Direction(self.direction)

    def __repr__(self) -> str:
        return (
            f"Repetition(id={self.id!r}, word_id={self.word_id!r}, "
            f"direction={self.direction!r}, practiced_at={self.practiced_at!r}, "
            f"remembered={self.remembered!r})"
        )
