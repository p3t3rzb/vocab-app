import enum

from sqlalchemy import Boolean, ForeignKey, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Direction(enum.IntEnum):
    FORWARD = 0  # source → target
    REVERSE = 1  # target → source


class LanguagePair(Base):
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
        return Direction(self.direction)

    def __repr__(self) -> str:
        return (
            f"Repetition(id={self.id!r}, word_id={self.word_id!r}, "
            f"direction={self.direction!r}, practiced_at={self.practiced_at!r}, "
            f"remembered={self.remembered!r})"
        )
