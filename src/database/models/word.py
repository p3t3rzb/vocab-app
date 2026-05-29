"""Word model — one vocabulary entry plus per-direction due timestamps."""
from sqlalchemy import Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base
from .repetition import Repetition


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

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    next_rep_fwd_at: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)
    next_rep_rev_at: Mapped[int | None] = mapped_column(Integer, nullable=True, default=None)

    repetitions: Mapped[list[Repetition]] = relationship(
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
