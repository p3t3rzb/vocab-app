"""Word model — one vocabulary entry plus per-direction forgetting-curve params."""
from sqlalchemy import Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseORM
from .repetition import Repetition


class Word(BaseORM):
    """One vocabulary entry — a source/target text pair plus per-direction curve params.

    The id is database-assigned (autoincrement). Existing rows imported by the
    original migration keep their 0-based ids; new words continue from max+1.

    Instead of a precomputed due timestamp, each direction stores the three
    parameters of the forgetting curve ``R(Δt) = p0·(1 + Δt/S)**(−d)`` emitted by
    the trained model. Recall score and the next-review time are derived from
    these *live* (see :mod:`src.model.curve`), so the recall threshold can change
    without recomputing anything. ``None`` means the params have not been computed
    yet — no model has been trained, or the word has no history in that direction.

    Attributes:
        fwd_p0, fwd_s, fwd_d: FORWARD (source→target) forgetting-curve params.
        rev_p0, rev_s, rev_d: REVERSE (target→source) forgetting-curve params.
        repetitions: All practice events for this word, in any direction.
            Cascades on delete so removing a Word also removes its history.
    """

    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    fwd_p0: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_s: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_d: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_p0: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_s: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_d: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

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
