"""Word model — one vocabulary entry plus per-direction forgetting-curve half-lives."""
from sqlalchemy import Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseORM
from .repetition import Repetition


class Word(BaseORM):
    """One vocabulary entry — a source/target text pair plus per-direction half-lives.

    The id is database-assigned (autoincrement). Existing rows imported by the
    original migration keep their 0-based ids; new words continue from max+1.

    Instead of a precomputed due timestamp, each direction stores the half-life
    ``H`` of the forgetting curve ``R(Δt) = 2**(−Δt/H)`` emitted by the trained
    model. Recall score and the next-review time are derived from it *live* (see
    :mod:`src.model.curve`), so the recall threshold can change without
    recomputing anything. ``None`` means the half-life has not been computed yet —
    no model has been trained, or the word has no history in that direction.

    Each direction also stores the **two curves the word would have if it were
    practised now** — one for a remembered answer, one for a forgotten one. The
    practice queue needs them to rank cards by how much recall a review *adds*
    (:func:`src.model.curve.expected_gain`), and forwarding every card's two
    hypothetical histories through the model is far too slow to do at session
    start, so they are precomputed by :class:`~src.model.inference.ParamScheduler`
    alongside the current half-life. Storing the half-life rather than the
    resulting score keeps it independent of the user's threshold and horizon
    settings, which stay applied live. Unlike the current half-life these are also
    computed for a direction with *no* history, since a never-practised card still
    has to be ranked.

    Attributes:
        fwd_h: FORWARD (source→target) forgetting-curve half-life, in seconds.
        rev_h: REVERSE (target→source) forgetting-curve half-life, in seconds.
        fwd_ok_h, fwd_no_h: FORWARD half-life after a hypothetical remembered /
            forgotten answer.
        rev_ok_h, rev_no_h: the same for REVERSE.
        repetitions: All practice events for this word, in any direction.
            Cascades on delete so removing a Word also removes its history.
    """

    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    fwd_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    fwd_ok_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_no_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_ok_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_no_h: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

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
