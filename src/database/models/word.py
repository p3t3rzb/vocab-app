"""Word model — one vocabulary entry plus per-direction forgetting-curve time constants."""
from sqlalchemy import Float, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import BaseORM
from .repetition import Repetition


class Word(BaseORM):
    """One vocabulary entry — a source/target text pair plus per-direction time constants.

    The id is database-assigned (autoincrement). Existing rows imported by the
    original migration keep their 0-based ids; new words continue from max+1.

    Instead of a precomputed due timestamp, each direction stores **both**
    parameters of the forgetting curve ``R(Δt) = p0·exp(−Δt/τ)`` emitted by the
    trained model — the time constant ``τ`` and the ceiling ``p0``, which the
    model predicts per cell. They are written and read as a pair
    (:class:`~src.model.curve.Curve`): a ``τ`` beside the wrong ``p0`` names a
    different curve and is silently wrong. Recall score
    and the next-review time are derived from them *live* (see
    :mod:`src.model.curve`), so the recall threshold can change without
    recomputing anything. ``None`` means the time constant has not been computed yet —
    no model has been trained, or the word has no history in that direction.

    Each direction also stores the **two curves the word would have if it were
    practised now** — one for a remembered answer, one for a forgotten one. The
    practice queue needs them to rank cards by how much recall a review *adds*
    (:func:`src.model.curve.expected_gain`), and forwarding every card's two
    hypothetical histories through the model is far too slow to do at session
    start, so they are precomputed by :class:`~src.model.inference.ParamScheduler`
    alongside the current time constant. Storing the time constant rather than the
    resulting score keeps it independent of the user's recall-threshold setting,
    which stays applied live. Unlike the current time constant these are also
    computed for a direction with *no* history, since a never-practised card still
    has to be ranked.

    Attributes:
        fwd_tau / fwd_ceiling: FORWARD (source→target) forgetting-curve time constant
            in seconds, and the ceiling it was emitted beside.
        rev_tau / rev_ceiling: the same for REVERSE (target→source).
        fwd_ok_tau / fwd_ok_ceiling, fwd_no_tau / fwd_no_ceiling: the FORWARD curve after a
            hypothetical remembered / forgotten answer.
        rev_ok_tau / rev_ok_ceiling, rev_no_tau / rev_no_ceiling: the same for REVERSE.
        repetitions: All practice events for this word, in any direction.
            Cascades on delete so removing a Word also removes its history.
    """

    __tablename__ = "words"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    target_text: Mapped[str] = mapped_column(Text, nullable=False)
    fwd_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

    fwd_ok_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_no_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_ok_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_no_tau: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_ok_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    fwd_no_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_ok_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)
    rev_no_ceiling: Mapped[float | None] = mapped_column(Float, nullable=True, default=None)

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
