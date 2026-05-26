"""LanguagePair model — single-row table identifying the database's languages."""
from sqlalchemy import Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..base import Base


class LanguagePair(Base):
    """Single-row table identifying the languages stored in this database.

    Every database has exactly one language pair row, pinned to
    :attr:`SINGLETON_ID`.
    """

    __tablename__ = "language_pair"

    SINGLETON_ID: int = 1

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_language: Mapped[str] = mapped_column(Text, nullable=False)
    target_language: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return (
            f"LanguagePair(id={self.id!r}, "
            f"source_language={self.source_language!r}, "
            f"target_language={self.target_language!r})"
        )
