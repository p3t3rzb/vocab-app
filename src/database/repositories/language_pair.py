"""Read access for the single-row ``language_pair`` table."""
from sqlalchemy import select

from ..models import LanguagePair
from .base import BaseRepository


class LanguagePairRepository(BaseRepository):
    """Read-only access to the single-row ``language_pair`` table."""

    def get(self) -> LanguagePair | None:
        """Return the language pair stored in this database, or ``None`` if absent."""
        return self._session.scalars(select(LanguagePair)).first()
