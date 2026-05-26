"""Module-level shims preserving the historical ``init_db`` / ``get_session`` API.

These wrap a single process-wide :class:`Database` instance so callers don't
need to thread one through. Tests or future multi-database scenarios can
import :class:`Database` directly from :mod:`src.database.database`.
"""
from contextlib import AbstractContextManager

from sqlalchemy.orm import Session

from .database import Database

_db = Database()


def init_db(
    database_url: str,
    source_language: str,
    target_language: str,
) -> None:
    """Initialize the process-wide :class:`Database`. See :meth:`Database.init`."""
    _db.init(database_url, source_language, target_language)


def get_session() -> AbstractContextManager[Session]:
    """Yield a session from the process-wide :class:`Database`. See :meth:`Database.session`.

    Usage::

        with get_session() as session:
            repo = WordRepository(session)
            word = repo.get_by_id(42)
    """
    return _db.session()
