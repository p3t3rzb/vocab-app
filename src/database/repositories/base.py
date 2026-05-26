"""Shared base class for repository implementations."""
from sqlalchemy.orm import Session


class BaseRepository:
    """Holds the SQLAlchemy :class:`Session` shared by every repository subclass.

    Repositories never commit on their own; callers obtain a session via
    :func:`src.database.get_session`, which commits on clean exit and rolls
    back on exception.
    """

    def __init__(self, session: Session) -> None:
        self._session = session
