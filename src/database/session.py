from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, Engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from .base import Base
from .models import LanguagePair

_engine: Engine | None = None
_SessionFactory: sessionmaker[Session] | None = None


def init_db(
    database_url: str,
    source_language: str,
    target_language: str,
) -> None:
    """Initialize the engine, create all tables, and seed the language pair row.

    Must be called once at application startup before any repository use.
    Calling it again with a different URL replaces the active engine.
    """
    global _engine, _SessionFactory

    _engine = create_engine(
        database_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    Base.metadata.create_all(_engine)
    _SessionFactory = sessionmaker(bind=_engine, autocommit=False, autoflush=False, expire_on_commit=False)

    with _engine.connect() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(words)"))]
        added = False
        for col_name in ("next_rep_fwd_at", "next_rep_rev_at"):
            if col_name not in cols:
                conn.execute(text(f"ALTER TABLE words ADD COLUMN {col_name} INTEGER"))
                added = True
        if added:
            conn.commit()

    with _SessionFactory() as session:
        existing = session.scalars(select(LanguagePair)).first()
        if existing is None:
            session.add(LanguagePair(id=1, source_language=source_language, target_language=target_language))
            session.commit()


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy Session, committing on clean exit and rolling back on exception.

    Usage:
        with get_session() as session:
            repo = WordRepository(session)
            word = repo.get_by_id(42)
    """
    if _SessionFactory is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
