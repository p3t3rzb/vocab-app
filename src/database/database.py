"""Engine + session lifecycle, encapsulated in a single class.

The free-function public API (``init_db`` / ``get_session``) in
:mod:`src.database.session` delegates to a module-level :class:`Database`
singleton so existing call sites stay unchanged. The class is also importable
directly for tests or future multi-database scenarios.
"""
import threading
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

from .base import Base
from .models import LanguagePair


class Database:
    """Owns one SQLAlchemy engine + session factory; safe to swap at runtime.

    A single lock guards reassignment of the engine / session factory so a
    background :meth:`init` call (e.g. the settings recalc worker iterating
    across DBs) cannot swap the factory out from under a caller that is
    mid-``with db.session()``.
    """

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._lock = threading.Lock()

    def init(
        self,
        database_url: str,
        source_language: str,
        target_language: str,
    ) -> None:
        """Initialize the engine, create all tables, and seed the language pair row.

        Must be called once at application startup before any repository use.
        Calling it again with a different URL replaces the active engine; the
        previous engine is disposed.

        Also performs an idempotent migration: if an older database is missing
        the per-direction ``next_rep_fwd_at`` / ``next_rep_rev_at`` columns,
        they are added via ``ALTER TABLE``.

        Args:
            database_url: SQLAlchemy URL such as
                ``sqlite:///storage/french_polish.db``.
            source_language: Human-readable name (e.g. ``"French"``) — used
                only when seeding a brand-new database.
            target_language: Same, for the target language (e.g. ``"Polish"``).
        """
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(engine)
        factory = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

        self._run_migrations(engine)
        self._seed_language_pair(factory, source_language, target_language)
        self._swap(engine, factory)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Yield a :class:`Session`, committing on clean exit / rolling back on exception.

        The session uses ``expire_on_commit=False`` so ORM objects remain
        readable after the ``with`` block ends — important for the GUI which
        holds word and repetition objects across screen redraws.

        Raises:
            RuntimeError: if :meth:`init` has not been called yet.
        """
        with self._lock:
            factory = self._session_factory
        if factory is None:
            raise RuntimeError("Database not initialized. Call init_db() first.")
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _run_migrations(engine: Engine) -> None:
        """Add per-direction next-rep columns to pre-existing databases."""
        with engine.connect() as conn:
            cols = [row[1] for row in conn.execute(text("PRAGMA table_info(words)"))]
            added = False
            for col_name in ("next_rep_fwd_at", "next_rep_rev_at"):
                if col_name not in cols:
                    conn.execute(text(f"ALTER TABLE words ADD COLUMN {col_name} INTEGER"))
                    added = True
            if added:
                conn.commit()

    @staticmethod
    def _seed_language_pair(
        factory: sessionmaker[Session],
        source_language: str,
        target_language: str,
    ) -> None:
        """Insert the singleton ``LanguagePair`` row if it isn't already there."""
        with factory() as session:
            existing = session.scalars(select(LanguagePair)).first()
            if existing is None:
                session.add(
                    LanguagePair(
                        id=LanguagePair.SINGLETON_ID,
                        source_language=source_language,
                        target_language=target_language,
                    )
                )
                session.commit()

    def _swap(self, engine: Engine, factory: sessionmaker[Session]) -> None:
        """Atomically install ``engine`` / ``factory`` and dispose the old engine."""
        with self._lock:
            old_engine = self._engine
            self._engine = engine
            self._session_factory = factory
        if old_engine is not None:
            old_engine.dispose()
