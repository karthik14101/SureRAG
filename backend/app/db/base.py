"""SQLAlchemy engine/session setup tuned for concurrent SQLite access.

The ingestion worker and the API request handlers write to the same database
file from different threads, so WAL journalling and a busy timeout are not
optional here -- without them you get intermittent "database is locked" errors.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    """Declarative base for every ORM model."""


engine: Engine = create_engine(
    settings.sqlite_url,
    echo=False,
    future=True,
    connect_args={
        # FastAPI hands requests to a threadpool; the worker has its own thread.
        "check_same_thread": False,
        # Wait up to 30s for a writer lock instead of raising immediately.
        "timeout": 30,
    },
)


@event.listens_for(engine, "connect")
def _configure_sqlite(dbapi_connection, _connection_record) -> None:
    """Apply per-connection pragmas. Runs for every new pooled connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")       # concurrent readers + 1 writer
    cursor.execute("PRAGMA synchronous=NORMAL")     # safe with WAL, much faster
    cursor.execute("PRAGMA foreign_keys=ON")        # enforce ON DELETE CASCADE
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for background work outside the request cycle."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
