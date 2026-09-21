"""Table creation and the embedding-compatibility guard.

Called once from the FastAPI lifespan handler. Safe to run repeatedly.
"""
from __future__ import annotations

from sqlalchemy import select

from app.config import settings
from app.db import models  # noqa: F401  (import registers every model on Base)
from app.db.base import Base, engine, session_scope
from app.logging_conf import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = "1"


class EmbeddingMismatchError(RuntimeError):
    """Raised when .env asks for a different embedding model than the stored index."""


def create_tables() -> None:
    Base.metadata.create_all(bind=engine)
    logger.info("SQLite schema ready at %s", settings.sqlite_path)


def _get_meta(db, key: str) -> str | None:
    row = db.execute(select(models.AppMeta).where(models.AppMeta.key == key)).scalar_one_or_none()
    return row.value if row else None


def _set_meta(db, key: str, value: str) -> None:
    row = db.execute(select(models.AppMeta).where(models.AppMeta.key == key)).scalar_one_or_none()
    if row is None:
        db.add(models.AppMeta(key=key, value=value))
    else:
        row.value = value


def verify_embedding_compatibility() -> None:
    """Refuse to start if the embedding model changed under an existing index.

    Swapping models silently is the single nastiest failure mode in a RAG stack:
    old vectors stay in Qdrant, new queries are embedded differently, and search
    quietly returns garbage. We fail loudly with the fix instead.
    """
    with session_scope() as db:
        stored_model = _get_meta(db, "embedding_model")
        stored_dim = _get_meta(db, "embedding_dim")

        if stored_model is None:
            _set_meta(db, "embedding_model", settings.embedding_model)
            _set_meta(db, "embedding_dim", str(settings.embedding_dim))
            _set_meta(db, "schema_version", SCHEMA_VERSION)
            logger.info("Recorded embedding fingerprint: %s", settings.embedding_model)
            return

        has_data = db.execute(select(models.Chunk.id).limit(1)).first() is not None

        if stored_model == settings.embedding_model and stored_dim == str(settings.embedding_dim):
            return

        if not has_data:
            # Nothing indexed yet, so adopting the new model is harmless.
            _set_meta(db, "embedding_model", settings.embedding_model)
            _set_meta(db, "embedding_dim", str(settings.embedding_dim))
            logger.info("Embedding model updated to %s (index was empty)", settings.embedding_model)
            return

        raise EmbeddingMismatchError(
            "\n"
            + "=" * 78
            + "\n  EMBEDDING MODEL MISMATCH -- refusing to start\n"
            + "=" * 78
            + "\n"
            + "  Indexed with : {} (dim {})\n".format(stored_model, stored_dim)
            + "  .env asks for: {} (dim {})\n\n".format(
                settings.embedding_model, settings.embedding_dim
            )
            + "  Existing vectors are not comparable to new ones, so search would\n"
            + "  silently return wrong results.\n\n"
            + "  Fix by EITHER restoring the old values in .env, OR wiping the index:\n"
            + "    1. docker compose down -v\n"
            + "    2. delete data/sure.db\n"
            + "    3. docker compose up -d  and re-upload your documents\n"
            + "=" * 78
        )


def init_database() -> None:
    create_tables()
    verify_embedding_compatibility()
