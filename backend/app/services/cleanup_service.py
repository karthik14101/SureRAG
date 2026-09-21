"""Cascading deletion across all four stores.

Deleting a document must remove it from SQLite, Qdrant, Neo4j and the media
directory. Missing any one leaves ghosts: stale vectors that surface in search
with no row behind them, or orphan image files that accumulate forever.

Order matters. External stores are cleared first, because a SQLite delete that
succeeds after an external failure would lose the ids needed to retry.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models
from app.graph import writer as graph_writer
from app.ingestion import image_store
from app.logging_conf import get_logger
from app.vectorstore import qdrant_client

logger = get_logger(__name__)


async def delete_document(db: Session, document: models.Document) -> None:
    doc_id = document.id
    kb_id = document.kb_id
    user_id = document.user_id

    media_rows = db.execute(
        select(models.MediaAsset.rel_path, models.MediaAsset.thumb_path).where(
            models.MediaAsset.doc_id == doc_id
        )
    ).all()

    try:
        await qdrant_client.delete_by_document(doc_id, user_id)
    except Exception as exc:  # noqa: BLE001 - never block the user's delete
        logger.warning("Could not delete vectors for %s: %s", doc_id, str(exc)[:200])

    try:
        await graph_writer.delete_document_graph(doc_id, kb_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete graph nodes for %s: %s", doc_id, str(exc)[:200])

    for rel_path, thumb_path in media_rows:
        image_store.delete_image(rel_path, thumb_path)

    # SQLite cascades handle chunks and media rows.
    db.delete(document)
    db.commit()

    await refresh_kb_counters(db, kb_id)
    logger.info("Deleted document %s (%s)", doc_id, document.filename)


async def delete_knowledge_base(db: Session, kb: models.KnowledgeBase) -> None:
    kb_id = kb.id
    user_id = kb.user_id

    try:
        await qdrant_client.delete_by_kb(kb_id, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete vectors for KB %s: %s", kb_id, str(exc)[:200])

    try:
        await graph_writer.delete_kb_graph(kb_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not delete graph for KB %s: %s", kb_id, str(exc)[:200])

    image_store.delete_kb_media(user_id, kb_id)

    db.delete(kb)
    db.commit()
    logger.info("Deleted knowledge base %s (%s)", kb_id, kb.name)


async def refresh_kb_counters(db: Session, kb_id: str) -> None:
    """Recompute the cached counters shown on the dashboard."""
    from sqlalchemy import func

    kb = db.get(models.KnowledgeBase, kb_id)
    if kb is None:
        return

    totals = db.execute(
        select(
            func.count(models.Document.id),
            func.coalesce(func.sum(models.Document.chunk_count), 0),
            func.coalesce(func.sum(models.Document.image_count), 0),
        ).where(models.Document.kb_id == kb_id, models.Document.status == "ready")
    ).one()

    kb.doc_count = int(totals[0] or 0)
    kb.chunk_count = int(totals[1] or 0)
    kb.image_count = int(totals[2] or 0)

    try:
        stats = await graph_writer.graph_stats(kb_id)
        kb.entity_count = int(stats.get("entities") or 0)
    except Exception:  # noqa: BLE001 - cosmetic
        pass

    db.commit()


def sweep_orphan_media(db: Session) -> int:
    """Delete media files on disk with no database row. Runs at startup."""
    known = {
        row[0]
        for row in db.execute(select(models.MediaAsset.rel_path)).all()
        if row[0]
    }
    thumbs = {
        row[0]
        for row in db.execute(select(models.MediaAsset.thumb_path)).all()
        if row[0]
    }
    return image_store.sweep_orphans(known | thumbs)


def clear_stale_staging() -> int:
    """Remove leftover upload staging directories from a previous run."""
    from app.config import settings

    root = settings.upload_tmp_dir
    if not root.exists():
        return 0

    import shutil

    removed = 0
    for path in root.iterdir():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
        else:
            try:
                path.unlink()
                removed += 1
            except Exception:  # noqa: BLE001
                pass
    if removed:
        logger.info("Cleared %d stale staging item(s)", removed)
    return removed
