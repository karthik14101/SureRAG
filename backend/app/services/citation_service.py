"""Citation lookup for the UI's citation drawer."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import models


def expand_citation(db: Session, chunk_id: str, user_id: str) -> dict | None:
    """Full chunk text plus its neighbours, so a citation can be read in context."""
    chunk = db.execute(
        select(models.Chunk).where(
            models.Chunk.id == chunk_id, models.Chunk.user_id == user_id
        )
    ).scalar_one_or_none()
    if chunk is None:
        return None

    document = db.get(models.Document, chunk.doc_id)

    neighbours = db.execute(
        select(models.Chunk)
        .where(
            models.Chunk.doc_id == chunk.doc_id,
            models.Chunk.ordinal.between(chunk.ordinal - 1, chunk.ordinal + 1),
            models.Chunk.id != chunk.id,
        )
        .order_by(models.Chunk.ordinal)
    ).scalars().all()

    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "filename": document.filename if document else "unknown",
        "page_no": chunk.page_no,
        "section": chunk.section,
        "modality": chunk.modality,
        "text": chunk.text,
        "context_before": next(
            (c.text for c in neighbours if c.ordinal < chunk.ordinal), None
        ),
        "context_after": next(
            (c.text for c in neighbours if c.ordinal > chunk.ordinal), None
        ),
        "media_ids": chunk.media_ids or [],
    }
