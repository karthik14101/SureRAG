"""Chunk explorer: browse the index in order, or query it directly.

Two distinct modes, deliberately kept separate:

  browse  -- reads SQLite in document/ordinal order. This is the ground truth of
             what was indexed, so it is the right thing to show when someone asks
             "what did the chunker actually do to my PDF?"

  search  -- runs the same hybrid dense+BM25 retrieval the agent uses, and
             returns the fusion scores. This is the retrieval layer with the
             agent taken off the top, which makes it genuinely useful for
             debugging a bad answer.
"""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import models
from app.embeddings import sparse
from app.logging_conf import get_logger
from app.schemas.chunk import ChunkFacets, ChunkListResponse, ChunkOut
from app.vectorstore import search as vsearch

logger = get_logger(__name__)

MAX_LIMIT = 100
SEARCH_LIMIT = 40


def _filename_map(db: Session, kb_id: str, user_id: str) -> dict[str, str]:
    rows = db.execute(
        select(models.Document.id, models.Document.filename).where(
            models.Document.kb_id == kb_id, models.Document.user_id == user_id
        )
    ).all()
    return {row[0]: row[1] for row in rows}


def browse(
    db: Session,
    kb_id: str,
    user_id: str,
    *,
    doc_id: str | None = None,
    modality: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ChunkListResponse:
    limit = max(1, min(MAX_LIMIT, limit))
    offset = max(0, offset)

    conditions = [models.Chunk.kb_id == kb_id, models.Chunk.user_id == user_id]
    if doc_id:
        conditions.append(models.Chunk.doc_id == doc_id)
    if modality:
        conditions.append(models.Chunk.modality == modality)

    total = db.execute(
        select(func.count(models.Chunk.id)).where(*conditions)
    ).scalar_one()

    rows = db.execute(
        select(models.Chunk)
        .where(*conditions)
        .order_by(models.Chunk.doc_id, models.Chunk.ordinal)
        .limit(limit)
        .offset(offset)
    ).scalars().all()

    names = _filename_map(db, kb_id, user_id)

    items = [
        ChunkOut(
            id=row.id,
            doc_id=row.doc_id,
            filename=names.get(row.doc_id, "unknown"),
            ordinal=row.ordinal,
            page_no=row.page_no,
            section=row.section,
            modality=row.modality,
            token_count=row.token_count,
            char_count=len(row.text or ""),
            text=row.text or "",
            media_ids=list(row.media_ids or []),
            score=None,
        )
        for row in rows
    ]

    return ChunkListResponse(
        items=items,
        total=int(total or 0),
        limit=limit,
        offset=offset,
        mode="browse",
    )


async def search(
    db: Session,
    kb_id: str,
    user_id: str,
    query: str,
    *,
    doc_id: str | None = None,
    modality: str | None = None,
    limit: int = SEARCH_LIMIT,
) -> ChunkListResponse:
    limit = max(1, min(MAX_LIMIT, limit))

    hits = await vsearch.hybrid_search(
        query,
        user_id=user_id,
        kb_id=kb_id,
        top_k=limit,
        doc_id=doc_id,
        modality=modality,
    )

    names = _filename_map(db, kb_id, user_id)

    # Token counts live in SQLite, not the Qdrant payload; fetch them in one go
    # so the explorer can show the same numbers the ingestion pipeline recorded.
    token_counts: dict[str, int] = {}
    if hits:
        rows = db.execute(
            select(models.Chunk.id, models.Chunk.token_count).where(
                models.Chunk.id.in_([h.chunk_id for h in hits])
            )
        ).all()
        token_counts = {row[0]: row[1] for row in rows}

    items = [
        ChunkOut(
            id=hit.chunk_id,
            doc_id=hit.doc_id,
            filename=names.get(hit.doc_id, hit.filename or "unknown"),
            ordinal=hit.ordinal,
            page_no=hit.page_no,
            section=hit.section,
            modality=hit.modality,
            token_count=token_counts.get(hit.chunk_id, 0),
            char_count=len(hit.text or ""),
            text=hit.text or "",
            media_ids=list(hit.media_ids or []),
            score=round(float(hit.score), 5),
        )
        for hit in hits
    ]

    return ChunkListResponse(
        items=items,
        total=len(items),
        limit=limit,
        offset=0,
        mode="search",
        query=query,
        sparse_used=sparse.is_enabled(),
    )


def facets(db: Session, kb_id: str, user_id: str) -> ChunkFacets:
    """Filter options built from what is actually in the index, not a fixed list."""
    per_doc = db.execute(
        select(models.Chunk.doc_id, func.count(models.Chunk.id))
        .where(models.Chunk.kb_id == kb_id, models.Chunk.user_id == user_id)
        .group_by(models.Chunk.doc_id)
    ).all()

    names = _filename_map(db, kb_id, user_id)
    documents = [
        {"id": doc_id, "filename": names.get(doc_id, "unknown"), "chunks": int(count)}
        for doc_id, count in per_doc
    ]
    documents.sort(key=lambda d: d["filename"].casefold())

    per_modality = db.execute(
        select(models.Chunk.modality, func.count(models.Chunk.id))
        .where(models.Chunk.kb_id == kb_id, models.Chunk.user_id == user_id)
        .group_by(models.Chunk.modality)
    ).all()
    modalities = sorted(
        ({"name": name or "text", "chunks": int(count)} for name, count in per_modality),
        key=lambda m: m["chunks"],
        reverse=True,
    )

    return ChunkFacets(
        documents=documents,
        modalities=modalities,
        total_chunks=sum(d["chunks"] for d in documents),
    )
