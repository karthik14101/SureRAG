"""Chunk explorer endpoints.

Exposes the retrieval layer directly: browse exactly what was indexed, or run
the same hybrid dense+BM25 search the agent uses and see the fusion scores.
Useful for verifying chunking quality and for diagnosing a disappointing answer.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.core.deps import CurrentUser, DbSession, client_key, get_owned_kb
from app.core.errors import NotFoundError
from app.core.rate_limit import chat_limiter
from app.schemas.chunk import ChunkFacets, ChunkListResponse
from app.services import chunk_service

router = APIRouter(tags=["chunks"])


@router.get("/kb/{kb_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    key: Annotated[str, Depends(client_key)],
    q: str | None = Query(default=None, max_length=500, description="Hybrid search query"),
    doc_id: str | None = Query(default=None),
    modality: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ChunkListResponse:
    """Browse the index in order, or search it when `q` is supplied."""
    kb = get_owned_kb(kb_id, db, user)

    query = (q or "").strip()
    if query:
        # Searching embeds the query, so it carries real CPU cost -- rate limit it
        # the same way chat is limited.
        chat_limiter.check(key)
        return await chunk_service.search(
            db, kb.id, user.id, query, doc_id=doc_id, modality=modality, limit=limit
        )

    return chunk_service.browse(
        db, kb.id, user.id, doc_id=doc_id, modality=modality, limit=limit, offset=offset
    )


@router.get("/kb/{kb_id}/chunks/facets", response_model=ChunkFacets)
def chunk_facets(kb_id: str, db: DbSession, user: CurrentUser) -> ChunkFacets:
    """Documents and modalities present in the index, with their chunk counts."""
    kb = get_owned_kb(kb_id, db, user)
    return chunk_service.facets(db, kb.id, user.id)


@router.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, db: DbSession, user: CurrentUser) -> dict:
    """One chunk with its neighbours, for reading a passage in context."""
    from app.services import citation_service

    result = citation_service.expand_citation(db, chunk_id, user.id)
    if result is None:
        raise NotFoundError("That chunk no longer exists.")
    return result
