"""Schemas for the chunk explorer -- browsing and searching the raw index."""
from __future__ import annotations

from pydantic import BaseModel


class ChunkOut(BaseModel):
    id: str
    doc_id: str
    filename: str
    ordinal: int
    page_no: int | None = None
    section: str | None = None
    modality: str = "text"
    token_count: int = 0
    char_count: int = 0
    text: str
    media_ids: list[str] = []
    # Populated in search mode only; None when browsing.
    score: float | None = None


class ChunkListResponse(BaseModel):
    items: list[ChunkOut]
    total: int
    limit: int
    offset: int
    # browse = index order from SQLite, search = hybrid retrieval from Qdrant
    mode: str
    query: str | None = None
    # True when the sparse (BM25) half of hybrid search took part.
    sparse_used: bool = False


class ChunkFacets(BaseModel):
    """Filter options for the explorer, derived from what is actually indexed."""

    documents: list[dict]
    modalities: list[dict]
    total_chunks: int
