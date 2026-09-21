"""Hybrid retrieval against Qdrant.

A single `query_points` call carries two prefetch branches -- dense semantic and
BM25 sparse -- fused server-side with Reciprocal Rank Fusion. One round trip,
no client-side merging, and no need to normalise two incompatible score scales.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from qdrant_client import models as qm

from app.config import settings
from app.embeddings import encoder, sparse
from app.logging_conf import get_logger
from app.vectorstore.qdrant_client import DENSE, SPARSE, get_client

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieval hit, in the shape the agent and the UI both consume."""

    chunk_id: str
    doc_id: str
    kb_id: str
    filename: str
    text: str
    score: float
    page_no: int | None = None
    section: str | None = None
    modality: str = "text"
    ordinal: int = 0
    media_ids: list[str] = field(default_factory=list)
    # vector | graph | expand -- shown on the citation chip
    source: str = "vector"

    @property
    def location(self) -> str:
        if self.page_no:
            return "{} p.{}".format(self.filename, self.page_no)
        if self.section:
            return "{} > {}".format(self.filename, self.section)
        return self.filename

    def snippet(self, limit: int = 320) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _tenant_filter(
    user_id: str,
    kb_id: str,
    exclude_ids: list[str] | None = None,
    doc_id: str | None = None,
    modality: str | None = None,
) -> qm.Filter:
    must: list = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(key="kb_id", match=qm.MatchValue(value=kb_id)),
    ]
    # Optional narrowing, used by the chunk explorer. The agent never sets these.
    if doc_id:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
    if modality:
        must.append(qm.FieldCondition(key="modality", match=qm.MatchValue(value=modality)))

    must_not: list = []
    if exclude_ids:
        must_not.append(qm.HasIdCondition(has_id=list(exclude_ids)))
    return qm.Filter(must=must, must_not=must_not or None)


def _to_chunk(point, source: str = "vector") -> RetrievedChunk:
    payload = point.payload or {}
    # `query_points` returns ScoredPoint (has .score); `retrieve` returns Record
    # (no .score at all), so this must not assume the attribute exists.
    raw_score = getattr(point, "score", None)
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id") or point.id),
        doc_id=str(payload.get("doc_id") or ""),
        kb_id=str(payload.get("kb_id") or ""),
        filename=str(payload.get("filename") or "unknown"),
        text=str(payload.get("text") or ""),
        score=float(raw_score) if raw_score is not None else 0.0,
        page_no=payload.get("page_no"),
        section=payload.get("section"),
        modality=str(payload.get("modality") or "text"),
        ordinal=int(payload.get("ordinal") or 0),
        media_ids=list(payload.get("media_ids") or []),
        source=source,
    )


async def hybrid_search(
    query: str,
    *,
    user_id: str,
    kb_id: str,
    top_k: int | None = None,
    exclude_ids: list[str] | None = None,
    source_label: str = "vector",
    doc_id: str | None = None,
    modality: str | None = None,
) -> list[RetrievedChunk]:
    """Dense + sparse retrieval fused with RRF, scoped to one knowledge base."""
    limit = top_k or settings.retrieval_top_k
    client = get_client()
    query_filter = _tenant_filter(user_id, kb_id, exclude_ids, doc_id, modality)

    dense_vector = await encoder.embed_query(query)
    sparse_vector = await sparse.embed_query_sparse(query)

    # Over-fetch in each branch so fusion has enough candidates to reorder.
    prefetch_limit = max(limit * 2, 20)

    try:
        if sparse_vector.is_empty:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                query=dense_vector,
                using=DENSE,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=[
                    qm.Prefetch(
                        query=dense_vector,
                        using=DENSE,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(
                            indices=sparse_vector.indices, values=sparse_vector.values
                        ),
                        using=SPARSE,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [_to_chunk(p, source_label) for p in response.points]
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant hybrid search failed: %s", str(exc)[:300])
        raise


async def multi_query_search(
    queries: list[str],
    *,
    user_id: str,
    kb_id: str,
    top_k_each: int = 6,
    exclude_ids: list[str] | None = None,
    source_label: str = "expand",
) -> list[RetrievedChunk]:
    """Run several expansion queries and merge, keeping each chunk's best score."""
    merged: dict[str, RetrievedChunk] = {}
    for query in queries:
        if not query.strip():
            continue
        try:
            hits = await hybrid_search(
                query,
                user_id=user_id,
                kb_id=kb_id,
                top_k=top_k_each,
                exclude_ids=exclude_ids,
                source_label=source_label,
            )
        except Exception:  # noqa: BLE001 - one bad expansion must not kill the turn
            continue
        for hit in hits:
            existing = merged.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                merged[hit.chunk_id] = hit
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


async def fetch_chunks_by_id(
    chunk_ids: list[str], *, user_id: str, kb_id: str, source_label: str = "graph"
) -> list[RetrievedChunk]:
    """Load specific chunks by id -- used after a graph traversal picks them out."""
    if not chunk_ids:
        return []
    client = get_client()
    try:
        records = await client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=list(dict.fromkeys(chunk_ids))[:64],
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant retrieve by id failed: %s", str(exc)[:200])
        return []

    out: list[RetrievedChunk] = []
    for record in records:
        payload = record.payload or {}
        # Retrieval by id bypasses filters, so re-check ownership here.
        if payload.get("user_id") != user_id or payload.get("kb_id") != kb_id:
            continue
        chunk = _to_chunk(record, source_label)
        # Graph hits carry no similarity score; give them a neutral prior so
        # they can be ranked alongside vector hits without dominating.
        chunk.score = 0.5
        out.append(chunk)
    return out


async def apply_reranking(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder with the local cross-encoder when RERANK_ENABLED=true."""
    if not settings.rerank_enabled or len(chunks) < 2:
        return chunks
    scores = await encoder.rerank_scores(query, [c.text for c in chunks])
    if scores is None:
        return chunks
    for chunk, score in zip(chunks, scores):
        chunk.score = float(score)
    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    return ranked[: settings.rerank_top_n] if settings.rerank_top_n else ranked


def deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop repeats, preferring the higher-scoring copy of each chunk."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda c: c.score, reverse=True)
