"""Qdrant connection, collection bootstrap and write path.

One collection holds every user's chunks, isolated by payload filters rather
than by collection-per-user. That keeps a single well-built HNSW graph (better
recall, flat memory) while `kb_id` carries `is_tenant=True` so Qdrant stores each
knowledge base's points contiguously on disk.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models as qm

from app.config import settings
from app.core.errors import UpstreamError
from app.embeddings.sparse import SparseVector
from app.logging_conf import get_logger

logger = get_logger(__name__)

DENSE = "dense"
SPARSE = "bm25"

_client: AsyncQdrantClient | None = None
_lock = threading.Lock()


@dataclass
class ChunkPoint:
    """Everything needed to index and later display one chunk."""

    chunk_id: str
    doc_id: str
    kb_id: str
    user_id: str
    filename: str
    text: str
    ordinal: int
    page_no: int | None = None
    section: str | None = None
    modality: str = "text"
    media_ids: list[str] | None = None
    dense: list[float] | None = None
    sparse: SparseVector | None = None


def get_client() -> AsyncQdrantClient:
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            _client = AsyncQdrantClient(
                url=settings.qdrant_url,
                timeout=60,
                prefer_grpc=False,
            )
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        try:
            await _client.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        _client = None


async def ensure_collection() -> None:
    """Create the collection and payload indexes if they do not exist."""
    client = get_client()
    name = settings.qdrant_collection

    try:
        exists = await client.collection_exists(collection_name=name)
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            "Cannot reach Qdrant at {}. Is `docker compose up -d` running? ({})".format(
                settings.qdrant_url, str(exc)[:160]
            )
        ) from exc

    if not exists:
        await client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE: qm.VectorParams(
                    size=settings.embedding_dim,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                # IDF modifier makes Qdrant apply BM25 inverse-document-frequency
                # weighting at query time using corpus-wide statistics.
                SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)
            },
        )
        logger.info("Created Qdrant collection '%s' (dim=%d)", name, settings.embedding_dim)
    else:
        # Guard against a dimension change against an existing collection.
        info = await client.get_collection(collection_name=name)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict) and DENSE in vectors:
            existing_dim = vectors[DENSE].size
            if existing_dim != settings.embedding_dim:
                raise UpstreamError(
                    "Qdrant collection '{}' has {}-dim vectors but the configured "
                    "embedding model produces {}. Run `docker compose down -v` and "
                    "re-index.".format(name, existing_dim, settings.embedding_dim)
                )

    for field, schema in (
        ("user_id", qm.PayloadSchemaType.KEYWORD),
        ("doc_id", qm.PayloadSchemaType.KEYWORD),
        ("modality", qm.PayloadSchemaType.KEYWORD),
    ):
        try:
            await client.create_payload_index(
                collection_name=name, field_name=field, field_schema=schema
            )
        except Exception:  # noqa: BLE001 - already exists
            pass

    try:
        await client.create_payload_index(
            collection_name=name,
            field_name="kb_id",
            field_schema=qm.KeywordIndexParams(
                type=qm.KeywordIndexType.KEYWORD,
                is_tenant=True,
            ),
        )
    except Exception:  # noqa: BLE001 - already exists
        pass


def _to_point(point: ChunkPoint) -> qm.PointStruct:
    vectors: dict = {}
    if point.dense:
        vectors[DENSE] = point.dense
    if point.sparse and not point.sparse.is_empty:
        vectors[SPARSE] = qm.SparseVector(
            indices=point.sparse.indices, values=point.sparse.values
        )
    return qm.PointStruct(
        id=point.chunk_id,
        vector=vectors,
        payload={
            "chunk_id": point.chunk_id,
            "doc_id": point.doc_id,
            "kb_id": point.kb_id,
            "user_id": point.user_id,
            "filename": point.filename,
            "text": point.text,
            "ordinal": point.ordinal,
            "page_no": point.page_no,
            "section": point.section,
            "modality": point.modality,
            "media_ids": point.media_ids or [],
        },
    )


async def upsert_chunks(points: list[ChunkPoint], batch_size: int = 64) -> int:
    """Index chunks. Point ids are chunk ids, so re-ingesting overwrites cleanly."""
    if not points:
        return 0
    client = get_client()
    written = 0
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        await client.upsert(
            collection_name=settings.qdrant_collection,
            points=[_to_point(p) for p in batch],
            wait=True,
        )
        written += len(batch)
    return written


async def delete_by_document(doc_id: str, user_id: str) -> None:
    client = get_client()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                ]
            )
        ),
        wait=True,
    )


async def delete_by_kb(kb_id: str, user_id: str) -> None:
    client = get_client()
    await client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=qm.FilterSelector(
            filter=qm.Filter(
                must=[
                    qm.FieldCondition(key="kb_id", match=qm.MatchValue(value=kb_id)),
                    qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
                ]
            )
        ),
        wait=True,
    )


async def count_chunks(kb_id: str, user_id: str) -> int:
    client = get_client()
    result = await client.count(
        collection_name=settings.qdrant_collection,
        count_filter=qm.Filter(
            must=[
                qm.FieldCondition(key="kb_id", match=qm.MatchValue(value=kb_id)),
                qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
            ]
        ),
        exact=True,
    )
    return result.count


async def health() -> tuple[bool, str]:
    try:
        client = get_client()
        collections = await client.get_collections()
        names = [c.name for c in collections.collections]
        return True, "collections: {}".format(", ".join(names) or "none")
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:200]
