"""BM25 sparse vectors: the keyword half of hybrid retrieval.

Dense vectors miss exact identifiers -- part numbers, error codes, acronyms --
because they embed meaning, not spelling. BM25 catches exactly those. The two
result lists are fused with RRF inside Qdrant.

fastembed's BM25 is a tokeniser plus IDF statistics (a few MB), not a neural
model. If it is unavailable for any reason the engine degrades to dense-only
search rather than failing.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger(__name__)

_encoder = None
_lock = threading.Lock()
_unavailable = False


@dataclass
class SparseVector:
    indices: list[int]
    values: list[float]

    @property
    def is_empty(self) -> bool:
        return not self.indices


def _load():
    global _encoder, _unavailable
    if _unavailable or not settings.sparse_enabled:
        return None
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None and not _unavailable:
            try:
                from fastembed import SparseTextEmbedding

                logger.info("Loading BM25 sparse encoder (Qdrant/bm25)")
                _encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
            except Exception as exc:  # noqa: BLE001
                _unavailable = True
                logger.warning(
                    "Sparse encoder unavailable, falling back to dense-only search: %s",
                    str(exc)[:200],
                )
                return None
    return _encoder


def is_enabled() -> bool:
    return settings.sparse_enabled and not _unavailable


def _encode_sync(texts: list[str], as_query: bool) -> list[SparseVector]:
    encoder = _load()
    if encoder is None:
        return [SparseVector([], []) for _ in texts]
    try:
        # query_embed weights terms for retrieval; embed builds document vectors.
        generator = encoder.query_embed(texts) if as_query else encoder.embed(texts)
        out: list[SparseVector] = []
        for item in generator:
            out.append(
                SparseVector(
                    indices=[int(i) for i in item.indices],
                    values=[float(v) for v in item.values],
                )
            )
        # Guard against a generator yielding fewer items than requested.
        while len(out) < len(texts):
            out.append(SparseVector([], []))
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sparse encoding failed: %s", str(exc)[:200])
        return [SparseVector([], []) for _ in texts]


async def embed_documents_sparse(texts: list[str]) -> list[SparseVector]:
    if not texts or not is_enabled():
        return [SparseVector([], []) for _ in texts]
    return await asyncio.to_thread(_encode_sync, texts, False)


async def embed_query_sparse(text: str) -> SparseVector:
    if not is_enabled():
        return SparseVector([], [])
    result = await asyncio.to_thread(_encode_sync, [text], True)
    return result[0]


def warmup() -> None:
    if not settings.sparse_enabled:
        return
    try:
        _encode_sync(["warmup"], True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Sparse warmup failed: %s", str(exc)[:200])
