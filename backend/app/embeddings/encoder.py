"""Local sentence-transformers embeddings and the optional cross-encoder reranker.

Everything here is free and runs on CPU. The models are loaded lazily on first
use and held for the process lifetime; encoding is pushed onto a worker thread so
the asyncio event loop keeps serving requests while the CPU crunches.
"""
from __future__ import annotations

import asyncio
import threading
import time

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger(__name__)

_model = None
_model_lock = threading.Lock()
_reranker = None
_reranker_lock = threading.Lock()


def _load_model():
    """Import torch/sentence-transformers lazily: they add ~5s to startup."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            started = time.perf_counter()
            logger.info(
                "Loading embedding model %s on %s (first run downloads ~90MB)",
                settings.embedding_model,
                settings.embedding_device,
            )
            model = SentenceTransformer(
                settings.embedding_model, device=settings.embedding_device
            )
            actual_dim = model.get_sentence_embedding_dimension()
            if actual_dim != settings.embedding_dim:
                raise RuntimeError(
                    "EMBEDDING_DIM is {} in .env but {} produces {}-dim vectors. "
                    "Fix EMBEDDING_DIM and re-index.".format(
                        settings.embedding_dim, settings.embedding_model, actual_dim
                    )
                )
            _model = model
            logger.info(
                "Embedding model ready in %.1fs (dim=%d)",
                time.perf_counter() - started,
                actual_dim,
            )
    return _model


def _encode_sync(texts: list[str], is_query: bool) -> list[list[float]]:
    model = _load_model()
    # normalize_embeddings=True lets cosine similarity reduce to a dot product,
    # which is what Qdrant's COSINE distance expects for best performance.
    vectors = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return [v.tolist() for v in vectors]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed document chunks. Returns one vector per input, order preserved."""
    if not texts:
        return []
    return await asyncio.to_thread(_encode_sync, texts, False)


async def embed_query(text: str) -> list[float]:
    vectors = await asyncio.to_thread(_encode_sync, [text], True)
    return vectors[0]


async def embed_queries(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    return await asyncio.to_thread(_encode_sync, texts, True)


def warmup() -> None:
    """Load the model at startup so the first user query is not slow."""
    try:
        _load_model()
        _encode_sync(["warmup"], True)
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail here
        logger.error("Embedding warmup failed: %s", exc)


# ---------------------------------------------------------------------------
# Optional reranker
# ---------------------------------------------------------------------------
def _load_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    with _reranker_lock:
        if _reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading reranker %s (~80MB on first run)", settings.rerank_model)
            _reranker = CrossEncoder(settings.rerank_model, device=settings.embedding_device)
    return _reranker


def _rerank_sync(query: str, passages: list[str]) -> list[float]:
    model = _load_reranker()
    pairs = [(query, passage) for passage in passages]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    return [float(s) for s in scores]


async def rerank_scores(query: str, passages: list[str]) -> list[float] | None:
    """Cross-encoder relevance scores, or None if reranking is off/unavailable."""
    if not settings.rerank_enabled or not passages:
        return None
    try:
        return await asyncio.to_thread(_rerank_sync, query, passages)
    except Exception as exc:  # noqa: BLE001 - degrade to vector order
        logger.warning("Reranking failed, keeping original order: %s", str(exc)[:200])
        return None
