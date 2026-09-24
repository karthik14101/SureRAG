"""Node 3a: pure vector retrieval (dense + BM25 fused by RRF in Qdrant)."""
from __future__ import annotations

import time

from app.agent.state import AgentState
from app.config import settings
from app.core.errors import UpstreamError
from app.logging_conf import get_logger
from app.vectorstore import search

logger = get_logger(__name__)


async def retrieve_vector(state: AgentState, top_k: int | None = None) -> AgentState:
    started = time.perf_counter()
    limit = top_k or settings.retrieval_top_k
    try:
        chunks = await state.take_prefetch(state.query, limit)
        if chunks is None:
            chunks = await search.hybrid_search(
                state.query,
                user_id=state.user_id,
                kb_id=state.kb_id,
                top_k=limit,
            )
    except Exception as exc:  # noqa: BLE001
        raise UpstreamError(
            "Vector search is unavailable. Check that Qdrant is running "
            "(`docker compose ps`). Details: {}".format(str(exc)[:200])
        ) from exc

    state.chunks = search.deduplicate(state.chunks + chunks)

    state.add_trace(
        "retrieve_vector",
        "Vector search",
        "Retrieved {} passage(s) from {} document(s).".format(
            len(chunks), len({c.doc_id for c in chunks})
        ),
        started,
        retrieved=len(chunks),
        top_score=round(chunks[0].score, 4) if chunks else 0.0,
    )
    return state
