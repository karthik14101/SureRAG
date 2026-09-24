"""Node 4: optional cross-encoder reranking.

Bi-encoder retrieval (what Qdrant does) embeds the query and the passage
separately, so it can only ever approximate relevance. A cross-encoder reads
both together and is far more accurate -- but it cannot be indexed, so it only
works as a second pass over a shortlist. Off by default; RERANK_ENABLED=true.
"""
from __future__ import annotations

import time

from app.agent.state import AgentState
from app.config import settings
from app.vectorstore import search


async def rerank(state: AgentState) -> AgentState:
    if not settings.rerank_enabled or len(state.chunks) < 2:
        return state

    started = time.perf_counter()
    pooled = len(state.chunks)
    before = [c.chunk_id for c in state.chunks[:5]]

    state.chunks = await search.apply_reranking(state.query, state.chunks)

    after = [c.chunk_id for c in state.chunks[:5]]
    promoted = len([cid for cid in after if cid not in before])

    if promoted:
        detail = "Cross-encoder rescored {} passage(s) against the question; {} new one(s) reached the top 5.".format(
            pooled, promoted
        )
    else:
        detail = "Cross-encoder rescored {} passage(s) and confirmed the order.".format(pooled)

    state.add_trace(
        "rerank",
        "Reranked results",
        detail,
        started,
        pooled=pooled,
        kept=len(state.chunks),
        reordered=before != after,
        promoted=promoted,
    )
    return state
