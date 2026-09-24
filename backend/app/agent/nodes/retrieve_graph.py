"""Node 3b: graph traversal retrieval.

Seeds entities from the question, walks the relationship edges, and pulls the
chunks that evidence those edges -- alongside an ordinary vector search, whose
results are merged in. The graph contributes the relational layer vector search
cannot see; it does not get to decide what evidence the answer is allowed.
"""
from __future__ import annotations

import asyncio
import time

from app.agent.state import AgentState
from app.config import settings
from app.graph import traversal
from app.graph.neo4j_client import is_available
from app.logging_conf import get_logger
from app.vectorstore import search

logger = get_logger(__name__)


async def _vector_support(state: AgentState) -> list:
    """The vector half of the GRAPH route. Never fatal: a failure just means
    the traversal stands on its own, exactly as it used to."""
    try:
        claimed = await state.take_prefetch(state.query, settings.retrieval_top_k)
        if claimed is not None:
            return claimed
        return await search.hybrid_search(
            state.query,
            user_id=state.user_id,
            kb_id=state.kb_id,
            source_label="vector",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Vector support for the graph route failed: %s", str(exc)[:160])
        return []


async def retrieve_graph(state: AgentState, fallback: bool = True) -> AgentState:
    started = time.perf_counter()

    if not is_available():
        state.warnings.append("The knowledge graph is offline; used vector search instead.")
        state.add_trace(
            "retrieve_graph",
            "Graph unavailable",
            "Neo4j is not reachable; falling back to vector search.",
            started,
            available=False,
        )
        if fallback:
            from app.agent.nodes.retrieve_vector import retrieve_vector

            return await retrieve_vector(state)
        return state

    # Traverse and search at the same time. The graph is an enhancement, not a
    # replacement: an entity the extractor never created cannot be traversed to,
    # and this corpus has no "article 359" node despite containing Article 359.
    # When traversal returns something irrelevant rather than nothing, the
    # empty-context fallback below never fires, and the route silently answers
    # from junk while the passage it needed sits one vector search away. Running
    # both costs a single extra Qdrant round trip against a traversal that was
    # going to take longer anyway.
    context, vector_chunks = await asyncio.gather(
        traversal.traverse(state.query, state.kb_id),
        _vector_support(state),
    )

    if context.is_empty:
        state.add_trace(
            "retrieve_graph",
            "Graph traversal",
            "No matching entities in the graph; falling back to vector search.",
            started,
            seeds=0,
        )
        if fallback:
            from app.agent.nodes.retrieve_vector import retrieve_vector

            return await retrieve_vector(state)
        return state

    state.graph_text = context.as_text()
    state.graph_entities = context.seeds

    chunks = await search.fetch_chunks_by_id(
        context.chunk_ids,
        user_id=state.user_id,
        kb_id=state.kb_id,
        source_label="graph",
    )
    state.chunks = search.deduplicate(state.chunks + chunks + vector_chunks)

    state.add_trace(
        "retrieve_graph",
        "Graph traversal",
        "Matched {} entit(ies), followed {} relationship(s), pulled {} passage(s); "
        "vector search added {} more.".format(
            len(context.seeds), len(context.paths), len(chunks), len(vector_chunks)
        ),
        started,
        seeds=len(context.seeds),
        entities=context.seeds[:8],
        paths=len(context.paths),
        chunks=len(chunks),
        vector_chunks=len(vector_chunks),
        hops=settings.graph_max_hops,
    )
    return state
