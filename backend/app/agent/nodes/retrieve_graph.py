"""Node 3b: graph traversal retrieval.

Seeds entities from the question, walks the relationship edges, and pulls the
chunks that evidence those edges. Degrades to vector search when the graph is
unavailable or the question matches no entity.
"""
from __future__ import annotations

import time

from app.agent.state import AgentState
from app.config import settings
from app.graph import traversal
from app.graph.neo4j_client import is_available
from app.logging_conf import get_logger
from app.vectorstore import search

logger = get_logger(__name__)


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

    context = await traversal.traverse(state.query, state.kb_id)

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
    state.chunks = search.deduplicate(state.chunks + chunks)

    state.add_trace(
        "retrieve_graph",
        "Graph traversal",
        "Matched {} entit(ies), followed {} relationship(s), pulled {} passage(s).".format(
            len(context.seeds), len(context.paths), len(chunks)
        ),
        started,
        seeds=len(context.seeds),
        entities=context.seeds[:8],
        paths=len(context.paths),
        chunks=len(chunks),
        hops=settings.graph_max_hops,
    )
    return state
