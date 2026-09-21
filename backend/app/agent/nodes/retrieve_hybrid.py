"""Node 3c: hybrid and multi-hop retrieval.

HYBRID runs vector and graph retrieval concurrently and merges the results --
the graph supplies relationship facts, the vector index supplies the prose that
explains them.

MULTIHOP first decomposes the question into sub-questions, retrieves for each,
and pools the evidence, so a comparison question gets passages about every item
being compared rather than only the first one mentioned.
"""
from __future__ import annotations

import asyncio
import time

from app.agent.prompts import DECOMPOSE_SYSTEM, DECOMPOSE_TEMPLATE
from app.agent.state import AgentState
from app.config import settings
from app.graph import traversal
from app.graph.neo4j_client import is_available
from app.llm.factory import get_llm
from app.logging_conf import get_logger
from app.vectorstore import search

logger = get_logger(__name__)

MAX_SUB_QUESTIONS = 4


async def retrieve_hybrid(state: AgentState) -> AgentState:
    started = time.perf_counter()

    async def _vector() -> list:
        try:
            return await search.hybrid_search(
                state.query,
                user_id=state.user_id,
                kb_id=state.kb_id,
                top_k=settings.retrieval_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Vector half of hybrid retrieval failed: %s", str(exc)[:160])
            return []

    async def _graph():
        if not is_available():
            return None
        try:
            return await traversal.traverse(state.query, state.kb_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph half of hybrid retrieval failed: %s", str(exc)[:160])
            return None

    vector_chunks, graph_context = await asyncio.gather(_vector(), _graph())

    graph_chunks: list = []
    if graph_context is not None and not graph_context.is_empty:
        state.graph_text = graph_context.as_text()
        state.graph_entities = graph_context.seeds
        graph_chunks = await search.fetch_chunks_by_id(
            graph_context.chunk_ids,
            user_id=state.user_id,
            kb_id=state.kb_id,
            source_label="graph",
        )

    state.chunks = search.deduplicate(state.chunks + vector_chunks + graph_chunks)

    detail = "Vector: {} passage(s).".format(len(vector_chunks))
    if graph_chunks:
        detail += " Graph: {} entit(ies), {} passage(s).".format(
            len(state.graph_entities), len(graph_chunks)
        )
    elif is_available():
        detail += " Graph: no entity match."
    else:
        detail += " Graph: offline."

    state.add_trace(
        "retrieve_hybrid",
        "Hybrid retrieval",
        detail,
        started,
        vector=len(vector_chunks),
        graph=len(graph_chunks),
        merged=len(state.chunks),
    )
    return state


async def decompose_question(state: AgentState) -> list[str]:
    """Split a complex question into independently answerable parts."""
    try:
        llm = get_llm()
        payload, usage = await llm.complete_json(
            DECOMPOSE_SYSTEM,
            DECOMPOSE_TEMPLATE.format(question=state.query),
            temperature=0.0,
            max_tokens=350,
        )
        state.spend_call(usage)
        raw = payload.get("sub_questions") or []
        questions = [
            str(q).strip()
            for q in raw
            if isinstance(q, (str, int)) and len(str(q).strip()) > 5
        ]
        return questions[:MAX_SUB_QUESTIONS]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Decomposition failed: %s", str(exc)[:160])
        return []


async def retrieve_multihop(state: AgentState) -> AgentState:
    """Decompose, retrieve per sub-question, then pool the evidence."""
    started = time.perf_counter()

    sub_questions = await decompose_question(state)

    if not sub_questions:
        state.add_trace(
            "retrieve_multihop",
            "Multi-hop retrieval",
            "Could not decompose; used hybrid retrieval.",
            started,
            sub_questions=0,
        )
        return await retrieve_hybrid(state)

    state.sub_questions = sub_questions

    # Always include the original question so the overall intent stays represented.
    queries = [state.query] + sub_questions
    per_query = max(4, settings.retrieval_top_k // max(1, len(queries)) + 2)

    pooled = await search.multi_query_search(
        queries,
        user_id=state.user_id,
        kb_id=state.kb_id,
        top_k_each=per_query,
        source_label="vector",
    )

    graph_chunks: list = []
    if is_available():
        try:
            context = await traversal.traverse(state.query, state.kb_id)
            if not context.is_empty:
                state.graph_text = context.as_text()
                state.graph_entities = context.seeds
                graph_chunks = await search.fetch_chunks_by_id(
                    context.chunk_ids,
                    user_id=state.user_id,
                    kb_id=state.kb_id,
                    source_label="graph",
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph step of multi-hop failed: %s", str(exc)[:160])

    state.chunks = search.deduplicate(state.chunks + pooled + graph_chunks)

    state.add_trace(
        "retrieve_multihop",
        "Multi-hop retrieval",
        "Split into {} sub-question(s); pooled {} passage(s).".format(
            len(sub_questions), len(state.chunks)
        ),
        started,
        sub_questions=sub_questions,
        pooled=len(pooled),
        graph=len(graph_chunks),
    )
    return state
