"""Node 6: query expansion, run only when the verifier reports a gap.

Three strategies fire together, because they fail in different ways:

  1. Targeted queries  -- generated from the verifier's `missing` list, using
     document vocabulary rather than the user's phrasing.
  2. HyDE              -- embed a hypothetical ANSWER instead of the question.
     Documents are written as statements, so a statement-shaped probe often
     lands closer in vector space than an interrogative one.
  3. Graph widening    -- pull one more hop out from the entities already found.

Whatever they return is merged with the existing evidence and re-verified.
"""
from __future__ import annotations

import asyncio
import time

from app.agent.prompts import EXPANSION_SYSTEM, EXPANSION_TEMPLATE, HYDE_SYSTEM
from app.agent.state import AgentState
from app.graph import traversal
from app.graph.neo4j_client import is_available
from app.llm.base import ChatMessage
from app.llm.factory import get_fast_llm
from app.logging_conf import get_logger
from app.vectorstore import search

logger = get_logger(__name__)

MAX_EXPANSION_QUERIES = 3


async def _generate_queries(state: AgentState) -> list[str]:
    """Use the verifier's gap list to write document-shaped search queries."""
    if state.expansion_queries:
        return state.expansion_queries[:MAX_EXPANSION_QUERIES]
    if not state.missing_aspects:
        return []

    try:
        llm = get_fast_llm()
        # The verifier's own account of what the evidence covers, not the list of
        # filenames this used to send. The template asks what is already covered,
        # and answering it with "curvv-ev-owners-manual.pdf" five times told the
        # query writer nothing. What it says instead is the vocabulary these
        # documents actually use, which is the whole difficulty when a question
        # names something by its shape or its symptom.
        covered = "\n".join("- " + c for c in state.covered_aspects) or "- (nothing yet)"
        missing = "\n".join("- " + m for m in state.missing_aspects)
        payload, usage = await llm.complete_json(
            EXPANSION_SYSTEM,
            EXPANSION_TEMPLATE.format(
                question=state.query, covered=covered, missing=missing
            ),
            temperature=0.3,
            max_tokens=300,
        )
        state.spend_call(usage, fast=True)
        raw = payload.get("queries") or []
        queries = [str(q).strip() for q in raw if str(q).strip()]
        return queries[:MAX_EXPANSION_QUERIES]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Expansion query generation failed: %s", str(exc)[:160])
        # Fall back to searching the missing aspects verbatim.
        return state.missing_aspects[:MAX_EXPANSION_QUERIES]


async def _hyde_probe(state: AgentState) -> str | None:
    """Generate a hypothetical answer to use as a retrieval probe."""
    if state.budget_exhausted():
        return None
    try:
        llm = get_fast_llm()
        result = await llm.complete(
            HYDE_SYSTEM,
            [ChatMessage(role="user", content=state.query)],
            temperature=0.4,
            max_tokens=300,
        )
        state.spend_call(result.usage, fast=True)
        text = result.text.strip()
        return text if len(text) > 40 else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("HyDE probe failed: %s", str(exc)[:160])
        return None


async def expand_and_retrieve(state: AgentState) -> AgentState:
    """Run the expansion pass and merge any new evidence into the state."""
    started = time.perf_counter()
    state.iterations += 1
    existing_ids = [c.chunk_id for c in state.chunks]

    queries, hyde = await asyncio.gather(_generate_queries(state), _hyde_probe(state))

    probes = list(queries)
    if hyde:
        probes.append(hyde)

    new_chunks: list = []
    if probes:
        new_chunks = await search.multi_query_search(
            probes,
            user_id=state.user_id,
            kb_id=state.kb_id,
            top_k_each=5,
            exclude_ids=existing_ids,
            source_label="expand",
        )

    # Widen the graph neighbourhood by one hop around the entities we know about.
    graph_chunks: list = []
    if is_available() and state.graph_entities:
        try:
            chunk_ids = await traversal.neighbourhood_chunks(
                state.graph_entities, state.kb_id, limit=10
            )
            fresh = [cid for cid in chunk_ids if cid not in set(existing_ids)]
            if fresh:
                graph_chunks = await search.fetch_chunks_by_id(
                    fresh, user_id=state.user_id, kb_id=state.kb_id, source_label="expand"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Graph widening failed: %s", str(exc)[:160])

    before = len(state.chunks)
    state.chunks = search.deduplicate(state.chunks + new_chunks + graph_chunks)
    added = len(state.chunks) - before
    # The orchestrator uses this to decide whether another pass is worth it.
    state.last_expansion_gain = added

    detail_parts = []
    if queries:
        detail_parts.append("{} targeted quer(ies)".format(len(queries)))
    if hyde:
        detail_parts.append("a hypothetical-answer probe")
    if graph_chunks:
        detail_parts.append("{} graph neighbour(s)".format(len(graph_chunks)))
    strategy = ", ".join(detail_parts) or "no viable strategy"

    state.add_trace(
        "expand",
        "Expansion pass {}".format(state.iterations),
        "Ran {}; found {} new passage(s).".format(strategy, added),
        started,
        iteration=state.iterations,
        queries=queries,
        used_hyde=bool(hyde),
        new_chunks=added,
    )

    # Nothing new is available: stop looping, the knowledge base simply lacks it.
    if added == 0:
        state.sufficient = True
        state.warnings.append(
            "The knowledge base does not appear to contain the missing details."
        )

    return state
