"""The SURE orchestrator: a bounded state machine over the retrieval nodes.

    condense -> route -> retrieve -> rerank -> VERIFY -+-> synthesize
                            ^                          |
                            +------ expand <-----------+
                                 (while insufficient, max N iterations)

Written as an explicit loop rather than on a graph framework because the shape
is fixed and small: there is exactly one cycle, the node set never changes at
runtime, and every transition is a plain `if`. That keeps the whole control flow
readable on one screen and makes the step trace trivial to emit for the UI.

Three hard budgets bound every turn: iterations, LLM calls, and wall-clock time.
Whichever trips first, the agent answers with what it has rather than hanging.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.agent.nodes import expand as expand_node
from app.agent.nodes import rerank as rerank_node
from app.agent.nodes import retrieve_graph as graph_node
from app.agent.nodes import retrieve_hybrid as hybrid_node
from app.agent.nodes import retrieve_vector as vector_node
from app.agent.nodes import synthesize as synth_node
from app.agent.nodes import verifier as verify_node
from app.agent.nodes.condense import condense
from app.agent.nodes.router import route_query
from app.agent.state import AgentState, Route
from app.config import settings
from app.core.errors import UpstreamError
from app.logging_conf import get_logger, log_event

logger = get_logger(__name__)


async def _retrieve(state: AgentState) -> AgentState:
    """Dispatch to the retriever the router selected."""
    if state.route == Route.VECTOR:
        return await vector_node.retrieve_vector(state)
    if state.route == Route.GRAPH:
        return await graph_node.retrieve_graph(state)
    if state.route == Route.MULTIHOP:
        return await hybrid_node.retrieve_multihop(state)
    return await hybrid_node.retrieve_hybrid(state)


async def _prepare(state: AgentState) -> AgentState:
    """Everything up to (but not including) answer generation.

    Shared by the streaming and non-streaming entry points so both take exactly
    the same path through routing, retrieval and verification.
    """
    await condense(state)
    await route_query(state)

    if state.route == Route.DIRECT:
        # Conversational turn: no retrieval, no verification, no citations.
        return state

    await _retrieve(state)
    await rerank_node.rerank(state)
    await verify_node.verify_sufficiency(state)

    # ---- the SURE loop ----------------------------------------------------
    while not state.sufficient and state.can_iterate():
        await expand_node.expand_and_retrieve(state)
        if state.sufficient:
            # Expansion found nothing new and gave up; stop here.
            break
        await rerank_node.rerank(state)
        await verify_node.verify_sufficiency(state)

    if not state.sufficient and state.chunks:
        state.warnings.append(
            "The retrieved evidence was judged incomplete ({:.0%} sufficiency). "
            "The answer below states what is missing.".format(state.sufficiency_score)
        )

    return state


def _budget_note(state: AgentState) -> None:
    if state.llm_calls >= settings.agent_max_llm_calls:
        state.add_trace(
            "budget",
            "Call budget reached",
            "Stopped after {} model calls (AGENT_MAX_LLM_CALLS).".format(state.llm_calls),
        )
    elif state.elapsed_seconds >= settings.agent_timeout_seconds:
        state.add_trace(
            "budget",
            "Time budget reached",
            "Stopped after {:.1f}s (AGENT_TIMEOUT_SECONDS).".format(state.elapsed_seconds),
        )


async def run(state: AgentState) -> AgentState:
    """Execute a full turn and return the completed state (non-streaming)."""
    try:
        await asyncio.wait_for(
            _prepare(state),
            timeout=settings.agent_timeout_seconds + 30,
        )
        _budget_note(state)
        await synth_node.synthesize(state)
    except asyncio.TimeoutError:
        state.error = (
            "This question took too long to process. Try narrowing it, or raise "
            "AGENT_TIMEOUT_SECONDS in .env."
        )
        state.answer = state.answer or state.error
        state.add_trace("error", "Timed out", state.error)
    except UpstreamError as exc:
        state.error = exc.message
        state.answer = exc.message
        state.add_trace("error", "Service unavailable", exc.message)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run failed")
        state.error = "Something went wrong while answering: {}".format(str(exc)[:200])
        state.answer = state.error
        state.add_trace("error", "Agent error", state.error)

    log_event(
        logger,
        "agent turn complete",
        route=state.route.value,
        chunks=len(state.chunks),
        citations=len(state.citations),
        sufficiency=round(state.sufficiency_score, 2),
        iterations=state.iterations,
        llm_calls=state.llm_calls,
        ms=state.elapsed_ms,
    )
    return state


async def run_stream(state: AgentState) -> AsyncIterator[tuple[str, object]]:
    """Execute a turn, yielding (event_name, payload) tuples for SSE.

    Events, in order:
      trace  -- one per node, as it completes (drives the Reasoning Trail)
      status -- routing/retrieval summary once preparation finishes
      token  -- answer text, incrementally
      done   -- final citations, images and metadata
      error  -- terminal failure; no further events follow
    """
    emitted_traces = 0

    try:
        prepare_task = asyncio.create_task(
            asyncio.wait_for(_prepare(state), timeout=settings.agent_timeout_seconds + 30)
        )

        # Stream traces as nodes complete, so the UI shows progress during the
        # slow part of the turn rather than after it.
        while not prepare_task.done():
            await asyncio.sleep(0.15)
            while emitted_traces < len(state.traces):
                yield "trace", state.traces[emitted_traces].as_dict()
                emitted_traces += 1

        await prepare_task

        while emitted_traces < len(state.traces):
            yield "trace", state.traces[emitted_traces].as_dict()
            emitted_traces += 1

        _budget_note(state)

        yield "status", {
            "route": state.route.value,
            "route_reason": state.route_reason,
            "chunks": len(state.chunks),
            "sufficiency_score": round(state.sufficiency_score, 3),
            "iterations": state.iterations,
            "warnings": state.warnings,
        }

        async for piece in synth_node.synthesize_stream(state):
            yield "token", piece

        while emitted_traces < len(state.traces):
            yield "trace", state.traces[emitted_traces].as_dict()
            emitted_traces += 1

        yield "done", {
            "answer": state.answer,
            "citations": [c.as_dict() for c in state.citations],
            "media_ids": state.media_ids,
            "route": state.route.value,
            "sufficiency_score": round(state.sufficiency_score, 3),
            "iterations": state.iterations,
            "llm_calls": state.llm_calls,
            "latency_ms": state.elapsed_ms,
            "token_usage": state.usage.as_dict(),
            "warnings": state.warnings,
            "trace": state.trace_dicts(),
        }

    except asyncio.TimeoutError:
        state.error = (
            "This question took too long to process. Try narrowing it, or raise "
            "AGENT_TIMEOUT_SECONDS in .env."
        )
        yield "error", {"message": state.error}
    except UpstreamError as exc:
        state.error = exc.message
        yield "error", {"message": exc.message}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Streaming agent run failed")
        state.error = "Something went wrong while answering: {}".format(str(exc)[:200])
        yield "error", {"message": state.error}

    log_event(
        logger,
        "agent stream complete",
        route=state.route.value,
        chunks=len(state.chunks),
        citations=len(state.citations),
        sufficiency=round(state.sufficiency_score, 2),
        iterations=state.iterations,
        llm_calls=state.llm_calls,
        ms=state.elapsed_ms,
        error=state.error,
    )
