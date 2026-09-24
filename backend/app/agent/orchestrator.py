"""The SURE orchestrator: a bounded state machine over the retrieval nodes.

    understand -> retrieve -> rerank -> VERIFY -+-> synthesize
     (condense       ^                          |
      + route)       +------ expand <-----------+
                          (while insufficient, max N iterations)

Understanding the question and the vector search run together where they can:
condensing and routing share one model call, and when the query needs no
rewriting the search starts while that call is still out.

Written as an explicit loop rather than on a graph framework because the shape
is fixed and small: there is exactly one cycle, the node set never changes at
runtime, and every transition is a plain `if`. That keeps the whole control flow
readable on one screen and makes the step trace trivial to emit for the UI.

Hard budgets bound every turn: iterations, wall-clock time, and two separate
call counts -- reasoning-model calls and fast-tier calls, which are counted
apart because they cost differently. Whichever trips first, the agent answers
with what it has rather than hanging.
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
from app.agent.nodes.condense import needs_rewrite
from app.agent.nodes.understand import understand
from app.agent.state import AgentState, Route
from app.config import settings
from app.core.errors import UpstreamError
from app.logging_conf import get_logger, log_event
from app.vectorstore import search as vector_search

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


# How much the sufficiency score must rise for an expansion pass to count as
# having worked.
MIN_SCORE_GAIN = 0.05

# Below this, after a full expansion pass has already failed to help, the gap
# is not a retrieval failure -- the corpus does not hold this material. The
# verifier is asked to say so itself via "in_scope", but it routinely omits the
# field, so the conclusion is also reached here from behaviour: targeted
# queries, a hypothetical-answer probe and a graph widening all ran, and the
# score is still on the floor. Searching a fourth time cannot conjure what was
# never ingested, and the reader is better served by one honest sentence than
# by a catalogue of everything absent.
OUT_OF_SCOPE_SCORE = 0.25


def _may_conclude_out_of_scope(state: AgentState) -> bool:
    """Whether a floor score is evidence of absence or just of a bad query.

    Both guards come from one turn. A Curvv EV owner asked what a yellow lamp
    with a turtle icon meant. The manual answers it -- "Limp Home Mode | Amber",
    and a table giving exactly that continuous chime at 5% charge -- but across
    364 pages it never writes the word "turtle", so the verifier scored the
    evidence at 20% twice while its own covered list named limp home mode. The
    loop then announced that the knowledge base held no such material and capped
    the answer at a short apology.

    So a number no longer overrules the verifier's explicit "yes, this is
    findable here", and the conclusion is only drawn about a corpus whose shape
    was actually measured. On that turn the verifier was shown "(not described)"
    and the loop still pronounced on what the corpus did not contain.

    When this returns False the loop still stops -- the score did not rise, and
    that has not changed -- but it stops saying what is missing rather than
    declaring the question unanswerable here.
    """
    if state.verifier_in_scope is True:
        return False
    return bool(state.corpus_shape)


def _should_stop(state: AgentState) -> bool:
    """Decide whether another expansion pass can plausibly help.

    One question, asked of the only number that cannot be argued with: did the
    last pass raise the sufficiency score? Expansion exists to close a gap the
    verifier named. If a full pass of targeted queries, a HyDE probe and a graph
    widening did not move the verdict, the next pass is not going to either.

    Everything softer than this has been tried and has failed on live traffic.
    Comparing the verifier's gap list as text failed because the model rephrases
    the same complaint every round. Continuing whenever *some* new passage
    arrived failed because new passages always arrive -- the corpus is large and
    the queries keep changing -- so the loop ran until a hard budget killed it,
    which is how a question that was answered well in 36 seconds came back
    fifty-percent worse in 72. Telling the verifier in its prompt not to demand
    a passage that states the conclusion failed too: it simply paraphrased its
    way around the instruction. A verifier can always invent another gap, so the
    bound on how long it may be humoured has to live here, in code, where no
    wording can talk its way past it.
    """
    if state.out_of_scope:
        return True  # the verifier already set stop_reason

    if state.previous_score is None:
        return False  # nothing to compare yet; the first pass always runs

    if state.sufficiency_score - state.previous_score >= MIN_SCORE_GAIN:
        return False  # it is working; let the budget decide when to stop

    if state.sufficiency_score < OUT_OF_SCOPE_SCORE and _may_conclude_out_of_scope(state):
        state.out_of_scope = True
        state.stop_reason = (
            "A full expansion pass left sufficiency at {:.0%}. The knowledge "
            "base does not appear to hold the kind of material this question "
            "needs, so searching again cannot find it.".format(state.sufficiency_score)
        )
        return True

    state.stop_reason = (
        "The last search did not improve the evidence (sufficiency {:.0%} then "
        "{:.0%}, with {} new passage(s) and {:.0%} of the judged window "
        "replaced), so another pass would not improve it either.".format(
            state.previous_score,
            state.sufficiency_score,
            state.last_expansion_gain,
            state.last_judged_turnover,
        )
    )
    return True


def _start_prefetch(state: AgentState) -> None:
    """Begin the vector search while the question is still being understood.

    Understanding the question costs up to one model call -- two to three
    seconds measured -- during which the vector index does nothing at all. When
    the question needs no rewriting, the query retrieval will use is already
    known, so that search can run alongside the decision and be waiting by the
    time a route is picked.

    Only started when the query is final. Speculating on a query that is about
    to be rewritten would fetch evidence for a question nobody asked, and
    `take_prefetch` refuses anything whose query does not match exactly. Three
    of the five routes claim it; MULTIHOP searches per sub-question at a
    different depth and DIRECT retrieves nothing, so both simply drop it.
    """
    if state.forced_route == Route.DIRECT or needs_rewrite(state.question, state.history):
        return

    query = state.question.strip()
    if not query:
        return

    limit = settings.retrieval_top_k
    state.prefetch_query = query
    state.prefetch_top_k = limit
    state.prefetch_task = asyncio.create_task(
        vector_search.hybrid_search(query, user_id=state.user_id, kb_id=state.kb_id, top_k=limit)
    )


async def _prepare(state: AgentState) -> AgentState:
    """Everything up to (but not including) answer generation.

    Shared by the streaming and non-streaming entry points so both take exactly
    the same path through routing, retrieval and verification.
    """
    _start_prefetch(state)
    try:
        return await _prepared(state)
    finally:
        # Whatever happened -- a conversational turn, a route that could not use
        # it, a retriever that raised -- the speculative task must not outlive
        # the turn that started it.
        state.discard_prefetch()


async def _prepared(state: AgentState) -> AgentState:
    """The turn itself. Wrapped by `_prepare` only so the speculative
    search is cleaned up on every path out of here, including a raise."""
    await understand(state)

    if state.route == Route.DIRECT:
        # Conversational turn: no retrieval, no verification, no citations.
        return state

    await _retrieve(state)
    await rerank_node.rerank(state)
    await verify_node.verify_sufficiency(state)

    # ---- the SURE loop ----------------------------------------------------
    while not state.sufficient and not _should_stop(state) and state.can_iterate():
        await expand_node.expand_and_retrieve(state)
        if state.sufficient:
            # Expansion found nothing new and gave up; stop here.
            break
        await rerank_node.rerank(state)
        await verify_node.verify_sufficiency(state)

    return _settle(state)


def _settle(state: AgentState) -> AgentState:
    """Everything between the last verification and writing the answer.

    Split out from `_prepare` so it can be exercised on its own: what the loop
    leaves behind -- which pool the answer is written from, whether the corpus
    is declared to lack the material, what the reader is warned about -- is
    decided here, and it decided wrongly on live traffic for longer than it
    should have.
    """
    if state.stop_reason:
        state.add_trace("budget", "Stopped expanding", state.stop_reason)

    # An expansion pass that made the verdict worse also made the evidence
    # worse; answer from the pool that scored best, not from the last one.
    if state.restore_best_window():
        state.add_trace(
            "recover",
            "Recovered the better evidence",
            "Expanding lowered the verdict, so the {} passage(s) that scored "
            "{:.0%} were ranked back to the front. Nothing found since was "
            "discarded.".format(len(state.best_window_ids), state.best_score),
            score=round(state.best_score, 3),
            restored=len(state.best_window_ids),
        )

    if state.out_of_scope and state.chunks:
        state.warnings.append(
            "This knowledge base does not contain the kind of material this "
            "question needs. The answer below says what is missing rather than "
            "inferring it."
        )
    elif not state.sufficient and state.chunks:
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
            "Stopped after {} reasoning-model call(s) (AGENT_MAX_LLM_CALLS).".format(
                state.llm_calls
            ),
        )
    elif state.fast_llm_calls >= settings.agent_max_fast_llm_calls:
        state.add_trace(
            "budget",
            "Call budget reached",
            "Stopped after {} fast-model call(s) (AGENT_MAX_FAST_LLM_CALLS).".format(
                state.fast_llm_calls
            ),
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
        llm_calls=state.total_llm_calls,
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
            "grounding": state.grounding,
            "out_of_scope": state.out_of_scope,
            "iterations": state.iterations,
            "llm_calls": state.total_llm_calls,
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
        llm_calls=state.total_llm_calls,
        ms=state.elapsed_ms,
        error=state.error,
    )
