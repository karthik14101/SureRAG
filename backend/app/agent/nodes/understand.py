"""Nodes 1 and 2, run in as few round trips as the question allows.

Condensing a follow-up and choosing a retrieval strategy are both short
structured judgements on the same input, and both were costing a separate
model call: 2.7s and 2.1s measured across live traffic, with 23 of 64 turns
paying for both in series before a single passage had been retrieved.

Neither depends on the other's answer, so when both are needed they go out as
one call. The cheap gates in front of each still run first, and the decisions
they reach are unchanged -- this removes a round trip, not a judgement:

    no rewrite needed, heuristic route  ->  0 calls   (as before)
    no rewrite needed, ambiguous route  ->  1 call    (as before)
    rewrite needed, heuristic route     ->  1 call    (as before)
    rewrite needed, ambiguous route     ->  1 call    (was 2)

The route heuristic still runs on the rewritten question and still wins when it
is confident, exactly as it did when it ran after a separate condense step.
"""
from __future__ import annotations

import time

from app.agent.nodes.condense import condense, needs_rewrite
from app.agent.nodes.router import heuristic_route, route_query, _coerce_route
from app.agent.prompts import UNDERSTAND_SYSTEM, UNDERSTAND_TEMPLATE, format_history
from app.agent.state import AgentState
from app.graph.neo4j_client import is_available as graph_available
from app.llm.factory import get_fast_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

# A rewrite that comes back as an essay, or empty, is not a question.
MIN_QUESTION_CHARS = 5
MAX_QUESTION_CHARS = 500


def _usable_rewrite(text: str) -> str:
    """The same guard the standalone condenser applies to a model's rewrite."""
    cleaned = (text or "").strip().strip('"').strip()
    if MIN_QUESTION_CHARS <= len(cleaned) <= MAX_QUESTION_CHARS and "\n" not in cleaned:
        return cleaned
    return ""


async def understand(state: AgentState) -> AgentState:
    """Settle the question's wording and its retrieval strategy."""
    # A forced route needs no classifier, so there is nothing to combine: the
    # ordinary two-step path is already one call at most.
    if state.forced_route is not None or not needs_rewrite(state.question, state.history):
        await condense(state)
        return await route_query(state)

    started = time.perf_counter()
    graph_on = graph_available()
    history_text = format_history(state.history)
    if state.history_summary:
        history_text = "Earlier context: {}\n\n{}".format(state.history_summary, history_text)

    try:
        llm = get_fast_llm()
        payload, usage = await llm.complete_json(
            UNDERSTAND_SYSTEM,
            UNDERSTAND_TEMPLATE.format(
                history=history_text,
                question=state.question,
                graph_available="yes" if graph_on else "no",
            ),
            temperature=0.0,
            max_tokens=400,
        )
        state.spend_call(usage, fast=True)
    except Exception as exc:  # noqa: BLE001 - never fail a turn on the preamble
        logger.warning("Combined condense/route failed: %s", str(exc)[:160])
        # Fall back to the separate steps rather than guessing at either.
        await condense(state)
        return await route_query(state)

    rewritten = _usable_rewrite(str(payload.get("question") or ""))
    state.condensed_question = rewritten or state.question

    if rewritten and rewritten != state.question:
        state.add_trace(
            "condense", "Question rewritten", rewritten, started,
            rewritten=True, original=state.question, combined=True,
        )
    else:
        state.add_trace(
            "condense", "Question understood",
            "Already standalone; used as-is." if rewritten else "Rewrite rejected; used the original.",
            started, rewritten=False, combined=True,
        )

    # The heuristic reads the rewritten question, which is the order it ran in
    # when condensing was its own step. A confident heuristic still wins: it is
    # deterministic, it was tuned against these routes, and the model's answer
    # cost nothing extra either way.
    route, reason, confidence = heuristic_route(state.query, graph_on)
    if route is not None:
        state.route = _coerce_route(route.value, graph_on)
        state.route_reason = reason
        state.route_decided_by = "heuristic"
    else:
        state.route = _coerce_route(payload.get("route"), graph_on)
        state.route_reason = str(payload.get("reason") or "Classified by the model.")[:200]
        state.route_decided_by = "llm"
        try:
            confidence = float(payload.get("confidence") or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5

    state.add_trace(
        "route",
        "Route: {}".format(state.route.value),
        state.route_reason,
        route=state.route.value,
        confidence=confidence,
        decided_by=state.route_decided_by,
        combined=True,
    )
    return state
