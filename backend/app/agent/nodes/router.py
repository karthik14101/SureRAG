"""Node 2: the dynamic query router.

Two-stage by design. A heuristic gate settles the obvious cases for free and
instantly; only genuinely ambiguous questions cost an LLM call. Routing every
query through the model would add latency and quota burn to "hello".
"""
from __future__ import annotations

import re
import time

from app.agent.prompts import ROUTER_SYSTEM, ROUTER_TEMPLATE
from app.agent.state import AgentState, Route
from app.graph.neo4j_client import is_available as graph_available
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

# --- DIRECT: conversational, no retrieval needed -----------------------------
GREETING_RE = re.compile(
    r"^\s*(hi|hey|hello|yo|good\s+(morning|afternoon|evening)|thanks?|thank\s+you|"
    r"ok(ay)?|cool|nice|great|bye|goodbye|test)\b[\s!.?]*$",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"\b(who are you|what are you|what can you do|how do you work|your name|"
    r"help me|what is this|how does this work)\b",
    re.IGNORECASE,
)

# --- GRAPH: relationship language -------------------------------------------
RELATION_WORDS = (
    "relationship", "related to", "connection", "connected", "link between",
    "linked to", "associated with", "depends on", "dependency", "affects",
    "affected by", "impacts", "impacted by", "influences", "caused by", "causes",
    "leads to", "results in", "works for", "belongs to", "part of", "owned by",
    "reports to", "interacts with", "who is involved", "network of", "downstream",
    "upstream", "between",
)

# --- MULTIHOP: comparison and multi-step language ----------------------------
MULTIHOP_WORDS = (
    "compare", "comparison", "versus", " vs ", "difference between", "differences",
    "contrast", "both", "all of the", "each of the", "step by step", "trace",
    "and then", "as well as", "in addition to", "which of these", "rank",
    "pros and cons", "trade-off", "tradeoff", "summarize all", "across all",
)

# --- VECTOR: plain factual lookup -------------------------------------------
LOOKUP_STARTS = (
    "what is", "what are", "what does", "define", "definition of", "explain",
    "describe", "summarize", "summarise", "tell me about", "how do i", "how to",
    "when was", "when did", "where is", "list the", "what was",
)


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    lowered = " " + text.casefold() + " "
    return sum(1 for marker in markers if marker in lowered)


def _proper_noun_count(text: str) -> int:
    """Capitalised tokens after the first word hint at named entities."""
    tokens = text.split()[1:]
    return sum(1 for token in tokens if token[:1].isupper() and len(token) > 2)


def heuristic_route(question: str, graph_on: bool) -> tuple[Route | None, str, float]:
    """Return a confident route, or None to escalate to the LLM classifier."""
    text = question.strip()
    if not text:
        return Route.DIRECT, "Empty question.", 1.0

    if GREETING_RE.match(text):
        return Route.DIRECT, "Greeting or acknowledgement.", 0.98
    if META_RE.search(text) and len(text.split()) <= 12:
        return Route.DIRECT, "Question about the assistant itself.", 0.9

    multihop_hits = _count_markers(text, MULTIHOP_WORDS)
    relation_hits = _count_markers(text, RELATION_WORDS)
    lowered = text.casefold()
    lookup_hit = any(lowered.startswith(prefix) for prefix in LOOKUP_STARTS)
    entity_hits = _proper_noun_count(text)

    if multihop_hits >= 1 and (entity_hits >= 2 or multihop_hits >= 2):
        return Route.MULTIHOP, "Comparative or multi-step phrasing.", 0.85

    if graph_on and relation_hits >= 1 and entity_hits >= 2:
        return Route.GRAPH, "Asks how named entities relate.", 0.85

    if lookup_hit and relation_hits == 0 and multihop_hits == 0 and entity_hits <= 1:
        return Route.VECTOR, "Direct factual lookup.", 0.8

    # Genuinely ambiguous: let the model decide.
    return None, "", 0.0


def _coerce_route(value: str | None, graph_on: bool) -> Route:
    route = Route.parse(value) or Route.HYBRID
    # Never route to the graph when it is unavailable or empty.
    if route == Route.GRAPH and not graph_on:
        return Route.VECTOR
    return route


async def route_query(state: AgentState) -> AgentState:
    started = time.perf_counter()
    graph_on = graph_available()

    if state.forced_route is not None:
        state.route = _coerce_route(state.forced_route.value, graph_on)
        state.route_reason = "Route forced by the request."
        state.route_decided_by = "forced"
        state.add_trace(
            "route", "Route: {}".format(state.route.value), state.route_reason, started,
            route=state.route.value, decided_by="forced",
        )
        return state

    route, reason, confidence = heuristic_route(state.query, graph_on)

    if route is not None:
        state.route = _coerce_route(route.value, graph_on)
        state.route_reason = reason
        state.route_decided_by = "heuristic"
        state.add_trace(
            "route",
            "Route: {}".format(state.route.value),
            reason,
            started,
            route=state.route.value,
            confidence=confidence,
            decided_by="heuristic",
        )
        return state

    # Escalate to the classifier.
    try:
        llm = get_llm()
        prompt = ROUTER_TEMPLATE.format(
            question=state.query, graph_available="yes" if graph_on else "no"
        )
        payload, usage = await llm.complete_json(
            ROUTER_SYSTEM, prompt, temperature=0.0, max_tokens=150
        )
        state.spend_call(usage)

        state.route = _coerce_route(payload.get("route"), graph_on)
        state.route_reason = str(payload.get("reason") or "Classified by the model.")[:200]
        state.route_decided_by = "llm"
        confidence = float(payload.get("confidence") or 0.5)
    except Exception as exc:  # noqa: BLE001 - routing must never fail a turn
        logger.warning("Router LLM call failed, defaulting to HYBRID: %s", str(exc)[:160])
        state.route = Route.HYBRID if graph_on else Route.VECTOR
        state.route_reason = "Classifier unavailable; used the safe default."
        state.route_decided_by = "fallback"
        confidence = 0.0

    state.add_trace(
        "route",
        "Route: {}".format(state.route.value),
        state.route_reason,
        started,
        route=state.route.value,
        confidence=confidence,
        decided_by=state.route_decided_by,
    )
    return state
