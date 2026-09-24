"""Understanding the question: how many round trips it costs, and when.

Condensing a follow-up and picking a retrieval strategy were two separate model
calls made back to back. Measured across 64 real turns they cost 2.7s and 2.1s,
and 23 turns paid both -- nearly five seconds of silence before a single passage
had been retrieved. Neither needs the other's answer.

Two things are pinned down here. That the cheap gates still settle what they
always settled, so this removes round trips and not judgements. And that the
gate in front of condensing stopped firing on the word "the".
"""
from __future__ import annotations

import asyncio

from app.agent.nodes import condense as condense_node
from app.agent.nodes import router as router_node
from app.agent.nodes import understand as understand_node
from app.agent.nodes.condense import needs_rewrite
from app.agent.state import AgentState, Route
from app.llm.base import ChatMessage, TokenUsage

HISTORY = [
    ChatMessage(role="user", content="What is the warranty on the Model B pump?"),
    ChatMessage(role="assistant", content="Three years from purchase."),
]


class FakeLLM:
    """Answers with a queued payload and counts how often it was asked."""

    def __init__(self, payload=None, fail=False):
        self.payload, self.fail = payload or {}, fail
        self.calls, self.prompts, self.systems = 0, [], []

    async def complete_json(self, system, prompt, **kw):
        self.calls += 1
        self.systems.append(system)
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("provider down")
        return dict(self.payload), TokenUsage(calls=1)

    async def complete(self, system, messages, **kw):
        self.calls += 1
        self.systems.append(system)
        if self.fail:
            raise RuntimeError("provider down")

        class R:
            text = self.payload.get("question", "rewritten question")
            usage = TokenUsage(calls=1)

        return R()


def drive(question, llm, history=HISTORY, graph_on=False, forced=None):
    saved = (understand_node.get_fast_llm, condense_node.get_fast_llm,
             router_node.get_fast_llm, understand_node.graph_available,
             router_node.graph_available)
    understand_node.get_fast_llm = lambda: llm
    condense_node.get_fast_llm = lambda: llm
    router_node.get_fast_llm = lambda: llm
    understand_node.graph_available = lambda: graph_on
    router_node.graph_available = lambda: graph_on
    try:
        state = AgentState(question=question, user_id="u", kb_id="kb", session_id="s")
        state.history = list(history)
        state.forced_route = forced
        asyncio.run(understand_node.understand(state))
        return state
    finally:
        (understand_node.get_fast_llm, condense_node.get_fast_llm,
         router_node.get_fast_llm, understand_node.graph_available,
         router_node.graph_available) = saved


def steps(state):
    return {t.name: t for t in state.traces}


# ---------------------------------------------------------------------------
# the gate that fired on "the"
# ---------------------------------------------------------------------------
def test_the_word_the_no_longer_looks_like_a_follow_up():
    """"he " was matched by substring, so it hit inside "t-h-e ". Every question
    containing the word "the" therefore looked like a follow-up: 44 of 47
    second-and-later turns paid a rewrite they did not need."""
    for question in [
        "What is the recommended tyre pressure for the Curvv EV?",
        "What is the exact battery cell chemistry used?",
        "What is the rating and description of fuse EF1 in the fuse box?",
        "What is the on-road price in Delhi?",
    ]:
        assert not needs_rewrite(question, HISTORY), question


def test_questions_that_genuinely_lean_on_the_conversation_still_do():
    for question in [
        "What about the second one?",
        "And its warranty?",
        "Is that covered?",
        "How long does it take?",
        "Tell me more about them",
        "What did you say earlier?",
        "Why?",
    ]:
        assert needs_rewrite(question, HISTORY), question


def test_the_first_question_of_a_session_is_never_a_follow_up():
    assert not needs_rewrite("What about it?", [])


# ---------------------------------------------------------------------------
# how many calls each shape of question costs
# ---------------------------------------------------------------------------
def test_a_standalone_question_the_heuristic_can_route_costs_nothing():
    llm = FakeLLM()
    state = drive("What is the refund window?", llm)

    assert llm.calls == 0, llm.calls
    assert state.route == Route.VECTOR
    assert state.route_decided_by == "heuristic"
    assert state.condensed_question == "What is the refund window?"


def test_a_follow_up_with_an_ambiguous_route_costs_one_call_not_two():
    llm = FakeLLM({
        "question": "Does the Model B pump support continuous duty?",
        "route": "HYBRID", "confidence": 0.7, "reason": "needs text and relationships",
    })
    state = drive("Does it support continuous duty?", llm)

    assert llm.calls == 1, llm.calls
    assert understand_node.UNDERSTAND_SYSTEM in llm.systems[0]
    assert state.condensed_question == "Does the Model B pump support continuous duty?"
    assert state.route == Route.HYBRID
    assert state.route_decided_by == "llm"
    # Both steps still appear in the trail; only the round trip was removed.
    assert steps(state)["condense"].meta["rewritten"] is True
    assert steps(state)["route"].meta["combined"] is True


def test_the_route_heuristic_still_wins_when_it_is_confident():
    """It reads the rewritten question, which is the order it ran in when
    condensing was its own step, and it is deterministic and free."""
    llm = FakeLLM({
        "question": "What is the warranty on the Model B pump?",
        "route": "MULTIHOP", "confidence": 0.9, "reason": "model thinks it is complex",
    })
    state = drive("What is its warranty?", llm)

    assert llm.calls == 1
    assert state.route == Route.VECTOR, state.route
    assert state.route_decided_by == "heuristic"


def test_a_forced_route_never_pays_for_a_classifier():
    llm = FakeLLM({"question": "What is the warranty on the Model B pump?"})
    state = drive("What is its warranty?", llm, forced=Route.GRAPH)

    assert llm.calls == 1, "only the rewrite"
    assert state.route_decided_by == "forced"


# ---------------------------------------------------------------------------
# it must not make things worse when the model misbehaves
# ---------------------------------------------------------------------------
def test_an_unusable_rewrite_keeps_the_original_question():
    llm = FakeLLM({
        "question": "Here is my reasoning:\nfirst, the pump\nsecond, the warranty",
        "route": "VECTOR", "confidence": 0.8, "reason": "lookup",
    })
    state = drive("What is its warranty?", llm)

    assert state.condensed_question == "What is its warranty?"
    assert steps(state)["condense"].meta["rewritten"] is False


def test_an_unparseable_route_falls_back_to_the_safe_default():
    llm = FakeLLM({
        "question": "Does the Model B pump support continuous duty?",
        "route": "SOMETHING_ELSE", "confidence": "high",
    })
    state = drive("Does it support continuous duty?", llm)

    assert state.route == Route.HYBRID
    assert steps(state)["route"].meta["confidence"] == 0.5


def test_a_dead_provider_falls_back_to_the_separate_steps():
    """The combined call is an optimisation. When it fails the turn still has to
    happen, so both steps run their own way and the route defaults safely."""
    llm = FakeLLM(fail=True)
    state = drive("Does it support continuous duty?", llm)

    assert state.condensed_question == "Does it support continuous duty?"
    assert state.route in (Route.HYBRID, Route.VECTOR)
    assert state.route_decided_by in ("fallback", "heuristic")
    assert "condense" in steps(state) and "route" in steps(state)


def test_the_graph_is_never_routed_to_when_it_is_offline():
    llm = FakeLLM({
        "question": "How does the Model B pump relate to the Q3 recall?",
        "route": "GRAPH", "confidence": 0.9, "reason": "relational",
    })
    state = drive("How does it relate to the recall?", llm, graph_on=False)

    assert state.route == Route.VECTOR, state.route
