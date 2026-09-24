"""The SURE loop: corpus-aware verification and when to stop expanding.

A fake LLM replays the verifier verdicts seen in real traces -- the Right to
Education, MRP and Article 368 turns -- so the loop runs end to end without a
provider. Every stop rule softer than "did the score go up?" has been tried and
failed here, and each failure is preserved below as its own test.
"""
from __future__ import annotations

import asyncio
import contextlib
import json

from app.agent import orchestrator
from app.agent.nodes import expand as expand_node
from app.agent.nodes import verifier as verify_node
from app.agent.state import AgentState, Route
from app.config import settings
from app.llm.base import LLMResult, TokenUsage
from app.vectorstore.search import RetrievedChunk

INSUFFICIENT = {
    "sufficient": False, "score": 0.3, "in_scope": True,
    "missing": ["the 86th Amendment", "case law"],
    "suggested_queries": ["86th amendment"], "reason": "gaps remain",
}
OUT_OF_SCOPE = {
    "sufficient": False, "score": 0.1, "in_scope": False,
    "missing": [], "suggested_queries": [], "reason": "corpus holds no case law",
}
GOOD = {"sufficient": True, "score": 0.95, "in_scope": True, "missing": [], "reason": "covered"}


def chunk(i):
    return RetrievedChunk(
        chunk_id=f"c{i}", doc_id="d1", kb_id="kb1", filename="constitution_of_india.json",
        text=f"Article {i}. Text of the article.", score=0.5, ordinal=i, source="vector",
    )


class FakeLLM:
    """Returns queued verifier verdicts; records every prompt it was given."""

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.prompts = []
        self.calls = 0

    async def complete_json(self, system, prompt, **kw):
        self.calls += 1
        self.prompts.append(prompt)
        return (self.verdicts.pop(0) if self.verdicts else {}), TokenUsage(calls=1)

    async def complete(self, system, messages, **kw):
        self.calls += 1
        return LLMResult(text=json.dumps({"queries": ["q1"]}), usage=TokenUsage(calls=1))


def state_with(chunks=14):
    state = AgentState(
        question="q", user_id="u", kb_id="kb1", session_id="s",
        corpus_profile='Knowledge base: "Constitution". 1 document: constitution_of_india.json',
    )
    state.route = Route.VECTOR
    state.chunks = [chunk(i) for i in range(chunks)]
    return state


@contextlib.contextmanager
def patched(llm, gain=0, promote=False):
    """Point the nodes at the fake LLM and stub retrieval, then put it all back.

    `promote` decides whether new passages reach the top of the list, which is
    what the reranker does for evidence that genuinely answers the question --
    and what the discarded turnover rule used to measure.

    Restoring matters: these are module globals, and leaving them patched would
    make other test files pass or fail depending on collection order.
    """
    import app.llm.factory as factory

    # The verifier imports get_fast_llm INSIDE its function, so the factory is
    # the only place worth patching -- assigning verify_node.get_fast_llm just
    # creates an attribute nothing ever reads.
    saved = (
        factory.get_fast_llm,
        expand_node.expand_and_retrieve,
        orchestrator.rerank_node.rerank,
    )

    async def fake_expand(state):
        state.iterations += 1
        fresh = [chunk(1000 + state.iterations * 50 + n) for n in range(gain)]
        state.chunks = (fresh + state.chunks) if promote else (state.chunks + fresh)
        state.last_expansion_gain = gain
        state.add_trace("expand", f"Expansion pass {state.iterations}", "")
        return state

    async def fake_rerank(state):
        return state

    factory.get_fast_llm = lambda: llm
    expand_node.expand_and_retrieve = fake_expand
    orchestrator.rerank_node.rerank = fake_rerank
    try:
        yield
    finally:
        (factory.get_fast_llm, expand_node.expand_and_retrieve,
         orchestrator.rerank_node.rerank) = saved


async def run_loop(state):
    """The loop body from `_prepare`, without retrieval."""
    await verify_node.verify_sufficiency(state)
    while not state.sufficient and not orchestrator._should_stop(state) and state.can_iterate():
        await expand_node.expand_and_retrieve(state)
        if state.sufficient:
            break
        await orchestrator.rerank_node.rerank(state)
        await verify_node.verify_sufficiency(state)
    if state.stop_reason:
        state.add_trace("budget", "Stopped expanding", state.stop_reason)
    return state


def drive(llm, gain=0, promote=False, chunks=14):
    settings.agent_max_llm_calls = 6
    settings.agent_max_iterations = 2
    state = state_with(chunks)
    with patched(llm, gain=gain, promote=promote):
        asyncio.run(run_loop(state))
    return state


def test_the_corpus_profile_reaches_the_verifier():
    llm = FakeLLM([GOOD])
    state = drive(llm)

    assert "constitution_of_india.json" in llm.prompts[0]
    assert state.sufficient and state.iterations == 0
    # Gaps are only actionable while the loop can still act on them.
    assert state.traces[0].meta["missing"] == []


def test_out_of_scope_stops_immediately():
    """The MRP turn: searching again cannot conjure what was never ingested."""
    llm = FakeLLM([OUT_OF_SCOPE] * 3)
    state = drive(llm, gain=20, promote=True)

    assert llm.calls == 1, llm.calls
    assert state.iterations == 0, state.iterations
    assert state.out_of_scope
    assert state.traces[0].label.endswith("outside this knowledge base")


def test_reworded_gaps_no_longer_defeat_the_stop_rule():
    """The Article 368 turn. Comparing the gap list as text failed because the
    model rephrases the same complaint every round."""
    reworded = [
        dict(INSUFFICIENT, score=0.30, missing=["text of Article 368 is absent"]),
        dict(INSUFFICIENT, score=0.32, missing=["the specific proviso in 368(2) is not present"]),
        dict(INSUFFICIENT, score=0.33, missing=["no direct textual link to the Seventh Schedule"]),
    ]
    llm = FakeLLM(reworded)
    state = drive(llm, gain=20, promote=False)

    assert state.iterations == 1, state.iterations
    assert llm.calls == 2, llm.calls
    assert "did not improve the evidence" in state.stop_reason, state.stop_reason
    assert not state.budget_exhausted()


def test_churn_without_progress_no_longer_buys_another_pass():
    """The case that regressed on live traffic. Fresh passages arrive on every
    pass -- the corpus is large and the queries keep changing -- so treating
    turnover as progress meant the loop ran until a hard budget killed it. A
    question answered well in 36s came back 50% worse in 72s, having discarded
    evidence it had already found."""
    llm = FakeLLM([
        dict(INSUFFICIENT, score=0.30),
        dict(INSUFFICIENT, score=0.31, missing=["still thin"]),   # no score gain...
        GOOD,
    ])
    state = drive(llm, gain=20, promote=True)                     # ...but the window changed

    assert state.iterations == 1, state.iterations
    assert llm.calls == 2, llm.calls
    assert "did not improve" in state.stop_reason, state.stop_reason
    assert not state.budget_exhausted()


def test_an_improving_score_keeps_the_loop_alive():
    llm = FakeLLM([
        dict(INSUFFICIENT, score=0.30),
        dict(INSUFFICIENT, score=0.50, missing=["only dates now"]),
        dict(GOOD, score=0.80),
    ])
    state = drive(llm, gain=20, promote=False)

    assert state.iterations == 2, state.iterations
    assert not state.stop_reason, state.stop_reason


def test_an_expansion_that_finds_nothing_stops_the_loop():
    llm = FakeLLM([dict(INSUFFICIENT, score=0.3), dict(INSUFFICIENT, score=0.3)])
    state = drive(llm, gain=0, promote=False)

    assert state.iterations == 1, state.iterations
    # The real expand node also short-circuits on zero gain; this exercises the
    # orchestrator's own guard, which must hold even if that one is bypassed.
    assert "did not improve the evidence" in state.stop_reason, state.stop_reason
    assert "0 new passage" in state.stop_reason, state.stop_reason


def test_the_iteration_budget_still_caps_a_loop_that_keeps_improving():
    llm = FakeLLM([
        dict(INSUFFICIENT, score=0.30),
        dict(INSUFFICIENT, score=0.45, missing=["a"]),
        dict(INSUFFICIENT, score=0.60, missing=["b"]),
        dict(INSUFFICIENT, score=0.64, missing=["c"]),
    ])
    state = drive(llm, gain=20, promote=True)

    assert state.iterations == 2, state.iterations


def test_a_failed_verification_still_answers():
    class Broken(FakeLLM):
        async def complete_json(self, system, prompt, **kw):
            raise RuntimeError("provider down")

    state = drive(Broken([]))

    # A failed check must not block the answer; the synthesis grounding rules
    # carry the load instead.
    assert state.sufficient
    assert not state.out_of_scope


def test_the_synthesis_note_matches_the_verdict():
    from app.agent.nodes import synthesize

    state = state_with()
    state.out_of_scope, state.sufficient = True, False
    _, prompt, _ = synthesize.build_prompt(state)
    assert prompt.startswith("Scope note:")
    assert "under 150 words" in prompt

    state.out_of_scope, state.sufficient = False, False
    _, prompt, _ = synthesize.build_prompt(state)
    assert prompt.startswith("Note:") and "Do not pad" in prompt

    state.sufficient = True
    _, prompt, _ = synthesize.build_prompt(state)
    assert prompt.startswith("Question:")


def test_the_verifier_reads_more_evidence_than_the_answer_does():
    from app.agent.nodes import synthesize

    assert verify_node.VERIFY_CHUNK_LIMIT > synthesize.SYNTHESIS_CHUNK_LIMIT
    # 0 means "keep the whole pool". Any other value must clear the verifier's
    # window, or reranking discards evidence between expansion passes.
    assert settings.rerank_top_n == 0 or settings.rerank_top_n > verify_node.VERIFY_CHUNK_LIMIT
