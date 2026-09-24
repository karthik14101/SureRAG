"""Expansion must not lose evidence an earlier pass already credited.

Reranking was made non-destructive after it deleted a passage the verifier had
confirmed. This is the same fault one level up: the verifier reads only the
leading twelve passages, and that window slides as expansion merges new ones in.

The turn that showed it. Asked for the recommended tyre pressure of a Curvv EV,
the first verification credited "recommended tyre pressures are printed on a
sticker near the driver seat" and scored 0.3. Expansion added sixteen passages,
the sticker fell past twelfth, the second verification scored 0.2 without it,
and the answer was written from the poorer pool -- never mentioning the sticker,
which is the one genuinely useful thing the manual says on the subject. The same
question asked eight minutes earlier kept that passage and answered well, so the
difference between the two runs was which passages survived the window.
"""
from __future__ import annotations

import asyncio

from app.agent import orchestrator
from app.agent.nodes import expand as expand_node
from app.agent.state import AgentState
from app.llm.base import TokenUsage
from app.vectorstore.search import RetrievedChunk


def chunk(i, score=0.5):
    return RetrievedChunk(
        chunk_id=f"c{i}", doc_id="d1", kb_id="kb1",
        filename="curvv-ev-owners-manual.pdf",
        text=f"Passage {i}.", score=score, ordinal=i, source="vector",
    )


def state_with(chunks):
    state = AgentState(question="q", user_id="u", kb_id="kb1", session_id="s")
    state.chunks = list(chunks)
    return state


def ids(state):
    return [c.chunk_id for c in state.chunks]


# ---------------------------------------------------------------------------
# recording the best window
# ---------------------------------------------------------------------------
def test_the_best_verdict_of_the_turn_is_what_is_remembered():
    state = state_with([chunk(i) for i in range(3)])

    state.record_verdict(0.3, ["c0", "c1"])
    state.record_verdict(0.2, ["c9"])

    assert state.best_score == 0.3
    assert state.best_window_ids == ["c0", "c1"]


def test_an_equal_verdict_takes_the_later_window():
    """A pass that scored no worse judged fresher evidence, and the loop is about
    to stop on it either way."""
    state = state_with([chunk(i) for i in range(3)])

    state.record_verdict(0.3, ["c0"])
    state.record_verdict(0.3, ["c1"])

    assert state.best_window_ids == ["c1"]


# ---------------------------------------------------------------------------
# restoring it
# ---------------------------------------------------------------------------
def test_a_pass_that_lowered_the_verdict_gives_the_earlier_evidence_back():
    """The tyre-pressure turn. c2 is the sticker passage."""
    state = state_with([chunk(i) for i in range(3)])
    state.record_verdict(0.3, ["c2", "c0"])

    # Expansion merges sixteen new passages in and pushes c2 down the pool.
    state.chunks = [chunk(100 + n) for n in range(16)] + state.chunks
    state.sufficiency_score = 0.2

    assert state.restore_best_window()
    assert ids(state)[:2] == ["c2", "c0"], ids(state)[:4]
    # The score follows the pool: reporting the last verdict for evidence that
    # is no longer the evidence would misstate what the answer rests on.
    assert state.sufficiency_score == 0.3


def test_recovery_keeps_every_passage_expansion_found():
    """Reordering is safe; discarding is not. Whatever the later pass turned up
    stays in the pool, ranked behind the evidence that earned the better
    verdict."""
    state = state_with([chunk(i) for i in range(3)])
    state.record_verdict(0.3, ["c2"])
    state.chunks = [chunk(100 + n) for n in range(16)] + state.chunks
    state.sufficiency_score = 0.2

    before = set(ids(state))
    state.restore_best_window()

    assert set(ids(state)) == before
    assert len(state.chunks) == 19


def test_an_improving_verdict_recovers_nothing():
    state = state_with([chunk(i) for i in range(3)])
    state.record_verdict(0.3, ["c2"])
    state.sufficiency_score = 0.7

    assert not state.restore_best_window()
    assert ids(state) == ["c0", "c1", "c2"]
    assert state.sufficiency_score == 0.7


def test_a_turn_that_was_never_verified_recovers_nothing():
    state = state_with([chunk(i) for i in range(3)])
    state.sufficiency_score = 0.2

    assert not state.restore_best_window()


def test_passages_that_left_the_pool_entirely_are_not_conjured_back():
    """Recovery reorders what is there; it cannot resurrect what a retrieval
    path dropped, and must not pretend otherwise."""
    state = state_with([chunk(9)])
    state.record_verdict(0.3, ["c2", "c0"])
    state.sufficiency_score = 0.2

    assert not state.restore_best_window()
    assert state.sufficiency_score == 0.2


# ---------------------------------------------------------------------------
# through the orchestrator, where it actually runs
# ---------------------------------------------------------------------------
def test_the_loop_answers_from_the_recovered_pool_and_says_so():
    state = state_with([chunk(i) for i in range(3)])
    state.record_verdict(0.3, ["c2", "c0"])
    state.chunks = [chunk(100 + n) for n in range(16)] + state.chunks
    state.sufficiency_score = 0.2
    state.stop_reason = "the last search did not improve the evidence"

    orchestrator._settle(state)

    names = [t.name for t in state.traces]
    assert "recover" in names, names
    # The reader is told the answer rests on the recovered pool, at its score.
    assert "30%" in state.warnings[0], state.warnings
    assert ids(state)[:2] == ["c2", "c0"]


def test_a_healthy_turn_is_left_alone():
    state = state_with([chunk(i) for i in range(3)])
    state.record_verdict(0.9, ["c0"])
    state.sufficiency_score, state.sufficient = 0.9, True

    orchestrator._settle(state)

    assert [t.name for t in state.traces] == []
    assert not state.warnings


# ---------------------------------------------------------------------------
# what expansion is told about the evidence it already has
# ---------------------------------------------------------------------------
class RecordingLLM:
    def __init__(self):
        self.prompt = ""

    async def complete_json(self, system, prompt, **kw):
        self.prompt = prompt
        return {"queries": ["limp home mode telltale amber"]}, TokenUsage(calls=1)


def test_query_generation_is_given_the_documents_own_vocabulary():
    """It used to be handed the filenames of the top five passages, five times
    over, in the slot the template labels "already covered by the retrieved
    context". The verifier's covered list goes there instead, because that is
    where a document's own term for the thing shows up -- ask about a turtle icon
    and it comes back saying "limp home mode", which is what is worth searching
    for."""
    import app.llm.factory as factory

    saved = factory.get_fast_llm
    llm = RecordingLLM()
    factory.get_fast_llm = lambda: llm
    expand_node.get_fast_llm = lambda: llm
    try:
        state = state_with([chunk(i) for i in range(3)])
        state.covered_aspects = ["warning lamps and chimes, including limp home mode"]
        state.missing_aspects = ["the meaning of the turtle symbol"]
        queries = asyncio.run(expand_node._generate_queries(state))
    finally:
        factory.get_fast_llm = saved
        expand_node.get_fast_llm = saved

    assert "limp home mode" in llm.prompt, llm.prompt
    assert "curvv-ev-owners-manual.pdf" not in llm.prompt, llm.prompt
    assert queries == ["limp home mode telltale amber"]


def test_the_verifiers_own_suggestions_still_win_when_it_made_any():
    state = state_with([chunk(0)])
    state.expansion_queries = ["limp home telltale"]
    state.missing_aspects = ["anything"]

    assert asyncio.run(expand_node._generate_queries(state)) == ["limp home telltale"]
