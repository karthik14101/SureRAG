"""Work that does not depend on other work must not wait for it.

Dense and sparse never posed this question -- they go to Qdrant as two prefetch
branches of one `query_points` call and are fused server-side, so there is one
round trip rather than two to sequence. The places that did pose it were
quieter: a multi-query pass looped its queries one at a time, the two query
encoders were awaited back to back, and multi-hop retrieval traversed the graph
only once every search had finished.

None of this changes what comes back. The tests below pin that down first,
because a latency fix that quietly reorders evidence is not a latency fix.
"""
from __future__ import annotations

import asyncio
import time

from app.agent import orchestrator
from app.agent.nodes import retrieve_hybrid as multihop_node
from app.agent.nodes import retrieve_vector as vector_node
from app.agent.state import AgentState, Route
from app.config import settings
from app.llm.base import ChatMessage
from app.vectorstore import search


def chunk(cid, score, source="vector"):
    return search.RetrievedChunk(
        chunk_id=cid, doc_id="d1", kb_id="kb1", filename="manual.pdf",
        text=f"Passage {cid}.", score=score, source=source,
    )


class Recorder:
    """Stands in for hybrid_search, logging when each call was in flight."""

    def __init__(self, hits, delay=0.05, fail_on=()):
        self.hits, self.delay, self.fail_on = hits, delay, set(fail_on)
        self.spans: list[tuple[str, float, float]] = []
        self.live, self.peak = 0, 0

    async def __call__(self, query, **kw):
        self.live += 1
        self.peak = max(self.peak, self.live)
        started = time.perf_counter()
        try:
            await asyncio.sleep(self.delay)
            if query in self.fail_on:
                raise RuntimeError("Qdrant unavailable")
            return list(self.hits.get(query, []))
        finally:
            self.spans.append((query, started, time.perf_counter()))
            self.live -= 1


def install(recorder):
    saved = search.hybrid_search
    search.hybrid_search = recorder
    return saved


def run(queries, recorder, **kw):
    saved = install(recorder)
    try:
        return asyncio.run(
            search.multi_query_search(queries, user_id="u", kb_id="kb1", **kw)
        )
    finally:
        search.hybrid_search = saved


# ---------------------------------------------------------------------------
# the result must not change
# ---------------------------------------------------------------------------
HITS = {
    "q1": [chunk("a", 0.9), chunk("b", 0.4)],
    "q2": [chunk("b", 0.7), chunk("c", 0.5)],
    "q3": [chunk("a", 0.2), chunk("d", 0.8)],
}


def sequential_merge(queries):
    """What the old loop produced, kept here as the thing to match."""
    merged = {}
    for query in queries:
        for hit in HITS.get(query, []):
            existing = merged.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                merged[hit.chunk_id] = hit
    return [(c.chunk_id, c.score) for c in
            sorted(merged.values(), key=lambda c: c.score, reverse=True)]


def test_the_merged_result_is_exactly_what_the_loop_produced():
    """Each chunk keeps its highest score across the queries, which cannot
    depend on the order they ran in, and `gather` returns results in the order
    they were requested, so ties resolve as before."""
    queries = ["q1", "q2", "q3"]
    out = run(queries, Recorder(HITS))

    assert [(c.chunk_id, c.score) for c in out] == sequential_merge(queries)
    # b was found twice; the better score is the one that survives.
    assert dict((c.chunk_id, c.score) for c in out)["b"] == 0.7


def test_blank_queries_are_skipped_and_an_empty_list_costs_nothing():
    recorder = Recorder(HITS)
    assert run(["", "   "], recorder) == []
    assert run([], recorder) == []
    assert recorder.spans == []


def test_one_failing_query_does_not_take_the_others_with_it():
    recorder = Recorder(HITS, fail_on=["q2"])
    out = run(["q1", "q2", "q3"], recorder)

    assert {c.chunk_id for c in out} == {"a", "b", "d"}
    assert len(recorder.spans) == 3, "the failure must not cancel its siblings"


# ---------------------------------------------------------------------------
# and it must actually be concurrent
# ---------------------------------------------------------------------------
def test_the_queries_overlap_instead_of_queueing():
    recorder = Recorder(HITS, delay=0.05)
    started = time.perf_counter()
    run(["q1", "q2", "q3"], recorder)
    elapsed = time.perf_counter() - started

    assert recorder.peak == 3, recorder.peak
    # Three 50ms searches in series is 150ms; overlapped they are one 50ms wait.
    assert elapsed < 0.12, elapsed


def test_concurrency_is_capped_so_a_wide_turn_cannot_thrash_the_pool():
    """The Qdrant round trips parallelise cleanly, but the local encoders that
    precede them compete for the same cores."""
    queries = [f"q{n}" for n in range(12)]
    recorder = Recorder({}, delay=0.02)
    run(queries, recorder)

    assert recorder.peak == search.MAX_CONCURRENT_QUERIES, recorder.peak
    assert len(recorder.spans) == 12


# ---------------------------------------------------------------------------
# multi-hop: the graph half no longer waits for the search half
# ---------------------------------------------------------------------------
def test_multihop_traverses_the_graph_while_it_searches():
    state = AgentState(question="q", user_id="u", kb_id="kb1", session_id="s")
    state.condensed_question = "how does X relate to Y"
    order: list[str] = []

    class Context:
        is_empty = False
        seeds = ["X", "Y"]
        chunk_ids = ["g1"]

        def as_text(self):
            return "X -> Y"

    async def fake_decompose(_state):
        return ["what is X", "what is Y"]

    async def fake_multi(queries, **kw):
        order.append("search:start")
        await asyncio.sleep(0.05)
        order.append("search:end")
        return [chunk("v1", 0.9)]

    async def fake_traverse(query, kb_id):
        order.append("graph:start")
        await asyncio.sleep(0.05)
        order.append("graph:end")
        return Context()

    async def fake_fetch(ids, **kw):
        return [chunk("g1", 0.5, source="graph")]

    saved = (multihop_node.decompose_question, multihop_node.search.multi_query_search,
             multihop_node.traversal.traverse, multihop_node.search.fetch_chunks_by_id,
             multihop_node.is_available)
    (multihop_node.decompose_question, multihop_node.search.multi_query_search,
     multihop_node.traversal.traverse, multihop_node.search.fetch_chunks_by_id,
     multihop_node.is_available) = (
        fake_decompose, fake_multi, fake_traverse, fake_fetch, lambda: True)
    try:
        started = time.perf_counter()
        asyncio.run(multihop_node.retrieve_multihop(state))
        elapsed = time.perf_counter() - started
    finally:
        (multihop_node.decompose_question, multihop_node.search.multi_query_search,
         multihop_node.traversal.traverse, multihop_node.search.fetch_chunks_by_id,
         multihop_node.is_available) = saved

    # Both had started before either finished.
    assert order[:2] == ["search:start", "graph:start"], order
    assert elapsed < 0.09, elapsed
    assert {c.chunk_id for c in state.chunks} == {"v1", "g1"}
    assert state.graph_entities == ["X", "Y"]


def test_multihop_survives_a_graph_that_is_offline():
    state = AgentState(question="q", user_id="u", kb_id="kb1", session_id="s")

    async def fake_decompose(_state):
        return ["a", "b"]

    async def fake_multi(queries, **kw):
        return [chunk("v1", 0.9)]

    saved = (multihop_node.decompose_question, multihop_node.search.multi_query_search,
             multihop_node.is_available)
    (multihop_node.decompose_question, multihop_node.search.multi_query_search,
     multihop_node.is_available) = (fake_decompose, fake_multi, lambda: False)
    try:
        asyncio.run(multihop_node.retrieve_multihop(state))
    finally:
        (multihop_node.decompose_question, multihop_node.search.multi_query_search,
         multihop_node.is_available) = saved

    assert [c.chunk_id for c in state.chunks] == ["v1"]
    assert state.traces[-1].meta["graph"] == 0


# ---------------------------------------------------------------------------
# the vector search that runs while the question is still being understood
# ---------------------------------------------------------------------------
def state_for(question, history=()):
    state = AgentState(question=question, user_id="u", kb_id="kb1", session_id="s")
    state.history = list(history)
    return state


def with_prefetch(state, recorder, body):
    """Start the speculative search, run `body`, then make sure nothing leaks."""
    saved = search.hybrid_search
    search.hybrid_search = recorder

    async def go():
        orchestrator._start_prefetch(state)
        try:
            return await body(state)
        finally:
            state.discard_prefetch()

    try:
        return asyncio.run(go())
    finally:
        search.hybrid_search = saved


def test_the_speculative_search_is_claimed_by_the_vector_route():
    """Routing can cost a model call, and retrieval cannot start until it lands.
    When the query needs no rewriting it is already final, so the search runs
    alongside the decision -- and the route then finds it waiting."""
    recorder = Recorder({"What is the refund window?": [chunk("a", 0.9)]}, delay=0.01)
    state = state_for("What is the refund window?")

    with_prefetch(state, recorder, vector_node.retrieve_vector)

    assert [c.chunk_id for c in state.chunks] == ["a"]
    assert len(recorder.spans) == 1, "the route must not search again"
    assert state.prefetch_task is None


def test_a_question_that_is_about_to_be_rewritten_is_never_speculated_on():
    """Evidence fetched for a question nobody asked is not a shortcut."""
    recorder = Recorder({}, delay=0.01)
    history = [ChatMessage(role="user", content="a"), ChatMessage(role="assistant", content="b")]
    state = state_for("What about its warranty?", history)

    with_prefetch(state, recorder, lambda s: asyncio.sleep(0))

    assert state.prefetch_task is None
    assert recorder.spans == []


def test_a_prefetch_for_a_different_query_is_refused_and_cancelled():
    recorder = Recorder({}, delay=0.01)
    state = state_for("What is the refund window?")

    async def body(s):
        s.condensed_question = "something else entirely"
        return await s.take_prefetch(s.condensed_question, settings.retrieval_top_k)

    assert with_prefetch(state, recorder, body) is None


def test_a_prefetch_for_a_different_depth_is_refused():
    """A shallower or deeper result is not the one retrieval asked for."""
    recorder = Recorder({}, delay=0.01)
    state = state_for("What is the refund window?")

    async def body(s):
        return await s.take_prefetch(s.query, settings.retrieval_top_k + 5)

    assert with_prefetch(state, recorder, body) is None


def test_a_failed_prefetch_lets_the_route_do_its_own_search():
    """A speculative failure must be invisible. The route searches for real and
    the turn proceeds as though nothing had been attempted."""

    class FlakyOnce(Recorder):
        async def __call__(self, query, **kw):
            first = not self.spans
            self.spans.append((query, 0.0, 0.0))
            if first:
                raise RuntimeError("Qdrant blinked")
            return [chunk("a", 0.9)]

    recorder = FlakyOnce({}, delay=0)
    state = state_for("What is the refund window?")

    with_prefetch(state, recorder, vector_node.retrieve_vector)

    assert len(recorder.spans) == 2, "prefetch failed, then the route searched"
    assert [c.chunk_id for c in state.chunks] == ["a"]


def test_a_conversational_turn_cancels_the_search_it_did_not_need():
    recorder = Recorder({}, delay=0.05)
    state = state_for("What is the refund window?")

    async def body(s):
        s.route = Route.DIRECT
        s.discard_prefetch()
        return None

    with_prefetch(state, recorder, body)
    assert state.prefetch_task is None
