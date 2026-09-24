"""The state object threaded through every node of the SURE loop.

Nodes read and mutate one AgentState, exactly like a graph framework's channel
state, but with a plain dataclass so it is trivially inspectable and can be
serialised into the message trace the UI renders.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.llm.base import ChatMessage, TokenUsage
from app.vectorstore.search import RetrievedChunk


class Route(str, Enum):
    """Retrieval strategy chosen by the router."""

    VECTOR = "VECTOR"      # semantic lookup in Qdrant
    GRAPH = "GRAPH"        # entity/relationship traversal in Neo4j
    HYBRID = "HYBRID"      # both, merged
    MULTIHOP = "MULTIHOP"  # decompose into sub-questions, retrieve per step
    DIRECT = "DIRECT"      # no retrieval needed (greetings, meta-questions)

    @classmethod
    def parse(cls, value: str | None) -> "Route | None":
        if not value:
            return None
        try:
            return cls(str(value).strip().upper())
        except ValueError:
            return None


def _fast_tier_configured() -> bool:
    """Whether the cheap tier is a genuinely different (cheaper) model.

    Imported lazily: the factory pulls in provider SDKs, and the state object
    is the one module every node already depends on.
    """
    try:
        from app.llm.factory import has_fast_tier

        return has_fast_tier()
    except Exception:  # noqa: BLE001 - budgeting must never break a turn
        return False


@dataclass
class StepTrace:
    """One node execution, surfaced in the UI's Reasoning Trail."""

    name: str
    label: str
    detail: str = ""
    duration_ms: int = 0
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "detail": self.detail,
            "duration_ms": self.duration_ms,
            "meta": self.meta,
        }


@dataclass
class Citation:
    marker_index: int
    chunk: RetrievedChunk

    def as_dict(self) -> dict:
        return {
            "marker_index": self.marker_index,
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "filename": self.chunk.filename,
            "page_no": self.chunk.page_no,
            "section": self.chunk.section,
            "snippet": self.chunk.snippet(),
            "score": round(float(self.chunk.score), 4),
            "retrieval_source": self.chunk.source,
        }


@dataclass
class AgentState:
    # ---- inputs -------------------------------------------------------------
    question: str
    user_id: str
    kb_id: str
    session_id: str
    history: list[ChatMessage] = field(default_factory=list)
    history_summary: str | None = None
    forced_route: Route | None = None
    # What this knowledge base actually contains, in one line. The verifier reads
    # it so it can tell "we missed it" apart from "this corpus never had it".
    corpus_profile: str = ""

    # ---- working values -----------------------------------------------------
    condensed_question: str = ""
    route: Route = Route.HYBRID
    route_reason: str = ""
    route_decided_by: str = "heuristic"
    chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_text: str = ""
    graph_entities: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    # A vector search started speculatively while the router was still deciding.
    # Only ever set when the query is already final, so whatever it returns is
    # exactly what retrieval would have asked for.
    prefetch_task: object | None = None
    prefetch_query: str = ""
    prefetch_top_k: int = 0

    # ---- verification -------------------------------------------------------
    sufficiency_score: float = 0.0
    sufficient: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    # What the verifier said the evidence DOES answer. Kept because it is where
    # the documents' own vocabulary shows up: a reader asking about a turtle
    # icon gets back "limp home mode", which is the term worth searching for.
    covered_aspects: list[str] = field(default_factory=list)
    expansion_queries: list[str] = field(default_factory=list)
    iterations: int = 0
    # Set when the verifier judges the gap to be outside this corpus entirely.
    # Searching harder cannot fix that, so the loop stops and says so.
    out_of_scope: bool = False
    # The verifier's own answer to "is this findable here", kept as a tri-state
    # so that "it said yes" can be told from "it did not say". A number must not
    # overrule an explicit yes.
    verifier_in_scope: bool | None = None
    # What the shape probe managed to establish about this corpus, if anything.
    # Concluding that a corpus lacks a kind of material, without having
    # established what it does hold, is a guess wearing a verdict's clothes.
    corpus_shape: str = ""
    # Why the expansion loop stopped, for the trace and the UI.
    stop_reason: str = ""
    # Previous round's verdict, used to detect an expansion that changed nothing.
    previous_score: float | None = None
    previous_missing: list[str] = field(default_factory=list)
    # Passages the last expansion pass actually added, and how many of them
    # reached the window the verifier reads. New evidence nobody looks at is
    # the same as no new evidence.
    last_expansion_gain: int = 0
    judged_chunk_ids: list[str] = field(default_factory=list)
    last_judged_turnover: float = 1.0
    # The best verdict of the turn and the window that earned it. Expansion can
    # push a passage the verifier already credited out of the window it reads,
    # and the answer is then written without it -- see restore_best_window.
    best_score: float = 0.0
    best_window_ids: list[str] = field(default_factory=list)

    # ---- outputs ------------------------------------------------------------
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    media_ids: list[str] = field(default_factory=list)
    # Share of the written answer's claims that carry a citation. Measured
    # after synthesis, unlike sufficiency_score, which judges the evidence pool
    # before a word is written. None when there is nothing to measure.
    grounding: float | None = None

    # ---- budget and bookkeeping --------------------------------------------
    # Counted apart because they cost differently: `llm_calls` is full-reasoning
    # work, `fast_llm_calls` is the cheap tier the loop runs on.
    llm_calls: int = 0
    fast_llm_calls: int = 0
    usage: TokenUsage = field(default_factory=TokenUsage)
    traces: list[StepTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    # ---- derived ------------------------------------------------------------
    @property
    def query(self) -> str:
        """The text retrieval should use: rewritten when available."""
        return self.condensed_question or self.question

    @property
    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def total_llm_calls(self) -> int:
        """Every model call this turn made, whichever tier served it."""
        return self.llm_calls + self.fast_llm_calls

    def budget_exhausted(self) -> bool:
        """Hard stops so a pathological question cannot loop or drain quota."""
        if self.llm_calls >= settings.agent_max_llm_calls:
            return True
        if self.fast_llm_calls >= settings.agent_max_fast_llm_calls:
            return True
        if self.elapsed_seconds >= settings.agent_timeout_seconds:
            return True
        return False

    def can_iterate(self) -> bool:
        return (
            self.iterations < settings.agent_max_iterations
            and not self.budget_exhausted()
        )

    def spend_call(self, usage: TokenUsage | None = None, *, fast: bool = False) -> None:
        """Record a model call against the budget its cost belongs to.

        A call only counts as cheap when a fast tier is actually configured:
        without one, `get_fast_llm()` hands back the main model, and charging
        those calls to the lenient budget would quietly let a turn spend twice
        what the operator asked for.
        """
        if fast and _fast_tier_configured():
            self.fast_llm_calls += 1
        else:
            self.llm_calls += 1
        if usage is not None:
            self.usage.add(usage)

    def add_trace(
        self, name: str, label: str, detail: str = "", started: float | None = None, **meta
    ) -> None:
        duration = int((time.perf_counter() - started) * 1000) if started else 0
        self.traces.append(
            StepTrace(name=name, label=label, detail=detail, duration_ms=duration, meta=meta)
        )

    async def take_prefetch(self, query: str, top_k: int) -> list[RetrievedChunk] | None:
        """Claim the speculative search, if it asked exactly this question.

        Routing can need a model call, and retrieval cannot begin until it
        lands -- two seconds of measured silence during which the vector index
        sits idle. When the question needs no rewriting the query is already
        final, so the search can run alongside the decision whatever the router
        then picks.

        The query and top_k must match, because a result fetched for a
        different question is not a shortcut, it is the wrong evidence.
        """
        task, self.prefetch_task = self.prefetch_task, None
        if task is None:
            return None
        if self.prefetch_query != query or self.prefetch_top_k != top_k:
            task.cancel()
            return None
        try:
            return await task
        except Exception:  # noqa: BLE001 - let the caller do its own search and report
            return None

    def discard_prefetch(self) -> None:
        """Cancel a speculative search nothing is going to claim."""
        task, self.prefetch_task = self.prefetch_task, None
        if task is not None:
            task.cancel()

    def record_verdict(self, score: float, window: list[str]) -> None:
        """Remember the best-judged window of the turn."""
        if score >= self.best_score:
            self.best_score = score
            self.best_window_ids = list(window)

    def restore_best_window(self) -> bool:
        """Rank the best-judged passages back to the front of the pool.

        Reranking was already made non-destructive, but the verifier's window is
        only the leading twelve passages and it still slides. A tyre-pressure
        turn showed the cost: pass one credited "recommended pressures are
        printed on a sticker near the driver seat" and scored 0.3, expansion
        added sixteen passages, the sticker fell past twelfth, pass two scored
        0.2 without it, and the answer was written from the poorer pool. The
        same question asked minutes earlier kept that passage and answered well,
        so the difference between the two runs was which passages survived the
        window rather than anything about the question.

        Nothing is discarded here either: the passages expansion found stay in
        the pool, just behind the ones that earned the better verdict. The score
        moves with the pool, because reporting the last verdict for evidence
        that is no longer the evidence would misstate what the answer rests on.
        """
        if not self.best_window_ids or self.sufficiency_score >= self.best_score:
            return False

        order = {cid: i for i, cid in enumerate(self.best_window_ids)}
        front = sorted(
            (c for c in self.chunks if c.chunk_id in order),
            key=lambda c: order[c.chunk_id],
        )
        if not front:
            return False

        self.chunks = front + [c for c in self.chunks if c.chunk_id not in order]
        self.sufficiency_score = self.best_score
        return True

    def context_chunks(self, limit: int | None = None) -> list[RetrievedChunk]:
        cap = limit or settings.retrieval_top_k
        return self.chunks[:cap]

    def trace_dicts(self) -> list[dict]:
        return [t.as_dict() for t in self.traces]
