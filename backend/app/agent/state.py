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

    # ---- working values -----------------------------------------------------
    condensed_question: str = ""
    route: Route = Route.HYBRID
    route_reason: str = ""
    route_decided_by: str = "heuristic"
    chunks: list[RetrievedChunk] = field(default_factory=list)
    graph_text: str = ""
    graph_entities: list[str] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)

    # ---- verification -------------------------------------------------------
    sufficiency_score: float = 0.0
    sufficient: bool = False
    missing_aspects: list[str] = field(default_factory=list)
    expansion_queries: list[str] = field(default_factory=list)
    iterations: int = 0

    # ---- outputs ------------------------------------------------------------
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    media_ids: list[str] = field(default_factory=list)

    # ---- budget and bookkeeping --------------------------------------------
    llm_calls: int = 0
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

    def budget_exhausted(self) -> bool:
        """Hard stops so a pathological question cannot loop or drain quota."""
        if self.llm_calls >= settings.agent_max_llm_calls:
            return True
        if self.elapsed_seconds >= settings.agent_timeout_seconds:
            return True
        return False

    def can_iterate(self) -> bool:
        return (
            self.iterations < settings.agent_max_iterations
            and not self.budget_exhausted()
        )

    def spend_call(self, usage: TokenUsage | None = None) -> None:
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

    def context_chunks(self, limit: int | None = None) -> list[RetrievedChunk]:
        cap = limit or settings.retrieval_top_k
        return self.chunks[:cap]

    def trace_dicts(self) -> list[dict]:
        return [t.as_dict() for t in self.traces]
