"""Schemas describing the agent's internal decisions, exposed for the UI trail."""
from __future__ import annotations

from pydantic import BaseModel


class RouteDecision(BaseModel):
    route: str
    confidence: float = 0.0
    reason: str = ""
    decided_by: str = "heuristic"  # heuristic | llm | forced | fallback


class SufficiencyVerdict(BaseModel):
    sufficient: bool
    score: float
    covered: list[str] = []
    missing: list[str] = []
    suggested_queries: list[str] = []
    reason: str = ""


class HealthComponent(BaseModel):
    name: str
    ok: bool
    detail: str = ""
    latency_ms: int = 0


class DeepHealth(BaseModel):
    ok: bool
    components: list[HealthComponent]
    config: dict
