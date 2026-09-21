"""Chat session, message and answer schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.schemas.citation import CitationOut
from app.schemas.document import MediaOut


class ChatSessionCreate(BaseModel):
    kb_id: str
    title: str | None = Field(default=None, max_length=200)


class ChatSessionUpdate(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ChatSessionOut(BaseModel):
    id: str
    kb_id: str
    kb_name: str | None = None
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TraceStep(BaseModel):
    name: str
    label: str
    detail: str | None = None
    duration_ms: int = 0
    meta: dict = {}


class MessageOut(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    route_used: str | None = None
    sufficiency_score: float | None = None
    iterations: int = 0
    latency_ms: int = 0
    error: str | None = None
    created_at: datetime
    citations: list[CitationOut] = []
    media: list[MediaOut] = []
    trace: list[TraceStep] = []


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    # Off by default: forcing a route is a debugging aid, not normal use.
    force_route: str | None = None


class AskResponse(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut
