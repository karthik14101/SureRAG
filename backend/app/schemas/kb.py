"""Knowledge base schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class KBCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class KBUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)


class KBOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    doc_count: int
    chunk_count: int
    image_count: int
    entity_count: int
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class KBStats(BaseModel):
    id: str
    name: str
    doc_count: int
    chunk_count: int
    image_count: int
    entity_count: int
    relation_count: int
    ready_docs: int
    failed_docs: int
    pending_docs: int
    total_bytes: int
    graph_available: bool
