"""Citation schemas surfaced to the UI as clickable chips."""
from __future__ import annotations

from pydantic import BaseModel


class CitationOut(BaseModel):
    marker_index: int
    chunk_id: str | None = None
    doc_id: str | None = None
    filename: str
    page_no: int | None = None
    section: str | None = None
    snippet: str
    score: float = 0.0
    retrieval_source: str = "vector"

    model_config = {"from_attributes": True}
