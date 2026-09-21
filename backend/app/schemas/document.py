"""Document, media and ingestion-job schemas."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: str
    kb_id: str
    filename: str
    ext: str
    mime: str | None = None
    size_bytes: int
    page_count: int
    chunk_count: int
    image_count: int
    status: str
    error: str | None = None
    source_zip_id: str | None = None
    source_path: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class MediaOut(BaseModel):
    id: str
    doc_id: str
    filename: str
    page_no: int | None = None
    caption: str | None = None
    width: int
    height: int
    mime: str
    origin: str
    url: str
    thumb_url: str | None = None


class UploadAccepted(BaseModel):
    job_id: str
    kb_id: str
    accepted: list[str]
    skipped: list[dict]
    total_files: int


class JobOut(BaseModel):
    id: str
    kb_id: str
    state: str
    total_files: int
    processed_files: int
    failed_files: int
    current_file: str | None = None
    stage: str | None = None
    errors: list = []
    progress: float = 0.0
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None

    model_config = {"from_attributes": True}
