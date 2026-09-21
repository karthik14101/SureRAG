"""Ingestion job progress endpoints."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.core.deps import CurrentUser, DbSession, get_owned_job, get_owned_kb
from app.db import models
from app.db.base import SessionLocal
from app.schemas.document import JobOut
from app.services.document_service import job_progress

router = APIRouter(prefix="/jobs", tags=["jobs"])

TERMINAL_STATES = ("completed", "completed_with_errors", "failed")


def _to_out(job: models.IngestionJob) -> JobOut:
    return JobOut(
        id=job.id,
        kb_id=job.kb_id,
        state=job.state,
        total_files=job.total_files,
        processed_files=job.processed_files,
        failed_files=job.failed_files,
        current_file=job.current_file,
        stage=job.stage,
        errors=job.errors or [],
        progress=job_progress(job),
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: DbSession, user: CurrentUser) -> JobOut:
    return _to_out(get_owned_job(job_id, db, user))


@router.get("/kb/{kb_id}/active", response_model=list[JobOut])
def active_jobs(kb_id: str, db: DbSession, user: CurrentUser) -> list[JobOut]:
    """Jobs still running for a knowledge base, so the UI can resume polling
    after a page refresh mid-upload."""
    kb = get_owned_kb(kb_id, db, user)
    rows = db.execute(
        select(models.IngestionJob)
        .where(
            models.IngestionJob.kb_id == kb.id,
            models.IngestionJob.user_id == user.id,
            models.IngestionJob.state.in_(["queued", "running"]),
        )
        .order_by(models.IngestionJob.created_at.desc())
    ).scalars().all()
    return [_to_out(row) for row in rows]


@router.get("/{job_id}/stream")
def stream_job(job_id: str, db: DbSession, user: CurrentUser) -> StreamingResponse:
    """Server-sent progress for one job, ending when it reaches a terminal state."""
    job = get_owned_job(job_id, db, user)
    job_id_checked = job.id

    async def generator():
        db_stream = SessionLocal()
        last_payload = None
        try:
            # Bounded so an abandoned browser tab cannot hold a connection open
            # forever: 600 polls at 1s = 10 minutes.
            for _ in range(600):
                db_stream.expire_all()
                current = db_stream.get(models.IngestionJob, job_id_checked)
                if current is None:
                    yield "event: error\ndata: {}\n\n".format(
                        json.dumps({"message": "Job not found."})
                    )
                    return

                payload = json.loads(_to_out(current).model_dump_json())
                if payload != last_payload:
                    yield "event: progress\ndata: {}\n\n".format(json.dumps(payload))
                    last_payload = payload

                if current.state in TERMINAL_STATES:
                    yield "event: done\ndata: {}\n\n".format(json.dumps(payload))
                    return

                await asyncio.sleep(1.0)

            yield "event: timeout\ndata: {}\n\n".format(
                json.dumps({"message": "Progress stream timed out; refresh to continue."})
            )
        finally:
            db_stream.close()

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )
