"""Background ingestion worker.

Uploads return HTTP 202 immediately and the real work happens here, on an
asyncio queue with bounded concurrency. Embedding is CPU-bound and the LLM calls
for graph extraction are rate-limited, so running two files at once is the sweet
spot on a laptop -- more just thrashes and burns quota faster.
"""
from __future__ import annotations

import asyncio
import pathlib
from dataclasses import dataclass
from datetime import datetime, timezone

from app.config import settings
from app.db import models
from app.db.base import session_scope
from app.ingestion import dispatcher, pipeline
from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedFile:
    job_id: str
    kb_id: str
    user_id: str
    document_id: str
    path: str
    filename: str


class IngestionWorker:
    """Owns the queue and the worker tasks for the process lifetime."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[QueuedFile | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        count = max(1, settings.ingest_concurrency)
        for index in range(count):
            self._tasks.append(asyncio.create_task(self._run(index), name="ingest-{}".format(index)))
        logger.info("Ingestion worker started (%d parallel slot(s))", count)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        for _ in self._tasks:
            await self._queue.put(None)
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Ingestion worker stopped")

    async def enqueue(self, item: QueuedFile) -> None:
        await self._queue.put(item)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    async def _run(self, index: int) -> None:
        while True:
            try:
                item = await self._queue.get()
            except asyncio.CancelledError:
                return
            if item is None:
                self._queue.task_done()
                return
            try:
                await self._process(item)
            except asyncio.CancelledError:
                self._queue.task_done()
                return
            except Exception:  # noqa: BLE001 - a worker must never die
                logger.exception("Worker %d crashed on %s", index, item.filename)
            finally:
                self._queue.task_done()

    async def _process(self, item: QueuedFile) -> None:
        path = pathlib.Path(item.path)

        with session_scope() as db:
            job = db.get(models.IngestionJob, item.job_id)
            document = db.get(models.Document, item.document_id)
            if document is None:
                logger.warning("Document %s vanished before ingestion", item.document_id)
                return

            if job is not None and job.state == "queued":
                job.state = "running"
                job.started_at = datetime.now(timezone.utc)
            if job is not None:
                job.current_file = item.filename
                job.stage = "parsing"
                db.commit()

            if not path.exists():
                document.status = "failed"
                document.error = "The staged upload disappeared before processing."
                db.commit()
                outcome = pipeline.IngestOutcome(
                    document_id=document.id,
                    filename=document.filename,
                    error=document.error,
                )
            else:
                outcome = await pipeline.ingest_document(db, document, path)

            self._record_result(db, job, outcome)

        # Remove the staged copy once the content is safely indexed.
        try:
            path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

        await self._finalise_if_done(item.job_id)

    @staticmethod
    def _record_result(db, job: models.IngestionJob | None, outcome) -> None:
        if job is None:
            return
        job.processed_files += 1
        if not outcome.ok:
            job.failed_files += 1
            errors = list(job.errors or [])
            errors.append({"file": outcome.filename, "error": outcome.error})
            job.errors = errors[:50]
        elif outcome.warnings:
            errors = list(job.errors or [])
            for warning in outcome.warnings:
                errors.append({"file": outcome.filename, "warning": warning})
            job.errors = errors[:50]
        job.current_file = None
        job.stage = None
        db.commit()

    async def _finalise_if_done(self, job_id: str) -> None:
        """Close out the job and refresh the KB counters once all files land."""
        with session_scope() as db:
            job = db.get(models.IngestionJob, job_id)
            if job is None or job.state in ("completed", "completed_with_errors", "failed"):
                return
            if job.processed_files < job.total_files:
                return

            job.state = "completed_with_errors" if job.failed_files else "completed"
            job.finished_at = datetime.now(timezone.utc)
            job.current_file = None
            job.stage = None

            kb = db.get(models.KnowledgeBase, job.kb_id)
            if kb is not None:
                from sqlalchemy import func, select

                totals = db.execute(
                    select(
                        func.count(models.Document.id),
                        func.coalesce(func.sum(models.Document.chunk_count), 0),
                        func.coalesce(func.sum(models.Document.image_count), 0),
                    ).where(
                        models.Document.kb_id == kb.id,
                        models.Document.status == "ready",
                    )
                ).one()
                kb.doc_count = int(totals[0] or 0)
                kb.chunk_count = int(totals[1] or 0)
                kb.image_count = int(totals[2] or 0)

                try:
                    from app.graph.writer import graph_stats

                    stats = await graph_stats(kb.id)
                    kb.entity_count = int(stats.get("entities") or 0)
                except Exception:  # noqa: BLE001 - counters are cosmetic
                    pass

            db.commit()
            logger.info(
                "Job %s finished: %d processed, %d failed",
                job_id,
                job.processed_files,
                job.failed_files,
            )

        dispatcher.cleanup_staging(job_id)


worker = IngestionWorker()


async def recover_stuck_jobs() -> None:
    """Mark jobs abandoned by a previous process as failed, so the UI is honest."""
    from sqlalchemy import select

    with session_scope() as db:
        stale_jobs = db.execute(
            select(models.IngestionJob).where(
                models.IngestionJob.state.in_(["queued", "running"])
            )
        ).scalars().all()

        for job in stale_jobs:
            job.state = "failed"
            job.finished_at = datetime.now(timezone.utc)
            errors = list(job.errors or [])
            errors.append(
                {"file": job.current_file or "job", "error": "Interrupted by a server restart."}
            )
            job.errors = errors[:50]
            job.current_file = None
            dispatcher.cleanup_staging(job.id)

        stale_docs = db.execute(
            select(models.Document).where(
                models.Document.status.in_(["pending", "parsing", "embedding", "graphing"])
            )
        ).scalars().all()

        for document in stale_docs:
            document.status = "failed"
            document.error = "Interrupted by a server restart. Delete and re-upload this file."

        if stale_jobs or stale_docs:
            db.commit()
            logger.info(
                "Recovered %d interrupted job(s) and %d document(s)",
                len(stale_jobs),
                len(stale_docs),
            )
