"""Upload and document management endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import settings
from app.core.deps import CurrentUser, DbSession, client_key, get_owned_document, get_owned_kb
from app.core.errors import ValidationError
from app.core.rate_limit import upload_limiter
from app.db import models
from app.ingestion import dispatcher, pipeline
from app.ingestion.worker import QueuedFile, worker
from app.logging_conf import get_logger
from app.schemas.document import DocumentOut, MediaOut, UploadAccepted
from app.services import cleanup_service, document_service

logger = get_logger(__name__)
router = APIRouter(tags=["documents"])

MAX_FILES_PER_REQUEST = 50


@router.post("/kb/{kb_id}/documents", response_model=UploadAccepted, status_code=202)
async def upload_documents(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    key: Annotated[str, Depends(client_key)],
    files: list[UploadFile] = File(...),
) -> UploadAccepted:
    """Accept uploads, stage them, and hand the work to the background worker.

    Returns 202 immediately: parsing, embedding and graph building can take
    minutes for a large PDF and must not hold the request open.
    """
    upload_limiter.check(key, cost=max(1.0, len(files) * 0.5))
    kb = get_owned_kb(kb_id, db, user)

    if not files:
        raise ValidationError("No files were uploaded.")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise ValidationError(
            "At most {} files per upload. Try a ZIP archive instead.".format(
                MAX_FILES_PER_REQUEST
            )
        )

    job = models.IngestionJob(kb_id=kb.id, user_id=user.id, state="queued", errors=[])
    db.add(job)
    db.commit()
    db.refresh(job)

    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        content = await upload.read()
        payloads.append((upload.filename or "upload", content))
        await upload.close()

    try:
        intake = dispatcher.stage_uploads(
            db, kb_id=kb.id, user_id=user.id, job_id=job.id, files=payloads
        )
    except Exception:
        job.state = "failed"
        job.errors = [{"file": "upload", "error": "Staging failed."}]
        db.commit()
        dispatcher.cleanup_staging(job.id)
        raise

    if not intake.staged:
        job.state = "completed_with_errors" if intake.skipped else "completed"
        job.total_files = 0
        job.errors = [
            {"file": item["name"], "error": item["reason"]} for item in intake.skipped
        ][:50]
        db.commit()
        dispatcher.cleanup_staging(job.id)
        return UploadAccepted(
            job_id=job.id,
            kb_id=kb.id,
            accepted=[],
            skipped=intake.skipped,
            total_files=0,
        )

    documents: list[models.Document] = []
    for staged in intake.staged:
        document = pipeline.make_document_row(
            kb_id=kb.id,
            user_id=user.id,
            filename=staged.filename,
            path=staged.path,
            sha256=staged.sha256,
            size_bytes=staged.size,
            source_path=staged.source_path,
        )
        db.add(document)
        documents.append(document)

    job.total_files = len(documents)
    if intake.skipped:
        job.errors = [
            {"file": item["name"], "error": item["reason"]} for item in intake.skipped
        ][:50]
    db.commit()

    for document, staged in zip(documents, intake.staged):
        db.refresh(document)
        await worker.enqueue(
            QueuedFile(
                job_id=job.id,
                kb_id=kb.id,
                user_id=user.id,
                document_id=document.id,
                path=str(staged.path),
                filename=staged.filename,
            )
        )

    logger.info(
        "Queued %d file(s) for KB '%s' (job %s)", len(documents), kb.name, job.id
    )

    return UploadAccepted(
        job_id=job.id,
        kb_id=kb.id,
        accepted=[d.filename for d in documents],
        skipped=intake.skipped,
        total_files=len(documents),
    )


@router.get("/kb/{kb_id}/documents", response_model=list[DocumentOut])
def list_documents(kb_id: str, db: DbSession, user: CurrentUser) -> list[DocumentOut]:
    kb = get_owned_kb(kb_id, db, user)
    return document_service.list_documents(db, kb.id, user.id)


@router.get("/kb/{kb_id}/media", response_model=list[MediaOut])
def list_media(kb_id: str, db: DbSession, user: CurrentUser) -> list[MediaOut]:
    kb = get_owned_kb(kb_id, db, user)
    return document_service.list_kb_media(db, kb.id, user.id)


@router.delete("/documents/{doc_id}", status_code=204)
async def delete_document(doc_id: str, db: DbSession, user: CurrentUser) -> None:
    document = get_owned_document(doc_id, db, user)
    await cleanup_service.delete_document(db, document)


@router.post("/kb/{kb_id}/rebuild-graph")
async def rebuild_graph(kb_id: str, db: DbSession, user: CurrentUser) -> dict:
    """Rebuild the knowledge graph for a knowledge base from its stored chunks.

    Useful after turning GRAPH_EXTRACTION on, switching LLM provider, or bringing
    Neo4j up for the first time. The source files are gone by then, but the chunk
    text is in SQLite, so the graph can be rebuilt without re-uploading anything.
    """
    from sqlalchemy import select

    from app.graph import extractor as graph_extractor
    from app.graph import writer as graph_writer
    from app.graph.neo4j_client import is_available

    kb = get_owned_kb(kb_id, db, user)

    if not is_available():
        raise ValidationError(
            "Neo4j is not reachable. Start it with `docker compose up -d`, then retry."
        )
    if settings.graph_extraction == "off":
        raise ValidationError(
            "Graph extraction is disabled. Set GRAPH_EXTRACTION=selective in .env "
            "and restart the backend."
        )

    chunks = db.execute(
        select(models.Chunk)
        .where(models.Chunk.kb_id == kb.id, models.Chunk.user_id == user.id)
        .order_by(models.Chunk.doc_id, models.Chunk.ordinal)
    ).scalars().all()

    if not chunks:
        raise ValidationError("This knowledge base has no indexed content yet.")

    await graph_writer.delete_kb_graph(kb.id)

    eligible = [c for c in chunks if graph_extractor.should_extract(c.text)]
    filenames = {
        doc.id: doc.filename
        for doc in db.execute(
            select(models.Document).where(models.Document.kb_id == kb.id)
        ).scalars().all()
    }

    entities = 0
    if eligible:
        extractions = await graph_extractor.extract_batch([c.text for c in eligible])
        for chunk, extraction in zip(eligible, extractions):
            node = graph_writer.ChunkNode(
                chunk_id=chunk.id,
                doc_id=chunk.doc_id,
                kb_id=kb.id,
                user_id=user.id,
                filename=filenames.get(chunk.doc_id, "unknown"),
                page_no=chunk.page_no,
                snippet=chunk.text[:500],
            )
            entities += await graph_writer.write_extraction(node, extraction)

    await cleanup_service.refresh_kb_counters(db, kb.id)
    stats = await graph_writer.graph_stats(kb.id)

    logger.info("Rebuilt graph for KB '%s': %d entities", kb.name, entities)
    return {
        "kb_id": kb.id,
        "chunks_considered": len(chunks),
        "chunks_extracted": len(eligible),
        "entities": int(stats.get("entities") or 0),
        "relations": int(stats.get("relations") or 0),
    }


@router.get("/config/limits")
def upload_limits() -> dict:
    """Surfaced in the UI so the dropzone can reject files before uploading."""
    from app.ingestion.parsers.base import SUPPORTED_EXTENSIONS

    return {
        "max_upload_mb": settings.max_upload_mb,
        "max_files_per_request": MAX_FILES_PER_REQUEST,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "max_zip_depth": settings.max_zip_depth,
        "max_zip_entries": settings.max_zip_entries,
    }
