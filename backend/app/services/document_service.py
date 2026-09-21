"""Document listing and the media URL helper used across the API."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db import models
from app.schemas.document import DocumentOut, MediaOut


def media_url(asset: models.MediaAsset) -> MediaOut:
    """Build the authenticated URLs the UI fetches images through.

    Deliberately NOT a static path: data/media is never mounted publicly, so
    every byte goes through an ownership check in the media route.
    """
    base = "{}/media".format(settings.api_prefix)
    return MediaOut(
        id=asset.id,
        doc_id=asset.doc_id,
        filename=asset.rel_path.rsplit("/", 1)[-1],
        page_no=asset.page_no,
        caption=asset.caption or (asset.ocr_text[:200] if asset.ocr_text else None),
        width=asset.width,
        height=asset.height,
        mime=asset.mime,
        origin=asset.origin,
        url="{}/{}".format(base, asset.id),
        thumb_url="{}/{}/thumb".format(base, asset.id) if asset.thumb_path else None,
    )


def list_documents(db: Session, kb_id: str, user_id: str) -> list[DocumentOut]:
    rows = db.execute(
        select(models.Document)
        .where(models.Document.kb_id == kb_id, models.Document.user_id == user_id)
        .order_by(models.Document.created_at.desc())
    ).scalars().all()
    return [DocumentOut.model_validate(row) for row in rows]


def list_kb_media(db: Session, kb_id: str, user_id: str, limit: int = 200) -> list[MediaOut]:
    rows = db.execute(
        select(models.MediaAsset)
        .where(models.MediaAsset.kb_id == kb_id, models.MediaAsset.user_id == user_id)
        .order_by(models.MediaAsset.created_at.desc())
        .limit(limit)
    ).scalars().all()
    return [media_url(row) for row in rows]


def job_progress(job: models.IngestionJob) -> float:
    if not job.total_files:
        return 0.0
    return round(min(1.0, job.processed_files / job.total_files), 3)
