"""Authenticated image serving.

data/media is deliberately NOT a static mount. Every byte is served through here
so ownership is verified per request -- a static mount would make one user's
extracted figures readable by anyone who guessed the URL, which would defeat the
data isolation the rest of the app enforces.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import FileResponse

from app.core.deps import CurrentUser, DbSession, get_owned_media
from app.core.errors import NotFoundError
from app.ingestion import image_store
from app.logging_conf import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/media", tags=["media"])

# Images are immutable once written (UUID filenames), so they cache forever.
CACHE_HEADERS = {"Cache-Control": "private, max-age=31536000, immutable"}


@router.get("/{asset_id}")
def get_media(asset_id: str, db: DbSession, user: CurrentUser) -> FileResponse:
    asset = get_owned_media(asset_id, db, user)
    try:
        path = image_store.absolute_path(asset.rel_path)
    except ValueError:
        logger.error("Blocked media path escape for asset %s", asset_id)
        raise NotFoundError("Image not found.") from None

    if not path.exists():
        raise NotFoundError("The image file is missing from disk.")

    return FileResponse(path, media_type=asset.mime, headers=CACHE_HEADERS)


@router.get("/{asset_id}/thumb")
def get_thumbnail(asset_id: str, db: DbSession, user: CurrentUser) -> FileResponse:
    asset = get_owned_media(asset_id, db, user)
    relative = asset.thumb_path or asset.rel_path
    try:
        path = image_store.absolute_path(relative)
    except ValueError:
        raise NotFoundError("Image not found.") from None

    if not path.exists():
        # Fall back to the full-size image rather than showing a broken tile.
        try:
            path = image_store.absolute_path(asset.rel_path)
        except ValueError:
            raise NotFoundError("Image not found.") from None
        if not path.exists():
            raise NotFoundError("The image file is missing from disk.")
        return FileResponse(path, media_type=asset.mime, headers=CACHE_HEADERS)

    media_type = "image/jpeg" if asset.thumb_path else asset.mime
    return FileResponse(path, media_type=media_type, headers=CACHE_HEADERS)


@router.get("/{asset_id}/info")
def get_media_info(asset_id: str, db: DbSession, user: CurrentUser) -> dict:
    from app.db import models

    asset = get_owned_media(asset_id, db, user)
    document = db.get(models.Document, asset.doc_id)
    return {
        "id": asset.id,
        "filename": asset.rel_path.rsplit("/", 1)[-1],
        "source_document": document.filename if document else None,
        "page_no": asset.page_no,
        "caption": asset.caption,
        "ocr_text": asset.ocr_text,
        "width": asset.width,
        "height": asset.height,
        "mime": asset.mime,
        "origin": asset.origin,
    }
