"""On-disk media storage with content-addressed deduplication.

Files are stored under data/media/<user_id>/<kb_id>/<uuid>.<ext> with an
unguessable name, and they are never served by a static mount -- the API checks
ownership on every read. The same figure appearing in ten documents is stored
once, keyed by SHA-256.
"""
from __future__ import annotations

import io
import pathlib
import uuid
from dataclasses import dataclass

from app.config import settings
from app.ingestion.parsers.base import ExtractedImage, sha256_bytes
from app.logging_conf import get_logger

logger = get_logger(__name__)

THUMB_MAX = 480
MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".png",
}


@dataclass
class StoredImage:
    rel_path: str
    thumb_path: str | None
    sha256: str
    mime: str
    width: int
    height: int
    size_bytes: int


def _extension_for(mime: str) -> str:
    return MIME_EXTENSIONS.get(mime.lower(), ".png")


def _kb_dir(user_id: str, kb_id: str) -> pathlib.Path:
    directory = settings.media_dir / user_id / kb_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def absolute_path(rel_path: str) -> pathlib.Path:
    """Resolve a stored relative path, refusing anything outside the media root."""
    base = settings.media_dir.resolve()
    candidate = (base / rel_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        raise ValueError("Media path escapes the media directory.") from None
    return candidate


def _make_thumbnail(data: bytes, destination: pathlib.Path) -> bool:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            if img.width <= THUMB_MAX and img.height <= THUMB_MAX:
                return False
            # Palette and transparency modes cannot be saved as JPEG.
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            img.thumbnail((THUMB_MAX, THUMB_MAX))
            img.save(destination, format="JPEG", quality=82, optimize=True)
        return True
    except Exception as exc:  # noqa: BLE001 - a thumbnail is a nicety
        logger.debug("Thumbnail generation failed: %s", str(exc)[:160])
        return False


def store_image(
    image: ExtractedImage,
    *,
    user_id: str,
    kb_id: str,
    existing_by_hash: dict[str, StoredImage] | None = None,
) -> StoredImage:
    """Write an image to disk, reusing the file when the bytes are already stored."""
    digest = sha256_bytes(image.data)

    if existing_by_hash is not None and digest in existing_by_hash:
        return existing_by_hash[digest]

    directory = _kb_dir(user_id, kb_id)
    extension = _extension_for(image.mime)
    name = uuid.uuid4().hex + extension
    destination = directory / name
    destination.write_bytes(image.data)

    thumb_rel: str | None = None
    thumb_name = destination.stem + "_thumb.jpg"
    if _make_thumbnail(image.data, directory / thumb_name):
        thumb_rel = "{}/{}/{}".format(user_id, kb_id, thumb_name)

    stored = StoredImage(
        rel_path="{}/{}/{}".format(user_id, kb_id, name),
        thumb_path=thumb_rel,
        sha256=digest,
        mime=image.mime,
        width=image.width,
        height=image.height,
        size_bytes=len(image.data),
    )

    if existing_by_hash is not None:
        existing_by_hash[digest] = stored
    return stored


def delete_image(rel_path: str, thumb_path: str | None = None) -> None:
    for path in (rel_path, thumb_path):
        if not path:
            continue
        try:
            absolute_path(path).unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.debug("Could not delete media file %s: %s", path, exc)


def delete_kb_media(user_id: str, kb_id: str) -> int:
    """Remove a knowledge base's whole media directory. Returns files deleted."""
    directory = settings.media_dir / user_id / kb_id
    if not directory.exists():
        return 0
    removed = 0
    for path in sorted(directory.rglob("*"), reverse=True):
        try:
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                path.rmdir()
        except Exception:  # noqa: BLE001
            continue
    try:
        directory.rmdir()
    except Exception:  # noqa: BLE001
        pass
    return removed


def sweep_orphans(known_paths: set[str]) -> int:
    """Delete media files with no database row. Runs once at startup."""
    root = settings.media_dir
    if not root.exists():
        return 0
    removed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel in known_paths:
            continue
        try:
            path.unlink()
            removed += 1
        except Exception:  # noqa: BLE001
            continue
    if removed:
        logger.info("Removed %d orphaned media file(s)", removed)
    return removed
