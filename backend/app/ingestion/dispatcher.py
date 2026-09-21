"""Upload intake: validation, deduplication, ZIP expansion and job creation.

Runs inside the request so the user gets immediate feedback on rejected files,
but does no parsing or embedding -- that is handed to the background worker.
"""
from __future__ import annotations

import pathlib
import shutil
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import ValidationError
from app.db import models
from app.ingestion.parsers.base import SUPPORTED_EXTENSIONS, ParserError, sha256_file
from app.ingestion.parsers.zip_parser import extract_archive
from app.logging_conf import get_logger

logger = get_logger(__name__)

# Magic-byte signatures, so a renamed .exe cannot masquerade as a .pdf.
MAGIC_SIGNATURES = {
    ".pdf": [b"%PDF-"],
    ".docx": [b"PK\x03\x04"],
    ".zip": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".gif": [b"GIF87a", b"GIF89a"],
    ".bmp": [b"BM"],
    ".webp": [b"RIFF"],
}


@dataclass
class StagedFile:
    path: pathlib.Path
    filename: str
    sha256: str
    size: int
    source_zip_name: str | None = None
    source_path: str | None = None


@dataclass
class IntakeResult:
    staged: list[StagedFile] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def staging_dir(job_id: str) -> pathlib.Path:
    directory = settings.upload_tmp_dir / job_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cleanup_staging(job_id: str) -> None:
    directory = settings.upload_tmp_dir / job_id
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


def validate_extension(filename: str) -> str:
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValidationError(
            "'{}' has an unsupported type. Allowed: {}".format(
                filename, ", ".join(sorted(SUPPORTED_EXTENSIONS))
            )
        )
    return suffix


def check_magic_bytes(path: pathlib.Path, suffix: str) -> bool:
    """True when the content matches the extension (or we have no signature)."""
    signatures = MAGIC_SIGNATURES.get(suffix)
    if not signatures:
        return True  # text/json/markdown have no reliable magic
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
        return any(head.startswith(sig) for sig in signatures)
    except Exception:  # noqa: BLE001
        return False


def existing_hashes(db: Session, kb_id: str, user_id: str) -> dict[str, str]:
    rows = db.execute(
        select(models.Document.sha256, models.Document.filename).where(
            models.Document.kb_id == kb_id,
            models.Document.user_id == user_id,
            models.Document.status != "failed",
        )
    ).all()
    return {row[0]: row[1] for row in rows}


def stage_uploads(
    db: Session,
    *,
    kb_id: str,
    user_id: str,
    job_id: str,
    files: list[tuple[str, bytes]],
) -> IntakeResult:
    """Write uploads to the staging directory, expanding archives as we go."""
    result = IntakeResult()
    directory = staging_dir(job_id)
    known = existing_hashes(db, kb_id, user_id)
    seen_this_batch: set[str] = set()

    for original_name, content in files:
        safe_name = pathlib.Path(original_name).name or "upload"

        try:
            suffix = validate_extension(safe_name)
        except ValidationError as exc:
            result.skipped.append({"name": safe_name, "reason": exc.message})
            continue

        if not content:
            result.skipped.append({"name": safe_name, "reason": "The file is empty."})
            continue

        if len(content) > settings.max_upload_bytes:
            result.skipped.append(
                {
                    "name": safe_name,
                    "reason": "Larger than the {} MB limit.".format(settings.max_upload_mb),
                }
            )
            continue

        staged_path = directory / "{}_{}".format(uuid.uuid4().hex[:8], safe_name)
        staged_path.write_bytes(content)

        if not check_magic_bytes(staged_path, suffix):
            staged_path.unlink(missing_ok=True)
            result.skipped.append(
                {
                    "name": safe_name,
                    "reason": "File content does not match its '{}' extension.".format(suffix),
                }
            )
            continue

        if suffix == ".zip":
            extract_dir = directory / (staged_path.stem + "_extracted")
            try:
                extraction = extract_archive(staged_path, extract_dir)
            except ParserError as exc:
                staged_path.unlink(missing_ok=True)
                result.skipped.append({"name": safe_name, "reason": str(exc)})
                continue

            result.skipped.extend(
                {"name": "{} > {}".format(safe_name, item["name"]), "reason": item["reason"]}
                for item in extraction.skipped
            )

            for entry in extraction.entries:
                digest = sha256_file(entry.path)
                if digest in known:
                    result.skipped.append(
                        {
                            "name": "{} > {}".format(safe_name, entry.relative_name),
                            "reason": "Already in this knowledge base as '{}'.".format(
                                known[digest]
                            ),
                        }
                    )
                    continue
                if digest in seen_this_batch:
                    result.skipped.append(
                        {
                            "name": "{} > {}".format(safe_name, entry.relative_name),
                            "reason": "Duplicate of another file in this upload.",
                        }
                    )
                    continue
                seen_this_batch.add(digest)
                result.staged.append(
                    StagedFile(
                        path=entry.path,
                        filename=pathlib.Path(entry.relative_name).name,
                        sha256=digest,
                        size=entry.size,
                        source_zip_name=safe_name,
                        source_path=entry.relative_name,
                    )
                )

            staged_path.unlink(missing_ok=True)
            continue

        digest = sha256_file(staged_path)
        if digest in known:
            staged_path.unlink(missing_ok=True)
            result.skipped.append(
                {
                    "name": safe_name,
                    "reason": "Already in this knowledge base as '{}'.".format(known[digest]),
                }
            )
            continue
        if digest in seen_this_batch:
            staged_path.unlink(missing_ok=True)
            result.skipped.append(
                {"name": safe_name, "reason": "Duplicate of another file in this upload."}
            )
            continue

        seen_this_batch.add(digest)
        result.staged.append(
            StagedFile(
                path=staged_path,
                filename=safe_name,
                sha256=digest,
                size=len(content),
            )
        )

    return result
