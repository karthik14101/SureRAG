"""Safe recursive ZIP extraction.

Archive handling is the one ingestion path that is genuinely dangerous, so every
known attack is blocked explicitly rather than assumed away:

  * Zip-slip  -- entries like `../../etc/passwd` escaping the extract directory
  * Zip bombs -- a few KB inflating to gigabytes
  * Zip quines / nested archives -- recursion capped at MAX_ZIP_DEPTH
  * Symlink entries -- refused outright
  * Entry floods -- thousands of tiny files exhausting inodes

Nothing is written to disk until an entry has passed every check.
"""
from __future__ import annotations

import pathlib
import zipfile
from dataclasses import dataclass

from app.config import settings
from app.ingestion.parsers.base import SUPPORTED_EXTENSIONS, ParserError
from app.logging_conf import get_logger

logger = get_logger(__name__)

# Compressed-to-uncompressed ratio above which an entry is treated as a bomb.
MAX_COMPRESSION_RATIO = 120
MIN_SIZE_FOR_RATIO_CHECK = 256 * 1024
# Directories that are archive metadata, not user content.
JUNK_PREFIXES = ("__MACOSX/", ".git/", "node_modules/")
JUNK_NAMES = {".DS_Store", "Thumbs.db", "desktop.ini"}


@dataclass
class ExtractedEntry:
    path: pathlib.Path      # where the file now lives on disk
    relative_name: str      # path as it appeared inside the archive
    size: int
    depth: int


@dataclass
class ExtractionResult:
    entries: list[ExtractedEntry]
    skipped: list[dict]
    total_uncompressed: int


def _is_junk(name: str) -> bool:
    if any(name.startswith(prefix) for prefix in JUNK_PREFIXES):
        return True
    base = name.rsplit("/", 1)[-1]
    if base in JUNK_NAMES or base.startswith("._"):
        return True
    return not base  # directory entry


def _safe_destination(base: pathlib.Path, name: str) -> pathlib.Path | None:
    """Resolve an entry inside `base`, or return None if it escapes."""
    # Normalise separators and strip drive letters / leading slashes.
    cleaned = name.replace("\\", "/").lstrip("/")
    if ":" in cleaned.split("/")[0]:
        cleaned = cleaned.split(":", 1)[1].lstrip("/")
    if not cleaned:
        return None

    candidate = (base / cleaned).resolve()
    base_resolved = base.resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError:
        return None
    return candidate


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    # Unix mode lives in the high 16 bits of external_attr; 0xA000 == symlink.
    mode = info.external_attr >> 16
    return bool(mode & 0o170000 == 0o120000)


def extract_archive(
    zip_path: pathlib.Path,
    dest_dir: pathlib.Path,
    *,
    depth: int = 0,
    budget: dict | None = None,
) -> ExtractionResult:
    """Extract one archive, recursing into nested archives up to the depth cap."""
    if budget is None:
        budget = {"bytes": 0, "entries": 0}

    entries: list[ExtractedEntry] = []
    skipped: list[dict] = []

    if depth > settings.max_zip_depth:
        skipped.append(
            {
                "name": zip_path.name,
                "reason": "Nested archives deeper than {} levels are not processed.".format(
                    settings.max_zip_depth
                ),
            }
        )
        return ExtractionResult(entries, skipped, budget["bytes"])

    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise ParserError(
            "'{}' is not a valid ZIP archive or is corrupt.".format(zip_path.name)
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)

    with archive:
        infos = archive.infolist()
        if len(infos) + budget["entries"] > settings.max_zip_entries:
            raise ParserError(
                "Archive contains more than {} entries, which exceeds the safety "
                "limit. Split it into smaller archives.".format(settings.max_zip_entries)
            )

        nested_archives: list[tuple[pathlib.Path, str]] = []

        for info in infos:
            name = info.filename

            if info.is_dir() or _is_junk(name):
                continue

            if _is_symlink(info):
                skipped.append({"name": name, "reason": "Symlinks are not extracted."})
                continue

            budget["entries"] += 1

            suffix = pathlib.Path(name).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                skipped.append(
                    {"name": name, "reason": "Unsupported file type '{}'.".format(suffix or "none")}
                )
                continue

            # Bomb check BEFORE writing anything.
            if info.file_size > MIN_SIZE_FOR_RATIO_CHECK and info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    skipped.append(
                        {
                            "name": name,
                            "reason": "Suspicious compression ratio ({}x) -- skipped.".format(
                                int(ratio)
                            ),
                        }
                    )
                    continue

            if budget["bytes"] + info.file_size > settings.max_zip_uncompressed_bytes:
                skipped.append(
                    {
                        "name": name,
                        "reason": "Archive exceeds the {} MB uncompressed limit.".format(
                            settings.max_zip_uncompressed_mb
                        ),
                    }
                )
                continue

            destination = _safe_destination(dest_dir, name)
            if destination is None:
                skipped.append(
                    {"name": name, "reason": "Unsafe path in archive -- skipped."}
                )
                logger.warning("Blocked zip-slip attempt: %s in %s", name, zip_path.name)
                continue

            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                with archive.open(info) as source, destination.open("wb") as target:
                    written = 0
                    limit = settings.max_zip_uncompressed_bytes - budget["bytes"]
                    while True:
                        block = source.read(1 << 16)
                        if not block:
                            break
                        written += len(block)
                        if written > limit:
                            raise ParserError(
                                "Archive exceeded the uncompressed size limit while "
                                "extracting '{}'.".format(name)
                            )
                        target.write(block)
                budget["bytes"] += written
            except ParserError:
                destination.unlink(missing_ok=True)
                raise
            except Exception as exc:  # noqa: BLE001
                destination.unlink(missing_ok=True)
                skipped.append(
                    {"name": name, "reason": "Extraction failed: {}".format(str(exc)[:120])}
                )
                continue

            if suffix == ".zip":
                nested_archives.append((destination, name))
            else:
                entries.append(
                    ExtractedEntry(
                        path=destination,
                        relative_name=name,
                        size=written,
                        depth=depth,
                    )
                )

        # Recurse only after the parent archive is fully processed and closed.
        for nested_path, nested_name in nested_archives:
            nested_dir = nested_path.parent / (nested_path.stem + "_unzipped")
            try:
                nested = extract_archive(
                    nested_path, nested_dir, depth=depth + 1, budget=budget
                )
                for entry in nested.entries:
                    entry.relative_name = "{}/{}".format(nested_name, entry.relative_name)
                entries.extend(nested.entries)
                skipped.extend(nested.skipped)
            except ParserError as exc:
                skipped.append({"name": nested_name, "reason": str(exc)})
            finally:
                nested_path.unlink(missing_ok=True)

    if not entries and not skipped:
        raise ParserError(
            "'{}' contained no supported files. Supported types: {}".format(
                zip_path.name,
                ", ".join(sorted(e for e in SUPPORTED_EXTENSIONS if e != ".zip")),
            )
        )

    return ExtractionResult(entries, skipped, budget["bytes"])
