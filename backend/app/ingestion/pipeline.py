"""The ingestion pipeline: parse -> chunk -> store images -> embed -> index -> graph.

Ordering matters for crash safety. Vectors go into Qdrant before the SQLite rows
are committed, so a failure mid-run leaves orphaned vectors (harmless, filtered
out by the ownership check and cleaned on document delete) rather than SQLite
rows pointing at vectors that do not exist (which would produce empty citations).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import settings
from app.db import models
from app.graph import extractor as graph_extractor
from app.graph import writer as graph_writer
from app.ingestion import image_store
from app.ingestion.chunker import Chunk, chunk_blocks
from app.ingestion.parsers.base import (
    IMAGE_EXTENSIONS,
    ParsedDocument,
    ParserError,
    guess_mime,
)
from app.logging_conf import get_logger
from app.vectorstore.qdrant_client import ChunkPoint, upsert_chunks

logger = get_logger(__name__)


@dataclass
class IngestOutcome:
    document_id: str
    filename: str
    chunks: int = 0
    images: int = 0
    entities: int = 0
    pages: int = 0
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


async def parse_file(path: pathlib.Path, filename: str) -> ParsedDocument:
    """Dispatch to the right parser. Sync parsers run on a worker thread."""
    import asyncio

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        from app.ingestion.parsers.pdf_parser import parse_pdf

        return await asyncio.to_thread(parse_pdf, path, filename)

    if suffix == ".docx":
        from app.ingestion.parsers.docx_parser import parse_docx

        return await asyncio.to_thread(parse_docx, path, filename)

    if suffix in (".txt", ".md", ".markdown"):
        from app.ingestion.parsers.text_parser import parse_text

        return await asyncio.to_thread(parse_text, path, filename)

    if suffix == ".json":
        from app.ingestion.parsers.json_parser import parse_json

        return await asyncio.to_thread(parse_json, path, filename)

    if suffix in IMAGE_EXTENSIONS:
        from app.ingestion.parsers.image_parser import parse_image

        # Natively async: it may call a vision model.
        return await parse_image(path, filename)

    raise ParserError(
        "Unsupported file type '{}'. Supported: PDF, DOCX, TXT, MD, JSON, "
        "PNG/JPG/WEBP images, and ZIP archives.".format(suffix or "unknown")
    )


def _persist_images(
    db: Session,
    parsed: ParsedDocument,
    document: models.Document,
) -> dict[str, models.MediaAsset]:
    """Write images to disk and SQLite. Returns parser ref -> MediaAsset."""
    ref_to_asset: dict[str, models.MediaAsset] = {}
    dedupe: dict[str, image_store.StoredImage] = {}

    for image in parsed.images:
        try:
            stored = image_store.store_image(
                image,
                user_id=document.user_id,
                kb_id=document.kb_id,
                existing_by_hash=dedupe,
            )
        except Exception as exc:  # noqa: BLE001 - one bad image must not stop the file
            logger.warning("Could not store an image from %s: %s", document.filename, exc)
            continue

        asset = models.MediaAsset(
            doc_id=document.id,
            kb_id=document.kb_id,
            user_id=document.user_id,
            rel_path=stored.rel_path,
            thumb_path=stored.thumb_path,
            sha256=stored.sha256,
            mime=stored.mime,
            width=stored.width,
            height=stored.height,
            page_no=image.page_no,
            caption=image.caption,
            ocr_text=image.ocr_text,
            origin=image.origin,
        )
        db.add(asset)
        db.flush()  # assign the id without committing
        ref_to_asset[image.ref] = asset

    return ref_to_asset


async def _build_graph(
    db: Session,
    document: models.Document,
    chunk_rows: list[models.Chunk],
) -> int:
    """Extract entities/relations for eligible chunks and write them to Neo4j."""
    from app.graph.neo4j_client import is_available

    if settings.graph_extraction == "off" or not is_available():
        return 0

    eligible = [c for c in chunk_rows if graph_extractor.should_extract(c.text)]
    if not eligible:
        return 0

    document.status = "graphing"
    db.commit()

    extractions = await graph_extractor.extract_batch([c.text for c in eligible])

    total_entities = 0
    for chunk_row, extraction in zip(eligible, extractions):
        node = graph_writer.ChunkNode(
            chunk_id=chunk_row.id,
            doc_id=document.id,
            kb_id=document.kb_id,
            user_id=document.user_id,
            filename=document.filename,
            page_no=chunk_row.page_no,
            snippet=chunk_row.text[:500],
        )
        total_entities += await graph_writer.write_extraction(node, extraction)

    return total_entities


async def ingest_document(
    db: Session,
    document: models.Document,
    file_path: pathlib.Path,
) -> IngestOutcome:
    """Run one file all the way through. Never raises; errors land on the row."""
    outcome = IngestOutcome(document_id=document.id, filename=document.filename)

    try:
        document.status = "parsing"
        db.commit()

        parsed = await parse_file(file_path, document.filename)
        outcome.warnings.extend(parsed.warnings)
        outcome.pages = parsed.page_count

        if parsed.is_empty:
            raise ParserError(
                "No readable content was found in '{}'.".format(document.filename)
            )

        ref_to_asset = _persist_images(db, parsed, document)
        outcome.images = len(ref_to_asset)

        chunks: list[Chunk] = chunk_blocks(parsed.blocks)
        if not chunks and not ref_to_asset:
            raise ParserError(
                "'{}' produced no indexable text.".format(document.filename)
            )

        document.status = "embedding"
        document.page_count = parsed.page_count
        db.commit()

        # --- create chunk rows (ids first: Qdrant point ids are chunk ids) ----
        chunk_rows: list[models.Chunk] = []
        for chunk in chunks:
            media_ids = [
                ref_to_asset[ref].id for ref in chunk.media_refs if ref in ref_to_asset
            ]
            row = models.Chunk(
                doc_id=document.id,
                kb_id=document.kb_id,
                user_id=document.user_id,
                ordinal=chunk.ordinal,
                text=chunk.text,
                token_count=chunk.token_count,
                page_no=chunk.page_no,
                section=chunk.section,
                modality=chunk.modality,
                media_ids=media_ids,
            )
            row.qdrant_point_id = None
            db.add(row)
            chunk_rows.append(row)
        db.flush()

        for row in chunk_rows:
            row.qdrant_point_id = row.id

        # Link images back to the chunk that describes them, for the UI gallery.
        for chunk, row in zip(chunks, chunk_rows):
            for ref in chunk.media_refs:
                asset = ref_to_asset.get(ref)
                if asset is not None and asset.linked_chunk_id is None:
                    asset.linked_chunk_id = row.id

        # --- embed and index --------------------------------------------------
        if chunk_rows:
            from app.embeddings import encoder, sparse

            texts = [row.text for row in chunk_rows]
            dense_vectors = await encoder.embed_texts(texts)
            sparse_vectors = await sparse.embed_documents_sparse(texts)

            points = [
                ChunkPoint(
                    chunk_id=row.id,
                    doc_id=document.id,
                    kb_id=document.kb_id,
                    user_id=document.user_id,
                    filename=document.filename,
                    text=row.text,
                    ordinal=row.ordinal,
                    page_no=row.page_no,
                    section=row.section,
                    modality=row.modality,
                    media_ids=row.media_ids or [],
                    dense=dense,
                    sparse=sparse_vector,
                )
                for row, dense, sparse_vector in zip(
                    chunk_rows, dense_vectors, sparse_vectors
                )
            ]
            await upsert_chunks(points)

        document.chunk_count = len(chunk_rows)
        document.image_count = len(ref_to_asset)
        db.commit()

        outcome.chunks = len(chunk_rows)

        # --- knowledge graph ---------------------------------------------------
        try:
            outcome.entities = await _build_graph(db, document, chunk_rows)
        except Exception as exc:  # noqa: BLE001 - graph is optional
            logger.warning("Graph building failed for %s: %s", document.filename, exc)
            outcome.warnings.append("The knowledge graph could not be built for this file.")

        document.status = "ready"
        document.error = None
        db.commit()
        return outcome

    except ParserError as exc:
        db.rollback()
        document.status = "failed"
        document.error = str(exc)
        db.commit()
        outcome.error = str(exc)
        return outcome
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("Ingestion failed for %s", document.filename)
        message = "Processing failed: {}".format(str(exc)[:300])
        document.status = "failed"
        document.error = message
        db.commit()
        outcome.error = message
        return outcome


def make_document_row(
    *,
    kb_id: str,
    user_id: str,
    filename: str,
    path: pathlib.Path,
    sha256: str,
    size_bytes: int,
    source_zip_id: str | None = None,
    source_path: str | None = None,
) -> models.Document:
    return models.Document(
        kb_id=kb_id,
        user_id=user_id,
        filename=filename,
        ext=path.suffix.lower().lstrip("."),
        mime=guess_mime(filename),
        size_bytes=size_bytes,
        sha256=sha256,
        status="pending",
        source_zip_id=source_zip_id,
        source_path=source_path,
    )
