"""Word document parsing: headings, paragraphs, tables and inline images.

python-docx exposes paragraphs and tables as separate collections, which loses
their interleaving. We walk the raw body XML instead so a table stays attached to
the paragraph that introduced it, and images keep their position in the flow.
"""
from __future__ import annotations

import io

import docx
from docx.document import Document as DocxDocument
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.config import settings
from app.ingestion.parsers.base import (
    ExtractedImage,
    ParsedDocument,
    ParserError,
    TextBlock,
    clean_text,
)
from app.logging_conf import get_logger

logger = get_logger(__name__)

HEADING_STYLES = ("heading 1", "heading 2", "heading 3", "heading 4", "title", "subtitle")


def _iter_body(document: DocxDocument):
    """Yield Paragraph and Table objects in true document order."""
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int | None:
    style_name = (paragraph.style.name or "").strip().casefold() if paragraph.style else ""
    if style_name in ("title",):
        return 1
    if style_name.startswith("heading"):
        tail = style_name.replace("heading", "").strip()
        if tail.isdigit():
            return int(tail)
        return 1
    return None


def _paragraph_image_ids(paragraph: Paragraph) -> list[str]:
    """Relationship ids of images embedded in this paragraph's runs."""
    ids: list[str] = []
    for blip in paragraph._element.iter(qn("a:blip")):
        rel_id = blip.get(qn("r:embed")) or blip.get(qn("r:link"))
        if rel_id:
            ids.append(rel_id)
    return ids


def _table_to_markdown(table: Table) -> str:
    """Markdown pipes: compact, and LLMs read the structure reliably."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [" ".join(cell.text.split()) for cell in row.cells]
        rows.append(cells)
    if not rows:
        return ""

    # Drop trailing all-empty rows that Word leaves behind.
    while rows and not any(cell.strip() for cell in rows[-1]):
        rows.pop()
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _image_dimensions(data: bytes) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as img:
            return img.width, img.height
    except Exception:  # noqa: BLE001
        return 0, 0


def _collect_images(document: DocxDocument) -> dict[str, ExtractedImage]:
    """Map relationship id -> image, skipping decorative assets."""
    images: dict[str, ExtractedImage] = {}
    for rel_id, rel in document.part.rels.items():
        if "image" not in rel.reltype:
            continue
        try:
            blob = rel.target_part.blob
        except Exception:  # noqa: BLE001 - linked (not embedded) image
            continue
        width, height = _image_dimensions(blob)
        if width and height:
            if width < settings.min_image_pixels or height < settings.min_image_pixels:
                continue
            if width / max(1, height) > 20 or height / max(1, width) > 20:
                continue

        content_type = getattr(rel.target_part, "content_type", "image/png")
        images[rel_id] = ExtractedImage(
            ref=rel_id,
            data=blob,
            mime=content_type or "image/png",
            width=width,
            height=height,
            page_no=None,
            origin="embedded",
        )
    return images


def parse_docx(path, filename: str) -> ParsedDocument:
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ParserError(
            "Could not open Word document '{}': {}. Note that legacy .doc files "
            "are not supported -- re-save as .docx.".format(filename, str(exc)[:160])
        ) from exc

    result = ParsedDocument()
    available_images = _collect_images(document)
    used_refs: set[str] = set()

    section_stack: list[str] = []
    buffer: list[str] = []
    buffer_refs: list[str] = []
    order = 0

    def current_section() -> str | None:
        return " > ".join(section_stack) if section_stack else None

    def flush() -> None:
        nonlocal buffer, buffer_refs, order
        text = clean_text("\n\n".join(buffer))
        if text:
            order += 1
            result.blocks.append(
                TextBlock(
                    text=text,
                    page_no=None,
                    section=current_section(),
                    modality="text",
                    media_refs=list(dict.fromkeys(buffer_refs)),
                    order=order,
                )
            )
        buffer = []
        buffer_refs = []

    for item in _iter_body(document):
        if isinstance(item, Paragraph):
            text = item.text.strip()
            rel_ids = [r for r in _paragraph_image_ids(item) if r in available_images]
            for rel_id in rel_ids:
                used_refs.add(rel_id)
                # A short paragraph next to an image is almost always its caption.
                if text and len(text) < 300 and available_images[rel_id].caption is None:
                    available_images[rel_id].caption = text

            level = _heading_level(item)
            if level is not None and text:
                flush()
                del section_stack[level - 1 :]
                section_stack.append(text)
                order += 1
                result.blocks.append(
                    TextBlock(
                        text=text,
                        section=current_section(),
                        modality="text",
                        order=order,
                    )
                )
                continue

            if rel_ids:
                buffer_refs.extend(rel_ids)
            if text:
                buffer.append(text)

        elif isinstance(item, Table):
            flush()
            markdown = _table_to_markdown(item)
            if markdown:
                order += 1
                result.blocks.append(
                    TextBlock(
                        text=markdown,
                        section=current_section(),
                        modality="table",
                        order=order,
                    )
                )

    flush()

    # Images referenced inline keep their position; the rest are appended so
    # nothing embedded in headers, footers or shapes is silently dropped.
    result.images = [available_images[r] for r in used_refs if r in available_images]
    result.images.extend(img for ref, img in available_images.items() if ref not in used_refs)

    for image in result.images:
        if image.caption:
            order += 1
            result.blocks.append(
                TextBlock(
                    text=image.caption,
                    section=None,
                    modality="caption",
                    media_refs=[image.ref],
                    order=order,
                )
            )

    core = document.core_properties
    result.metadata = {
        "title": core.title or "",
        "author": core.author or "",
        "paragraphs": len(result.blocks),
    }
    result.page_count = 0

    if result.is_empty:
        result.warnings.append("The document appears to be empty.")

    return result
