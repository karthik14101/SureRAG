"""PDF parsing with PyMuPDF: tables, reading-order text, and embedded figures.

Chunk quality is decided here, and it decides retrieval quality. Four things
matter more than they look:

  1. Tables are detected and emitted as markdown, BEFORE text extraction runs.
     Generic block sorting linearises a grid into column-order soup -- headers
     detach from values and the numbers become unretrievable.
  2. Table regions are masked out of the prose flow, so a cell is never also
     emitted as loose text.
  3. Text is NEVER removed from the page in order to caption an image. An
     earlier version did, which let a photo inside a table steal the cell
     beside it out of the surrounding paragraph.
  4. Only an explicit "Figure 3 - ..." style caption becomes its own chunk.
     Everything else stays in the prose where it has context; images still get a
     display caption, and are still reachable through their parent chunk.
"""
from __future__ import annotations

import io
import re
from collections import Counter

import fitz  # PyMuPDF

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

# Only these openers earn a standalone caption chunk.
CAPTION_PREFIX_RE = re.compile(
    r"^(figure|fig\.?|table|chart|diagram|exhibit|illustration|photo|plate)\s*[\dIVX]",
    re.IGNORECASE,
)
MIN_CAPTION_WORDS = 6          # shorter than this is a label, not a caption
CAPTION_MAX_DISTANCE = 120.0   # points below an image to look for its caption
DISPLAY_CAPTION_MAX_CHARS = 300

HEADER_ZONE = 0.07             # fraction of page height treated as header/footer
MIN_REPEAT_RATIO = 0.6         # a line on 60%+ of pages is furniture

TABLE_OVERLAP_RATIO = 0.55     # block counts as "inside" a table above this
TABLE_TITLE_MAX_DISTANCE = 90.0
TABLE_TITLE_MAX_CHARS = 90
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _rect(block: tuple) -> fitz.Rect:
    return fitz.Rect(block[0], block[1], block[2], block[3])


def _overlap_ratio(rect: fitz.Rect, other: fitz.Rect) -> float:
    """Fraction of `rect` that lies inside `other`."""
    area = rect.get_area()
    if area <= 0:
        return 0.0
    intersection = rect & other
    if intersection.is_empty:
        return 0.0
    return intersection.get_area() / area


def _inside_any(rect: fitz.Rect, regions: list[fitz.Rect]) -> bool:
    return any(_overlap_ratio(rect, region) >= TABLE_OVERLAP_RATIO for region in regions)


# ---------------------------------------------------------------------------
# Text blocks
# ---------------------------------------------------------------------------
def _page_blocks(page) -> list[tuple[float, float, float, float, str]]:
    """Text blocks sorted into reading order: top-to-bottom, then left-to-right."""
    raw = page.get_text("blocks")
    blocks = []
    for item in raw:
        if len(item) < 5:
            continue
        x0, y0, x1, y1, text = item[0], item[1], item[2], item[3], item[4]
        # block_type 1 is an image placeholder; images are handled separately.
        if len(item) >= 7 and item[6] != 0:
            continue
        if not str(text).strip():
            continue
        blocks.append((float(x0), float(y0), float(x1), float(y1), str(text)))
    # Round y to a 12pt band so side-by-side columns stay on the same visual row.
    blocks.sort(key=lambda b: (round(b[1] / 12.0), b[0]))
    return blocks


def _detect_furniture(pages_lines: list[list[str]], page_count: int) -> set[str]:
    """Lines repeating across most pages are headers/footers/watermarks."""
    if page_count < 3:
        return set()
    counter: Counter[str] = Counter()
    for lines in pages_lines:
        for line in set(lines):
            normalised = re.sub(r"\d+", "#", line.strip())
            if 3 <= len(normalised) <= 120:
                counter[normalised] += 1
    threshold = max(2, int(page_count * MIN_REPEAT_RATIO))
    return {line for line, count in counter.items() if count >= threshold}


def _is_furniture(text: str, furniture: set[str]) -> bool:
    normalised = re.sub(r"\d+", "#", text.strip())
    if normalised in furniture:
        return True
    # A bare page number.
    return bool(re.fullmatch(r"[-–\s]*\d{1,4}[-–\s]*", text.strip()))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _cell_text(value) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    # A literal pipe would break the markdown row.
    return text.replace("|", "\\|")


def _merge_rows(rows: list[list[str]]) -> list[str]:
    """Collapse several header rows into one, column by column."""
    if not rows:
        return []
    width = max(len(row) for row in rows)
    merged: list[str] = []
    for column in range(width):
        parts = [row[column].strip() for row in rows if column < len(row) and row[column].strip()]
        merged.append(" ".join(parts))
    return merged


def _split_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Separate a possibly multi-line header from the body.

    PDF tables routinely wrap a header across two or three physical rows
    ("Charging" / "Time In" / "SoC Band"). Taking only the first row as the
    header leaves the columns mislabelled, which is worse than no header at all
    because the model then reads the values under the wrong names.
    """
    if not rows:
        return [], []

    # Rule 1: a numbered first column marks unambiguously where data begins.
    for index, row in enumerate(rows[:5]):
        first = (row[0] or "").strip()
        if re.fullmatch(r"\d{1,3}[.)]?", first):
            if index >= 1:
                return _merge_rows(rows[:index]), rows[index:]
            break

    # Rule 2: leading rows with many empty cells are header fragments; the first
    # dense row after them completes the header.
    def is_sparse(row: list[str]) -> bool:
        empties = sum(1 for cell in row if not cell.strip())
        return empties >= max(1, len(row) // 3)

    if is_sparse(rows[0]) and len(rows) > 2:
        end = 1
        while end < min(4, len(rows)) and is_sparse(rows[end]):
            end += 1
        if end < len(rows) - 1:
            end += 1
        return _merge_rows(rows[:end]), rows[end:]

    return rows[0], rows[1:]


def _table_markdown(rows: list[list]) -> str:
    """Render extracted cells as a markdown table with a real header row."""
    cleaned = [[_cell_text(cell) for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    if len(cleaned) < MIN_TABLE_ROWS:
        return ""

    width = max(len(row) for row in cleaned)
    if width < MIN_TABLE_COLS:
        return ""
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]

    header, body = _split_header(cleaned)
    header = (header + [""] * width)[:width]
    if not body:
        return ""
    # A blank header row would make every column nameless; synthesise one.
    if not any(cell.strip() for cell in header):
        header = ["Column {}".format(i + 1) for i in range(width)]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def _find_tables(page) -> list[tuple[fitz.Rect, str]]:
    """Detect tables and render each as markdown. Returns (bbox, markdown)."""
    try:
        finder = page.find_tables()
    except Exception as exc:  # noqa: BLE001 - detection must never fail a page
        logger.debug("find_tables failed on a page: %s", str(exc)[:160])
        return []

    results: list[tuple[fitz.Rect, str]] = []
    for table in getattr(finder, "tables", []) or []:
        try:
            if table.row_count < MIN_TABLE_ROWS or table.col_count < MIN_TABLE_COLS:
                continue
            rows = table.extract()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Table extraction failed: %s", str(exc)[:160])
            continue

        markdown = _table_markdown(rows)
        if not markdown:
            continue

        # Reject "tables" that are really layout grids holding almost no text.
        text_volume = sum(len(_cell_text(c)) for row in rows for c in row)
        if text_volume < 40:
            continue

        results.append((fitz.Rect(table.bbox), markdown))

    return results


def _table_title(table_rect: fitz.Rect, blocks: list[tuple], table_rects: list[fitz.Rect]) -> str | None:
    """Nearest short line above the table -- usually its heading.

    Worth the effort: a bare grid of numbers embeds poorly, but the same grid
    under "TYPE OF CHARGING" is findable by a natural-language question.
    """
    best: str | None = None
    best_distance = TABLE_TITLE_MAX_DISTANCE

    for block in blocks:
        rect = _rect(block)
        if _inside_any(rect, table_rects):
            continue
        distance = table_rect.y0 - rect.y1
        if distance < -4 or distance > TABLE_TITLE_MAX_DISTANCE:
            continue
        # Require horizontal alignment with the table.
        if rect.x1 < table_rect.x0 - 40 or rect.x0 > table_rect.x1 + 40:
            continue
        text = " ".join(str(block[4]).split())
        if not text or len(text) > TABLE_TITLE_MAX_CHARS:
            continue
        if re.fullmatch(r"[-–\s]*\d{1,4}[-–\s]*", text):
            continue  # page number
        if distance < best_distance:
            best_distance = distance
            best = text

    return best


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def _is_meaningful_image(data: bytes, width: int, height: int) -> bool:
    """Reject icons, rules, spacers and near-blank panels."""
    if width < settings.min_image_pixels or height < settings.min_image_pixels:
        return False
    if width * height < settings.min_image_pixels**2:
        return False
    # Extreme aspect ratios are almost always decorative rules.
    if width / max(1, height) > 20 or height / max(1, width) > 20:
        return False
    try:
        from PIL import Image, ImageStat

        with Image.open(io.BytesIO(data)) as img:
            sample = img.convert("L").resize((32, 32))
            stat = ImageStat.Stat(sample)
            # A near-uniform panel carries no information.
            if stat.stddev and stat.stddev[0] < 6.0:
                return False
    except Exception:  # noqa: BLE001 - keep the image if inspection fails
        return True
    return True


def _display_caption(
    image_rect: fitz.Rect, blocks: list[tuple], table_rects: list[fitz.Rect]
) -> str | None:
    """A short label for the figure, for display only.

    Crucially this does NOT claim the block: the same text still appears in the
    page's prose. Nothing is removed from the text flow to build a caption.
    """
    explicit: tuple[float, str] | None = None
    fallback: tuple[float, str] | None = None

    for block in blocks:
        rect = _rect(block)
        # Never caption from inside a table -- those are cells, not captions.
        if _inside_any(rect, table_rects):
            continue

        distance = rect.y0 - image_rect.y1
        if distance < -8 or distance > CAPTION_MAX_DISTANCE:
            continue
        if rect.x1 < image_rect.x0 - 40 or rect.x0 > image_rect.x1 + 40:
            continue

        text = " ".join(str(block[4]).split())
        if not text or len(text) > DISPLAY_CAPTION_MAX_CHARS:
            continue
        if re.fullmatch(r"[-–\s]*\d{1,4}[-–\s]*", text):
            continue  # a page number is not a caption

        if CAPTION_PREFIX_RE.match(text):
            if explicit is None or distance < explicit[0]:
                explicit = (distance, text)
        elif fallback is None or distance < fallback[0]:
            fallback = (distance, text)

    chosen = explicit or fallback
    return chosen[1] if chosen else None


def _extract_images(
    doc, page, page_index: int, blocks: list[tuple], table_rects: list[fitz.Rect]
) -> tuple[list[ExtractedImage], dict[str, fitz.Rect]]:
    images: list[ExtractedImage] = []
    ref_rects: dict[str, fitz.Rect] = {}

    try:
        image_list = page.get_images(full=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("get_images failed on page %d: %s", page_index + 1, exc)
        return images, ref_rects

    seen_xrefs: set[int] = set()
    for entry in image_list:
        xref = entry[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            pixmap = fitz.Pixmap(doc, xref)
            # CMYK / separation colourspaces cannot be written straight to PNG.
            if pixmap.n - pixmap.alpha >= 4:
                pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
            width, height = pixmap.width, pixmap.height
            data = pixmap.tobytes("png")
            pixmap = None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Could not decode image xref %s: %s", xref, exc)
            continue

        if not _is_meaningful_image(data, width, height):
            continue

        ref = "p{}_x{}".format(page_index + 1, xref)
        caption = None
        try:
            rects = page.get_image_rects(xref)
            if rects:
                image_rect = fitz.Rect(rects[0])
                ref_rects[ref] = image_rect
                caption = _display_caption(image_rect, blocks, table_rects)
        except Exception:  # noqa: BLE001 - placement is a bonus
            pass

        images.append(
            ExtractedImage(
                ref=ref,
                data=data,
                mime="image/png",
                width=width,
                height=height,
                page_no=page_index + 1,
                caption=clean_text(caption) if caption else None,
                origin="embedded",
            )
        )

    return images, ref_rects


_BARE_NUMBER_LINE = re.compile(r"^[-–\s]*\d{1,4}[-–\s]*$")


def _strip_page_number(text: str) -> str:
    """Remove a trailing line that is only a page number.

    The margin-zone furniture filter catches most of these, but a page number
    that sits slightly inside the margin band survives, and it is pure noise in
    an embedding.
    """
    lines = text.split("\n")
    while lines and _BARE_NUMBER_LINE.fullmatch(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).strip()


def _is_standalone_caption(text: str | None) -> bool:
    """Only an explicit, substantial figure caption earns its own chunk."""
    if not text:
        return False
    if not CAPTION_PREFIX_RE.match(text.strip()):
        return False
    return len(text.split()) >= MIN_CAPTION_WORDS


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_pdf(path, filename: str) -> ParsedDocument:
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        raise ParserError("Could not open PDF '{}': {}".format(filename, str(exc)[:200])) from exc

    result = ParsedDocument()

    try:
        if doc.needs_pass:
            raise ParserError(
                "'{}' is password protected. Remove the password and re-upload.".format(filename)
            )

        page_count = doc.page_count
        result.page_count = page_count
        result.metadata = {
            "title": (doc.metadata or {}).get("title") or "",
            "author": (doc.metadata or {}).get("author") or "",
            "pages": page_count,
        }

        all_page_blocks: list[list[tuple]] = []
        page_heights: list[float] = []
        for index in range(page_count):
            page = doc.load_page(index)
            all_page_blocks.append(_page_blocks(page))
            page_heights.append(float(page.rect.height))

        furniture = _detect_furniture(
            [[b[4].strip() for b in blocks] for blocks in all_page_blocks], page_count
        )

        order = 0
        table_count = 0

        for index in range(page_count):
            page = doc.load_page(index)
            blocks = all_page_blocks[index]
            height = page_heights[index] or 1.0

            # --- tables first: they claim their region ------------------------
            tables = _find_tables(page)
            table_rects = [rect for rect, _ in tables]

            images, ref_rects = _extract_images(doc, page, index, blocks, table_rects)
            result.images.extend(images)

            # Images sitting inside a table belong to that table, not the prose.
            table_image_refs: set[str] = set()
            for rect, markdown in tables:
                refs = [
                    ref
                    for ref, image_rect in ref_rects.items()
                    if _overlap_ratio(image_rect, rect) >= TABLE_OVERLAP_RATIO
                ]
                table_image_refs.update(refs)
                title = _table_title(rect, blocks, table_rects)
                order += 1
                table_count += 1
                result.blocks.append(
                    TextBlock(
                        text=markdown,
                        page_no=index + 1,
                        section=title,
                        modality="table",
                        media_refs=refs,
                        order=order,
                    )
                )

            # --- standalone figure captions -----------------------------------
            for image in images:
                if _is_standalone_caption(image.caption):
                    order += 1
                    result.blocks.append(
                        TextBlock(
                            text=image.caption,
                            page_no=index + 1,
                            section=None,
                            modality="caption",
                            media_refs=[image.ref],
                            order=order,
                        )
                    )

            # --- prose: everything not inside a table -------------------------
            page_parts: list[str] = []
            for block in blocks:
                rect = _rect(block)
                if _inside_any(rect, table_rects):
                    continue  # already emitted as part of the table
                stripped = str(block[4]).strip()
                if not stripped:
                    continue
                in_margin = rect.y0 < height * HEADER_ZONE or rect.y1 > height * (1 - HEADER_ZONE)
                if in_margin and _is_furniture(stripped, furniture):
                    continue
                page_parts.append(stripped)

            page_text = _strip_page_number(clean_text("\n".join(page_parts)))
            prose_images = [img.ref for img in images if img.ref not in table_image_refs]

            # On a page whose content is entirely tabular, the only prose left is
            # the running section header. That is noise on its own -- but keep it
            # when it anchors images, which would otherwise have no text at all.
            trivial = len(page_text.split()) <= 3
            if page_text and not (trivial and not prose_images):
                order += 1
                result.blocks.append(
                    TextBlock(
                        text=page_text,
                        page_no=index + 1,
                        modality="text",
                        media_refs=prose_images,
                        order=order,
                    )
                )

        if table_count:
            result.metadata["tables"] = table_count
            logger.info("%s: extracted %d table(s)", filename, table_count)

        if not result.blocks and result.images:
            result.warnings.append(
                "No selectable text found -- this PDF looks scanned. "
                "Figures were still indexed; enable OCR for the page text."
            )
        elif not result.blocks:
            result.warnings.append("No text or images could be extracted from this PDF.")

        return result
    finally:
        doc.close()
