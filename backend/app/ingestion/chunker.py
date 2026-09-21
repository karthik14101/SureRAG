"""Chunking strategies, one per modality.

Chunk boundaries decide retrieval quality more than any model choice: a fact
split across two chunks is usually unfindable. So we split on the largest
natural boundary that fits (paragraph, then sentence, then word), overlap
adjacent chunks, and never merge across a page or section boundary.

Token counts are approximated at ~4 characters per token. That avoids a tokeniser
dependency that would be wrong for at least one of the two providers anyway, and
the estimate only needs to be good enough to size a window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.ingestion.parsers.base import TextBlock

CHARS_PER_TOKEN = 4
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
# A table row must never be cut in half.
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Chunk:
    text: str
    ordinal: int
    page_no: int | None = None
    section: str | None = None
    modality: str = "text"
    media_refs: list[str] = field(default_factory=list)
    token_count: int = 0


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


def _target_chars() -> int:
    return settings.chunk_size_tokens * CHARS_PER_TOKEN


def _overlap_chars() -> int:
    return settings.chunk_overlap_tokens * CHARS_PER_TOKEN


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """Split oversized text on the largest boundary that fits."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    for paragraph in PARAGRAPH_SPLIT.split(text):
        if not paragraph.strip():
            continue
        if len(paragraph) <= max_chars:
            pieces.append(paragraph)
            continue

        sentences = SENTENCE_SPLIT.split(paragraph)
        buffer = ""
        for sentence in sentences:
            if len(sentence) > max_chars:
                # A single monstrous sentence (minified JSON, a wall of CSV).
                if buffer:
                    pieces.append(buffer)
                    buffer = ""
                words = sentence.split(" ")
                current = ""
                for word in words:
                    if len(current) + len(word) + 1 > max_chars:
                        if current:
                            pieces.append(current)
                        # A single word longer than the window: hard split.
                        while len(word) > max_chars:
                            pieces.append(word[:max_chars])
                            word = word[max_chars:]
                        current = word
                    else:
                        current = current + " " + word if current else word
                if current:
                    buffer = current
                continue

            if len(buffer) + len(sentence) + 1 > max_chars:
                if buffer:
                    pieces.append(buffer)
                buffer = sentence
            else:
                buffer = buffer + " " + sentence if buffer else sentence
        if buffer:
            pieces.append(buffer)

    return [p.strip() for p in pieces if p.strip()]


def _with_overlap(pieces: list[str], overlap_chars: int) -> list[str]:
    """Prefix each chunk with the tail of the previous one, on a word boundary."""
    if overlap_chars <= 0 or len(pieces) < 2:
        return pieces
    out = [pieces[0]]
    for index in range(1, len(pieces)):
        previous = pieces[index - 1]
        tail = previous[-overlap_chars:]
        space = tail.find(" ")
        if space > 0:
            tail = tail[space + 1 :]
        out.append((tail + " " + pieces[index]).strip() if tail.strip() else pieces[index])
    return out


def _table_heading(block: TextBlock) -> str:
    """A language prefix for a table chunk.

    A bare grid of numbers embeds terribly -- there is almost no natural language
    in it for a question to match against. Prefixing the table's own heading and
    page makes it findable by a question phrased in words rather than figures.
    """
    parts: list[str] = []
    if block.section:
        parts.append(block.section.strip())
    if block.page_no:
        parts.append("(table, page {})".format(block.page_no))
    elif parts:
        parts.append("(table)")
    return " ".join(parts).strip()


def _chunk_table(block: TextBlock, max_chars: int, start_ordinal: int) -> list[Chunk]:
    """Split a markdown table by rows, repeating the header in every chunk."""
    heading = _table_heading(block)
    prefix = heading + "\n\n" if heading else ""
    # Leave room for the heading and the repeated header row.
    max_chars = max(400, max_chars - len(prefix))

    lines = [line for line in block.text.split("\n") if line.strip()]
    if len(lines) < 2:
        text = prefix + block.text
        return [
            Chunk(
                text=text,
                ordinal=start_ordinal,
                page_no=block.page_no,
                section=block.section,
                modality="table",
                media_refs=list(block.media_refs),
                token_count=estimate_tokens(text),
            )
        ]

    header = lines[0]
    separator = lines[1] if TABLE_ROW.match(lines[1]) or "---" in lines[1] else ""
    body = lines[2:] if separator else lines[1:]
    # Repeated at the top of every piece so each chunk is a readable table.
    header_prefix = header + ("\n" + separator if separator else "")

    chunks: list[Chunk] = []
    buffer: list[str] = []
    ordinal = start_ordinal
    for row in body:
        candidate_len = len(header_prefix) + sum(len(r) + 1 for r in buffer) + len(row)
        if buffer and candidate_len > max_chars:
            text = prefix + header_prefix + "\n" + "\n".join(buffer)
            chunks.append(
                Chunk(
                    text=text,
                    ordinal=ordinal,
                    page_no=block.page_no,
                    section=block.section,
                    modality="table",
                    media_refs=list(block.media_refs),
                    token_count=estimate_tokens(text),
                )
            )
            ordinal += 1
            buffer = [row]
        else:
            buffer.append(row)

    if buffer:
        text = prefix + header_prefix + "\n" + "\n".join(buffer)
        chunks.append(
            Chunk(
                text=text,
                ordinal=ordinal,
                page_no=block.page_no,
                section=block.section,
                modality="table",
                media_refs=list(block.media_refs),
                token_count=estimate_tokens(text),
            )
        )
    return chunks


def chunk_blocks(blocks: list[TextBlock]) -> list[Chunk]:
    """Turn parsed blocks into embeddable chunks.

    Small adjacent blocks sharing a page and section are merged so a two-line
    paragraph does not become its own near-useless chunk.
    """
    max_chars = _target_chars()
    overlap = _overlap_chars()
    min_merge_chars = max(200, max_chars // 5)

    chunks: list[Chunk] = []
    ordinal = 0

    # Tables, captions and JSON records are structural; keep them intact.
    pending: list[TextBlock] = []

    def flush_pending() -> None:
        nonlocal pending, ordinal
        if not pending:
            return
        merged_text = "\n\n".join(b.text for b in pending if b.text.strip())
        if not merged_text.strip():
            pending = []
            return
        first = pending[0]
        media_refs: list[str] = []
        for block in pending:
            media_refs.extend(block.media_refs)

        pieces = _split_long_text(merged_text, max_chars)
        pieces = _with_overlap(pieces, overlap)
        for piece in pieces:
            if not piece.strip():
                continue
            ordinal += 1
            chunks.append(
                Chunk(
                    text=piece,
                    ordinal=ordinal,
                    page_no=first.page_no,
                    section=first.section,
                    modality="text",
                    media_refs=list(dict.fromkeys(media_refs)),
                    token_count=estimate_tokens(piece),
                )
            )
        pending = []

    for block in blocks:
        if not block.text.strip():
            continue

        if block.modality in ("table", "caption", "json", "ocr"):
            flush_pending()
            if block.modality == "table":
                produced = _chunk_table(block, max_chars, ordinal + 1)
                chunks.extend(produced)
                ordinal += len(produced)
            else:
                pieces = _split_long_text(block.text, max_chars)
                for piece in pieces:
                    ordinal += 1
                    chunks.append(
                        Chunk(
                            text=piece,
                            ordinal=ordinal,
                            page_no=block.page_no,
                            section=block.section,
                            modality=block.modality,
                            media_refs=list(block.media_refs),
                            token_count=estimate_tokens(piece),
                        )
                    )
            continue

        # Page or section change breaks the merge run so provenance stays exact.
        if pending:
            previous = pending[-1]
            same_page = previous.page_no == block.page_no
            same_section = previous.section == block.section
            current_len = sum(len(b.text) for b in pending)
            if not (same_page and same_section) or current_len >= min_merge_chars:
                flush_pending()

        pending.append(block)

        if sum(len(b.text) for b in pending) >= max_chars:
            flush_pending()

    flush_pending()
    return chunks
