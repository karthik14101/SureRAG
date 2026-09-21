"""Plain text and Markdown.

Markdown headings are tracked so chunks carry a section path, which makes
citations far more useful than a bare line number.
"""
from __future__ import annotations

import pathlib
import re

from app.ingestion.parsers.base import ParsedDocument, ParserError, TextBlock, clean_text

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

# Tried in order; the first decoder that succeeds wins.
ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def read_text_file(path: pathlib.Path) -> str:
    raw = path.read_bytes()
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot fail, but keep an explicit last resort for clarity.
    return raw.decode("utf-8", errors="replace")


def parse_text(path, filename: str) -> ParsedDocument:
    path = pathlib.Path(path)
    try:
        content = read_text_file(path)
    except Exception as exc:  # noqa: BLE001
        raise ParserError(
            "Could not read '{}': {}".format(filename, str(exc)[:200])
        ) from exc

    result = ParsedDocument()
    is_markdown = path.suffix.lower() in {".md", ".markdown"}

    if not is_markdown:
        text = clean_text(content)
        if text:
            result.blocks.append(TextBlock(text=text, modality="text", order=1))
        else:
            result.warnings.append("The file is empty.")
        result.metadata = {"characters": len(text)}
        return result

    section_stack: list[str] = []
    buffer: list[str] = []
    order = 0

    def section_path() -> str | None:
        return " > ".join(section_stack) if section_stack else None

    def flush() -> None:
        nonlocal buffer, order
        text = clean_text("\n".join(buffer))
        if text:
            order += 1
            result.blocks.append(
                TextBlock(text=text, section=section_path(), modality="text", order=order)
            )
        buffer = []

    for line in content.splitlines():
        match = HEADING_RE.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            del section_stack[level - 1 :]
            if title:
                section_stack.append(title)
                order += 1
                result.blocks.append(
                    TextBlock(
                        text=title, section=section_path(), modality="text", order=order
                    )
                )
            continue
        buffer.append(line)

    flush()

    if result.is_empty:
        result.warnings.append("The file is empty.")
    result.metadata = {"characters": len(content), "markdown": True}
    return result
