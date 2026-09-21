"""Shared contracts every parser returns.

A parser's job is to turn one file into ordered TextBlocks plus ExtractedImages.
Chunking, embedding and storage all happen downstream, so adding a new format
means writing one parser and registering it -- nothing else changes.
"""
from __future__ import annotations

import hashlib
import mimetypes
import pathlib
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """A contiguous run of text with the provenance needed for citations."""

    text: str
    page_no: int | None = None
    section: str | None = None
    # text | table | caption | ocr | json
    modality: str = "text"
    # Ids of images this block describes or sits next to.
    media_refs: list[str] = field(default_factory=list)
    order: int = 0


@dataclass
class ExtractedImage:
    """An image pulled out of a document, or an uploaded image file itself."""

    # Stable key used to link this image to blocks before DB ids exist.
    ref: str
    data: bytes
    mime: str = "image/png"
    width: int = 0
    height: int = 0
    page_no: int | None = None
    caption: str | None = None
    ocr_text: str | None = None
    # embedded | uploaded | page_render
    origin: str = "embedded"


@dataclass
class ParsedDocument:
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    page_count: int = 0
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def text_length(self) -> int:
        return sum(len(b.text) for b in self.blocks)

    @property
    def is_empty(self) -> bool:
        return not self.blocks and not self.images


class ParserError(Exception):
    """A file could not be parsed. Carries a message meant for the end user."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def guess_mime(filename: str, default: str = "application/octet-stream") -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or default


def clean_text(raw: str) -> str:
    """Normalise whitespace without destroying paragraph structure."""
    if not raw:
        return ""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("﻿", "")
    # Repair words split across a line break by hyphenation.
    text = text.replace("-\n", "")
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    blank = 0
    for line in lines:
        if not line.strip():
            blank += 1
            if blank <= 1:
                out.append("")
            continue
        blank = 0
        out.append(" ".join(line.split()))
    return "\n".join(out).strip()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".json",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".zip",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
