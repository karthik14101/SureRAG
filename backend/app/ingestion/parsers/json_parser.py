"""JSON ingestion.

Raw JSON embeds badly -- braces and quotes carry no semantic signal. We flatten
it into readable `path = value` lines and keep the JSON path as the chunk's
section, so a citation can point at `orders[3].customer.name` precisely.

Arrays of records (the common export shape) are split one record per block, so
retrieval returns whole records instead of fragments of several.
"""
from __future__ import annotations

import json
import pathlib

from app.ingestion.parsers.base import ParsedDocument, ParserError, TextBlock, clean_text
from app.ingestion.parsers.text_parser import read_text_file

MAX_DEPTH = 12
MAX_LINES_PER_BLOCK = 400
ARRAY_RECORD_THRESHOLD = 3  # arrays at least this long become one block per item


def _format_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    return " ".join(text.split())


def flatten(node, prefix: str = "", depth: int = 0, out: list[str] | None = None) -> list[str]:
    """Depth-first flatten into `path = value` lines."""
    if out is None:
        out = []
    if depth > MAX_DEPTH:
        out.append("{} = [nesting too deep]".format(prefix or "root"))
        return out

    if isinstance(node, dict):
        if not node:
            out.append("{} = {{}}".format(prefix or "root"))
        for key, value in node.items():
            safe_key = str(key)
            path = "{}.{}".format(prefix, safe_key) if prefix else safe_key
            flatten(value, path, depth + 1, out)
    elif isinstance(node, list):
        if not node:
            out.append("{} = []".format(prefix or "root"))
        for index, value in enumerate(node):
            path = "{}[{}]".format(prefix, index)
            flatten(value, path, depth + 1, out)
    else:
        out.append("{} = {}".format(prefix or "root", _format_scalar(node)))
    return out


def _record_title(record, index: int) -> str:
    """A human label for a record block, taken from a likely name field."""
    if isinstance(record, dict):
        for key in ("name", "title", "id", "key", "label", "subject", "question"):
            value = record.get(key)
            if isinstance(value, (str, int)) and str(value).strip():
                return "{}".format(str(value).strip())[:120]
    return "record {}".format(index)


def parse_json(path, filename: str) -> ParsedDocument:
    path = pathlib.Path(path)
    try:
        raw = read_text_file(path)
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParserError(
            "'{}' is not valid JSON (line {}, column {}): {}".format(
                filename, exc.lineno, exc.colno, exc.msg
            )
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise ParserError("Could not read '{}': {}".format(filename, str(exc)[:200])) from exc

    result = ParsedDocument()
    order = 0

    # Shape 1: a top-level array of records.
    if isinstance(data, list) and len(data) >= ARRAY_RECORD_THRESHOLD:
        for index, record in enumerate(data):
            lines = flatten(record, "[{}]".format(index))
            text = clean_text("\n".join(lines[:MAX_LINES_PER_BLOCK]))
            if not text:
                continue
            order += 1
            result.blocks.append(
                TextBlock(
                    text=text,
                    section="{}[{}] ({})".format(
                        path.stem, index, _record_title(record, index)
                    ),
                    modality="json",
                    order=order,
                )
            )
        result.metadata = {"records": len(data), "shape": "array"}

    # Shape 2: an object whose values are large collections -> one block per key.
    elif isinstance(data, dict) and any(
        isinstance(v, (list, dict)) and len(v) >= ARRAY_RECORD_THRESHOLD for v in data.values()
    ):
        for key, value in data.items():
            lines = flatten(value, str(key))
            text = clean_text("\n".join(lines[:MAX_LINES_PER_BLOCK]))
            if not text:
                continue
            order += 1
            result.blocks.append(
                TextBlock(text=text, section=str(key), modality="json", order=order)
            )
        result.metadata = {"keys": list(data.keys())[:50], "shape": "object"}

    # Shape 3: anything else -> a single flattened block.
    else:
        lines = flatten(data)
        text = clean_text("\n".join(lines[:MAX_LINES_PER_BLOCK * 4]))
        if text:
            order += 1
            result.blocks.append(
                TextBlock(text=text, section=path.stem, modality="json", order=order)
            )
        result.metadata = {"shape": type(data).__name__}

    if result.is_empty:
        result.warnings.append("The JSON file contained no indexable values.")

    return result
