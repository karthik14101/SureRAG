"""Stripping <think>...</think> reasoning preambles from model output.

Some reasoning models (DeepSeek-R1 on Groq or on Azure Foundry, among others)
emit their chain of thought inline, wrapped in <think> tags. That text must never
reach the user or the citation parser. Shared here so every provider applies the
same rule, including across streamed chunk boundaries.
"""
from __future__ import annotations

import re

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN = "<think>"
_CLOSE = "</think>"
# Held back at the end of the buffer in case a tag is split across chunks.
# Must be at least len("</think>") - 1.
_TAIL_GUARD = 16


def strip_reasoning(text: str) -> str:
    """Remove complete reasoning blocks from a finished response."""
    cleaned = _THINK_RE.sub("", text or "")
    # An unterminated block means the model was cut off mid-thought.
    if _OPEN in cleaned.lower():
        cleaned = cleaned[: cleaned.lower().index(_OPEN)]
    return cleaned.strip()


class ThinkTagFilter:
    """Incremental version of `strip_reasoning` for streamed output.

    Feed each chunk as it arrives and emit whatever `feed` returns; call `flush`
    once the stream ends. Tags that straddle a chunk boundary are handled by
    holding back a short tail until the next chunk resolves it.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_think = False

    def feed(self, piece: str) -> str:
        if not piece:
            return ""
        self._buffer += piece
        emitted: list[str] = []

        while self._buffer:
            lowered = self._buffer.lower()
            if self._in_think:
                end = lowered.find(_CLOSE)
                if end == -1:
                    # Still inside the reasoning block: discard, but keep a tail
                    # in case the closing tag is being split.
                    self._buffer = self._buffer[-_TAIL_GUARD:]
                    break
                self._buffer = self._buffer[end + len(_CLOSE) :]
                self._in_think = False
                continue

            start = lowered.find(_OPEN)
            if start == -1:
                if len(self._buffer) > _TAIL_GUARD:
                    emitted.append(self._buffer[:-_TAIL_GUARD])
                    self._buffer = self._buffer[-_TAIL_GUARD:]
                break

            if start > 0:
                emitted.append(self._buffer[:start])
            self._buffer = self._buffer[start + len(_OPEN) :]
            self._in_think = True

        return "".join(emitted)

    def flush(self) -> str:
        """Release anything held back. Returns nothing if still mid-thought."""
        remainder = "" if self._in_think else self._buffer
        self._buffer = ""
        return remainder
