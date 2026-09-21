"""Provider-agnostic LLM interface.

Every call site in the agent talks to this ABC, never to a vendor SDK. Swapping
LLM_PROVIDER in .env changes the concrete class and nothing else.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls

    def as_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }


@dataclass
class LLMResult:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    model: str = ""


class LLMProvider(ABC):
    """Minimum surface the SURE engine needs from any provider."""

    name: str = "base"
    supports_vision: bool = False

    @abstractmethod
    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        """One-shot completion."""

    @abstractmethod
    def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Yield answer text incrementally. Must be an async generator."""

    async def describe_image(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> str | None:
        """Caption/read an image. Returns None when the provider has no vision model."""
        return None

    async def complete_json(
        self,
        system: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> tuple[dict, TokenUsage]:
        """Completion constrained to a JSON object, parsed defensively."""
        result = await self.complete(
            system,
            [ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return parse_json_object(result.text), result.usage

    async def warmup(self) -> None:
        """Optional startup hook, run in the background. Local providers use it
        to load the model so the first question does not pay for it."""
        return None

    async def health(self) -> tuple[bool, str]:
        try:
            result = await self.complete(
                "You are a health probe.",
                [ChatMessage(role="user", content="Reply with the single word: ok")],
                temperature=0.0,
                max_tokens=8,
            )
            return (bool(result.text.strip()), result.text.strip()[:40] or "empty response")
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return (False, str(exc)[:200])


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(raw: str) -> dict:
    """Best-effort extraction of a JSON object from model output.

    Models wrap JSON in prose or fences even when told not to, and a hard parse
    failure here would break the verifier and router. We degrade to {} instead,
    and callers treat {} as "no verdict" rather than crashing the request.
    """
    if not raw:
        return {}
    text = raw.strip()

    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass

    # Fall back to the outermost {...} span.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            # Trailing commas are the most common malformation; try once more.
            cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(cleaned)
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}
