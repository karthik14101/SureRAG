"""GroqCloud provider (OpenAI-compatible chat completions).

Model note (2026): `deepseek-r1-distill-llama-70b` was retired, and
`llama-3.3-70b-versatile` is enterprise-tier only. The default in .env is
`openai/gpt-oss-120b`. Groq currently exposes no vision model, so
`describe_image` returns None and image understanding falls back to local OCR.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from groq import AsyncGroq

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMResult, TokenUsage
from app.llm.reasoning import ThinkTagFilter, strip_reasoning
from app.llm.retry import with_retry
from app.logging_conf import get_logger

logger = get_logger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"
    supports_vision = False

    def __init__(self) -> None:
        if not settings.groq_api_key:
            raise UpstreamError(
                "GROQ_API_KEY is empty. Add your GroqCloud key to .env, "
                "or set LLM_PROVIDER=gemini."
            )
        self._client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=float(settings.llm_timeout_seconds),
            max_retries=0,  # retries are handled by app.llm.retry
        )
        self.model = settings.groq_model

    @staticmethod
    def _to_messages(system: str, messages: list[ChatMessage]) -> list[dict]:
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
        for message in messages:
            role = "assistant" if message.role == "assistant" else "user"
            payload.append({"role": role, "content": message.content})
        return payload

    @staticmethod
    def _usage(response) -> TokenUsage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return TokenUsage(calls=1)
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            calls=1,
        )

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        payload = self._to_messages(system, messages)
        kwargs: dict = {
            "model": self.model,
            "messages": payload,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        async def _call() -> LLMResult:
            response = await self._client.chat.completions.create(**kwargs)
            choice = response.choices[0] if response.choices else None
            raw = (choice.message.content if choice and choice.message else "") or ""
            return LLMResult(
                text=strip_reasoning(raw),
                usage=self._usage(response),
                model=self.model,
            )

        return await with_retry(
            _call,
            max_attempts=settings.llm_max_retries,
            provider="Groq",
            model=self.model,
            operation="completion",
        )

    async def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = self._to_messages(system, messages)
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=payload,
                temperature=settings.llm_temperature if temperature is None else temperature,
                max_tokens=max_tokens or settings.llm_max_tokens,
                stream=True,
            )
            # Reasoning tags can straddle chunk boundaries; the filter holds back
            # a short tail until each tag is resolved.
            think = ThinkTagFilter()
            async for chunk in stream:
                if not chunk.choices:
                    continue
                piece = getattr(chunk.choices[0].delta, "content", None)
                visible = think.feed(piece or "")
                if visible:
                    yield visible

            tail = think.flush()
            if tail:
                yield tail
        except Exception as exc:  # noqa: BLE001
            from app.llm.retry import friendly_message

            raise UpstreamError(friendly_message(exc, "Groq", self.model)) from exc

    async def describe_image(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> str | None:
        # No vision model on GroqCloud today; caller falls back to OCR.
        return None
