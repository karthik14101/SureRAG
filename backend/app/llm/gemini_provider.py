"""Google AI Studio (Gemini) provider, built on the current `google-genai` SDK.

Note this is NOT the retired `google-generativeai` package. The entrypoint is
`genai.Client(...)`, and the async surface lives under `client.aio`.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMResult, TokenUsage
from app.llm.retry import with_retry
from app.logging_conf import get_logger

logger = get_logger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"
    supports_vision = True

    def __init__(self, fast: bool = False) -> None:
        if not settings.gemini_api_key:
            raise UpstreamError(
                "GEMINI_API_KEY is empty. Add your Google AI Studio key to .env, "
                "or set LLM_PROVIDER=groq."
            )
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self.fast = fast
        self.model = (settings.gemini_fast_model if fast else "") or settings.gemini_model
        self.vision_model = settings.gemini_vision_model

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _to_contents(messages: list[ChatMessage]) -> list[types.Content]:
        """Gemini calls the assistant role "model"."""
        contents: list[types.Content] = []
        for message in messages:
            role = "model" if message.role == "assistant" else "user"
            contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=message.content)])
            )
        return contents

    def _config(
        self,
        system: str,
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
    ) -> types.GenerateContentConfig:
        kwargs: dict = {
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_output_tokens": max_tokens or settings.llm_max_tokens,
        }
        if system:
            kwargs["system_instruction"] = system
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return types.GenerateContentConfig(**kwargs)

    @staticmethod
    def _usage(response) -> TokenUsage:
        meta = getattr(response, "usage_metadata", None)
        if meta is None:
            return TokenUsage(calls=1)
        return TokenUsage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            total_tokens=getattr(meta, "total_token_count", 0) or 0,
            calls=1,
        )

    @staticmethod
    def _text_of(response) -> str:
        text = getattr(response, "text", None)
        if text:
            return text
        # Safety blocks and tool-only turns leave .text empty; dig into parts.
        collected: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                part_text = getattr(part, "text", None)
                if part_text:
                    collected.append(part_text)
        return "".join(collected)

    # ---- interface ----------------------------------------------------------
    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        contents = self._to_contents(messages)
        config = self._config(system, temperature, max_tokens, json_mode)

        async def _call() -> LLMResult:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            return LLMResult(
                text=self._text_of(response),
                usage=self._usage(response),
                model=self.model,
            )

        return await with_retry(
            _call,
            max_attempts=settings.llm_max_retries,
            provider="Gemini",
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
        contents = self._to_contents(messages)
        config = self._config(system, temperature, max_tokens, False)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            )
            async for chunk in stream:
                piece = self._text_of(chunk)
                if piece:
                    yield piece
        except Exception as exc:  # noqa: BLE001
            from app.llm.retry import friendly_message

            raise UpstreamError(friendly_message(exc, "Gemini", self.model)) from exc

    async def describe_image(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> str | None:
        config = types.GenerateContentConfig(temperature=0.1, max_output_tokens=400)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    types.Part.from_text(text=prompt),
                ],
            )
        ]

        async def _call() -> str:
            response = await self._client.aio.models.generate_content(
                model=self.vision_model, contents=contents, config=config
            )
            return self._text_of(response).strip()

        try:
            return await with_retry(
                _call,
                max_attempts=2,
                provider="Gemini",
                model=self.vision_model,
                operation="vision",
            )
        except Exception as exc:  # noqa: BLE001
            # A failed caption must never fail an ingestion run.
            logger.warning("Gemini vision captioning failed: %s", str(exc)[:200])
            return None
