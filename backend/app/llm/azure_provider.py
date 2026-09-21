"""Azure OpenAI / Azure AI Foundry provider.

Azure differs from calling OpenAI directly in ways that break a naive port:

  * Requests route by DEPLOYMENT name, which is chosen in Foundry and is often
    not the model name.
  * There are two API surfaces. The v1 API (GA August 2025) takes the standard
    OpenAI client with base_url=<host>/openai/v1/ and no api-version; the legacy
    surface needs AzureOpenAI plus a dated api-version. v1 is the default.
  * Reasoning deployments (GPT-5, o1/o3/o4) reject `max_tokens` and
    `temperature` with a 400, and even the GPT-4o family has moved to
    `max_completion_tokens`. They also spend output budget on hidden reasoning,
    so a small budget can return an EMPTY answer rather than a short one.
  * Streams open with a chunk carrying only prompt-filter results and no
    choices, and the content filter can end a response mid-way.

Because the deployment behind the endpoint is unknown until it is called, this
provider does not hard-code any of those rules. It starts from a sensible guess,
and when Azure answers 400 "unsupported parameter", it learns which rule applies,
adjusts, and retries -- once per rule, remembered for the process lifetime.
"""
from __future__ import annotations

import base64
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMResult, TokenUsage
from app.llm.reasoning import ThinkTagFilter, strip_reasoning
from app.llm.retry import friendly_message, with_retry
from app.logging_conf import get_logger

logger = get_logger(__name__)

PROVIDER_LABEL = "Azure OpenAI"

# Reasoning models bill hidden "thinking" tokens against the same output budget.
# Every call gets at least this much room, or short tasks (routing, titles, a
# health probe asking for 8 tokens) come back empty. It is a ceiling, not spend:
# with reasoning_effort=low, actual usage stays small.
REASONING_TOKEN_FLOOR = 4000

# A deployment name that mentions a reasoning family. Only a starting guess --
# names are user-chosen, so the API's own 400s are the real authority.
_REASONING_NAME_RE = re.compile(r"(^|[-_/.\s])(o1|o3|o4|gpt-?5)", re.IGNORECASE)

# "Unsupported parameter: 'max_tokens' ..." / "Unsupported value: 'temperature' ..."
_PARAM_IN_MESSAGE_RE = re.compile(r"(?:parameter|value)[:\s]+'([A-Za-z0-9_.\[\]]+)'")

# Bounds the learn-and-retry loop so a broken deployment cannot spin forever.
MAX_NEGOTIATION_ROUNDS = 5

CONTENT_FILTER_MESSAGE = (
    "Azure's content filter blocked this response. Rephrase the question, or "
    "review the content filter policy attached to the deployment in Foundry."
)


@dataclass
class _Capabilities:
    """What this deployment accepts. Starts as a guess and is corrected by the API."""

    reasoning: bool
    token_param: str = "max_completion_tokens"
    supports_temperature: bool = True
    send_reasoning_effort: bool = True
    supports_json_mode: bool = True
    supports_system_role: bool = True


def _error_details(exc: Exception) -> tuple[str, str, str]:
    """(code, param, message) from an SDK error, read defensively.

    The SDK documents status_code and response on its errors, but not `code` or
    `param`. Try the attributes, then the parsed body, then the message text, so
    this keeps working whichever of them a given SDK version exposes.
    """
    code = getattr(exc, "code", None)
    param = getattr(exc, "param", None)
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            code = code or error.get("code")
            param = param or error.get("param")

    message = str(exc)
    if not param:
        match = _PARAM_IN_MESSAGE_RE.search(message)
        if match:
            param = match.group(1)

    return str(code or "").lower(), str(param or ""), message.lower()


def _is_bad_request(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 400


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


class AzureOpenAIProvider(LLMProvider):
    name = "azure"

    def __init__(self) -> None:
        host, deployment, api_version = settings.azure_target

        if not settings.azure_openai_api_key:
            raise UpstreamError(
                "AZURE_OPENAI_API_KEY is empty. Copy a key from your resource in "
                "the Azure portal (Keys and Endpoint) into .env."
            )
        if not host:
            raise UpstreamError(
                "AZURE_OPENAI_ENDPOINT is empty. Paste the endpoint from Foundry, "
                "e.g. https://your-resource.openai.azure.com"
            )
        if not deployment:
            raise UpstreamError(
                "AZURE_OPENAI_DEPLOYMENT is empty. Set it to the deployment name "
                "shown in Foundry's Deployments tab -- this is the name you chose, "
                "which is not always the model name."
            )

        self.host = host
        self.deployment = deployment
        self.api_version = api_version
        self.model = deployment

        mode = settings.azure_openai_reasoning
        if mode == "true":
            reasoning = True
        elif mode == "false":
            reasoning = False
        else:
            reasoning = bool(_REASONING_NAME_RE.search(deployment))
        self._reasoning_mode = mode
        self._caps = _Capabilities(reasoning=reasoning, supports_temperature=not reasoning)

        self._vision = bool(settings.azure_openai_vision)
        effort = (settings.azure_openai_reasoning_effort or "").strip().lower()
        self._effort = effort or None

        self._client = self._build_client()

        logger.info(
            "Azure OpenAI: host=%s deployment=%s api=%s reasoning=%s (%s)",
            self.host,
            self.deployment,
            "v1" if self._uses_v1 else self.api_version,
            reasoning,
            "configured" if mode != "auto" else "inferred from name, will adapt",
        )

    # ---- setup --------------------------------------------------------------
    @property
    def _uses_v1(self) -> bool:
        return self.api_version.lower() in ("v1", "latest", "")

    @property
    def supports_vision(self) -> bool:  # type: ignore[override]
        return self._vision

    def _build_client(self):
        common = {
            "api_key": settings.azure_openai_api_key,
            "timeout": float(settings.llm_timeout_seconds),
            "max_retries": 0,  # retries are handled by app.llm.retry
        }
        if self._uses_v1:
            from openai import AsyncOpenAI

            return AsyncOpenAI(base_url="{}/openai/v1/".format(self.host), **common)

        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise UpstreamError(
                "The installed openai package has no AsyncAzureOpenAI client. Set "
                "AZURE_OPENAI_API_VERSION=v1 in .env to use the v1 API instead."
            ) from exc
        return AsyncAzureOpenAI(
            azure_endpoint=self.host, api_version=self.api_version, **common
        )

    # ---- request construction -----------------------------------------------
    def _messages(self, system: str, messages: list[ChatMessage]) -> list[dict]:
        turns = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in messages
        ]
        if not system:
            return turns
        if self._caps.supports_system_role:
            return [{"role": "system", "content": system}] + turns

        # Older reasoning models reject the system role: fold it into the first
        # user turn so the instructions still arrive.
        for turn in turns:
            if turn["role"] == "user":
                turn["content"] = "{}\n\n{}".format(system, turn["content"])
                return turns
        return [{"role": "user", "content": system}] + turns

    def _build_request(self, spec: dict) -> dict:
        """Translate a provider-neutral spec into what this deployment accepts.

        Rebuilt on every attempt, so a capability learned from a 400 takes
        effect on the immediate retry.
        """
        if spec.get("raw_messages") is not None:
            messages = spec["raw_messages"]
        else:
            messages = self._messages(spec.get("system", ""), spec.get("messages", []))

        request: dict = {"model": self.deployment, "messages": messages}

        tokens = int(spec.get("max_tokens") or settings.llm_max_tokens)
        if self._caps.reasoning:
            tokens = max(tokens, REASONING_TOKEN_FLOOR)
            if self._caps.send_reasoning_effort and self._effort:
                request["reasoning_effort"] = self._effort
        request[self._caps.token_param] = tokens

        temperature = spec.get("temperature")
        if temperature is not None and self._caps.supports_temperature:
            request["temperature"] = temperature

        if spec.get("json_mode") and self._caps.supports_json_mode:
            request["response_format"] = {"type": "json_object"}

        if spec.get("stream"):
            request["stream"] = True

        return request

    def _adapt(self, exc: Exception) -> bool:
        """Learn from a 400. Returns True if something changed and a retry may help."""
        code, param, message = _error_details(exc)
        caps = self._caps
        before = repr(caps)
        unsupported = (
            "unsupported" in code
            or "unsupported" in message
            or "not supported" in message
            or "does not support" in message
        )

        if param == "max_tokens" or ("'max_tokens'" in message and unsupported):
            if caps.token_param == "max_tokens":
                caps.token_param = "max_completion_tokens"
        elif param == "max_completion_tokens" or (
            "max_completion_tokens" in message and ("unrecognized" in message or unsupported)
        ):
            # A pre-2024 dated api-version that predates max_completion_tokens.
            if caps.token_param == "max_completion_tokens":
                caps.token_param = "max_tokens"
        elif param == "temperature" or ("temperature" in message and unsupported):
            if caps.supports_temperature:
                caps.supports_temperature = False
                # Rejecting sampling controls is the signature of a reasoning
                # model: switch on the token floor so answers do not come back
                # empty after the model spends its budget thinking.
                if self._reasoning_mode == "auto":
                    caps.reasoning = True
        elif param == "reasoning_effort" or "reasoning_effort" in message:
            caps.send_reasoning_effort = False
        elif param == "response_format" or "response_format" in message or "json_object" in message:
            # JSON is still requested in the prompt; parse_json_object copes
            # with fences and prose around it.
            caps.supports_json_mode = False
        elif "role" in param or ("'system'" in message and unsupported):
            caps.supports_system_role = False

        changed = repr(caps) != before
        if changed:
            logger.info(
                "Azure deployment '%s' rejected %s; adapted and retrying. Now: %s",
                self.deployment,
                "'{}'".format(param) if param else "a request parameter",
                caps,
            )
        return changed

    async def _create(self, spec: dict):
        """Call the API, adapting to 400 'unsupported parameter' replies."""
        last_exc: Exception | None = None
        for _ in range(MAX_NEGOTIATION_ROUNDS):
            request = self._build_request(spec)
            try:
                return await self._client.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001 - classified below
                if not _is_bad_request(exc) or not self._adapt(exc):
                    raise
                last_exc = exc
        raise UpstreamError(
            "Azure deployment '{}' kept rejecting the request after adapting its "
            "parameters. Set AZURE_OPENAI_REASONING=true or false explicitly in .env. "
            "Last error: {}".format(self.deployment, str(last_exc)[:200])
        )

    @staticmethod
    def _extract(response) -> tuple[str, str | None]:
        """(text, finish_reason) from a completion, surfacing content filtering."""
        choices = getattr(response, "choices", None) or []
        if not choices:
            return "", None
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        if finish == "content_filter":
            raise UpstreamError(CONTENT_FILTER_MESSAGE)
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content or "", finish

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
        spec = {
            "system": system,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "json_mode": json_mode,
        }

        async def _call() -> LLMResult:
            response = await self._create(spec)
            text, finish = self._extract(response)
            usage = _usage(response)

            # A reasoning model can exhaust its budget before writing a word.
            # Retry once with far more room rather than hand back nothing.
            if not text.strip() and finish == "length" and self._caps.reasoning:
                logger.info(
                    "Azure deployment '%s' used its whole token budget reasoning; "
                    "retrying once with more room",
                    self.deployment,
                )
                roomier = dict(
                    spec, max_tokens=max(int(spec["max_tokens"]) * 4, REASONING_TOKEN_FLOOR * 2)
                )
                response = await self._create(roomier)
                text, finish = self._extract(response)
                usage.add(_usage(response))

            return LLMResult(text=strip_reasoning(text), usage=usage, model=self.deployment)

        return await with_retry(
            _call,
            max_attempts=settings.llm_max_retries,
            provider=PROVIDER_LABEL,
            model=self.deployment,
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
        spec = {
            "system": system,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
        }
        try:
            stream = await self._create(spec)
            think = ThinkTagFilter()
            async for chunk in stream:
                # Azure's first chunk carries only prompt_filter_results and no
                # choices at all; indexing it blindly raises IndexError.
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    continue
                choice = choices[0]
                if getattr(choice, "finish_reason", None) == "content_filter":
                    raise UpstreamError(CONTENT_FILTER_MESSAGE)
                delta = getattr(choice, "delta", None)
                piece = getattr(delta, "content", None) if delta is not None else None
                visible = think.feed(piece or "")
                if visible:
                    yield visible

            tail = think.flush()
            if tail:
                yield tail
        except UpstreamError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(friendly_message(exc, PROVIDER_LABEL, self.deployment)) from exc

    async def describe_image(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> str | None:
        if not self._vision:
            return None

        data_url = "data:{};base64,{}".format(
            mime_type or "image/png", base64.b64encode(image_bytes).decode("ascii")
        )
        spec = {
            "raw_messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 400,
        }

        try:
            response = await self._create(spec)
            text, _ = self._extract(response)
            return strip_reasoning(text) or None
        except Exception as exc:  # noqa: BLE001 - a caption must never fail ingestion
            if _is_bad_request(exc) and "image" in str(exc).lower():
                # A text-only deployment. Stop trying for the rest of the session
                # rather than failing on every image in the upload.
                self._vision = False
                logger.warning(
                    "Azure deployment '%s' does not accept images; image captioning "
                    "is off for this session (OCR still applies). Set "
                    "AZURE_OPENAI_VISION=false to skip this probe.",
                    self.deployment,
                )
            else:
                logger.warning("Azure vision captioning failed: %s", str(exc)[:200])
            return None
