"""Ollama provider: any model served by a local (or remote) Ollama server.

Talks to Ollama's native REST API (/api/chat) over httpx, so no extra package is
needed. The native API is used rather than Ollama's OpenAI-compatible endpoint
because only the native one accepts `options.num_ctx` -- and Ollama's default
context window (2-4k tokens) would otherwise silently cut a RAG prompt in half.

Things handled here that a naive port gets wrong:
  * Connection refused means "Ollama is not running", and a 404 means "model not
    pulled". Both become a one-line fix-it message, not a stack trace.
  * Local generation is slow, so a timeout is not retried: waiting three times as
    long for the same too-slow model helps nobody.
  * Thinking models (qwen3, deepseek-r1) get `think: false` by default. Older
    Ollama builds and some models reject the field, in which case it is dropped
    once and never sent again.
  * Vision is detected from the model's capabilities (`/api/show`), because a
    text-only model given an image does not error -- it just makes things up.
"""
from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator

import httpx

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMResult, TokenUsage
from app.llm.reasoning import ThinkTagFilter, strip_reasoning
from app.llm.retry import friendly_message, with_retry
from app.logging_conf import get_logger

logger = get_logger(__name__)

PROVIDER_LABEL = "Ollama"


class _ThinkRejected(Exception):
    """The server or model does not accept the `think` field."""


def _parse_think(raw: str) -> bool | str | None:
    value = (raw or "").strip().lower()
    if value in ("", "auto", "default"):
        return None
    if value in ("true", "on", "yes", "1"):
        return True
    if value in ("false", "off", "no", "0"):
        return False
    if value in ("low", "medium", "high"):
        return value  # gpt-oss takes an effort level instead of a boolean
    logger.warning("Ignoring unrecognised OLLAMA_THINK=%r", raw)
    return None


def _keep_alive(raw: str) -> str | int:
    value = (raw or "").strip() or "30m"
    # Ollama parses strings as Go durations, which require a unit; a bare
    # number such as -1 must be sent as a number.
    return int(value) if value.lstrip("-").isdigit() else value


def _error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])
    except ValueError:
        pass
    return (response.text or "").strip()[:300] or "HTTP {}".format(response.status_code)


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self) -> None:
        self.model = (settings.ollama_model or "").strip()
        if not self.model:
            raise UpstreamError(
                "OLLAMA_MODEL is empty. Pull a model and name it in .env, e.g. "
                "`ollama pull qwen2.5:7b` then OLLAMA_MODEL=qwen2.5:7b"
            )
        self.base_url = settings.ollama_url
        self.vision_model = (settings.ollama_vision_model or "").strip() or self.model

        headers = {}
        if settings.ollama_api_key:
            headers["Authorization"] = "Bearer " + settings.ollama_api_key
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            # Connect fails fast (server down); reading waits for slow generation.
            timeout=httpx.Timeout(float(settings.local_llm_timeout_seconds), connect=5.0),
        )

        mode = settings.ollama_vision
        self._vision: bool | None = None if mode == "auto" else mode == "true"
        self._think = _parse_think(settings.ollama_think)
        self._send_think = self._think is not None
        self._keep_alive = _keep_alive(settings.ollama_keep_alive)

        logger.info(
            "Ollama: url=%s model=%s vision_model=%s num_ctx=%s think=%s",
            self.base_url,
            self.model,
            self.vision_model,
            settings.ollama_num_ctx,
            self._think,
        )

    # ------------------------------------------------------------------ vision
    @property
    def supports_vision(self) -> bool:  # type: ignore[override]
        # Unknown (auto, not yet probed) counts as "maybe": describe_image
        # probes on first use and answers None if the model cannot see.
        return self._vision is not False

    async def _detect_vision(self, model: str) -> bool:
        try:
            response = await self._client.post("/api/show", json={"model": model, "name": model})
            if response.status_code != 200:
                return False
            info = response.json()
        except Exception:  # noqa: BLE001 - detection is best-effort
            return False

        capabilities = info.get("capabilities")
        if isinstance(capabilities, list):
            return "vision" in capabilities
        # Ollama builds before the capabilities field: look for a vision
        # projector in the model's families instead.
        families = (info.get("details") or {}).get("families") or []
        return bool(info.get("projector_info")) or any(f in ("clip", "mllama") for f in families)

    # ---------------------------------------------------------------- messages
    def _options(self, temperature: float | None, max_tokens: int | None) -> dict:
        options: dict = {
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "num_predict": max_tokens or settings.llm_max_tokens,
        }
        if settings.ollama_num_ctx > 0:
            options["num_ctx"] = settings.ollama_num_ctx
        return options

    def _body(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        json_mode: bool = False,
    ) -> dict:
        payload: list[dict] = []
        if system:
            payload.append({"role": "system", "content": system})
        for message in messages:
            role = "assistant" if message.role == "assistant" else "user"
            payload.append({"role": role, "content": message.content})

        body: dict = {
            "model": self.model,
            "messages": payload,
            "stream": stream,
            "options": self._options(temperature, max_tokens),
            "keep_alive": self._keep_alive,
        }
        if json_mode:
            body["format"] = "json"
        if self._send_think:
            body["think"] = self._think
        return body

    @staticmethod
    def _usage(data: dict) -> TokenUsage:
        prompt = int(data.get("prompt_eval_count") or 0)
        completion = int(data.get("eval_count") or 0)
        return TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
            calls=1,
        )

    # ------------------------------------------------------------------ errors
    def _not_running(self) -> str:
        return (
            "Cannot reach Ollama at {}. Start it (open the Ollama app, or run "
            "`ollama serve`), or fix OLLAMA_BASE_URL in .env.".format(self.base_url)
        )

    def _too_slow(self) -> str:
        return (
            "Ollama model '{}' did not answer within {}s. Use a smaller model "
            "(e.g. llama3.2:3b), lower OLLAMA_NUM_CTX, or raise "
            "LOCAL_LLM_TIMEOUT_SECONDS in .env.".format(
                self.model, settings.local_llm_timeout_seconds
            )
        )

    def _raise_for(self, status: int, message: str, model: str, body: dict) -> None:
        low = message.lower()
        if status == 400 and "think" in low and "think" in body:
            raise _ThinkRejected(message)
        if status == 404 or ("not found" in low and "model" in low) or "pull" in low:
            raise UpstreamError(
                "Ollama model '{}' is not installed. Run:  ollama pull {}".format(model, model)
            )
        if "memory" in low:
            raise UpstreamError(
                "Ollama could not load '{}': {}. Use a smaller or more quantised "
                "model, or lower OLLAMA_NUM_CTX.".format(model, message[:200])
            )
        if status in (401, 403):
            raise UpstreamError(
                "Ollama at {} rejected the request ({}). For a protected remote "
                "server, set OLLAMA_API_KEY.".format(self.base_url, status)
            )
        if status >= 500:
            # Possibly transient (model still loading, server busy): with_retry
            # sees the status code in the message and backs off.
            raise RuntimeError("Ollama {}: {}".format(status, message))
        raise UpstreamError("Ollama rejected the request: {}".format(message[:300]))

    def _drop_think(self, body: dict, reason: str) -> None:
        self._send_think = False
        body.pop("think", None)
        logger.info(
            "Ollama model '%s' does not accept the think option (%s); no longer sending it.",
            body.get("model"),
            reason[:120],
        )

    async def _post_chat(self, body: dict) -> dict:
        for _ in range(2):
            try:
                response = await self._client.post("/api/chat", json=body)
            except httpx.ConnectError as exc:
                raise UpstreamError(self._not_running()) from exc
            except httpx.TimeoutException as exc:
                raise UpstreamError(self._too_slow()) from exc

            if response.status_code < 400:
                return response.json()
            try:
                self._raise_for(response.status_code, _error_text(response), body["model"], body)
            except _ThinkRejected as exc:
                self._drop_think(body, str(exc))
        raise UpstreamError("Ollama kept rejecting the request.")

    # -------------------------------------------------------------- interface
    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResult:
        async def _call() -> LLMResult:
            body = self._body(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                json_mode=json_mode,
            )
            data = await self._post_chat(body)
            raw = ((data.get("message") or {}).get("content")) or ""
            text = strip_reasoning(raw)
            if not text and data.get("done_reason") == "length":
                logger.warning(
                    "Ollama model '%s' hit num_predict before answering; consider "
                    "OLLAMA_THINK=false or a larger LLM_MAX_TOKENS.",
                    self.model,
                )
            return LLMResult(text=text, usage=self._usage(data), model=self.model)

        return await with_retry(
            _call,
            max_attempts=settings.llm_max_retries,
            provider=PROVIDER_LABEL,
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
        body = self._body(
            system, messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        think = ThinkTagFilter()
        try:
            for _ in range(2):
                try:
                    async with self._client.stream("POST", "/api/chat", json=body) as response:
                        if response.status_code >= 400:
                            await response.aread()
                            self._raise_for(
                                response.status_code, _error_text(response), self.model, body
                            )
                        # Ollama streams newline-delimited JSON objects.
                        async for line in response.aiter_lines():
                            if not line.strip():
                                continue
                            chunk = json.loads(line)
                            if chunk.get("error"):
                                raise UpstreamError(
                                    "Ollama failed mid-answer: {}".format(chunk["error"])
                                )
                            piece = (chunk.get("message") or {}).get("content") or ""
                            visible = think.feed(piece)
                            if visible:
                                yield visible
                            if chunk.get("done"):
                                break
                    break
                except _ThinkRejected as exc:
                    # Rejected before any text was produced, so retrying is safe.
                    self._drop_think(body, str(exc))

            tail = think.flush()
            if tail:
                yield tail
        except UpstreamError:
            raise
        except httpx.ConnectError as exc:
            raise UpstreamError(self._not_running()) from exc
        except httpx.TimeoutException as exc:
            raise UpstreamError(self._too_slow()) from exc
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(friendly_message(exc, PROVIDER_LABEL, self.model)) from exc

    async def describe_image(
        self, image_bytes: bytes, mime_type: str, prompt: str
    ) -> str | None:
        if self._vision is False:
            return None
        if self._vision is None:
            self._vision = await self._detect_vision(self.vision_model)
            if not self._vision:
                logger.info(
                    "Ollama model '%s' has no vision capability; images fall back to "
                    "OCR. Set OLLAMA_VISION_MODEL (e.g. gemma3:4b) to caption images.",
                    self.vision_model,
                )
                return None

        body = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(image_bytes).decode("ascii")],
                }
            ],
            "stream": False,
            "options": self._options(0.0, 512),
            "keep_alive": self._keep_alive,
        }
        try:
            data = await self._post_chat(body)
        except UpstreamError as exc:
            low = exc.message.lower()
            if "image" in low or "vision" in low or "multimodal" in low:
                self._vision = False
                logger.warning(
                    "Ollama model '%s' does not accept images; captioning disabled.",
                    self.vision_model,
                )
            else:
                logger.warning("Ollama image captioning failed: %s", exc.message[:200])
            return None
        except Exception as exc:  # noqa: BLE001 - captioning is best-effort
            logger.warning("Ollama image captioning failed: %s", str(exc)[:200])
            return None

        text = strip_reasoning(((data.get("message") or {}).get("content")) or "")
        return text or None

    # --------------------------------------------------------------- lifecycle
    async def _installed_models(self) -> set[str]:
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        names: set[str] = set()
        for entry in response.json().get("models") or []:
            for key in ("name", "model"):
                if entry.get(key):
                    names.add(entry[key])
        return names

    @staticmethod
    def _is_installed(model: str, installed: set[str]) -> bool:
        # "llama3.1" and "llama3.1:latest" are the same model to Ollama.
        return model in installed or (":" not in model and model + ":latest" in installed)

    async def warmup(self) -> None:
        try:
            installed = await self._installed_models()
        except httpx.ConnectError:
            logger.warning(self._not_running())
            return
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama warmup skipped: %s", str(exc)[:200])
            return

        if not self._is_installed(self.model, installed):
            logger.warning(
                "Ollama model '%s' is not installed. Run:  ollama pull %s",
                self.model,
                self.model,
            )
            return
        if self._vision is None:
            self._vision = await self._detect_vision(self.vision_model)

        # An empty message list makes Ollama load the model into memory and
        # return immediately, so the first question skips the load time.
        try:
            await self._client.post(
                "/api/chat",
                json={"model": self.model, "messages": [], "keep_alive": self._keep_alive},
            )
            logger.info("Ollama model '%s' loaded (vision=%s)", self.model, self._vision)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama could not preload '%s': %s", self.model, str(exc)[:200])

    async def health(self) -> tuple[bool, str]:
        try:
            installed = await self._installed_models()
        except httpx.ConnectError:
            return False, self._not_running()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return False, str(exc)[:200]
        if not self._is_installed(self.model, installed):
            return False, "model not installed -- run: ollama pull {}".format(self.model)
        return await super().health()
