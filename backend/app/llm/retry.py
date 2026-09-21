"""Retry with exponential backoff for flaky/rate-limited provider calls.

Free tiers throttle aggressively (Gemini free is roughly 10 requests/minute), so
a 429 is an expected condition, not an error. We back off and retry; only after
exhausting retries does the user see a message.
"""
from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.core.errors import UpstreamError
from app.logging_conf import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Substrings that indicate a transient condition worth retrying.
_RETRYABLE_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "resource exhausted",
    "resource_exhausted",
    "quota",
    "500",
    "502",
    "503",
    "504",
    "overloaded",
    "unavailable",
    "deadline",
    "timeout",
    "timed out",
    "connection reset",
    "connection error",
    "temporarily",
)

# Substrings that are permanent: retrying only wastes the user's time.
_FATAL_MARKERS = (
    "api key not valid",
    "api_key_invalid",
    "invalid api key",
    "permission denied",
    "unauthorized",
    "401",
    "403",
    "not found",
    "404",
    "model_not_found",
    "does not exist",
    # Azure-specific permanent failures.
    "deploymentnotfound",
    "content_filter",
    "content management policy",
    "responsibleaipolicyviolation",
    # OpenAI-format 400s (Azure and Groq both use them). A malformed request
    # fails identically on every attempt, so retrying only adds latency.
    "invalid_request_error",
    "unsupported_parameter",
    "unsupported_value",
    "context_length_exceeded",
)


def _classify(exc: Exception) -> str:
    message = str(exc).lower()
    for marker in _FATAL_MARKERS:
        if marker in message:
            return "fatal"
    for marker in _RETRYABLE_MARKERS:
        if marker in message:
            return "retryable"
    return "unknown"


def friendly_message(exc: Exception, provider: str, model: str) -> str:
    message = str(exc)
    low = message.lower()
    is_azure = provider.lower().startswith("azure")

    if (
        "content_filter" in low
        or "content management policy" in low
        or "responsibleaipolicyviolation" in low
    ):
        return (
            "{}'s content filter blocked this request. Rephrase it, or review the "
            "content filter policy on the deployment.".format(provider)
        )
    if "api key" in low or "401" in low or "unauthorized" in low or "access denied" in low:
        if is_azure:
            # On Azure a 401 is as often a key/endpoint mismatch as a bad key.
            return (
                "Azure rejected the credentials. Check that AZURE_OPENAI_API_KEY "
                "belongs to the same resource as AZURE_OPENAI_ENDPOINT."
            )
        return (
            "The {} API key was rejected. Check the key in your .env file.".format(provider)
        )
    if is_azure and (
        "deploymentnotfound" in low
        or "404" in low
        or ("deployment" in low and ("does not exist" in low or "not found" in low))
    ):
        return (
            "Azure deployment '{}' was not found at this endpoint. "
            "AZURE_OPENAI_DEPLOYMENT must be the deployment name from Foundry's "
            "Deployments tab, which is not always the model name. A deployment "
            "created in the last few minutes may also still be provisioning.".format(model)
        )
    if "not found" in low or "does not exist" in low or "model" in low and "404" in low:
        return (
            "Model '{}' is not available on {}. Update the model id in .env "
            "(see the provider console for the current list).".format(model, provider)
        )
    if "429" in low or "quota" in low or "rate limit" in low or "rate_limit" in low:
        return (
            "{} is rate limiting this key. Wait a moment and try again, or switch "
            "LLM_PROVIDER in .env.".format(provider)
        )
    return "The {} API call failed: {}".format(provider, message[:300])


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    provider: str,
    model: str,
    operation: str = "completion",
    base_delay: float = 1.5,
) -> T:
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise
        except UpstreamError:
            # Raised deliberately with a user-facing message (a content-filter
            # block, a misconfiguration). It is final: retrying cannot help, and
            # re-wrapping it would bury the message inside a generic one.
            raise
        except Exception as exc:  # noqa: BLE001 - classified below
            last_exc = exc
            kind = _classify(exc)

            if kind == "fatal" or attempt >= max_attempts:
                break

            # Full jitter avoids retry stampedes when several chunks fail together.
            delay = min(20.0, base_delay * (2 ** (attempt - 1)))
            delay = random.uniform(delay * 0.5, delay)
            logger.warning(
                "%s %s attempt %d/%d failed (%s); retrying in %.1fs",
                provider,
                operation,
                attempt,
                max_attempts,
                str(exc)[:160],
                delay,
            )
            await asyncio.sleep(delay)

    assert last_exc is not None
    raise UpstreamError(friendly_message(last_exc, provider, model)) from last_exc
