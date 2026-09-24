"""Provider factory: the single place LLM_PROVIDER is read.

Two tiers, one provider. Every turn makes a handful of small JSON calls --
condensing a follow-up, picking a route, decomposing a question, judging
evidence -- and one call that writes the answer. Those are not the same job: the
first four need obedience and speed, the last needs the best model available.

The fast tier is therefore the same provider with a cheaper configuration: a
smaller model where one is named, and on Azure the same deployment told to spend
minimal effort reasoning. Configure nothing and both tiers are the same object,
so the default behaviour is exactly what it was before tiers existed.
"""
from __future__ import annotations

import threading

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import LLMProvider
from app.logging_conf import get_logger

logger = get_logger(__name__)

_instances: dict[bool, LLMProvider] = {}
_lock = threading.Lock()


def _build(provider_name: str, fast: bool) -> LLMProvider:
    if provider_name == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider(fast=fast)
    if provider_name == "groq":
        from app.llm.groq_provider import GroqProvider

        return GroqProvider(fast=fast)
    if provider_name == "azure":
        from app.llm.azure_provider import AzureOpenAIProvider

        return AzureOpenAIProvider(fast=fast)
    if provider_name == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider(fast=fast)
    if provider_name == "huggingface":
        from app.llm.huggingface_provider import HuggingFaceProvider

        return HuggingFaceProvider(fast=fast)
    raise UpstreamError(
        "Unknown LLM_PROVIDER '{}'. Use 'gemini', 'groq', 'azure', 'ollama' or "
        "'huggingface'.".format(provider_name)
    )


def has_fast_tier() -> bool:
    """Whether the fast tier is configured to differ from the main model.

    Azure is the exception: even with one deployment the fast tier is distinct,
    because it asks for minimal reasoning effort, and on Azure reasoning tokens
    are billed and rate-limited as output.
    """
    provider = settings.llm_provider
    if provider == "azure":
        return True
    return bool(
        {
            "gemini": settings.gemini_fast_model,
            "groq": settings.groq_fast_model,
            "ollama": settings.ollama_fast_model,
            "huggingface": settings.hf_fast_model,
        }.get(provider, "").strip()
    )


def get_llm(tier: str = "main") -> LLMProvider:
    """Process-wide singleton per tier. Clients hold pools worth reusing."""
    fast = tier == "fast" and has_fast_tier()

    instance = _instances.get(fast)
    if instance is not None:
        return instance

    with _lock:
        if fast not in _instances:
            built = _build(settings.llm_provider, fast)
            _instances[fast] = built
            logger.info(
                "LLM provider ready [%s]: %s (model=%s, vision=%s)",
                "fast" if fast else "main",
                built.name,
                getattr(built, "model", settings.active_model),
                built.supports_vision,
            )
    return _instances[fast]


def get_fast_llm() -> LLMProvider:
    """The tier for routing, condensing, decomposition and verification."""
    return get_llm(tier="fast")


def reset_llm() -> None:
    """Drop the cached clients. Used by tests and after a config change."""
    with _lock:
        _instances.clear()


def provider_is_configured() -> bool:
    if settings.llm_provider == "azure":
        # Azure needs three values, not just a key.
        host, deployment, _ = settings.azure_target
        return bool(settings.azure_openai_api_key and host and deployment)
    if settings.llm_provider == "ollama":
        return bool(settings.ollama_model.strip())
    if settings.llm_provider == "huggingface":
        return bool(settings.hf_model.strip())
    return bool(settings.active_api_key())
