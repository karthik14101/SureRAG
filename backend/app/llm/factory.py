"""Provider factory: the single place LLM_PROVIDER is read."""
from __future__ import annotations

import threading

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import LLMProvider
from app.logging_conf import get_logger

logger = get_logger(__name__)

_instance: LLMProvider | None = None
_lock = threading.Lock()


def _build(provider_name: str) -> LLMProvider:
    if provider_name == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    if provider_name == "groq":
        from app.llm.groq_provider import GroqProvider

        return GroqProvider()
    if provider_name == "azure":
        from app.llm.azure_provider import AzureOpenAIProvider

        return AzureOpenAIProvider()
    if provider_name == "ollama":
        from app.llm.ollama_provider import OllamaProvider

        return OllamaProvider()
    if provider_name == "huggingface":
        from app.llm.huggingface_provider import HuggingFaceProvider

        return HuggingFaceProvider()
    raise UpstreamError(
        "Unknown LLM_PROVIDER '{}'. Use 'gemini', 'groq', 'azure', 'ollama' or "
        "'huggingface'.".format(provider_name)
    )


def get_llm() -> LLMProvider:
    """Process-wide singleton. Clients hold connection pools worth reusing."""
    global _instance
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = _build(settings.llm_provider)
            logger.info(
                "LLM provider ready: %s (model=%s, vision=%s)",
                _instance.name,
                settings.active_model,
                _instance.supports_vision,
            )
    return _instance


def reset_llm() -> None:
    """Drop the cached client. Used by tests and after a config change."""
    global _instance
    with _lock:
        _instance = None


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
