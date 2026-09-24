"""The fast tier: a cheaper model for the small JSON calls.

Routing, condensing, decomposition and verification are short structured calls
that do not need a reasoning model. On Azure they can run on the SAME
deployment at minimal reasoning effort, which needs no extra quota -- the
constraint that motivated this, since reasoning tokens are billed as output and
a router call that deliberates costs as much as a paragraph of answer.

Leave the fast settings empty and every provider behaves exactly as before.
"""
from __future__ import annotations

from app.config import settings
from app.llm import azure_provider as AZ
from app.llm import factory


def use_azure(deployment="gpt-5.1", fast_deployment=""):
    settings.llm_provider = "azure"
    settings.azure_openai_endpoint = "https://res.openai.azure.com"
    settings.azure_openai_api_key = "k"
    settings.azure_openai_deployment = deployment
    settings.azure_openai_fast_deployment = fast_deployment
    settings.azure_openai_api_version = "v1"
    settings.azure_openai_reasoning = "auto"
    settings.azure_openai_reasoning_effort = "low"
    settings.azure_openai_fast_reasoning_effort = "minimal"
    AZ.AzureOpenAIProvider._build_client = lambda self: object()
    factory.reset_llm()


class Rejected(Exception):
    """Azure's real 400 when a deployment does not accept 'minimal'."""

    status_code = 400
    param = "reasoning_effort"
    body = {"message": "Unsupported value: 'reasoning_effort' does not support 'minimal'"}

    def __str__(self):
        return self.body["message"]


def test_the_fast_tier_runs_on_one_azure_deployment():
    use_azure()
    main_llm, fast_llm = factory.get_llm(), factory.get_fast_llm()

    assert fast_llm.deployment == main_llm.deployment == "gpt-5.1"
    assert main_llm is not fast_llm
    assert fast_llm._effort == "minimal", fast_llm._effort
    assert main_llm._effort == "low", main_llm._effort
    assert AZ.FAST_TOKEN_FLOOR < AZ.REASONING_TOKEN_FLOOR

    fast_request = fast_llm._build_request({"system": "s", "messages": [], "max_tokens": 150})
    main_request = main_llm._build_request({"system": "s", "messages": [], "max_tokens": 150})

    assert fast_request["max_completion_tokens"] == 1500, fast_request
    assert main_request["max_completion_tokens"] == 4000, main_request
    assert fast_request["reasoning_effort"] == "minimal"
    assert main_request["reasoning_effort"] == "low"
    # Small JSON calls never send images, and claiming vision would only invite
    # a request the cheap path cannot serve.
    assert not fast_llm.supports_vision


def test_a_rejected_minimal_effort_steps_down_rather_than_off():
    """The adaptation returned False after stepping the effort down, because
    only the capability flags were compared -- a real bug these tests caught."""
    use_azure()
    fast_llm = factory.get_fast_llm()

    assert fast_llm._adapt(Rejected())
    assert fast_llm._effort == "low", fast_llm._effort
    assert fast_llm._caps.send_reasoning_effort

    # A second rejection gives up on the parameter entirely.
    fast_llm._adapt(Rejected())
    assert not fast_llm._caps.send_reasoning_effort


def test_a_second_azure_deployment_is_used_when_configured():
    use_azure(fast_deployment="gpt-4o-mini")

    assert factory.get_fast_llm().deployment == "gpt-4o-mini"
    assert factory.get_llm().deployment == "gpt-5.1"


def test_an_unconfigured_fast_tier_changes_nothing():
    settings.llm_provider = "groq"
    settings.groq_api_key = "k"
    settings.groq_model = "qwen/qwen3.8-27b"
    settings.groq_fast_model = ""
    factory.reset_llm()

    assert not factory.has_fast_tier()
    # The very same object, so nothing can accidentally cost twice.
    assert factory.get_fast_llm() is factory.get_llm()

    settings.groq_fast_model = "llama-3.1-8b-instant"
    factory.reset_llm()

    assert factory.has_fast_tier()
    assert factory.get_fast_llm().model == "llama-3.1-8b-instant"
    assert factory.get_llm().model == "qwen/qwen3.8-27b"
    assert factory.get_fast_llm() is not factory.get_llm()


def test_every_provider_accepts_the_fast_flag():
    settings.gemini_api_key = "k"
    settings.gemini_fast_model = "gemini-2.5-flash-lite"
    settings.ollama_fast_model = "llama3.2:3b"
    settings.hf_fast_model = "Qwen/Qwen2.5-0.5B-Instruct"

    for provider, attribute, expected in [
        ("gemini", "model", "gemini-2.5-flash-lite"),
        ("ollama", "model", "llama3.2:3b"),
        ("huggingface", "model_id", "Qwen/Qwen2.5-0.5B-Instruct"),
    ]:
        settings.llm_provider = provider
        factory.reset_llm()
        built = factory.get_fast_llm()
        assert getattr(built, attribute) == expected, (provider, getattr(built, attribute))
