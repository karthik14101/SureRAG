"""Azure provider adaptation.

A fake client reproduces Azure's real 400 messages, so every negotiation
path runs without an Azure account: reasoning deployments that reject
temperature, opaque names that hide a reasoning model, dated api-versions
that predate max_completion_tokens, content filters, and a model that
burns its whole budget thinking.

Kept as one test because the sections build on each other -- the client
negotiated in one is the client asserted against in the next. Splitting
them would mean rewriting the negotiation, and rewriting is how you lose
what tests like these are protecting. `check` names the exact assertion
that failed."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace as NS


from app.config import settings
from app.core.errors import UpstreamError
from app.llm import azure_provider as AZ
from app.llm.base import ChatMessage

settings.llm_max_retries = 1


def check(name, cond, detail=""):
    """Assert, keeping the readable name the section gave it."""
    assert cond, "{}  {}".format(name, detail)


class BadRequest(Exception):
    """Mimics openai.BadRequestError. with_param=False forces the regex fallback."""

    status_code = 400

    def __init__(self, message, param=None, code=None, with_param=True):
        super().__init__(message)
        if with_param:
            self.param, self.code = param, code
            self.body = {"message": message, "param": param, "code": code}


def ok(text, finish="stop"):
    return NS(
        choices=[NS(finish_reason=finish, message=NS(content=text))],
        usage=NS(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


class FakeAzure:
    def __init__(self, rules):
        self.rules, self.sent = rules, []
        self.chat = NS(completions=NS(create=self._create))

    async def _create(self, **req):
        self.sent.append(dict(req))
        return self.rules(req)


def provider(deployment, rules, reasoning="auto", vision=True, endpoint="https://res.openai.azure.com"):
    settings.azure_openai_endpoint = endpoint
    settings.azure_openai_api_key = "test-key"
    settings.azure_openai_deployment = deployment
    settings.azure_openai_api_version = "v1"
    settings.azure_openai_reasoning = reasoning
    settings.azure_openai_vision = vision
    fake = FakeAzure(rules)
    AZ.AzureOpenAIProvider._build_client = lambda self: fake
    return AZ.AzureOpenAIProvider(), fake


MSG = [ChatMessage(role="user", content="hi")]


async def _run():
    print("\n[1] Endpoint parsing -- every form the Foundry portal shows")
    settings.azure_openai_api_version = "v1"
    cases = [
        ("https://res.openai.azure.com", "https://res.openai.azure.com", ""),
        ("https://res.openai.azure.com/", "https://res.openai.azure.com", ""),
        ("https://res.openai.azure.com/openai/v1/", "https://res.openai.azure.com", ""),
        ("res.openai.azure.com", "https://res.openai.azure.com", ""),
        ("https://res.cognitiveservices.azure.com/", "https://res.cognitiveservices.azure.com", ""),
        ("https://res.services.ai.azure.com/api/projects/proj1", "https://res.services.ai.azure.com", ""),
        (
            "https://res.openai.azure.com/openai/deployments/my-gpt4o/chat/completions?api-version=2025-01-01-preview",
            "https://res.openai.azure.com",
            "my-gpt4o",
        ),
    ]
    for raw, want_host, want_dep in cases:
        settings.azure_openai_endpoint, settings.azure_openai_deployment = raw, ""
        host, dep, ver = settings.azure_target
        check(raw[:64], host == want_host and dep == want_dep and ver == "v1", f"-> {host!r} {dep!r} {ver!r}")
    settings.azure_openai_endpoint = "https://r.openai.azure.com/openai/deployments/from-url/chat/completions"
    settings.azure_openai_deployment = "explicit"
    check("explicit AZURE_OPENAI_DEPLOYMENT beats the one in the URL", settings.azure_target[1] == "explicit")

    print("\n[2] Reasoning deployment recognised by name (gpt-5-mini)")
    p, fake = provider("gpt-5-mini", lambda r: ok("fine"))
    await p.complete("sys", MSG, temperature=0.2, max_tokens=150)
    req = fake.sent[-1]
    check("uses max_completion_tokens, never max_tokens", "max_completion_tokens" in req and "max_tokens" not in req)
    check("temperature omitted", "temperature" not in req)
    check("150-token request raised to the 4000 floor", req["max_completion_tokens"] == 4000, req.get("max_completion_tokens"))
    check("reasoning_effort=low sent", req.get("reasoning_effort") == "low")
    check("routes by deployment name", req["model"] == "gpt-5-mini")
    check("single round trip", len(fake.sent) == 1)

    print("\n[3] Classic deployment (gpt-4o) -- temperature kept, no floor")
    p, fake = provider("gpt-4o", lambda r: ok("fine"))
    await p.complete("sys", MSG, temperature=0.2, max_tokens=150)
    req = fake.sent[-1]
    check("temperature sent", req.get("temperature") == 0.2)
    check("no reasoning_effort", "reasoning_effort" not in req)
    check("budget left at 150", req.get("max_completion_tokens") == 150)
    check("'gpt-4o' not misread as o4", p._caps.reasoning is False)

    print("\n[4] Opaque name hides a reasoning model -> learned from Azure's 400")

    def reasoning_rules(req):
        if "temperature" in req:
            raise BadRequest(
                "Unsupported value: 'temperature' does not support 0.2 with this model. "
                "Only the default (1) value is supported.",
                param="temperature",
                code="unsupported_value",
            )
        return ok("adapted answer")

    p, fake = provider("prod-chat", reasoning_rules)
    res = await p.complete("sys", MSG, temperature=0.2, max_tokens=150)
    check("answer returned after adapting", res.text == "adapted answer", res.text)
    check("exactly one retry", len(fake.sent) == 2, len(fake.sent))
    check("retry dropped temperature", "temperature" not in fake.sent[1])
    check("switched into reasoning mode -> floor applied", fake.sent[1]["max_completion_tokens"] == 4000)
    await p.complete("sys", MSG, temperature=0.2)
    check("capability remembered: next call right first time", len(fake.sent) == 3, len(fake.sent))

    print("\n[5] Error carries no .param -> parsed from the message text")

    def noparam_rules(req):
        if "max_tokens" in req:
            raise BadRequest(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead.",
                with_param=False,
            )
        return ok("x")

    p, fake = provider("classic-name", noparam_rules)
    p._caps.token_param = "max_tokens"
    await p.complete("sys", MSG)
    check("switched to max_completion_tokens", p._caps.token_param == "max_completion_tokens")

    print("\n[6] Pre-2024 dated api-version that predates max_completion_tokens")

    def old_api_rules(req):
        if "max_completion_tokens" in req:
            raise BadRequest("Unrecognized request argument supplied: max_completion_tokens", with_param=False)
        return ok("legacy ok")

    p, fake = provider("gpt-35-turbo", old_api_rules)
    res = await p.complete("sys", MSG)
    check("fell back to max_tokens", p._caps.token_param == "max_tokens" and res.text == "legacy ok")

    print("\n[7] JSON mode and system role both rejected")

    def picky_rules(req):
        if "response_format" in req:
            raise BadRequest(
                "Invalid parameter: 'response_format' of type 'json_object' is not supported with this model.",
                param="response_format",
            )
        if req["messages"][0]["role"] == "system":
            raise BadRequest(
                "Unsupported value: 'messages[0].role' does not support 'system' with this model.",
                param="messages[0].role",
                code="unsupported_value",
            )
        return ok('{"route":"VECTOR"}')

    p, fake = provider("o1-mini", picky_rules)
    data, _ = await p.complete_json("SYSTEM RULES", "classify this")
    check("JSON still parsed from the prompt alone", data == {"route": "VECTOR"}, data)
    last = fake.sent[-1]
    check(
        "system prompt folded into the user turn",
        last["messages"][0]["role"] == "user" and last["messages"][0]["content"].startswith("SYSTEM RULES"),
    )

    print("\n[8] Reasoning model burns its whole budget thinking -> retried with room")
    calls = {"n": 0}

    def empty_then_ok(req):
        calls["n"] += 1
        return ok("", finish="length") if calls["n"] == 1 else ok("final answer")

    p, fake = provider("o3", empty_then_ok)
    res = await p.complete("sys", MSG, max_tokens=150)
    check("did not return an empty answer", res.text == "final answer", repr(res.text))
    check("retry got a bigger budget", fake.sent[1]["max_completion_tokens"] == 8000, fake.sent[1].get("max_completion_tokens"))

    print("\n[9] Content filter -> clear message, not retried")
    settings.llm_max_retries = 3
    p, fake = provider("gpt-4o", lambda r: ok("", finish="content_filter"))
    try:
        await p.complete("sys", MSG)
        check("raised", False)
    except UpstreamError as exc:
        check("UpstreamError carries the filter message", "content filter" in exc.message, exc.message)
        check("not retried despite 3 allowed attempts", len(fake.sent) == 1, len(fake.sent))
    settings.llm_max_retries = 1

    print("\n[10] Streaming: filter-only first chunk + <think> split across chunks")

    async def gen():
        yield NS(choices=[])
        for piece in ["<thi", "nk>plan</th", "ink>The ans", "wer is 42 [1]."]:
            yield NS(choices=[NS(finish_reason=None, delta=NS(content=piece))])
        yield NS(choices=[NS(finish_reason="stop", delta=NS(content=None))])

    p, fake = provider("deepseek-r1", lambda r: gen())
    out = "".join([x async for x in p.stream("sys", MSG)])
    check("empty-choices chunk skipped, reasoning stripped", out == "The answer is 42 [1].", repr(out))

    print("\n[11] Text-only deployment rejects an image -> vision off, once")

    def no_vision(req):
        if isinstance(req["messages"][0]["content"], list):
            raise BadRequest("Invalid content type. image_url is only supported by certain models.")
        return ok("x")

    p, fake = provider("gpt-35-turbo", no_vision)
    first = await p.describe_image(b"\x89PNG", "image/png", "describe")
    check("returns None instead of failing ingestion", first is None)
    check("vision now disabled", p.supports_vision is False)
    await p.describe_image(b"\x89PNG", "image/png", "describe")
    check("second image not even attempted", len(fake.sent) == 1, len(fake.sent))

    print("\n[12] A deployment that rejects everything cannot loop forever")

    def hostile(req):
        name = "max_tokens" if "max_tokens" in req else "max_completion_tokens"
        raise BadRequest(f"Unsupported parameter: '{name}' is not supported.", param=name)

    p, fake = provider("broken", hostile)
    try:
        await p.complete("sys", MSG)
        check("raised", False)
    except UpstreamError as exc:
        check("gave up with a clear error", "kept rejecting" in exc.message, exc.message[:80])
        check(
            f"bounded at {AZ.MAX_NEGOTIATION_ROUNDS} rounds",
            len(fake.sent) == AZ.MAX_NEGOTIATION_ROUNDS,
            len(fake.sent),
        )

    print("\n[13] Factory wiring")
    from app.llm import factory

    settings.llm_provider = "azure"
    settings.azure_openai_endpoint = "https://res.openai.azure.com"
    settings.azure_openai_deployment = "gpt-4o"
    check("provider_is_configured with all three values", factory.provider_is_configured())
    settings.azure_openai_deployment = ""
    check("not configured without a deployment", not factory.provider_is_configured())
    settings.azure_openai_endpoint = (
        "https://res.openai.azure.com/openai/deployments/dep1/chat/completions?api-version=2024-10-21"
    )
    check("configured when the deployment is inside the pasted URL", factory.provider_is_configured())
    factory.reset_llm()
    check("factory builds AzureOpenAIProvider", type(factory.get_llm()).__name__ == "AzureOpenAIProvider")


def test_azure_negotiates_every_deployment_shape():
    asyncio.run(_run())
