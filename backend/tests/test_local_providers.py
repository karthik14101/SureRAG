"""The Ollama and HuggingFace providers -- no server, no model download.

Covers the request shape, thinking tags split across streaming chunks,
the think parameter being rejected and remembered, vision detection and
warmup. One test for the same reason as the Azure suite: the sections
share the provider they negotiated."""
from __future__ import annotations

import asyncio
import json
import sys

import httpx


from app.config import Settings, settings
from app.core.errors import UpstreamError
from app.llm import ollama_provider as OP
from app.llm.base import ChatMessage
from app.llm.huggingface_provider import _merge_turns

settings.llm_max_retries = 1


def check(name, cond, detail=""):
    """Assert, keeping the readable name the section gave it."""
    assert cond, "{}  {}".format(name, detail)


MSG = [ChatMessage(role="user", content="hi")]


def ollama(handler, **cfg):
    settings.ollama_model = cfg.get("model", "qwen2.5:7b")
    settings.ollama_vision = cfg.get("vision", "auto")
    settings.ollama_think = cfg.get("think", "false")
    settings.ollama_vision_model = cfg.get("vision_model", "")
    p = OP.OllamaProvider()
    p._client = httpx.AsyncClient(base_url=p.base_url, transport=httpx.MockTransport(handler))
    return p


async def _run():
    print("\n[1] Config")
    for raw, want in [
        ("localhost:11434", "http://localhost:11434"),
        ("http://0.0.0.0:11434/", "http://127.0.0.1:11434"),
        ("http://127.0.0.1:11434/api", "http://127.0.0.1:11434"),
        ("http://box:11434/v1/", "http://box:11434"),
    ]:
        settings.ollama_base_url = raw
        check(f"ollama_url {raw}", settings.ollama_url == want, settings.ollama_url)
    settings.ollama_base_url = "http://127.0.0.1:11434"
    s = Settings(llm_provider="HF", agent_timeout_seconds=45)
    check("'HF' normalised to huggingface", s.llm_provider == "huggingface")
    check("agent timeout raised for local", s.agent_timeout_seconds == s.local_agent_timeout_seconds)
    s = Settings(llm_provider="groq", agent_timeout_seconds=45)
    check("cloud timeout untouched", s.agent_timeout_seconds == 45)
    check("keep_alive -1 sent as int", OP._keep_alive("-1") == -1 and OP._keep_alive("30m") == "30m")

    print("\n[2] Ollama completion request shape")
    sent = []

    def ok(req):
        body = json.loads(req.content)
        sent.append(body)
        return httpx.Response(200, json={"message": {"content": "<think>x</think>Hello [1]"},
                                         "done": True, "prompt_eval_count": 7, "eval_count": 3})

    p = ollama(ok)
    r = await p.complete("SYS", MSG, temperature=0.1, max_tokens=99, json_mode=True)
    b = sent[-1]
    check("reasoning stripped", r.text == "Hello [1]", r.text)
    check("usage mapped", r.usage.total_tokens == 10)
    check("num_ctx sent", b["options"]["num_ctx"] == 8192)
    check("num_predict + temperature", b["options"]["num_predict"] == 99 and b["options"]["temperature"] == 0.1)
    check("json format", b["format"] == "json")
    check("think false sent", b["think"] is False)
    check("system first", b["messages"][0] == {"role": "system", "content": "SYS"})

    print("\n[3] think rejected -> dropped once, remembered")
    sent.clear()

    def no_think(req):
        body = json.loads(req.content)
        sent.append(body)
        if "think" in body:
            return httpx.Response(400, json={"error": '"llama3" does not support thinking'})
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True})

    p = ollama(no_think, think="true")
    r = await p.complete("s", MSG)
    check("answer after dropping think", r.text == "ok" and len(sent) == 2)
    await p.complete("s", MSG)
    check("not sent again", len(sent) == 3 and "think" not in sent[-1])

    print("\n[4] Errors")
    p = ollama(lambda req: httpx.Response(404, json={"error": 'model "foo" not found, try pulling it first'}), model="foo")
    try:
        await p.complete("s", MSG)
        check("raised", False)
    except UpstreamError as e:
        check("404 -> ollama pull hint", "ollama pull foo" in e.message, e.message)

    def refuse(req):
        raise httpx.ConnectError("refused")

    p = ollama(refuse)
    try:
        await p.complete("s", MSG)
        check("raised", False)
    except UpstreamError as e:
        check("connect error -> not running", "Cannot reach Ollama" in e.message, e.message)
    ok_h, detail = await p.health()
    check("health reports not running", not ok_h and "Cannot reach" in detail)

    print("\n[5] Streaming NDJSON with split think tags")

    def stream(req):
        lines = [{"message": {"content": c}, "done": False} for c in ["<thi", "nk>plan</th", "ink>The ans", "wer."]]
        lines.append({"message": {"content": ""}, "done": True})
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    p = ollama(stream)
    out = "".join([x async for x in p.stream("s", MSG)])
    check("stream text", out == "The answer.", repr(out))

    print("\n[6] Vision detection")
    calls = []

    def vis(req):
        calls.append(req.url.path)
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(200, json={"message": {"content": "a cat"}, "done": True})

    p = ollama(vis)
    check("auto = maybe before probe", p.supports_vision)
    res = await p.describe_image(b"img", "image/png", "describe")
    check("text-only model -> None, no chat call", res is None and calls == ["/api/show"], calls)
    check("vision now off", not p.supports_vision)
    calls.clear()

    def vis2(req):
        calls.append(req.url.path)
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion", "vision"]})
        body = json.loads(req.content)
        assert body["model"] == "gemma3:4b" and body["messages"][0]["images"]
        return httpx.Response(200, json={"message": {"content": "a cat"}, "done": True})

    p = ollama(vis2, vision_model="gemma3:4b")
    check("vision model captions", await p.describe_image(b"img", "image/png", "d") == "a cat")

    print("\n[7] Warmup")

    def warm(req):
        if req.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen2.5:7b", "model": "qwen2.5:7b"}]})
        if req.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        calls.append(json.loads(req.content))
        return httpx.Response(200, json={"done": True})

    calls.clear()
    p = ollama(warm)
    await p.warmup()
    check("preload sent empty messages", calls and calls[-1]["messages"] == [])
    check("latest alias", OP.OllamaProvider._is_installed("llama3", {"llama3:latest"}))

    print("\n[8] HuggingFace helpers + factory")
    m = _merge_turns("SYS", [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                             {"role": "assistant", "content": "c"}])
    check("system folded + alternation", m[0] == {"role": "user", "content": "SYS\n\na\n\nb"} and len(m) == 2, m)
    m = _merge_turns("SYS", [{"role": "assistant", "content": "c"}])
    check("leading user turn inserted", m[0]["role"] == "user" and m[0]["content"] == "SYS")

    from app.llm import factory
    settings.llm_provider = "huggingface"
    factory.reset_llm()
    p = factory.get_llm()
    check("factory builds HF lazily (no load)", type(p).__name__ == "HuggingFaceProvider" and p._model is None)
    ok_h, d = await p.health()
    check("health doesn't block on load", ok_h and "loading" in d)
    settings.llm_provider = "ollama"
    factory.reset_llm()
    check("factory builds Ollama", type(factory.get_llm()).__name__ == "OllamaProvider")
    check("configured", factory.provider_is_configured())


def test_ollama_and_huggingface_providers():
    asyncio.run(_run())
