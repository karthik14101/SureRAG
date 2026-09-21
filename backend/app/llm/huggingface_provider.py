"""Local HuggingFace provider: any causal-LM run in-process with transformers.

Set HF_MODEL to a Hub repo id (Qwen/Qwen2.5-1.5B-Instruct,
meta-llama/Llama-3.2-3B-Instruct, microsoft/Phi-3.5-mini-instruct, ...) or to a
local folder holding the weights. Nothing new to install: transformers and torch
already come in with sentence-transformers.

Design notes:
  * The model loads lazily and on a thread, so constructing the provider stays
    cheap and a multi-GB first download never blocks the event loop. `warmup`
    starts that load at app startup.
  * One generation runs at a time. Two concurrent generate() calls on one model
    double peak memory and are no faster on CPU; queuing them is the right trade.
  * Chat templates vary. Some reject a system role (Gemma), some need strict
    user/assistant alternation, base models have no template at all. Each case
    degrades to something that works instead of raising.
  * Hosted APIs enforce JSON mode; transformers cannot. JSON calls get an extra
    instruction and rely on the tolerant parser in base.parse_json_object.
  * Vision is not offered here (image-text models need different model classes);
    image understanding falls back to local OCR, as it does with Groq.
"""
from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator

from app.config import settings
from app.core.errors import UpstreamError
from app.llm.base import ChatMessage, LLMProvider, LLMResult, TokenUsage
from app.llm.reasoning import ThinkTagFilter, strip_reasoning
from app.logging_conf import get_logger

logger = get_logger(__name__)

PROVIDER_LABEL = "HuggingFace (local)"
JSON_INSTRUCTION = (
    "\n\nRespond with a single valid JSON object and nothing else: "
    "no prose, no code fences."
)
# Share of the prompt budget kept from the start when a prompt is trimmed; the
# rest comes from the end, where the question and the answer cue live.
HEAD_SHARE = 0.25
_DONE = object()


def _merge_turns(system: str, turns: list[dict]) -> list[dict]:
    """Fold the system prompt into the first user turn and enforce strict
    user/assistant alternation, for templates that accept nothing else."""
    merged: list[dict] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] += "\n\n" + turn["content"]
        else:
            merged.append(dict(turn))
    if not merged or merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": ""})
    if system:
        first = merged[0]["content"]
        merged[0]["content"] = system + ("\n\n" + first if first else "")
    return merged


class HuggingFaceProvider(LLMProvider):
    name = "huggingface"
    supports_vision = False

    def __init__(self) -> None:
        self.model_id = (settings.hf_model or "").strip()
        if not self.model_id:
            raise UpstreamError(
                "HF_MODEL is empty. Set it to a HuggingFace repo id such as "
                "Qwen/Qwen2.5-1.5B-Instruct, or to a local model folder."
            )
        self.model = self.model_id  # for logs and LLMResult
        self._model = None
        self._tokenizer = None
        self._device = "cpu"
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._generate_lock = asyncio.Lock()
        self._system_ok = True  # flips off for templates that reject a system role

    # --------------------------------------------------------------- loading
    def _resolve_device(self, torch) -> str:
        wanted = (settings.hf_device or "auto").strip().lower()
        if wanted == "auto":
            if torch.cuda.is_available():
                return "cuda"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if wanted.startswith("cuda") and not torch.cuda.is_available():
            raise UpstreamError(
                "HF_DEVICE={} but this torch build has no CUDA support (the PyPI "
                "wheel on Windows is CPU-only). Set HF_DEVICE=auto or cpu, or "
                "install a CUDA build of torch.".format(wanted)
            )
        return wanted

    @staticmethod
    def _resolve_dtype(torch, device: str):
        wanted = (settings.hf_dtype or "auto").strip().lower()
        named = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if wanted in named:
            return named[wanted]
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if device == "mps":
            return torch.float16
        # Half precision on CPU is slower than float32 for most kernels.
        return torch.float32

    def _explain_load_error(self, exc: Exception) -> str:
        text = str(exc)
        low = text.lower()
        if "gated" in low or "401" in low or "403" in low or "access to model" in low:
            return (
                "'{}' is a gated model. Accept its licence on huggingface.co, then "
                "put a read token in HF_TOKEN in .env.".format(self.model_id)
            )
        if "not a valid model identifier" in low or "404" in low or "repository not found" in low:
            return (
                "HF_MODEL '{}' was not found on the HuggingFace Hub or as a local "
                "folder. Check the repo id (it is case-sensitive).".format(self.model_id)
            )
        if "out of memory" in low or "not enough memory" in low or "defaultcpuallocator" in low:
            return (
                "Not enough memory to load '{}'. Pick a smaller model (e.g. "
                "Qwen/Qwen2.5-0.5B-Instruct), or on an NVIDIA GPU set "
                "HF_LOAD_IN_4BIT=true.".format(self.model_id)
            )
        if "trust_remote_code" in low:
            return (
                "'{}' ships custom model code. If you trust the repo, set "
                "HF_TRUST_REMOTE_CODE=true in .env.".format(self.model_id)
            )
        if "connection" in low or "offline" in low or "resolve" in low:
            return (
                "Could not download '{}' (no network?). Once a model has been "
                "downloaded it loads from the local cache.".format(self.model_id)
            )
        return "Could not load HuggingFace model '{}': {}".format(self.model_id, text[:300])

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise UpstreamError(
                    "transformers/torch are not installed. Run "
                    "`pip install -r requirements.txt` in the backend venv."
                ) from exc

            device = self._resolve_device(torch)
            dtype = self._resolve_dtype(torch, device)
            common: dict = {"trust_remote_code": settings.hf_trust_remote_code}
            if settings.hf_token:
                common["token"] = settings.hf_token

            model_kwargs: dict = dict(common, torch_dtype=dtype)
            if settings.hf_load_in_4bit:
                if not device.startswith("cuda"):
                    raise UpstreamError(
                        "HF_LOAD_IN_4BIT=true needs an NVIDIA GPU (bitsandbytes). "
                        "Set it to false on CPU."
                    )
                try:
                    import bitsandbytes  # noqa: F401
                    from transformers import BitsAndBytesConfig
                except ImportError as exc:
                    raise UpstreamError(
                        "HF_LOAD_IN_4BIT=true needs extra packages: "
                        "pip install bitsandbytes accelerate"
                    ) from exc
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.float16,
                )
                model_kwargs["device_map"] = {"": 0}

            logger.info(
                "Loading HuggingFace model %s on %s (%s)%s -- the first run downloads "
                "the weights, which can take several minutes.",
                self.model_id,
                device,
                str(dtype).replace("torch.", ""),
                ", 4-bit" if settings.hf_load_in_4bit else "",
            )
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id, **common)
                model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except UpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001 - translated for the user
                raise UpstreamError(self._explain_load_error(exc)) from exc

            if not settings.hf_load_in_4bit:
                model.to(device)
            model.eval()
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            self._tokenizer, self._model, self._device = tokenizer, model, device
            logger.info("HuggingFace model %s ready on %s", self.model_id, device)

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise UpstreamError(self._load_error)
        try:
            await asyncio.to_thread(self._load_sync)
        except Exception as exc:  # noqa: BLE001
            # Remember it: retrying a failed multi-GB load on every question
            # would stall each request for minutes. Restart after fixing .env.
            message = exc.message if isinstance(exc, UpstreamError) else self._explain_load_error(exc)
            self._load_error = message
            raise UpstreamError(message) from exc

    # --------------------------------------------------------------- prompting
    def _render(self, system: str, messages: list[ChatMessage], json_mode: bool) -> str:
        if json_mode:
            system = (system or "") + JSON_INSTRUCTION
        turns = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in messages
        ]
        tokenizer = self._tokenizer

        if not getattr(tokenizer, "chat_template", None):
            # Base (non-chat) model: a plain transcript is the best we can do.
            lines = [system] if system else []
            for turn in turns:
                lines.append("{}: {}".format(turn["role"].capitalize(), turn["content"]))
            lines.append("Assistant:")
            return "\n\n".join(lines)

        # Unused template variables are ignored, so this is safe for every model;
        # Qwen3-style templates read it to switch <think> blocks off.
        extra = {"enable_thinking": settings.hf_thinking}

        if system and self._system_ok:
            try:
                return tokenizer.apply_chat_template(
                    [{"role": "system", "content": system}] + turns,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra,
                )
            except Exception as exc:  # noqa: BLE001 - jinja TemplateError
                if "system" not in str(exc).lower():
                    raise
                self._system_ok = False
                logger.info(
                    "%s's chat template rejects a system role; folding it into the "
                    "user turn from now on.",
                    self.model_id,
                )

        try:
            return tokenizer.apply_chat_template(
                _merge_turns(system, turns),
                tokenize=False,
                add_generation_prompt=True,
                **extra,
            )
        except Exception as exc:  # noqa: BLE001
            raise UpstreamError(
                "{}'s chat template failed: {}".format(self.model_id, str(exc)[:200])
            ) from exc

    def _context_limit(self) -> int:
        limit = settings.hf_max_input_tokens if settings.hf_max_input_tokens > 0 else 8192
        model_max = getattr(self._model.config, "max_position_embeddings", None)
        if isinstance(model_max, int) and model_max > 0:
            limit = min(limit, model_max)
        return limit

    def _encode(self, text: str, max_new_tokens: int):
        """Tokenise, trimming the middle if the prompt exceeds the budget."""
        import torch

        tokenizer = self._tokenizer
        # Chat templates already include BOS; a bare transcript does not.
        add_special = not getattr(tokenizer, "chat_template", None)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=add_special)["input_ids"]

        budget = max(256, self._context_limit() - max_new_tokens)
        length = ids.shape[1]
        if length > budget:
            head = int(budget * HEAD_SHARE)
            tail = budget - head
            ids = torch.cat([ids[:, :head], ids[:, length - tail :]], dim=1)
            logger.warning(
                "Prompt of %d tokens exceeds the %d-token budget for %s; trimmed "
                "the middle. Lower RETRIEVAL_TOP_K or raise HF_MAX_INPUT_TOKENS.",
                length,
                budget,
                self.model_id,
            )
        return ids.to(self._model.device)

    def _generation_kwargs(
        self, input_ids, temperature: float | None, max_new_tokens: int, cancel: threading.Event
    ) -> dict:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _StopOnCancel(StoppingCriteria):
            def __call__(self, *args, **kwargs) -> bool:
                return cancel.is_set()

        temp = settings.llm_temperature if temperature is None else temperature
        kwargs: dict = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "stopping_criteria": StoppingCriteriaList([_StopOnCancel()]),
        }
        if temp and temp > 0:
            kwargs.update(do_sample=True, temperature=float(temp), top_p=0.9)
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _generate_sync(self, kwargs: dict, streamer=None):
        import torch

        if streamer is not None:
            kwargs = dict(kwargs, streamer=streamer)
        with torch.inference_mode():
            return self._model.generate(**kwargs)

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
        await self._ensure_loaded()
        max_new = max_tokens or settings.llm_max_tokens
        cancel = threading.Event()

        async with self._generate_lock:
            try:
                prompt = self._render(system, messages, json_mode)
                input_ids = self._encode(prompt, max_new)
                kwargs = self._generation_kwargs(input_ids, temperature, max_new, cancel)
                output = await asyncio.wait_for(
                    asyncio.to_thread(self._generate_sync, kwargs),
                    timeout=settings.local_llm_timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                raise UpstreamError(self._too_slow()) from exc
            except UpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise UpstreamError(self._explain_runtime_error(exc)) from exc
            finally:
                # Stops a generation that outlived a timeout or a cancelled request.
                cancel.set()

        prompt_len = input_ids.shape[1]
        new_tokens = output[0][prompt_len:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        usage = TokenUsage(
            prompt_tokens=int(prompt_len),
            completion_tokens=int(new_tokens.shape[0]),
            total_tokens=int(prompt_len + new_tokens.shape[0]),
            calls=1,
        )
        return LLMResult(text=strip_reasoning(raw), usage=usage, model=self.model_id)

    async def stream(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        await self._ensure_loaded()
        from transformers import TextIteratorStreamer

        max_new = max_tokens or settings.llm_max_tokens
        cancel = threading.Event()
        failure: list[BaseException] = []

        async with self._generate_lock:
            try:
                prompt = self._render(system, messages, json_mode=False)
                input_ids = self._encode(prompt, max_new)
                kwargs = self._generation_kwargs(input_ids, temperature, max_new, cancel)
            except UpstreamError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise UpstreamError(self._explain_runtime_error(exc)) from exc

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
                timeout=float(settings.local_llm_timeout_seconds),
            )

            def _run() -> None:
                try:
                    self._generate_sync(kwargs, streamer=streamer)
                except BaseException as exc:  # noqa: BLE001 - reported to the consumer
                    failure.append(exc)
                    streamer.end()  # unblock the consumer

            worker = threading.Thread(target=_run, name="hf-generate", daemon=True)
            worker.start()

            think = ThinkTagFilter()
            iterator = iter(streamer)
            try:
                while True:
                    try:
                        piece = await asyncio.to_thread(next, iterator, _DONE)
                    except queue.Empty as exc:
                        raise UpstreamError(self._too_slow()) from exc
                    if piece is _DONE:
                        break
                    visible = think.feed(piece)
                    if visible:
                        yield visible

                if failure:
                    raise UpstreamError(self._explain_runtime_error(failure[0]))
                tail = think.flush()
                if tail:
                    yield tail
            finally:
                cancel.set()
                # Let the generation thread notice the cancel before the next
                # request takes the lock; it stops within one token.
                await asyncio.to_thread(worker.join, 30)

    # ------------------------------------------------------------------ errors
    def _too_slow(self) -> str:
        return (
            "HuggingFace model '{}' did not finish within {}s on {}. Use a smaller "
            "model, lower LLM_MAX_TOKENS, or raise LOCAL_LLM_TIMEOUT_SECONDS.".format(
                self.model_id, settings.local_llm_timeout_seconds, self._device
            )
        )

    def _explain_runtime_error(self, exc: BaseException) -> str:
        low = str(exc).lower()
        if "out of memory" in low:
            return (
                "Ran out of memory generating with '{}'. Lower RETRIEVAL_TOP_K or "
                "HF_MAX_INPUT_TOKENS, or use a smaller model.".format(self.model_id)
            )
        return "Local generation with '{}' failed: {}".format(self.model_id, str(exc)[:300])

    # --------------------------------------------------------------- lifecycle
    async def warmup(self) -> None:
        try:
            await self._ensure_loaded()
        except UpstreamError as exc:
            logger.error("HuggingFace model not loaded: %s", exc.message)

    async def health(self) -> tuple[bool, str]:
        if self._load_error:
            return False, self._load_error
        if self._model is None:
            # Loading can take minutes on a first download; do not block the
            # health check on it. warmup() is already loading it.
            return True, "loading in the background (first run downloads the weights)"
        return await super().health()
