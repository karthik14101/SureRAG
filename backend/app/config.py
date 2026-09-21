"""Application configuration, loaded and validated from the project-root .env."""
from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> <project root>
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Every runtime knob. Anything absent from .env falls back to the default here."""

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- app ----------------------------------------------------------------
    app_name: str = "SURE-GraphRAG"
    api_prefix: str = "/api/v1"
    # Loopback only: the app is reachable from this machine, not from the network.
    # Set to 0.0.0.0 if you deliberately want to reach it from another device.
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    session_ttl_hours: int = 720

    # ---- llm ----------------------------------------------------------------
    llm_provider: Literal["gemini", "groq", "azure", "ollama", "huggingface"] = "gemini"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3-flash"
    gemini_vision_model: str = "gemini-3-flash"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Azure OpenAI / Azure AI Foundry. The endpoint may be pasted in any form the
    # Foundry portal shows, including a full "Target URI" -- see azure_target.
    azure_openai_endpoint: str = ""
    azure_openai_api_key: str = ""
    # The DEPLOYMENT name you chose in Foundry, which is not always the model name.
    azure_openai_deployment: str = ""
    # "v1" = the always-latest v1 API (recommended). A dated value such as
    # "2025-04-01-preview" switches to the legacy api-version routing instead.
    azure_openai_api_version: str = "v1"
    # auto = infer from the deployment and adapt to what the API accepts.
    azure_openai_reasoning: Literal["auto", "true", "false"] = "auto"
    # Thinking budget for reasoning deployments: minimal | low | medium | high.
    azure_openai_reasoning_effort: str = "low"
    # Set false if the deployment is text-only; images then fall back to OCR.
    azure_openai_vision: bool = True

    # Ollama (local server, https://ollama.com). The model must be pulled first:
    #   ollama pull qwen2.5:7b
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"
    # Optional separate model for image captioning (e.g. gemma3:4b, llava,
    # qwen2.5vl:7b). Empty = use OLLAMA_MODEL if it can see images.
    ollama_vision_model: str = ""
    # auto = ask Ollama whether the model has the "vision" capability.
    ollama_vision: Literal["auto", "true", "false"] = "auto"
    # Ollama's own default context (2-4k tokens) silently truncates RAG prompts.
    ollama_num_ctx: int = 8192
    # How long Ollama keeps the model in memory after a call ("30m", "-1" = forever).
    ollama_keep_alive: str = "30m"
    # Thinking models (qwen3, deepseek-r1, gpt-oss): false | true | low | medium |
    # high | auto (= do not send; model default). false keeps answers fast.
    ollama_think: str = "false"
    # Only for a protected remote Ollama or a proxy in front of it.
    ollama_api_key: str = ""

    # HuggingFace transformers, loaded in-process. Any causal-LM repo id or a
    # local folder path works; the weights download into the HF cache once.
    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # auto = cuda if available, then mps (Apple), then cpu.
    hf_device: str = "auto"
    # auto = bfloat16/float16 on GPU, float32 on CPU.
    hf_dtype: str = "auto"
    # Needed only for gated repos (Llama, Gemma): accept the licence on the
    # model page, then create a read token at huggingface.co/settings/tokens.
    hf_token: str = ""
    # Prompt budget in tokens. Longer prompts are trimmed from the middle.
    hf_max_input_tokens: int = 8192
    # 4-bit quantisation on an NVIDIA GPU. Needs: pip install bitsandbytes accelerate
    hf_load_in_4bit: bool = False
    # Some repos ship custom model code. Only enable for repos you trust.
    hf_trust_remote_code: bool = False
    # Qwen3-style templates can emit <think> blocks; off keeps answers fast.
    hf_thinking: bool = False

    # Local models are far slower than hosted APIs, especially on CPU. When a
    # local provider is active these replace the (much shorter) cloud timeouts.
    local_llm_timeout_seconds: int = 600
    local_agent_timeout_seconds: int = 300

    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048
    llm_timeout_seconds: int = 90
    llm_max_retries: int = 3

    # ---- embeddings ---------------------------------------------------------
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"
    rerank_enabled: bool = False
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_n: int = 8
    sparse_enabled: bool = True

    # ---- datastores ---------------------------------------------------------
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "sure_chunks"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sureGraph2024"
    neo4j_database: str = "neo4j"

    # ---- retrieval / agent --------------------------------------------------
    chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 150
    retrieval_top_k: int = 12
    graph_max_hops: int = 2
    graph_seed_entities: int = 6
    sure_threshold: float = 0.65
    agent_max_iterations: int = 2
    agent_max_llm_calls: int = 6
    agent_timeout_seconds: int = 45
    graph_extraction: Literal["on", "off", "selective"] = "selective"

    # ---- ingestion ----------------------------------------------------------
    max_upload_mb: int = 100
    max_zip_depth: int = 3
    max_zip_entries: int = 500
    max_zip_uncompressed_mb: int = 500
    ingest_concurrency: int = 2
    min_image_pixels: int = 64
    ocr_enabled: bool = True
    tesseract_cmd: str = ""

    # ---- paths (resolved in code, never read from env) ----------------------
    project_root: pathlib.Path = Field(default=PROJECT_ROOT, exclude=True)

    @field_validator("sure_threshold")
    @classmethod
    def _clamp_threshold(cls, v: float) -> float:
        return max(0.0, min(1.0, v))

    @field_validator("embedding_device")
    @classmethod
    def _normalise_device(cls, v: str) -> str:
        return (v or "cpu").strip().lower()

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, v: str) -> str:
        value = (v or "").strip().lower()
        # Accept the obvious spellings people type.
        return {"hf": "huggingface", "hugging_face": "huggingface", "transformers": "huggingface"}.get(
            value, value
        )

    @model_validator(mode="after")
    def _local_timeouts(self) -> "Settings":
        # A 45s agent budget suits a hosted API; a 7B model on a laptop CPU can
        # need that long for one answer. Never shorten what the user set.
        if self.is_local_llm:
            self.agent_timeout_seconds = max(
                self.agent_timeout_seconds, self.local_agent_timeout_seconds
            )
        return self

    # ---- derived ------------------------------------------------------------
    @property
    def data_dir(self) -> pathlib.Path:
        return self.project_root / "data"

    @property
    def media_dir(self) -> pathlib.Path:
        return self.data_dir / "media"

    @property
    def upload_tmp_dir(self) -> pathlib.Path:
        return self.data_dir / "uploads_tmp"

    @property
    def sqlite_path(self) -> pathlib.Path:
        return self.data_dir / "sure.db"

    @property
    def sqlite_url(self) -> str:
        # as_posix() keeps the URL valid on Windows (C:/... rather than C:\...)
        return "sqlite:///" + self.sqlite_path.as_posix()

    @property
    def browsable_host(self) -> str:
        """A host you can actually type into a browser.

        0.0.0.0 and :: are bind-all wildcards, not destinations -- a browser
        cannot connect to them. Map them onto loopback so every URL we print or
        log is one the user can click.
        """
        if self.backend_host in ("0.0.0.0", "::", "*", ""):
            return "127.0.0.1"
        return self.backend_host

    @property
    def browsable_url(self) -> str:
        return "http://{}:{}".format(self.browsable_host, self.backend_port)

    @property
    def qdrant_url(self) -> str:
        return "http://" + self.qdrant_host + ":" + str(self.qdrant_port)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def max_zip_uncompressed_bytes(self) -> int:
        return self.max_zip_uncompressed_mb * 1024 * 1024

    @property
    def azure_target(self) -> tuple[str, str, str]:
        """Resolve the Azure settings into (host, deployment, api_version).

        People paste whatever the Foundry portal shows them, which is rarely just
        the bare endpoint. All of these resolve to the same place:

            https://res.openai.azure.com
            https://res.openai.azure.com/openai/v1/
            https://res.cognitiveservices.azure.com/
            https://res.services.ai.azure.com/api/projects/my-project
            https://res.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2025-01-01-preview

        The host is kept and the path discarded. A deployment name embedded in a
        pasted Target URI is used when AZURE_OPENAI_DEPLOYMENT is empty. An
        api-version embedded in the URL is deliberately ignored: it is only
        whatever the portal's sample happened to use, and the v1 API is the
        better default for every current resource.
        """
        from urllib.parse import unquote, urlparse

        raw = (self.azure_openai_endpoint or "").strip()
        deployment = (self.azure_openai_deployment or "").strip()
        version = (self.azure_openai_api_version or "").strip() or "v1"

        if not raw:
            return "", deployment, version
        if "://" not in raw:
            raw = "https://" + raw

        parsed = urlparse(raw)
        host = "{}://{}".format(parsed.scheme or "https", parsed.netloc).rstrip("/")

        if not deployment:
            import re

            match = re.search(r"/openai/deployments/([^/?#]+)", parsed.path)
            if match:
                deployment = unquote(match.group(1))

        return host, deployment, version

    @property
    def ollama_url(self) -> str:
        """OLLAMA_BASE_URL normalised to a scheme+host+port with no API path.

        Accepts what people copy from Ollama's docs or their OLLAMA_HOST:
        "localhost:11434", "http://0.0.0.0:11434", ".../api", ".../v1/".
        """
        raw = (self.ollama_base_url or "").strip() or "http://127.0.0.1:11434"
        if "://" not in raw:
            raw = "http://" + raw
        raw = raw.rstrip("/")
        for suffix in ("/api", "/v1"):
            if raw.endswith(suffix):
                raw = raw[: -len(suffix)]
        # 0.0.0.0 is a listen address, not a destination.
        return raw.replace("//0.0.0.0", "//127.0.0.1")

    @property
    def is_local_llm(self) -> bool:
        return self.llm_provider in ("ollama", "huggingface")

    @property
    def active_model(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_model
        if self.llm_provider == "azure":
            return self.azure_target[1] or "(no deployment set)"
        if self.llm_provider == "ollama":
            return self.ollama_model
        if self.llm_provider == "huggingface":
            return self.hf_model
        return self.groq_model

    def active_api_key(self) -> str:
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        if self.llm_provider == "azure":
            return self.azure_openai_api_key
        if self.is_local_llm:
            return ""  # local providers need no key
        return self.groq_api_key

    @property
    def active_key_env_name(self) -> str:
        """The .env variable the user must fill in for the active provider."""
        return {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "azure": "AZURE_OPENAI_API_KEY (plus AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT)",
            "ollama": "OLLAMA_MODEL",
            "huggingface": "HF_MODEL",
        }[self.llm_provider]

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.media_dir, self.upload_tmp_dir):
            path.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """Safe-to-log summary. Deliberately never includes API keys."""
        return {
            "provider": self.llm_provider,
            "model": self.active_model,
            "api_key_present": (
                "not needed (local)" if self.is_local_llm else bool(self.active_api_key())
            ),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "device": self.embedding_device,
            "qdrant": self.qdrant_url,
            "neo4j": self.neo4j_uri,
            "graph_extraction": self.graph_extraction,
            "rerank": self.rerank_enabled,
            "sparse": self.sparse_enabled,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    # Keep the HuggingFace cache inside the project unless the user set HF_HOME.
    if not os.getenv("HF_HOME"):
        cache = s.project_root / ".hf_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    # Tokenizer forks warn loudly inside threadpools; we batch manually anyway.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return s


settings = get_settings()
