"""Health and diagnostics.

/health/deep is the single call that tells you which part of the stack is not
running -- it is what the README's validation step uses.
"""
from __future__ import annotations

import time

from fastapi import APIRouter

from app.config import settings
from app.schemas.agent import DeepHealth, HealthComponent

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": settings.app_name}


@router.get("/health/deep", response_model=DeepHealth)
async def deep_health() -> DeepHealth:
    components: list[HealthComponent] = []

    # --- SQLite --------------------------------------------------------------
    started = time.perf_counter()
    try:
        from sqlalchemy import text

        from app.db.base import engine

        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        components.append(
            HealthComponent(
                name="sqlite",
                ok=True,
                detail=str(settings.sqlite_path),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            HealthComponent(name="sqlite", ok=False, detail=str(exc)[:200])
        )

    # --- Qdrant --------------------------------------------------------------
    started = time.perf_counter()
    try:
        from app.vectorstore.qdrant_client import health as qdrant_health

        ok, detail = await qdrant_health()
        components.append(
            HealthComponent(
                name="qdrant",
                ok=ok,
                detail=detail
                if ok
                else "{} (is `docker compose up -d` running?)".format(detail),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(HealthComponent(name="qdrant", ok=False, detail=str(exc)[:200]))

    # --- Neo4j ---------------------------------------------------------------
    started = time.perf_counter()
    try:
        from app.graph.neo4j_client import health as neo4j_health

        ok, detail = await neo4j_health()
        components.append(
            HealthComponent(
                name="neo4j",
                ok=ok,
                detail=detail if ok else "{} (graph features disabled)".format(detail),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(HealthComponent(name="neo4j", ok=False, detail=str(exc)[:200]))

    # --- Embeddings ----------------------------------------------------------
    started = time.perf_counter()
    try:
        from app.embeddings.encoder import embed_query

        vector = await embed_query("health probe")
        components.append(
            HealthComponent(
                name="embeddings",
                ok=len(vector) == settings.embedding_dim,
                detail="{} ({} dims)".format(settings.embedding_model, len(vector)),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(
            HealthComponent(name="embeddings", ok=False, detail=str(exc)[:200])
        )

    # --- LLM provider --------------------------------------------------------
    started = time.perf_counter()
    try:
        from app.llm.factory import get_llm, provider_is_configured

        if not provider_is_configured():
            components.append(
                HealthComponent(
                    name="llm",
                    ok=False,
                    detail="Provider '{}' is not fully configured. Set {} in .env.".format(
                        settings.llm_provider, settings.active_key_env_name
                    ),
                )
            )
        else:
            ok, detail = await get_llm().health()
            components.append(
                HealthComponent(
                    name="llm",
                    ok=ok,
                    detail="{} / {} -- {}".format(
                        settings.llm_provider, settings.active_model, detail
                    ),
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            )
    except Exception as exc:  # noqa: BLE001
        components.append(HealthComponent(name="llm", ok=False, detail=str(exc)[:200]))

    # --- OCR (informational; never fails the health check) -------------------
    try:
        from app.ingestion.parsers.image_parser import ocr_available

        available = ocr_available()
        components.append(
            HealthComponent(
                name="ocr",
                ok=True,
                detail="Tesseract available" if available else "not installed (optional)",
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(HealthComponent(name="ocr", ok=True, detail=str(exc)[:120]))

    # Neo4j and OCR are optional: the app is usable without them.
    required = {"sqlite", "qdrant", "embeddings", "llm"}
    overall = all(c.ok for c in components if c.name in required)

    return DeepHealth(ok=overall, components=components, config=settings.describe())
