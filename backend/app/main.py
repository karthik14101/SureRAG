"""FastAPI application: lifespan, middleware and router mounting.

Startup order is deliberate. Local resources (SQLite, directories) come first so
their failures are reported clearly; external services are probed next but are
allowed to be down -- Qdrant failing is fatal for search, Neo4j failing is not,
and the app should start either way so the user can see the diagnosis in the UI
rather than staring at a crashed process.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.core.errors import register_error_handlers
from app.logging_conf import configure_logging, get_logger, new_request_id, request_id_ctx

configure_logging(settings.log_level)
logger = get_logger(__name__)

BANNER = r"""
  ___ _   _ ___ ___     ___               _    ___   _   ___
 / __| | | | _ \ __|__ / __|_ _ __ _ _ __| |_ | _ \ /_\ / __|
 \__ \ |_| |   / _|___| (_ | '_/ _` | '_ \ ' \|   //   \ (_ |
 |___/\___/|_|_\___|   \___|_| \__,_| .__/_||_|_|_\_/ \_\___|
                                    |_|
 Adaptive Hybrid Graph-Agentic Retrieval Engine
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(BANNER)
    logger.info("Starting %s", settings.app_name)
    for key, value in settings.describe().items():
        logger.info("  %-18s %s", key, value)

    # ---- local resources (fatal if broken) ---------------------------------
    settings.ensure_dirs()
    from app.db.init_db import init_database

    init_database()

    from app.db.base import session_scope
    from app.services import auth_service, cleanup_service

    with session_scope() as db:
        purged = auth_service.purge_expired_sessions(db)
        if purged:
            logger.info("Purged %d expired session(s)", purged)
        cleanup_service.sweep_orphan_media(db)
    cleanup_service.clear_stale_staging()

    from app.ingestion.worker import recover_stuck_jobs, worker

    await recover_stuck_jobs()

    # ---- external services (non-fatal) -------------------------------------
    try:
        from app.vectorstore.qdrant_client import ensure_collection

        await ensure_collection()
        logger.info("Qdrant collection '%s' ready", settings.qdrant_collection)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Qdrant is not available: %s\n"
            "  Search will not work until it is running. Start it with:\n"
            "    docker compose up -d",
            str(exc)[:300],
        )

    try:
        from app.graph.neo4j_client import ensure_schema

        await ensure_schema()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j setup skipped: %s", str(exc)[:200])

    from app.llm.factory import provider_is_configured

    if not provider_is_configured():
        logger.warning(
            "LLM_PROVIDER=%s is not fully configured. Add %s to .env -- retrieval "
            "will work but answering will not.",
            settings.llm_provider,
            settings.active_key_env_name,
        )

    # ---- background work ----------------------------------------------------
    await worker.start()

    # Load the embedding model now so the first question is not slow. Runs on a
    # thread so a cold HuggingFace download does not block the event loop.
    import asyncio

    from app.embeddings import encoder, sparse

    asyncio.create_task(asyncio.to_thread(encoder.warmup))
    asyncio.create_task(asyncio.to_thread(encoder.warmup_reranker))
    asyncio.create_task(asyncio.to_thread(sparse.warmup))

    # Local LLMs (Ollama, HuggingFace) load their model now, in the background.
    if provider_is_configured():
        async def _warm_llm() -> None:
            try:
                from app.llm.factory import get_llm

                await get_llm().warmup()
            except Exception as exc:  # noqa: BLE001 - never block startup
                logger.warning("LLM warmup skipped: %s", str(exc)[:200])

        asyncio.create_task(_warm_llm())

    logger.info("Ready on %s  (docs at %s/docs)", settings.browsable_url, settings.browsable_url)

    yield

    logger.info("Shutting down")
    await worker.stop()

    from app.graph.neo4j_client import close_driver
    from app.vectorstore.qdrant_client import close_client

    await close_client()
    await close_driver()


app = FastAPI(
    title=settings.app_name,
    description=(
        "Adaptive Hybrid Graph-Agentic RAG engine. Hybrid dense+BM25 vector "
        "retrieval over Qdrant, entity/relationship traversal over Neo4j, and a "
        "bounded agentic loop with evidence-sufficiency verification."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # The frontend reads these off streaming responses.
    expose_headers=["X-Request-Id", "X-Response-Time-Ms"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a correlation id and timing to every request."""
    request_id = request.headers.get("X-Request-Id") or new_request_id()
    token = request_id_ctx.set(request_id)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        request_id_ctx.reset(token)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)

    # Log the interesting requests only; polling would otherwise flood the log.
    path = request.url.path
    if not path.endswith(("/health", "/docs", "/openapi.json")) and elapsed_ms > 500:
        logger.info(
            "%s %s -> %d (%dms)", request.method, path, response.status_code, elapsed_ms
        )

    return response


register_error_handlers(app)
app.include_router(api_router, prefix=settings.api_prefix)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "app": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "{}/health/deep".format(settings.api_prefix),
        "api": settings.api_prefix,
    }
