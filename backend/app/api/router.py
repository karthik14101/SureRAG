"""Aggregates every route module under the versioned API prefix."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.routes import (
    auth,
    chat,
    chunks,
    documents,
    graph,
    health,
    jobs,
    knowledge_base,
    media,
)

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(auth.router)
api_router.include_router(knowledge_base.router)
api_router.include_router(documents.router)
api_router.include_router(jobs.router)
api_router.include_router(chunks.router)
api_router.include_router(graph.router)
api_router.include_router(chat.router)
api_router.include_router(media.router)
