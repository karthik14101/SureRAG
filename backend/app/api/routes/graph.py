"""Graph explorer endpoints.

Read-only views over the knowledge graph in Neo4j: the entities and relations
extracted during ingestion, the neighbourhood around any entity, and the chunks
each one came from. The main use is checking extraction quality -- duplicates,
junk entities and vague predicates all show up here and all weaken GRAPH and
MULTIHOP answers.

Entities are addressed by their normalised name, which can contain spaces, so it
travels as a query parameter rather than a path segment.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Query

from app.core.deps import CurrentUser, DbSession, get_owned_kb
from app.core.errors import NotFoundError
from app.schemas.graph import (
    EntityDetail,
    EntityListResponse,
    GraphFacets,
    GraphView,
    RelationListResponse,
)
from app.services import graph_service

router = APIRouter(tags=["graph"])


@router.get("/kb/{kb_id}/graph/facets", response_model=GraphFacets)
async def graph_facets(kb_id: str, db: DbSession, user: CurrentUser) -> GraphFacets:
    """Entity types and relation predicates present in the graph, with counts."""
    kb = get_owned_kb(kb_id, db, user)
    return await graph_service.facets(kb.id)


@router.get("/kb/{kb_id}/graph/entities", response_model=EntityListResponse)
async def list_entities(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    q: str | None = Query(default=None, max_length=200),
    type: str | None = Query(default=None, max_length=40),
    sort: Literal["degree", "mentions", "name"] = "degree",
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> EntityListResponse:
    kb = get_owned_kb(kb_id, db, user)
    return await graph_service.list_entities(
        kb.id, q=q, entity_type=type, sort=sort, limit=limit, offset=offset
    )


@router.get("/kb/{kb_id}/graph/entity", response_model=EntityDetail)
async def entity_detail(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    id: str = Query(..., min_length=1, max_length=300),
) -> EntityDetail:
    """One entity with its outgoing and incoming relations and source chunks."""
    kb = get_owned_kb(kb_id, db, user)
    detail = await graph_service.entity_detail(kb.id, id)
    if detail is None:
        raise NotFoundError("That entity is not in this knowledge base's graph.")
    return detail


@router.get("/kb/{kb_id}/graph/relations", response_model=RelationListResponse)
async def list_relations(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    q: str | None = Query(default=None, max_length=200),
    predicate: str | None = Query(default=None, max_length=120),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> RelationListResponse:
    kb = get_owned_kb(kb_id, db, user)
    return await graph_service.list_relations(
        kb.id, q=q, predicate=predicate, limit=limit, offset=offset
    )


@router.get("/kb/{kb_id}/graph/overview", response_model=GraphView)
async def graph_overview(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(default=40, ge=5, le=150),
) -> GraphView:
    """The most connected entities and the relations among them."""
    kb = get_owned_kb(kb_id, db, user)
    return await graph_service.overview(kb.id, limit=limit)


@router.get("/kb/{kb_id}/graph/neighbourhood", response_model=GraphView)
async def graph_neighbourhood(
    kb_id: str,
    db: DbSession,
    user: CurrentUser,
    id: str = Query(..., min_length=1, max_length=300),
    hops: int = Query(default=1, ge=1, le=3),
    limit: int = Query(default=40, ge=1, le=150),
) -> GraphView:
    """An entity and its neighbours up to `hops` away, strongest links first."""
    kb = get_owned_kb(kb_id, db, user)
    view = await graph_service.neighbourhood(kb.id, id, hops=hops, limit=limit)
    if view is None:
        raise NotFoundError("That entity is not in this knowledge base's graph.")
    return view
