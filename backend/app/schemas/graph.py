"""Schemas for the graph explorer -- browsing entities and relations in Neo4j."""
from __future__ import annotations

from pydantic import BaseModel


class EntityOut(BaseModel):
    # The normalised name is the entity's key inside a knowledge base.
    id: str
    name: str
    type: str = "OTHER"
    description: str | None = None
    mentions: int = 0
    degree: int = 0


class EntityListResponse(BaseModel):
    items: list[EntityOut]
    total: int
    limit: int
    offset: int
    available: bool = True


class RelationOut(BaseModel):
    id: str
    predicate: str
    description: str | None = None
    weight: int = 1
    chunk_ids: list[str] = []
    source_id: str
    source_name: str
    source_type: str = "OTHER"
    target_id: str
    target_name: str
    target_type: str = "OTHER"


class RelationListResponse(BaseModel):
    items: list[RelationOut]
    total: int
    limit: int
    offset: int
    available: bool = True


class MentionOut(BaseModel):
    chunk_id: str
    filename: str | None = None
    page_no: int | None = None
    snippet: str | None = None
    count: int = 1


class EntityDetail(BaseModel):
    entity: EntityOut
    outgoing: list[RelationOut]
    incoming: list[RelationOut]
    mentions: list[MentionOut]


class GraphNode(BaseModel):
    id: str
    name: str
    type: str = "OTHER"
    mentions: int = 0
    degree: int = 0
    # Hops from the focused entity; None in the overview.
    depth: int | None = None


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    predicate: str
    weight: int = 1


class GraphView(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center: str | None = None
    # True when the node limit cut off part of the neighbourhood.
    truncated: bool = False
    available: bool = True


class GraphFacets(BaseModel):
    types: list[dict]
    predicates: list[dict]
    total_entities: int
    total_relations: int
    available: bool = True
