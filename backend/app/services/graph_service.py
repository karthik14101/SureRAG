"""Read-only queries behind the graph explorer.

Everything is scoped by kb_id, which the route has already verified belongs to
the caller, so one user can never read another's entities. Neo4j is optional:
when it is down every function returns an empty result flagged
`available=False` rather than raising.

Search here is plain substring matching on the normalised name (and on the
description), not the fulltext index. An explorer should find exactly what you
typed; the fuzzy fulltext index is right for seeding retrieval, not for
browsing.
"""
from __future__ import annotations

from app.graph.extractor import normalise_name
from app.graph.neo4j_client import is_available, run_read
from app.schemas.graph import (
    EntityDetail,
    EntityListResponse,
    EntityOut,
    GraphEdge,
    GraphFacets,
    GraphNode,
    GraphView,
    MentionOut,
    RelationListResponse,
    RelationOut,
)

# ORDER BY cannot be parameterised in Cypher, so sort keys map onto fixed text.
_ENTITY_SORTS = {
    "degree": "degree DESC, mentions DESC, name",
    "mentions": "mentions DESC, degree DESC, name",
    "name": "toLower(name)",
}

MAX_GRAPH_NODES = 150
MAX_GRAPH_EDGES = 600

# Degree counts RELATED edges in both directions. A pattern comprehension
# rather than COUNT {} keeps this working on Neo4j 4.4 as well as 5.x.
_DEGREE = "size([(e)-[:RELATED]-(:Entity) | 1])"


def _search_terms(q: str | None) -> tuple[str | None, str | None]:
    """(normalised term for names, lowercase term for descriptions)."""
    raw = (q or "").strip()
    if not raw:
        return None, None
    return (normalise_name(raw) or raw.lower()), raw.lower()


def _entity(row: dict) -> EntityOut:
    return EntityOut(
        id=row["id"],
        name=row.get("name") or row["id"],
        type=row.get("type") or "OTHER",
        description=row.get("description") or None,
        mentions=int(row.get("mentions") or 0),
        degree=int(row.get("degree") or 0),
    )


def _relation(row: dict) -> RelationOut:
    return RelationOut(
        id=row["id"],
        predicate=row.get("predicate") or "RELATED_TO",
        description=row.get("description") or None,
        weight=int(row.get("weight") or 1),
        chunk_ids=list(row.get("chunk_ids") or []),
        source_id=row["source_id"],
        source_name=row.get("source_name") or row["source_id"],
        source_type=row.get("source_type") or "OTHER",
        target_id=row["target_id"],
        target_name=row.get("target_name") or row["target_id"],
        target_type=row.get("target_type") or "OTHER",
    )


# --------------------------------------------------------------------- facets
async def facets(kb_id: str) -> GraphFacets:
    if not is_available():
        return GraphFacets(types=[], predicates=[], total_entities=0, total_relations=0, available=False)

    types = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        RETURN coalesce(e.type, 'OTHER') AS name, count(*) AS count
        ORDER BY count DESC
        """,
        {"kb_id": kb_id},
    )
    predicates = await run_read(
        """
        MATCH (:Entity {kb_id: $kb_id})-[r:RELATED]->(:Entity {kb_id: $kb_id})
        RETURN r.predicate AS name, count(*) AS count
        ORDER BY count DESC, name
        """,
        {"kb_id": kb_id},
    )
    return GraphFacets(
        types=[{"name": t["name"], "count": int(t["count"])} for t in types],
        # Long tail of one-off predicates is noise in a dropdown; keep the top.
        predicates=[{"name": p["name"], "count": int(p["count"])} for p in predicates[:80]],
        total_entities=sum(int(t["count"]) for t in types),
        total_relations=sum(int(p["count"]) for p in predicates),
    )


# ------------------------------------------------------------------- entities
async def list_entities(
    kb_id: str,
    *,
    q: str | None,
    entity_type: str | None,
    sort: str,
    limit: int,
    offset: int,
) -> EntityListResponse:
    if not is_available():
        return EntityListResponse(items=[], total=0, limit=limit, offset=offset, available=False)

    name_term, desc_term = _search_terms(q)
    params = {
        "kb_id": kb_id,
        "type": entity_type or None,
        "name_term": name_term,
        "desc_term": desc_term,
        "limit": limit,
        "offset": offset,
    }
    where = """
        WHERE ($type IS NULL OR e.type = $type)
          AND ($name_term IS NULL
               OR e.norm_name CONTAINS $name_term
               OR toLower(coalesce(e.description, '')) CONTAINS $desc_term)
    """

    rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        %s
        WITH e, %s AS degree
        RETURN e.norm_name AS id, e.name AS name, e.type AS type,
               e.description AS description,
               coalesce(e.mentions, 0) AS mentions, degree
        ORDER BY %s
        SKIP $offset LIMIT $limit
        """
        % (where, _DEGREE, _ENTITY_SORTS.get(sort, _ENTITY_SORTS["degree"])),
        params,
    )
    counted = await run_read(
        "MATCH (e:Entity {kb_id: $kb_id}) %s RETURN count(e) AS total" % where, params
    )
    total = int(counted[0]["total"]) if counted else 0
    return EntityListResponse(
        items=[_entity(r) for r in rows], total=total, limit=limit, offset=offset
    )


_RELATION_FIELDS = """
    elementId(r) AS id, r.predicate AS predicate, r.description AS description,
    coalesce(r.weight, 1) AS weight, coalesce(r.chunk_ids, []) AS chunk_ids,
    a.norm_name AS source_id, a.name AS source_name, a.type AS source_type,
    b.norm_name AS target_id, b.name AS target_name, b.type AS target_type
"""


async def entity_detail(kb_id: str, entity_id: str, *, limit: int = 200) -> EntityDetail | None:
    if not is_available():
        return None

    found = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id, norm_name: $id})
        RETURN e.norm_name AS id, e.name AS name, e.type AS type,
               e.description AS description,
               coalesce(e.mentions, 0) AS mentions, %s AS degree
        """
        % _DEGREE,
        {"kb_id": kb_id, "id": entity_id},
    )
    if not found:
        return None

    params = {"kb_id": kb_id, "id": entity_id, "limit": limit}
    outgoing = await run_read(
        """
        MATCH (a:Entity {kb_id: $kb_id, norm_name: $id})-[r:RELATED]->(b:Entity {kb_id: $kb_id})
        RETURN %s
        ORDER BY weight DESC, predicate, target_name
        LIMIT $limit
        """
        % _RELATION_FIELDS,
        params,
    )
    incoming = await run_read(
        """
        MATCH (a:Entity {kb_id: $kb_id})-[r:RELATED]->(b:Entity {kb_id: $kb_id, norm_name: $id})
        RETURN %s
        ORDER BY weight DESC, predicate, source_name
        LIMIT $limit
        """
        % _RELATION_FIELDS,
        params,
    )
    mentions = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id, norm_name: $id})-[m:MENTIONED_IN]->(c:Chunk)
        RETURN c.id AS chunk_id, c.filename AS filename, c.page_no AS page_no,
               c.snippet AS snippet, coalesce(m.count, 1) AS count
        ORDER BY count DESC, filename, page_no
        LIMIT 50
        """,
        {"kb_id": kb_id, "id": entity_id},
    )

    return EntityDetail(
        entity=_entity(found[0]),
        outgoing=[_relation(r) for r in outgoing],
        incoming=[_relation(r) for r in incoming],
        mentions=[
            MentionOut(
                chunk_id=m["chunk_id"],
                filename=m.get("filename"),
                page_no=m.get("page_no"),
                snippet=m.get("snippet"),
                count=int(m.get("count") or 1),
            )
            for m in mentions
        ],
    )


# ------------------------------------------------------------------ relations
async def list_relations(
    kb_id: str,
    *,
    q: str | None,
    predicate: str | None,
    limit: int,
    offset: int,
) -> RelationListResponse:
    if not is_available():
        return RelationListResponse(items=[], total=0, limit=limit, offset=offset, available=False)

    name_term, raw_term = _search_terms(q)
    params = {
        "kb_id": kb_id,
        "predicate": predicate or None,
        "name_term": name_term,
        # Predicates are stored as WORKS_FOR; let "works for" find them.
        "pred_term": raw_term.replace(" ", "_") if raw_term else None,
        "limit": limit,
        "offset": offset,
    }
    match = """
        MATCH (a:Entity {kb_id: $kb_id})-[r:RELATED]->(b:Entity {kb_id: $kb_id})
        WHERE ($predicate IS NULL OR r.predicate = $predicate)
          AND ($name_term IS NULL
               OR a.norm_name CONTAINS $name_term
               OR b.norm_name CONTAINS $name_term
               OR toLower(r.predicate) CONTAINS $pred_term)
    """
    rows = await run_read(
        match
        + """
        RETURN %s
        ORDER BY weight DESC, source_name, predicate
        SKIP $offset LIMIT $limit
        """
        % _RELATION_FIELDS,
        params,
    )
    counted = await run_read(match + " RETURN count(r) AS total", params)
    total = int(counted[0]["total"]) if counted else 0
    return RelationListResponse(
        items=[_relation(r) for r in rows], total=total, limit=limit, offset=offset
    )


# ---------------------------------------------------------------- graph views
async def _materialise(
    kb_id: str, ids: list[str], depths: dict[str, int] | None, center: str | None, truncated: bool
) -> GraphView:
    """Nodes for `ids`, plus every relation whose both ends are among them."""
    if not ids:
        return GraphView(nodes=[], edges=[], center=center, truncated=truncated)

    node_rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        WHERE e.norm_name IN $ids
        RETURN e.norm_name AS id, e.name AS name, e.type AS type,
               coalesce(e.mentions, 0) AS mentions, %s AS degree
        """
        % _DEGREE,
        {"kb_id": kb_id, "ids": ids},
    )
    edge_rows = await run_read(
        """
        MATCH (a:Entity {kb_id: $kb_id})-[r:RELATED]->(b:Entity {kb_id: $kb_id})
        WHERE a.norm_name IN $ids AND b.norm_name IN $ids
        RETURN elementId(r) AS id, a.norm_name AS source, b.norm_name AS target,
               r.predicate AS predicate, coalesce(r.weight, 1) AS weight
        ORDER BY weight DESC
        LIMIT $edge_limit
        """,
        {"kb_id": kb_id, "ids": ids, "edge_limit": MAX_GRAPH_EDGES},
    )
    return GraphView(
        nodes=[
            GraphNode(
                id=n["id"],
                name=n.get("name") or n["id"],
                type=n.get("type") or "OTHER",
                mentions=int(n.get("mentions") or 0),
                degree=int(n.get("degree") or 0),
                depth=(depths or {}).get(n["id"]),
            )
            for n in node_rows
        ],
        edges=[
            GraphEdge(
                id=e["id"],
                source=e["source"],
                target=e["target"],
                predicate=e.get("predicate") or "RELATED_TO",
                weight=int(e.get("weight") or 1),
            )
            for e in edge_rows
        ],
        center=center,
        truncated=truncated or len(edge_rows) >= MAX_GRAPH_EDGES,
    )


async def overview(kb_id: str, *, limit: int) -> GraphView:
    """The most connected entities -- a starting view that is not a hairball."""
    if not is_available():
        return GraphView(nodes=[], edges=[], available=False)
    limit = min(limit, MAX_GRAPH_NODES)
    rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        WITH e, %s AS degree
        WHERE degree > 0
        RETURN e.norm_name AS id
        ORDER BY degree DESC, e.mentions DESC
        LIMIT $limit
        """
        % _DEGREE,
        {"kb_id": kb_id, "limit": limit + 1},
    )
    ids = [r["id"] for r in rows]
    return await _materialise(kb_id, ids[:limit], None, None, truncated=len(ids) > limit)


async def neighbourhood(kb_id: str, entity_id: str, *, hops: int, limit: int) -> GraphView | None:
    """An entity plus up to `limit` neighbours, breadth-first, strongest first."""
    if not is_available():
        return GraphView(nodes=[], edges=[], available=False)

    exists = await run_read(
        "MATCH (e:Entity {kb_id: $kb_id, norm_name: $id}) RETURN e.norm_name AS id",
        {"kb_id": kb_id, "id": entity_id},
    )
    if not exists:
        return None

    limit = min(limit, MAX_GRAPH_NODES)
    depths: dict[str, int] = {entity_id: 0}
    frontier = [entity_id]
    truncated = False

    for depth in range(1, hops + 1):
        budget = limit - (len(depths) - 1)
        if budget <= 0 or not frontier:
            truncated = truncated or bool(frontier)
            break
        rows = await run_read(
            """
            MATCH (a:Entity {kb_id: $kb_id})-[r:RELATED]-(n:Entity {kb_id: $kb_id})
            WHERE a.norm_name IN $frontier AND NOT n.norm_name IN $seen
            WITH n, sum(coalesce(r.weight, 1)) AS strength
            RETURN n.norm_name AS id
            ORDER BY strength DESC, coalesce(n.mentions, 0) DESC
            LIMIT $budget
            """,
            {
                "kb_id": kb_id,
                "frontier": frontier,
                "seen": list(depths),
                "budget": budget + 1,  # one extra reveals whether we cut anything
            },
        )
        found = [r["id"] for r in rows]
        if len(found) > budget:
            truncated = True
            found = found[:budget]
        for node_id in found:
            depths[node_id] = depth
        frontier = found

    return await _materialise(kb_id, list(depths), depths, entity_id, truncated)
