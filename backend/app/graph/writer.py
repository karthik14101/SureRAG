"""Idempotent writes into Neo4j.

Everything is MERGE-based and keyed on (kb_id, norm_name), so re-ingesting a
document updates the graph in place instead of duplicating it. Relationship
predicates are stored as a property on a single :RELATED type rather than as
dynamic relationship types -- dynamic types would need string-built Cypher,
which is both an injection risk and a schema explosion.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.graph.extractor import Extraction, normalise_name
from app.graph.neo4j_client import is_available, run_read, run_write
from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkNode:
    chunk_id: str
    doc_id: str
    kb_id: str
    user_id: str
    filename: str
    page_no: int | None
    snippet: str


UPSERT_DOCUMENT = """
MERGE (d:Document {id: $doc_id})
SET d.kb_id = $kb_id, d.user_id = $user_id, d.filename = $filename
"""

UPSERT_CHUNK = """
MERGE (c:Chunk {id: $chunk_id})
SET c.kb_id = $kb_id,
    c.user_id = $user_id,
    c.doc_id = $doc_id,
    c.filename = $filename,
    c.page_no = $page_no,
    c.snippet = $snippet
WITH c
MATCH (d:Document {id: $doc_id})
MERGE (c)-[:PART_OF]->(d)
"""

UPSERT_ENTITIES = """
UNWIND $entities AS row
MERGE (e:Entity {kb_id: $kb_id, norm_name: row.norm_name})
ON CREATE SET e.name = row.name,
              e.type = row.type,
              e.description = row.description,
              e.user_id = $user_id,
              e.mentions = 1
ON MATCH SET  e.mentions = coalesce(e.mentions, 0) + 1,
              e.description = CASE
                  WHEN e.description IS NULL OR e.description = ''
                  THEN row.description ELSE e.description END,
              e.type = CASE WHEN e.type = 'OTHER' THEN row.type ELSE e.type END
WITH e, row
MATCH (c:Chunk {id: $chunk_id})
MERGE (e)-[m:MENTIONED_IN]->(c)
ON CREATE SET m.count = 1
ON MATCH SET  m.count = coalesce(m.count, 0) + 1
"""

UPSERT_RELATIONS = """
UNWIND $relations AS row
MATCH (a:Entity {kb_id: $kb_id, norm_name: row.source_norm})
MATCH (b:Entity {kb_id: $kb_id, norm_name: row.target_norm})
MERGE (a)-[r:RELATED {predicate: row.predicate}]->(b)
ON CREATE SET r.description = row.description,
              r.weight = 1,
              r.chunk_ids = [$chunk_id]
ON MATCH SET  r.weight = coalesce(r.weight, 0) + 1,
              r.chunk_ids = CASE
                  WHEN $chunk_id IN coalesce(r.chunk_ids, [])
                  THEN r.chunk_ids
                  ELSE coalesce(r.chunk_ids, []) + $chunk_id END
"""


async def write_extraction(node: ChunkNode, extraction: Extraction) -> int:
    """Persist one chunk plus its entities/relations. Returns entities written."""
    if not is_available() or settings.graph_extraction == "off":
        return 0

    await run_write(
        UPSERT_DOCUMENT,
        {
            "doc_id": node.doc_id,
            "kb_id": node.kb_id,
            "user_id": node.user_id,
            "filename": node.filename,
        },
    )
    await run_write(
        UPSERT_CHUNK,
        {
            "chunk_id": node.chunk_id,
            "doc_id": node.doc_id,
            "kb_id": node.kb_id,
            "user_id": node.user_id,
            "filename": node.filename,
            "page_no": node.page_no,
            "snippet": node.snippet[:500],
        },
    )

    if extraction.is_empty:
        return 0

    entity_rows = []
    for entity in extraction.entities:
        norm = normalise_name(entity.name)
        if not norm:
            continue
        entity_rows.append(
            {
                "norm_name": norm,
                "name": entity.name,
                "type": entity.type,
                "description": entity.description,
            }
        )

    if entity_rows:
        await run_write(
            UPSERT_ENTITIES,
            {
                "entities": entity_rows,
                "kb_id": node.kb_id,
                "user_id": node.user_id,
                "chunk_id": node.chunk_id,
            },
        )

    relation_rows = []
    for relation in extraction.relations:
        source_norm = normalise_name(relation.source)
        target_norm = normalise_name(relation.target)
        if not source_norm or not target_norm or source_norm == target_norm:
            continue
        relation_rows.append(
            {
                "source_norm": source_norm,
                "target_norm": target_norm,
                "predicate": relation.predicate,
                "description": relation.description,
            }
        )

    if relation_rows:
        await run_write(
            UPSERT_RELATIONS,
            {
                "relations": relation_rows,
                "kb_id": node.kb_id,
                "chunk_id": node.chunk_id,
            },
        )

    return len(entity_rows)


async def delete_document_graph(doc_id: str, kb_id: str) -> None:
    """Remove a document's chunks, then sweep entities left with no mentions."""
    if not is_available():
        return
    await run_write(
        """
        MATCH (c:Chunk {doc_id: $doc_id, kb_id: $kb_id})
        DETACH DELETE c
        """,
        {"doc_id": doc_id, "kb_id": kb_id},
    )
    await run_write(
        "MATCH (d:Document {id: $doc_id}) DETACH DELETE d",
        {"doc_id": doc_id},
    )
    await run_write(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)
        DETACH DELETE e
        """,
        {"kb_id": kb_id},
    )


async def delete_kb_graph(kb_id: str) -> None:
    if not is_available():
        return
    for label in ("Chunk", "Document", "Entity"):
        await run_write(
            "MATCH (n:{} {{kb_id: $kb_id}}) DETACH DELETE n".format(label),
            {"kb_id": kb_id},
        )


async def graph_stats(kb_id: str) -> dict:
    if not is_available():
        return {"entities": 0, "relations": 0, "chunks": 0, "available": False}
    rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        WITH count(e) AS entities
        OPTIONAL MATCH (:Entity {kb_id: $kb_id})-[r:RELATED]->(:Entity {kb_id: $kb_id})
        WITH entities, count(r) AS relations
        OPTIONAL MATCH (c:Chunk {kb_id: $kb_id})
        RETURN entities, relations, count(c) AS chunks
        """,
        {"kb_id": kb_id},
    )
    if not rows:
        return {"entities": 0, "relations": 0, "chunks": 0, "available": True}
    row = rows[0]
    return {
        "entities": int(row.get("entities") or 0),
        "relations": int(row.get("relations") or 0),
        "chunks": int(row.get("chunks") or 0),
        "available": True,
    }
