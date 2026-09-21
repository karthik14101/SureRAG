"""Graph traversal: question -> seed entities -> k-hop subgraph -> evidence chunks.

This is what vector search cannot do. "Which suppliers are affected by the
Q3 recall?" needs edges followed, not passages matched, because no single
passage states the answer -- it is spread across the relationships.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.graph.extractor import normalise_name
from app.graph.neo4j_client import is_available, run_read
from app.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass
class GraphPath:
    source: str
    predicate: str
    target: str
    description: str = ""
    weight: int = 1
    chunk_ids: list[str] = field(default_factory=list)

    def as_sentence(self) -> str:
        verb = self.predicate.replace("_", " ").lower()
        base = "{} {} {}".format(self.source, verb, self.target)
        return base + (". " + self.description if self.description else ".")


@dataclass
class GraphContext:
    seeds: list[str]
    paths: list[GraphPath]
    chunk_ids: list[str]
    entity_summaries: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.paths and not self.chunk_ids

    def as_text(self, limit: int = 40) -> str:
        """Render the subgraph as facts the LLM can read."""
        if not self.paths:
            return ""
        lines = [p.as_sentence() for p in self.paths[:limit]]
        return "Knowledge graph facts:\n" + "\n".join("- " + line for line in lines)


# Neo4j's fulltext query language treats these as operators.
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}")
_QUESTION_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "does", "did", "do", "is", "are", "was", "were", "the", "and", "for",
    "with", "from", "that", "this", "these", "those", "about", "into", "than",
    "then", "there", "their", "they", "have", "has", "had", "been", "being",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "tell", "show", "give", "list", "explain", "describe", "compare", "between",
    "related", "relationship", "connection", "affect", "affected", "impact",
}


def _escape_lucene(term: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", term)


def build_fulltext_query(question: str) -> str:
    """Turn a question into a fuzzy OR query over entity names."""
    terms = []
    for word in _WORD.findall(question):
        if word.casefold() in _QUESTION_STOPWORDS:
            continue
        escaped = _escape_lucene(word)
        # ~1 edit distance absorbs plurals and minor spelling drift.
        terms.append("{}~1".format(escaped) if len(escaped) > 4 else escaped)
    return " OR ".join(terms[:12])


async def find_seed_entities(question: str, kb_id: str, limit: int | None = None) -> list[dict]:
    """Locate the entities a question is about, fulltext first then exact match."""
    if not is_available():
        return []
    cap = limit or settings.graph_seed_entities

    query = build_fulltext_query(question)
    seeds: list[dict] = []

    if query:
        seeds = await run_read(
            """
            CALL db.index.fulltext.queryNodes('entity_fulltext', $query)
            YIELD node, score
            WHERE node.kb_id = $kb_id
            RETURN node.norm_name AS norm_name,
                   node.name AS name,
                   node.type AS type,
                   node.description AS description,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query": query, "kb_id": kb_id, "limit": cap},
        )

    if not seeds:
        # Fulltext index may be missing or the query too exotic: fall back to
        # substring matching on the normalised names.
        candidates = [
            normalise_name(w)
            for w in _WORD.findall(question)
            if w.casefold() not in _QUESTION_STOPWORDS
        ]
        candidates = [c for c in candidates if len(c) > 2][:8]
        if candidates:
            seeds = await run_read(
                """
                MATCH (e:Entity {kb_id: $kb_id})
                WHERE any(term IN $terms WHERE e.norm_name CONTAINS term)
                RETURN e.norm_name AS norm_name,
                       e.name AS name,
                       e.type AS type,
                       e.description AS description,
                       coalesce(e.mentions, 1) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"kb_id": kb_id, "terms": candidates, "limit": cap},
            )

    return seeds


async def traverse(question: str, kb_id: str, max_hops: int | None = None) -> GraphContext:
    """Expand from seed entities across up to `max_hops` relationships."""
    if not is_available():
        return GraphContext([], [], [], [])

    hops = max(1, min(3, max_hops or settings.graph_max_hops))
    seeds = await find_seed_entities(question, kb_id)
    if not seeds:
        return GraphContext([], [], [], [])

    seed_names = [s["norm_name"] for s in seeds if s.get("norm_name")]
    entity_summaries = [
        "{} ({}): {}".format(
            s.get("name") or s["norm_name"],
            s.get("type") or "OTHER",
            (s.get("description") or "").strip() or "no description",
        )
        for s in seeds
    ]

    # Variable-length pattern; the upper bound is interpolated because Cypher
    # does not accept a parameter there. `hops` is clamped to 1..3 above, so no
    # user input ever reaches the query string.
    rows = await run_read(
        """
        MATCH (start:Entity {kb_id: $kb_id})
        WHERE start.norm_name IN $seeds
        MATCH path = (start)-[:RELATED*1..%d]-(other:Entity {kb_id: $kb_id})
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH DISTINCT r, startNode(r) AS a, endNode(r) AS b
        RETURN a.name AS source,
               b.name AS target,
               r.predicate AS predicate,
               coalesce(r.description, '') AS description,
               coalesce(r.weight, 1) AS weight,
               coalesce(r.chunk_ids, []) AS chunk_ids
        ORDER BY weight DESC
        LIMIT 80
        """
        % hops,
        {"kb_id": kb_id, "seeds": seed_names},
    )

    paths: list[GraphPath] = []
    chunk_ids: list[str] = []
    for row in rows:
        if not row.get("source") or not row.get("target"):
            continue
        path = GraphPath(
            source=row["source"],
            predicate=row.get("predicate") or "RELATED_TO",
            target=row["target"],
            description=row.get("description") or "",
            weight=int(row.get("weight") or 1),
            chunk_ids=list(row.get("chunk_ids") or []),
        )
        paths.append(path)
        chunk_ids.extend(path.chunk_ids)

    # Entities found but no edges yet: still surface the chunks mentioning them.
    if not chunk_ids:
        direct = await run_read(
            """
            MATCH (e:Entity {kb_id: $kb_id})-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.norm_name IN $seeds
            RETURN c.id AS chunk_id
            LIMIT 20
            """,
            {"kb_id": kb_id, "seeds": seed_names},
        )
        chunk_ids = [r["chunk_id"] for r in direct if r.get("chunk_id")]
    else:
        mentions = await run_read(
            """
            MATCH (e:Entity {kb_id: $kb_id})-[:MENTIONED_IN]->(c:Chunk)
            WHERE e.norm_name IN $seeds
            RETURN c.id AS chunk_id
            LIMIT 12
            """,
            {"kb_id": kb_id, "seeds": seed_names},
        )
        chunk_ids.extend(r["chunk_id"] for r in mentions if r.get("chunk_id"))

    unique_chunks = list(dict.fromkeys(chunk_ids))[:24]

    return GraphContext(
        seeds=[s.get("name") or s["norm_name"] for s in seeds],
        paths=paths,
        chunk_ids=unique_chunks,
        entity_summaries=entity_summaries,
    )


async def neighbourhood_chunks(entity_names: list[str], kb_id: str, limit: int = 15) -> list[str]:
    """Widen an existing traversal by one hop. Used by the expansion node."""
    if not is_available() or not entity_names:
        return []
    norms = [normalise_name(n) for n in entity_names if n]
    rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})-[:RELATED]-(n:Entity {kb_id: $kb_id})
        WHERE e.norm_name IN $norms
        MATCH (n)-[:MENTIONED_IN]->(c:Chunk)
        RETURN DISTINCT c.id AS chunk_id
        LIMIT $limit
        """,
        {"kb_id": kb_id, "norms": norms, "limit": limit},
    )
    return [r["chunk_id"] for r in rows if r.get("chunk_id")]
