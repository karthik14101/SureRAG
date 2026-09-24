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
    # Prepositions and connectives. They carry no entity meaning, but they do
    # occupy slots in the fulltext query and can look "distinctive" to the seed
    # filter simply because only one candidate happens to contain them.
    "under", "upon", "within", "without", "through", "during", "against",
    "above", "below", "across", "among", "along", "toward", "towards", "made",
}


# An entity whose name begins with a demonstrative or quantifier is not a thing
# in the world, it is a back-reference: "this Constitution", "any State", "such
# State", "the said article". Extraction produces plenty of them, they are
# mentioned everywhere, and as a seed each one drags in hundreds of unrelated
# relationships. They are still fine as neighbours -- this only stops them
# being the starting point.
_BACKREFERENCE = {
    "this", "that", "these", "those", "such", "any", "said", "other", "certain",
    "all", "some", "every", "each", "either", "both", "another", "same",
}


def _is_backreference(norm_name: str) -> bool:
    parts = norm_name.split()
    return len(parts) > 1 and parts[0] in _BACKREFERENCE


# A "hub" is an entity the corpus mentions everywhere: Parliament, President,
# CONSTITUTION. Being named in the question does not make one a useful starting
# point -- a question about the Constitution of India, asked of a corpus that is
# the Constitution of India, is not about the entity "CONSTITUTION". Seeding on
# one pulls its whole neighbourhood and crowds out the entities that actually
# distinguish the question. The threshold is measured against the candidates
# rather than fixed, so it scales with corpus size instead of guessing at it.
_HUB_FLOOR = 10
_HUB_MULTIPLE = 8


def _hub_threshold(seeds: list[dict]) -> float:
    counts = sorted(int(seed.get("mentions") or 1) for seed in seeds)
    if not counts:
        return float("inf")
    median = counts[len(counts) // 2]
    return max(_HUB_FLOOR, median * _HUB_MULTIPLE)


def _squash(norm_name: str) -> str:
    """Key that collapses near-duplicate entities: VicePresident vs Vice-President."""
    return re.sub(r"[^a-z0-9]", "", norm_name.casefold())


def _filter_seeds(question: str, seeds: list[dict]) -> list[dict]:
    """Keep the seeds that actually distinguish this question from any other.

    The fulltext index matches on any term, so a question about "Article 368 and
    the Seventh Schedule" seeds every entity containing the word "article" --
    article 5, article 292, and so on -- which then traverse into hundreds of
    irrelevant relationships. The useful seeds are the ones matched on a term
    that most candidates do NOT share, so shared-term frequency is measured
    across the candidate set itself rather than against a fixed word list. That
    calibrates to each corpus: "article" is generic here, "schedule" is not.
    """
    if not seeds:
        return seeds

    asked = {
        w.casefold()
        for w in _WORD.findall(question)
        if w.casefold() not in _QUESTION_STOPWORDS
    }

    overlaps: list[set[str]] = []
    frequency: dict[str, int] = {}
    for seed in seeds:
        shared = {t for t in (seed.get("norm_name") or "").split() if t in asked}
        overlaps.append(shared)
        for token in shared:
            frequency[token] = frequency.get(token, 0) + 1

    lowered = question.casefold()
    hub_at = _hub_threshold(seeds)
    strong: list[dict] = []
    weak: list[dict] = []
    seen: set[str] = set()

    for seed, shared in zip(seeds, overlaps):
        norm = seed.get("norm_name") or ""
        key = _squash(norm)
        if not key or key in seen:
            continue  # a near-duplicate of one already kept
        seen.add(key)

        if _is_backreference(norm):
            continue  # never a useful starting point, however it was ranked

        # A term that names three or more different candidates does not say
        # which of them the question is about: "article" matches article 5,
        # article 292 and article 368 alike, while "368" matches one.
        distinctive = norm in lowered or any(frequency[t] < 3 for t in shared)
        # A hub is demoted even when the question names it outright, which is
        # the case that matters: the question said "Constitution", and that is
        # exactly why the literal-name test used to promote the one entity in
        # the corpus that tells us nothing.
        if int(seed.get("mentions") or 1) >= hub_at:
            distinctive = False
        (strong if distinctive else weak).append(seed)

    # Drop the vague matches only when enough precise ones survive; otherwise a
    # question whose every match is vague would lose its graph entirely.
    if len(strong) >= 2:
        return strong
    return (strong + weak) or seeds[:2]


# Per-knowledge-base hub cutoff, computed once and kept for the process. The
# value only shifts as documents are ingested, and a stale cutoff merely lets a
# borderline entity through.
_hub_cutoffs: dict[str, float] = {}


async def kb_hub_cutoff(kb_id: str) -> float:
    """The mention count above which an entity is corpus furniture.

    Taken from the corpus rather than hard-coded, because "mentioned a lot"
    only means anything relative to the rest of the knowledge base. On the
    Constitution of India the 99th percentile lands at 26, which separates
    Parliament, President, State and CONSTITUTION from every entity that
    actually distinguishes one question from another.
    """
    cached = _hub_cutoffs.get(kb_id)
    if cached is not None:
        return cached
    rows = await run_read(
        """
        MATCH (e:Entity {kb_id: $kb_id})
        RETURN percentileCont(coalesce(e.mentions, 1), 0.99) AS p99
        """,
        {"kb_id": kb_id},
    )
    p99 = float(rows[0].get("p99") or 0) if rows else 0.0
    cutoff = max(float(_HUB_FLOOR), p99)
    _hub_cutoffs[kb_id] = cutoff
    return cutoff


def _escape_lucene(term: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", term)


MAX_QUERY_TERMS = 16


def _term_priority(word: str) -> tuple[int, int]:
    """Rank a question word by how likely it is to name a specific entity.

    Sorting matters because the query is capped: a question long enough to
    overflow the cap would otherwise lose its final words, and the words that
    pin a question down often arrive last. "What is the structural link between
    the procedure for amending the Constitution under Article 368 and the
    subjects listed in the Seventh Schedule?" put "Seventh" and "Schedule" in
    positions thirteen and fourteen, so a reading-order cap dropped the one
    phrase that identified half the question.

    No corpus statistics are needed for the ordering that matters here: a token
    containing a digit is almost always an identifier, and a capitalised word
    mid-sentence is almost always a proper noun.
    """
    has_digit = any(c.isdigit() for c in word)
    proper = word[:1].isupper() and not word.isupper()
    rank = 0 if has_digit else (1 if proper else 2)
    return (rank, -len(word))


def build_fulltext_query(question: str) -> str:
    """Turn a question into a fuzzy OR query over entity names."""
    seen: set[str] = set()
    words: list[str] = []
    for word in _WORD.findall(question):
        folded = word.casefold()
        # Repeats waste cap slots without adding recall: Lucene already scores a
        # term once however often the question says it.
        if folded in _QUESTION_STOPWORDS or folded in seen:
            continue
        seen.add(folded)
        words.append(word)

    terms = []
    for word in sorted(words, key=_term_priority)[:MAX_QUERY_TERMS]:
        escaped = _escape_lucene(word)
        # ~1 edit distance absorbs plurals and minor spelling drift.
        terms.append("{}~1".format(escaped) if len(escaped) > 4 else escaped)
    return " OR ".join(terms)


async def find_seed_entities(question: str, kb_id: str, limit: int | None = None) -> list[dict]:
    """Locate the entities a question is about, fulltext first then exact match."""
    if not is_available():
        return []
    cap = limit or settings.graph_seed_entities
    # Over-fetch: filtering happens after the database has ranked candidates, and
    # asking for exactly `cap` would leave nothing once the generic ones go.
    pool = cap * 4

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
                   coalesce(node.mentions, 1) AS mentions,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query": query, "kb_id": kb_id, "limit": pool},
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
                       coalesce(e.mentions, 1) AS mentions,
                       coalesce(e.mentions, 1) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"kb_id": kb_id, "terms": candidates, "limit": pool},
            )

    return _filter_seeds(question, seeds)[:cap]


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
    hub = await kb_hub_cutoff(kb_id)

    # Two hub guards, and they do different jobs. The first stops the walk from
    # travelling THROUGH furniture: everything in this corpus connects to
    # "State", so a two-hop path via State reaches every entity there is, which
    # is the same as reaching none of them. The second drops edges between two
    # hubs, which are true but say nothing -- "Legislature of a State part of
    # State" is not evidence. An edge from a real entity to a hub survives.
    rows = await run_read(
        """
        MATCH (start:Entity {kb_id: $kb_id})
        WHERE start.norm_name IN $seeds
        MATCH path = (start)-[:RELATED*1..%d]-(other:Entity {kb_id: $kb_id})
        WHERE all(mid IN nodes(path)[1..-1]
                  WHERE coalesce(mid.mentions, 1) < $hub)
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH DISTINCT r, startNode(r) AS a, endNode(r) AS b
        WHERE coalesce(a.mentions, 1) < $hub OR coalesce(b.mentions, 1) < $hub
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
        {"kb_id": kb_id, "seeds": seed_names, "hub": hub},
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
