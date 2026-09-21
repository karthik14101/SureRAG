"""Entity and relationship extraction for the knowledge graph.

Three modes, set by GRAPH_EXTRACTION in .env:
  on        -- one LLM call per chunk (richest graph, slowest, most quota)
  selective -- only entity-dense chunks get an LLM call (default)
  off       -- no graph is built

A SHA-256 cache means re-ingesting the same text is free, and if the LLM is
unavailable a zero-cost heuristic extractor keeps ingestion moving.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass

from app.config import settings
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

MAX_CACHE_ENTRIES = 4000
_cache: dict[str, "Extraction"] = {}

ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LOCATION",
    "PRODUCT",
    "TECHNOLOGY",
    "CONCEPT",
    "EVENT",
    "METRIC",
    "DOCUMENT",
    "OTHER",
)

SYSTEM_PROMPT = (
    "You extract a knowledge graph from text. Return ONLY a JSON object, no prose.\n"
    "Schema:\n"
    '{"entities":[{"name":"...","type":"ONE_OF_TYPES","description":"one short line"}],'
    '"relations":[{"source":"entity name","predicate":"SHORT_VERB_PHRASE",'
    '"target":"entity name","description":"one short line"}]}\n'
    "Rules:\n"
    "- Types must be one of: " + ", ".join(ENTITY_TYPES) + "\n"
    "- Use the exact surface name from the text; do not invent entities.\n"
    "- Every relation's source and target MUST appear in the entities array.\n"
    "- Predicates are uppercase with underscores, e.g. WORKS_FOR, PART_OF, CAUSES.\n"
    "- Extract at most 12 entities and 12 relations. Skip generic filler terms.\n"
    "- If the text carries no meaningful entities, return empty arrays."
)

_CAP_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9_-]{2,}\b")
_ACRONYM = re.compile(r"\b[A-Z]{2,6}\b")
_STOPWORDS = {
    "The", "This", "That", "These", "Those", "There", "Then", "They", "It", "Its",
    "We", "You", "He", "She", "His", "Her", "Their", "Our", "And", "But", "For",
    "With", "From", "Into", "Also", "However", "Therefore", "Figure", "Table",
    "Chapter", "Section", "Page", "Note", "Example", "Abstract", "Introduction",
    "Conclusion", "References", "Appendix", "January", "February", "March",
    "April", "May", "June", "July", "August", "September", "October", "November",
    "December", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
}


@dataclass
class ExtractedEntity:
    name: str
    type: str
    description: str = ""


@dataclass
class ExtractedRelation:
    source: str
    predicate: str
    target: str
    description: str = ""


@dataclass
class Extraction:
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]
    method: str = "llm"

    @property
    def is_empty(self) -> bool:
        return not self.entities and not self.relations


def normalise_name(name: str) -> str:
    """Key used for entity resolution inside a knowledge base."""
    cleaned = re.sub(r"[^\w\s-]", "", (name or "").strip().casefold())
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Drop a leading article so "The Acme Corp" resolves onto "Acme Corp".
    for article in ("the ", "a ", "an "):
        if cleaned.startswith(article):
            cleaned = cleaned[len(article) :]
    return cleaned.strip()


def entity_density(text: str) -> float:
    """Proxy for "is this chunk worth an LLM call" in selective mode."""
    words = text.split()
    if len(words) < 20:
        return 0.0
    candidates = {
        token
        for token in _CAP_TOKEN.findall(text) + _ACRONYM.findall(text)
        if token not in _STOPWORDS
    }
    return len(candidates) / max(1, len(words) / 100.0)


def should_extract(text: str) -> bool:
    mode = settings.graph_extraction
    if mode == "off":
        return False
    if mode == "on":
        return len(text.split()) >= 15
    # selective: roughly 4+ distinct proper nouns per 100 words
    return entity_density(text) >= 4.0


def heuristic_extract(text: str) -> Extraction:
    """Zero-LLM fallback: proper nouns become entities, co-occurrence a relation."""
    seen: dict[str, str] = {}
    for token in _CAP_TOKEN.findall(text):
        if token in _STOPWORDS or len(token) < 3:
            continue
        key = normalise_name(token)
        if key and key not in seen:
            seen[key] = token
    for token in _ACRONYM.findall(text):
        if token in _STOPWORDS:
            continue
        key = normalise_name(token)
        if key and key not in seen:
            seen[key] = token

    names = list(seen.values())[:10]
    entities = [ExtractedEntity(name=n, type="OTHER", description="") for n in names]
    relations: list[ExtractedRelation] = []
    # Link consecutive entities as co-mentions: weak signal, but it makes the
    # graph traversable instead of a bag of disconnected nodes.
    for left, right in zip(names, names[1:]):
        relations.append(
            ExtractedRelation(
                source=left,
                predicate="CO_MENTIONED_WITH",
                target=right,
                description="Appear together in the same passage.",
            )
        )
    return Extraction(entities=entities, relations=relations[:8], method="heuristic")


def _parse(payload: dict) -> Extraction:
    raw_entities = payload.get("entities") or []
    raw_relations = payload.get("relations") or []

    entities: list[ExtractedEntity] = []
    known: set[str] = set()
    for item in raw_entities[:12]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or len(name) > 120:
            continue
        etype = str(item.get("type") or "OTHER").strip().upper()
        if etype not in ENTITY_TYPES:
            etype = "OTHER"
        entities.append(
            ExtractedEntity(
                name=name,
                type=etype,
                description=str(item.get("description") or "").strip()[:300],
            )
        )
        known.add(normalise_name(name))

    relations: list[ExtractedRelation] = []
    for item in raw_relations[:12]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        predicate = str(item.get("predicate") or "RELATED_TO").strip().upper()
        predicate = re.sub(r"[^A-Z0-9_]+", "_", predicate).strip("_") or "RELATED_TO"
        if not source or not target or source == target:
            continue
        # Models hallucinate endpoints that were never declared as entities.
        if normalise_name(source) not in known or normalise_name(target) not in known:
            continue
        relations.append(
            ExtractedRelation(
                source=source,
                predicate=predicate[:60],
                target=target,
                description=str(item.get("description") or "").strip()[:300],
            )
        )

    return Extraction(entities=entities, relations=relations, method="llm")


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


async def extract_from_chunk(text: str) -> Extraction:
    """Extract one chunk's graph, using the cache and falling back gracefully."""
    if not text.strip():
        return Extraction([], [], method="skipped")

    key = _cache_key(text)
    cached = _cache.get(key)
    if cached is not None:
        return cached

    try:
        llm = get_llm()
        prompt = "Extract the knowledge graph from this text:\n\n" + text[:6000]
        payload, _usage = await llm.complete_json(
            SYSTEM_PROMPT, prompt, temperature=0.0, max_tokens=1200
        )
        extraction = _parse(payload)
        if extraction.is_empty and settings.graph_extraction == "on":
            extraction = heuristic_extract(text)
    except Exception as exc:  # noqa: BLE001 - ingestion must never hard-fail here
        logger.warning("LLM graph extraction failed, using heuristics: %s", str(exc)[:160])
        extraction = heuristic_extract(text)

    if len(_cache) >= MAX_CACHE_ENTRIES:
        # Cheap bounded cache: drop the oldest quarter rather than tracking LRU.
        for stale in list(_cache.keys())[: MAX_CACHE_ENTRIES // 4]:
            _cache.pop(stale, None)
    _cache[key] = extraction
    return extraction


async def extract_batch(texts: list[str], concurrency: int = 4) -> list[Extraction]:
    """Extract several chunks with bounded concurrency to respect rate limits."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(text: str) -> Extraction:
        async with semaphore:
            return await extract_from_chunk(text)

    return await asyncio.gather(*(_one(t) for t in texts))
