"""Hybrid retrieval against Qdrant.

A single `query_points` call carries two prefetch branches -- dense semantic and
BM25 sparse -- fused server-side with Reciprocal Rank Fusion. One round trip,
no client-side merging, and no need to normalise two incompatible score scales.
"""
from __future__ import annotations

import asyncio
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from qdrant_client import models as qm

from app.config import settings
from app.embeddings import encoder, sparse
from app.logging_conf import get_logger
from app.vectorstore.qdrant_client import DENSE, SPARSE, get_client

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """One retrieval hit, in the shape the agent and the UI both consume."""

    chunk_id: str
    doc_id: str
    kb_id: str
    filename: str
    text: str
    score: float
    page_no: int | None = None
    section: str | None = None
    modality: str = "text"
    ordinal: int = 0
    media_ids: list[str] = field(default_factory=list)
    # vector | graph | expand -- shown on the citation chip
    source: str = "vector"

    @property
    def location(self) -> str:
        if self.page_no:
            return "{} p.{}".format(self.filename, self.page_no)
        if self.section:
            return "{} > {}".format(self.filename, self.section)
        return self.filename

    def snippet(self, limit: int = 320) -> str:
        text = " ".join(self.text.split())
        return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _tenant_filter(
    user_id: str,
    kb_id: str,
    exclude_ids: list[str] | None = None,
    doc_id: str | None = None,
    modality: str | None = None,
) -> qm.Filter:
    must: list = [
        qm.FieldCondition(key="user_id", match=qm.MatchValue(value=user_id)),
        qm.FieldCondition(key="kb_id", match=qm.MatchValue(value=kb_id)),
    ]
    # Optional narrowing, used by the chunk explorer. The agent never sets these.
    if doc_id:
        must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
    if modality:
        must.append(qm.FieldCondition(key="modality", match=qm.MatchValue(value=modality)))

    must_not: list = []
    if exclude_ids:
        must_not.append(qm.HasIdCondition(has_id=list(exclude_ids)))
    return qm.Filter(must=must, must_not=must_not or None)


def _to_chunk(point, source: str = "vector") -> RetrievedChunk:
    payload = point.payload or {}
    # `query_points` returns ScoredPoint (has .score); `retrieve` returns Record
    # (no .score at all), so this must not assume the attribute exists.
    raw_score = getattr(point, "score", None)
    return RetrievedChunk(
        chunk_id=str(payload.get("chunk_id") or point.id),
        doc_id=str(payload.get("doc_id") or ""),
        kb_id=str(payload.get("kb_id") or ""),
        filename=str(payload.get("filename") or "unknown"),
        text=str(payload.get("text") or ""),
        score=float(raw_score) if raw_score is not None else 0.0,
        page_no=payload.get("page_no"),
        section=payload.get("section"),
        modality=str(payload.get("modality") or "text"),
        ordinal=int(payload.get("ordinal") or 0),
        media_ids=list(payload.get("media_ids") or []),
        source=source,
    )


async def hybrid_search(
    query: str,
    *,
    user_id: str,
    kb_id: str,
    top_k: int | None = None,
    exclude_ids: list[str] | None = None,
    source_label: str = "vector",
    doc_id: str | None = None,
    modality: str | None = None,
) -> list[RetrievedChunk]:
    """Dense + sparse retrieval fused with RRF, scoped to one knowledge base."""
    limit = top_k or settings.retrieval_top_k
    client = get_client()
    query_filter = _tenant_filter(user_id, kb_id, exclude_ids, doc_id, modality)

    # Two different local models with nothing to say to each other, so there is
    # no reason for the second to wait on the first. Both release the GIL during
    # inference, so the threads genuinely overlap.
    dense_vector, sparse_vector = await asyncio.gather(
        encoder.embed_query(query), sparse.embed_query_sparse(query)
    )

    # Over-fetch in each branch so fusion has enough candidates to reorder.
    prefetch_limit = max(limit * 2, 20)

    try:
        if sparse_vector.is_empty:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                query=dense_vector,
                using=DENSE,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
            )
        else:
            response = await client.query_points(
                collection_name=settings.qdrant_collection,
                prefetch=[
                    qm.Prefetch(
                        query=dense_vector,
                        using=DENSE,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    qm.Prefetch(
                        query=qm.SparseVector(
                            indices=sparse_vector.indices, values=sparse_vector.values
                        ),
                        using=SPARSE,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ],
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        return [_to_chunk(p, source_label) for p in response.points]
    except Exception as exc:  # noqa: BLE001
        logger.error("Qdrant hybrid search failed: %s", str(exc)[:300])
        raise


# Concurrent searches per multi-query pass. A multi-hop turn issues one per
# sub-question plus the original, and an expansion pass issues one per targeted
# query plus the HyDE probe. The Qdrant round trips parallelise cleanly; the
# local encodes that precede them compete for the same cores, so this is capped
# rather than unbounded -- five sub-questions should not thrash the thread pool.
MAX_CONCURRENT_QUERIES = 4


async def multi_query_search(
    queries: list[str],
    *,
    user_id: str,
    kb_id: str,
    top_k_each: int = 6,
    exclude_ids: list[str] | None = None,
    source_label: str = "expand",
) -> list[RetrievedChunk]:
    """Run several queries concurrently and merge, keeping each chunk's best score.

    These ran one after another, which put up to six round trips in series on
    the two slowest paths in the system -- multi-hop retrieval and every
    expansion pass -- for no reason at all: the queries do not depend on each
    other, and a single search measures around 400ms end to end.

    The merged result is unchanged, not merely equivalent. Each chunk keeps its
    highest score across the queries, which does not depend on the order they
    ran in, and `gather` hands results back in the order they were requested, so
    ties resolve exactly as the loop resolved them.
    """
    wanted = [query for query in queries if query.strip()]
    if not wanted:
        return []

    gate = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)

    async def _one(query: str) -> list[RetrievedChunk]:
        async with gate:
            try:
                return await hybrid_search(
                    query,
                    user_id=user_id,
                    kb_id=kb_id,
                    top_k=top_k_each,
                    exclude_ids=exclude_ids,
                    source_label=source_label,
                )
            except Exception:  # noqa: BLE001 - one bad query must not kill the turn
                return []

    results = await asyncio.gather(*(_one(query) for query in wanted))

    merged: dict[str, RetrievedChunk] = {}
    for hits in results:
        for hit in hits:
            existing = merged.get(hit.chunk_id)
            if existing is None or hit.score > existing.score:
                merged[hit.chunk_id] = hit
    return sorted(merged.values(), key=lambda c: c.score, reverse=True)


async def fetch_chunks_by_id(
    chunk_ids: list[str], *, user_id: str, kb_id: str, source_label: str = "graph"
) -> list[RetrievedChunk]:
    """Load specific chunks by id -- used after a graph traversal picks them out."""
    if not chunk_ids:
        return []
    client = get_client()
    try:
        records = await client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=list(dict.fromkeys(chunk_ids))[:64],
            with_payload=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Qdrant retrieve by id failed: %s", str(exc)[:200])
        return []

    out: list[RetrievedChunk] = []
    for record in records:
        payload = record.payload or {}
        # Retrieval by id bypasses filters, so re-check ownership here.
        if payload.get("user_id") != user_id or payload.get("kb_id") != kb_id:
            continue
        chunk = _to_chunk(record, source_label)
        # Graph hits carry no similarity score; give them a neutral prior so
        # they can be ranked alongside vector hits without dominating.
        chunk.score = 0.5
        out.append(chunk)
    return out


# Upper bound on the evidence pool, so a pathological turn cannot grow it
# without limit. Deliberately far above anything the verifier or the answer
# reads: its job is to stop runaway memory, not to select evidence.
POOL_CAP = 80


def _sigmoid(value: float) -> float:
    """Squash a cross-encoder logit into 0-1 so scores stay comparable.

    The logits run roughly -11..+11, while fused retrieval scores are 0-1. Mixing
    the two would corrupt every later sort and make the score bars in the UI
    meaningless, so the reranked scores are mapped onto the same scale.
    """
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _rescale(values: list[float]) -> list[float]:
    """Map a set of scores onto 0.05-1.0, preserving order.

    Scores only ever mean "more relevant than the next one in this answer's
    evidence", so the absolute numbers are free to be readable. This is also what
    the citation score bars display.
    """
    if not values:
        return []
    low, high = min(values), max(values)
    spread = high - low
    if spread <= 0:
        return [1.0] * len(values)
    return [0.05 + 0.95 * (v - low) / spread for v in values]


async def apply_reranking(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reorder a retrieved pool with the local cross-encoder.

    Retrieval fuses several queries -- sub-questions, expansion probes, graph
    neighbours -- whose scores are reciprocal ranks from different result sets
    and therefore not comparable with each other. A passage that ranked first for
    a narrow sub-question outranks a better passage that ranked second for the
    real question, so the top of the pool is close to arbitrary. The
    cross-encoder reads the question and each passage together and produces one
    comparable score, which is what makes the top of the list mean something.
    """
    if not settings.rerank_enabled or len(chunks) < 2:
        return chunks

    # Score the strongest candidates only; the tail rarely wins and each pair
    # costs CPU. Anything not scored keeps its place behind those that were.
    limit = settings.rerank_candidates or len(chunks)
    candidates, tail = chunks[:limit], chunks[limit:]

    passages = [c.text[: settings.rerank_max_chars] for c in candidates]
    scores = await encoder.rerank_scores(query, passages)
    if scores is None:
        return chunks

    # Blend rather than replace. The cross-encoder is trained on web passages
    # and is measurably weaker on structured text -- on a corpus of JSON-shaped
    # legal records it ranked the right article first but demoted two others the
    # question named. Retrieval order is a genuine prior, so the two are combined
    # instead of letting either one decide alone.
    #
    # Both sides are rescaled across this result set first: cross-encoder logits
    # are absolute and cluster near zero on such text, while the fused scores are
    # reciprocal ranks. Neither is a probability, and only their order matters.
    cross = _rescale([_sigmoid(float(s)) for s in scores])
    prior = _rescale([float(c.score) for c in candidates])
    weight = min(1.0, max(0.0, settings.rerank_weight))

    for chunk, cross_score, prior_score in zip(candidates, cross, prior):
        chunk.score = round(weight * cross_score + (1.0 - weight) * prior_score, 6)
    ranked = sorted(candidates, key=lambda c: c.score, reverse=True) + tail

    # Reordering is safe; discarding is not. This runs again after every
    # expansion pass, over the merged pool, so anything dropped here is dropped
    # for the rest of the turn. That cost a real answer: a verifier that had
    # already confirmed Article 51A(k) was present went on to expand, the pool
    # grew past the cut, and the passage was gone by the time the answer was
    # written -- the system lost evidence it had already found. The pool only
    # ever feeds a top-12 window, so carrying the tail costs nothing but memory.
    keep = settings.rerank_top_n or POOL_CAP
    return ranked[:keep]


# Structured records announce their own shape in their field names. Prose does
# not, and is described from what its payloads carry instead.
_FIELD_RE = re.compile(r"\]\.(\w+)\s*=")
CORPUS_SAMPLE = 200

# How a passage got into the corpus, in words a verifier can reason about.
_MODALITY_WORDS = {
    "text": "running text",
    "table": "tables rendered as text",
    "caption": "figure captions",
    "image": "text recovered from images",
}
# Keyed by (kb_id, profile). The profile carries the passage count, so
# re-ingesting a knowledge base changes the key and the shape is measured again.
_corpus_shapes: dict[tuple[str, str], str] = {}


async def describe_corpus_shape(kb_id: str, revision: str) -> str:
    """One factual sentence about the SHAPE of a corpus's passages.

    The verifier is told which files a knowledge base holds but nothing about
    what is inside them, and that gap costs real turns. Asked how Article 368
    relates to the Seventh Schedule, it demanded "the text of the Seventh
    Schedule itself" on every pass and scored the answer at 50% -- correctly,
    because this corpus is 480 article records and the schedules were never
    ingested, but with no way to tell that from a retrieval failure. So it kept
    searching for something that does not exist.

    Reported as measurement rather than conclusion: the verifier's own prompt
    already knows what to do with "the corpus does not hold this kind of
    material", it just needs to be able to see it. The sample is drawn
    independently of the question -- the retrieved chunks are all relevant by
    construction, and describing a corpus from them would make a mixed one look
    homogeneous. It is not a random sample either, but the leading passages by
    id, so the count it reports is stated rather than generalised from.
    """
    key = (kb_id, revision)
    cached = _corpus_shapes.get(key)
    if cached is not None:
        return cached

    shape = ""
    try:
        client = get_client()
        points, _ = await client.scroll(
            collection_name=settings.qdrant_collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="kb_id", match=qm.MatchValue(value=kb_id))]
            ),
            limit=CORPUS_SAMPLE,
            with_payload=True,
        )
        signatures: list[tuple[str, ...]] = []
        for point in points:
            text = str((point.payload or {}).get("text") or "")
            fields = tuple(dict.fromkeys(_FIELD_RE.findall(text)))
            if fields:
                signatures.append(fields)

        if points and len(signatures) >= 0.95 * len(points):
            common, count = Counter(signatures).most_common(1)[0]
            if count >= 0.95 * len(signatures):
                fields = ", ".join(common)
                # Say what was counted, not what it implies. A handful of
                # passages usually carry no field markers at all -- a preamble,
                # the tail of a record split across chunks -- and claiming
                # "every one" over them would be a plain untruth in the one
                # place the verifier has no way to check.
                if count == len(points):
                    shape = (
                        "Every one of the {} passages sampled from this corpus is a "
                        "structured record with exactly these fields: {}.".format(
                            len(points), fields
                        )
                    )
                else:
                    shape = (
                        "{} of the {} passages sampled from this corpus are structured "
                        "records with exactly these fields: {}; the rest carry no field "
                        "markers.".format(count, len(points), fields)
                    )

        if not shape:
            shape = _describe_prose(points)
    except Exception as exc:  # noqa: BLE001 - a missing hint must not fail a turn
        logger.warning("Corpus shape probe failed: %s", str(exc)[:200])

    _corpus_shapes[key] = shape
    return shape


def _describe_prose(points) -> str:
    """Describe a corpus that is not made of structured records.

    Prose used to get no sentence at all, so the verifier was shown "(not
    described)" and had no way to tell a retrieval miss from material that was
    never ingested -- the exact failure this probe exists to prevent. In
    practice that meant it fired on one JSON corpus and on nothing else, because
    hardly any corpus is JSON. A 364-page car manual got nothing.

    Only what the payload actually carries is reported: how many files, the page
    span, and how the passages were obtained. That last part is the useful one
    on a manual. Tables arrive flattened into text and figures arrive as their
    captions, so a verifier told this can reason about why a pictogram is
    nowhere described, instead of concluding the topic is absent.
    """
    files = {str((point.payload or {}).get("filename") or "") for point in points}
    files.discard("")
    pages = [
        n
        for n in ((point.payload or {}).get("page_no") for point in points)
        if isinstance(n, int) and n > 0
    ]

    # With neither a filename nor a page number there is nothing to say that is
    # not guesswork, and silence is the honest output.
    if not files and not pages:
        return ""

    where = "{} file{}".format(len(files), "" if len(files) == 1 else "s") if files else ""
    if pages:
        span = "pages {} to {}".format(min(pages), max(pages))
        where = "{} spanning {}".format(where, span) if where else "passages on {}".format(span)

    kinds = Counter(
        str((point.payload or {}).get("modality") or "text") for point in points
    )
    if len(kinds) == 1:
        kind = next(iter(kinds))
        mix = "all are {}".format(_MODALITY_WORDS.get(kind, kind))
    else:
        described = [
            "{} are {}".format(count, _MODALITY_WORDS.get(kind, kind))
            for kind, count in kinds.most_common()
        ]
        mix = ", ".join(described[:-1]) + " and " + described[-1]

    return "The {} passages sampled come from {}. Of those, {}.".format(
        len(points), where, mix
    )


def deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop repeats, preferring the higher-scoring copy of each chunk."""
    best: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        existing = best.get(chunk.chunk_id)
        if existing is None or chunk.score > existing.score:
            best[chunk.chunk_id] = chunk
    return sorted(best.values(), key=lambda c: c.score, reverse=True)
