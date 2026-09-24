"""Cross-encoder reranking, and why it blends rather than decides.

Retrieval fuses several queries -- sub-questions, expansion probes, graph
neighbours -- whose scores are reciprocal ranks from different result sets and
so are not comparable. A passage that ranked first for a narrow sub-question
outranks a better passage that ranked second for the real question, which makes
the top of the pool close to arbitrary.

The cross-encoder fixes that, but it is trained on web passages and is
measurably weaker on structured text: on this corpus of JSON-shaped legal
records, letting it decide alone demoted two articles the question named. So
retrieval order is kept as a prior and the two are combined.
"""
from __future__ import annotations

import asyncio

from app.config import settings
from app.embeddings import encoder
from app.vectorstore import search
from app.vectorstore.search import RetrievedChunk


def chunk(i, score):
    return RetrievedChunk(
        chunk_id=f"c{i}", doc_id="d", kb_id="kb", filename="f.json",
        text=f"passage {i} " + "x" * 3000, score=score, ordinal=i, source="vector",
    )


def pool():
    """Fifty passages whose incoming scores are reciprocal ranks from other
    queries -- the shape that makes the raw ordering meaningless."""
    return [chunk(i, 1.0 / (1 + i % 5)) for i in range(50)]


def install_scores():
    """Realistic cross-encoder logits (-11..+11); the last passage is the best."""
    scored = []

    async def fake_scores(query, passages):
        scored.append(passages)
        return [(i - len(passages) / 2) / 2 for i in range(len(passages))]

    encoder.rerank_scores = fake_scores
    settings.rerank_enabled = True
    settings.rerank_top_n = 20
    settings.rerank_candidates = 40
    return scored


def test_reranking_makes_the_pool_comparable():
    scored = install_scores()
    ranked = asyncio.run(search.apply_reranking("q", pool()))

    assert len(scored[-1]) == 40, len(scored[-1])      # candidates capped
    assert all(len(p) <= settings.rerank_max_chars for p in scored[-1])
    assert len(ranked) == 20, len(ranked)

    # Raw cross-encoder probabilities on this kind of text sit around 0.5 and
    # differ in the third decimal, so the displayed scores must spread out --
    # they are what the citation score bars show.
    assert ranked[0].score > 0.9
    assert ranked[0].score - ranked[-1].score > 0.3
    assert all(0.0 <= c.score <= 1.0 for c in ranked)
    assert len(ranked) >= 12                            # wider than the verifier reads

    unscored = {f"c{i}" for i in range(40, 50)}
    assert not [c for c in ranked if c.chunk_id in unscored], (
        "a passage that was never scored outranked scored evidence"
    )


def test_the_cross_encoder_informs_the_order_without_dictating_it():
    install_scores()

    settings.rerank_weight = 1.0
    pure = asyncio.run(search.apply_reranking("q", pool()))
    assert pure[0].chunk_id == "c39", pure[0].chunk_id

    settings.rerank_weight = 0.0
    prior_only = asyncio.run(search.apply_reranking("q", pool()))
    assert prior_only[0].score == 1.0
    assert prior_only[0].chunk_id in {f"c{i}" for i in range(0, 40, 5)}

    settings.rerank_weight = 0.6
    blended = asyncio.run(search.apply_reranking("q", pool()))
    assert blended[0].chunk_id != pure[0].chunk_id, blended[0].chunk_id
    # A passage strong on both signals wins, which is the whole point.
    assert blended[0].chunk_id in {f"c{i}" for i in range(30, 40)}, blended[0].chunk_id


def test_reranking_degrades_safely():
    install_scores()
    original = pool()[:5]

    async def no_scores(query, passages):
        return None

    encoder.rerank_scores = no_scores
    unchanged = asyncio.run(search.apply_reranking("q", list(original)))
    assert [c.chunk_id for c in unchanged] == [c.chunk_id for c in original]

    settings.rerank_enabled = False
    assert len(asyncio.run(search.apply_reranking("q", list(original)))) == 5
