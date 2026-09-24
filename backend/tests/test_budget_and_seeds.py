"""The split call budget, the graph seed query, and non-destructive reranking.

The budget was one number shared by every model call. On a live turn all six of
them were cheap fast-tier calls -- condense, verify, HyDE, verify, HyDE, verify
-- so the budget meant to bound expensive reasoning was spent entirely on cheap
work, and the loop stopped before the reasoning it was protecting ever ran.
"""
from __future__ import annotations

import asyncio

from app.agent.state import AgentState
from app.config import settings
from app.embeddings import encoder
from app.graph import traversal
from app.llm import azure_provider as AZ
from app.llm import factory
from app.vectorstore import search
from app.vectorstore.search import RetrievedChunk

QUESTION = (
    "What is the structural link between the procedure for amending the "
    "Constitution under Article 368 of the Indian Constitution and the subjects "
    "listed in the Seventh Schedule of the Indian Constitution?"
)


def fresh_state():
    return AgentState(question="q", user_id="u", kb_id="kb", session_id="s")


def use_azure_fast_tier():
    settings.llm_provider = "azure"
    settings.azure_openai_endpoint = "https://res.openai.azure.com"
    settings.azure_openai_api_key = "k"
    settings.azure_openai_deployment = "gpt-5.1"
    settings.azure_openai_fast_deployment = ""
    settings.azure_openai_api_version = "v1"
    AZ.AzureOpenAIProvider._build_client = lambda self: object()
    factory.reset_llm()


def use_no_fast_tier():
    settings.llm_provider = "groq"
    settings.groq_api_key = "k"
    settings.groq_model = "qwen/qwen3.8-27b"
    settings.groq_fast_model = ""
    factory.reset_llm()


# ---------------------------------------------------------------------------
# budgets
# ---------------------------------------------------------------------------
def test_the_two_budgets_are_counted_apart():
    use_azure_fast_tier()
    assert factory.has_fast_tier()

    state = fresh_state()
    # The exact live trace: condense, verify, HyDE, verify, HyDE, verify.
    for _ in range(6):
        state.spend_call(fast=True)

    assert state.fast_llm_calls == 6
    assert state.llm_calls == 0
    assert state.total_llm_calls == 6
    assert not state.budget_exhausted()
    assert state.can_iterate()

    state.spend_call()  # synthesis
    assert state.llm_calls == 1


def test_the_cheap_budget_still_has_a_ceiling():
    use_azure_fast_tier()
    state = fresh_state()
    for _ in range(settings.agent_max_fast_llm_calls):
        state.spend_call(fast=True)

    assert state.budget_exhausted()
    assert settings.agent_max_fast_llm_calls > settings.agent_max_llm_calls


def test_without_a_fast_tier_nothing_changes():
    """A call only counts as cheap when a fast tier exists; otherwise
    get_fast_llm() hands back the main model and charging it to the lenient
    budget would quietly let a turn spend twice what the operator asked for."""
    use_no_fast_tier()
    assert not factory.has_fast_tier()

    state = fresh_state()
    for _ in range(6):
        state.spend_call(fast=True)

    assert state.llm_calls == 6
    assert state.fast_llm_calls == 0
    assert state.budget_exhausted()


def test_the_main_budget_still_stops_the_loop():
    use_azure_fast_tier()
    state = fresh_state()
    for _ in range(settings.agent_max_llm_calls):
        state.spend_call()

    assert state.budget_exhausted()


# ---------------------------------------------------------------------------
# graph seeds
# ---------------------------------------------------------------------------
def test_the_seed_query_keeps_the_words_that_identify_the_question():
    terms = traversal.build_fulltext_query(QUESTION).split(" OR ")

    assert any(t.startswith("Seventh") for t in terms), terms
    assert any(t.startswith("Schedule") for t in terms), terms
    assert "368" in terms, terms
    assert len(terms) == len(set(terms)), terms
    assert len([t for t in terms if t.startswith("Constitution")]) == 1, terms
    assert len(terms) <= traversal.MAX_QUERY_TERMS
    assert not any(t.lower().startswith("under") for t in terms), terms


def test_the_old_builder_really_did_lose_the_seventh_schedule():
    """Reconstructed exactly: the old builder kept prepositions, kept duplicates,
    and cut at 12 in reading order -- which pushed "Seventh" to 13th and
    "Schedule" to 14th, so the graph was never asked about them at all."""
    added_stopwords = {
        "under", "upon", "within", "without", "through", "during", "against",
        "above", "below", "across", "among", "along", "toward", "towards", "made",
    }
    old_stopwords = traversal._QUESTION_STOPWORDS - added_stopwords
    old_terms = [
        w for w in traversal._WORD.findall(QUESTION)
        if w.casefold() not in old_stopwords
    ][:12]

    assert "Seventh" not in old_terms and "Schedule" not in old_terms, old_terms
    # ...and it wasted a slot on a repeated word.
    assert len(old_terms) != len({t.casefold() for t in old_terms}), old_terms


def test_identifiers_outrank_prose_in_the_ordering():
    short = traversal.build_fulltext_query("Tell me about article 368 and ratification")
    assert short.split(" OR ")[0] == "368", short


def test_a_hub_is_not_a_seed_even_when_the_question_names_it():
    """A question about the Constitution, asked of a corpus that IS the
    Constitution, is not about the entity "CONSTITUTION"."""
    seeds = [
        {"norm_name": "constitution", "name": "CONSTITUTION", "mentions": 82},
        {"norm_name": "article 368", "name": "article 368", "mentions": 8},
        {"norm_name": "seventh schedule", "name": "Seventh Schedule", "mentions": 3},
        {"norm_name": "article 5", "name": "article 5", "mentions": 2},
        {"norm_name": "article 292", "name": "article 292", "mentions": 1},
    ]
    kept = [s["norm_name"] for s in traversal._filter_seeds(QUESTION, seeds)]

    assert "constitution" not in kept, kept
    assert "constitution" in QUESTION.casefold()   # ...although the question says it
    assert "article 368" in kept and "seventh schedule" in kept, kept
    assert "article 5" not in kept, kept


def test_the_hub_cutoff_scales_with_the_corpus_not_a_magic_number():
    tiny = [{"norm_name": "a", "mentions": 1}, {"norm_name": "b", "mentions": 2}]
    dense = [{"norm_name": str(i), "mentions": 40} for i in range(9)]

    assert traversal._hub_threshold(tiny) >= traversal._HUB_FLOOR
    assert traversal._hub_threshold(dense) > traversal._hub_threshold(tiny)
    # Seeds with no mention count at all must still work.
    assert len(traversal._filter_seeds("article 368", [{"norm_name": "article 368"}])) == 1


# ---------------------------------------------------------------------------
# reranking must not delete evidence
# ---------------------------------------------------------------------------
def chunk(i, score=0.5):
    return RetrievedChunk(chunk_id=f"c{i}", doc_id="d", kb_id="kb", filename="f.json",
                          text=f"passage {i}", score=score, source="vector")


async def rerank_waves(top_n=0):
    """The Article 51A(k) scenario: the verifier confirms the needle is present,
    then two expansion passes flood the pool with better-scoring passages."""

    async def scores(query, passages):
        return [float(len(passages) - i) for i in range(len(passages))]

    encoder.rerank_scores = scores
    settings.rerank_enabled = True
    settings.rerank_weight = 0.6
    settings.rerank_candidates = 50
    settings.rerank_top_n = top_n

    pool = [chunk(i) for i in range(25)]
    pool = await search.apply_reranking("q", pool)
    survived_first = any(c.chunk_id == "c0" for c in pool)

    for wave in range(2):
        pool = search.deduplicate(pool + [chunk(100 + wave * 20 + i) for i in range(20)])
        pool = await search.apply_reranking("q", pool)

    return survived_first, pool


def test_reranking_reorders_evidence_it_does_not_delete_it():
    """Reranking runs again after every expansion pass over the merged pool, so
    anything dropped is dropped for the rest of the turn. That cost a real
    answer: a verifier had already confirmed Article 51A(k) was present."""
    survived_first, pool = asyncio.run(rerank_waves())

    assert survived_first
    assert any(c.chunk_id == "c0" for c in pool), len(pool)
    assert len(pool) <= search.POOL_CAP, len(pool)


def test_an_explicit_cut_still_truncates_for_anyone_who_wants_it():
    _, truncated = asyncio.run(rerank_waves(top_n=20))
    assert len(truncated) == 20, len(truncated)


# ---------------------------------------------------------------------------
# verifier prompt
# ---------------------------------------------------------------------------
def test_the_verifier_is_told_that_composing_an_answer_is_its_job():
    from app.agent import prompts

    system = prompts.VERIFIER_SYSTEM
    assert "COMPOSITION IS NOT A GAP" in system

    block = system.split("COMPOSITION IS NOT A GAP")[1].split("\n\n")[0]
    # The long version did not work and diluted the rest of the prompt.
    assert len(block) < 600, len(block)

    assert "JUDGE AGAINST THIS CORPUS" in system
    assert "inferring beyond what the context states" in system
    assert all(k in system for k in ('"sufficient"', '"in_scope"', '"missing"', '"score"'))
