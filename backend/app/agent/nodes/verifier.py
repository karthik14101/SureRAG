"""Node 5: the SURE Evidence Sufficiency Verifier.

Ordinary RAG answers from whatever retrieval returned, so weak context produces
a confident, wrong answer. This node interposes a judgement: before generating,
ask whether the retrieved evidence can actually support an answer, and if not,
name precisely what is missing so the expansion node can go find it.

The verifier never answers the question -- separating judging from answering is
what stops it rationalising the context it already has.
"""
from __future__ import annotations

import time

from app.agent.prompts import VERIFIER_SYSTEM, VERIFIER_TEMPLATE, format_context
from app.agent.state import AgentState
from app.config import settings
from app.logging_conf import get_logger
from app.vectorstore.search import describe_corpus_shape

logger = get_logger(__name__)

# Chunks fed to the verifier. More than synthesis reads, because the judgement
# is about coverage: a gap the verifier reports while the evidence sits just
# outside its window sends the whole loop chasing something it already has.
# Each one is trimmed hard below, so the wider window costs little.
VERIFY_CHUNK_LIMIT = 12
VERIFY_CHAR_LIMIT = 1200


def _condensed_context(chunks) -> str:
    """Trim each chunk so the verifier sees breadth rather than depth."""
    trimmed = []
    for chunk in chunks[:VERIFY_CHUNK_LIMIT]:
        text = chunk.text
        if len(text) > VERIFY_CHAR_LIMIT:
            # Head and tail: the middle of a chunk is the least diagnostic part.
            text = text[: VERIFY_CHAR_LIMIT // 2] + "\n[...]\n" + text[-VERIFY_CHAR_LIMIT // 2 :]
        clone = type(chunk)(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            kb_id=chunk.kb_id,
            filename=chunk.filename,
            text=text,
            score=chunk.score,
            page_no=chunk.page_no,
            section=chunk.section,
            modality=chunk.modality,
            ordinal=chunk.ordinal,
            media_ids=chunk.media_ids,
            source=chunk.source,
        )
        trimmed.append(clone)
    return format_context(trimmed)


def _tri_state(value) -> bool | None:
    """Read a JSON boolean a model may have rendered as a string, or omitted.

    `in_scope` is the one field the verifier is allowed to leave out, and it
    frequently does. `payload.get("in_scope") is False` then silently evaluates
    to False for a missing key, which is indistinguishable from "this question
    is in scope" -- so the out-of-scope path never once fired on live traffic.
    Returning None for "did not say" lets the caller tell the two apart.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().casefold()
        if lowered in {"true", "yes", "y", "1"}:
            return True
        if lowered in {"false", "no", "n", "0"}:
            return False
    return None


def _string_list(value, limit: int = 5) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, (str, int, float)) and str(item).strip():
            out.append(str(item).strip()[:200])
    return out[:limit]


async def verify_sufficiency(state: AgentState) -> AgentState:
    started = time.perf_counter()

    if not state.chunks:
        state.sufficient = False
        state.sufficiency_score = 0.0
        state.missing_aspects = ["No relevant passages were retrieved."]
        state.add_trace(
            "verify",
            "Evidence check: insufficient",
            "Nothing was retrieved for this question.",
            started,
            score=0.0,
            sufficient=False,
        )
        return state

    # Out of budget: proceed with what we have rather than stalling the user.
    if state.budget_exhausted():
        state.sufficient = True
        state.sufficiency_score = 0.5
        state.add_trace(
            "verify",
            "Evidence check skipped",
            "Time or call budget reached; answering with the current evidence.",
            started,
            score=0.5,
            sufficient=True,
            skipped=True,
        )
        return state

    # Remember the previous verdict before overwriting it: the orchestrator
    # compares the two to spot an expansion pass that changed nothing.
    if state.missing_aspects or state.sufficiency_score:
        state.previous_score = state.sufficiency_score
        state.previous_missing = list(state.missing_aspects)

    # How much of what the verifier reads is new since last time. A pass that
    # retrieved 20 passages but changed nothing inside this window cannot change
    # the verdict either, and that is what makes another pass pointless.
    judged = [c.chunk_id for c in state.chunks[:VERIFY_CHUNK_LIMIT]]
    if state.judged_chunk_ids:
        fresh = len([cid for cid in judged if cid not in set(state.judged_chunk_ids)])
        state.last_judged_turnover = fresh / max(1, len(judged))
    state.judged_chunk_ids = judged

    # What the corpus is made of, not just which files it came from. Measured
    # once per knowledge base and cached; without it the verifier cannot tell a
    # gap the corpus never had from one the retriever missed.
    corpus = state.corpus_profile or "(not described)"
    shape = await describe_corpus_shape(state.kb_id, state.corpus_profile)
    state.corpus_shape = shape
    if shape:
        corpus = "{} {}".format(corpus, shape)

    context = _condensed_context(state.chunks)
    prompt = VERIFIER_TEMPLATE.format(
        question=state.query,
        corpus=corpus,
        context=context,
    )

    try:
        from app.llm.factory import get_fast_llm

        llm = get_fast_llm()
        payload, usage = await llm.complete_json(
            VERIFIER_SYSTEM, prompt, temperature=0.0, max_tokens=600
        )
        state.spend_call(usage, fast=True)

        if not payload:
            raise ValueError("verifier returned no JSON")

        score = payload.get("score")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.5
        score = max(0.0, min(1.0, score))

        explicit = payload.get("sufficient")
        sufficient = bool(explicit) if isinstance(explicit, bool) else score >= settings.sure_threshold
        # The numeric score is the authority: a model that says "sufficient"
        # while scoring 0.3 is contradicting itself.
        if score < settings.sure_threshold:
            sufficient = False

        # A verifier that says "in_scope": false is reporting that the gap is not
        # the retriever's fault. Expanding cannot close it, so the loop stops and
        # the answer says what is absent instead of guessing at it.
        in_scope = _tri_state(payload.get("in_scope"))
        state.verifier_in_scope = in_scope
        state.out_of_scope = in_scope is False and not sufficient

        state.sufficiency_score = score
        state.sufficient = sufficient
        # Keep the best window of the turn, so a later pass cannot quietly lose
        # a passage this one has just credited.
        state.record_verdict(score, judged)
        state.missing_aspects = _string_list(payload.get("missing"))
        state.expansion_queries = _string_list(payload.get("suggested_queries"), limit=3)
        reason = str(payload.get("reason") or "").strip()[:300]

        if state.out_of_scope:
            state.stop_reason = (
                "The missing material is not the kind of thing this knowledge base "
                "holds, so searching again cannot find it."
            )

        covered = _string_list(payload.get("covered"))
        state.covered_aspects = covered

        if state.out_of_scope:
            label = "Evidence check: outside this knowledge base"
        elif sufficient:
            label = "Evidence check: sufficient"
        else:
            label = "Evidence check: insufficient"

        state.add_trace(
            "verify",
            label,
            reason
            or "Scored {:.2f} against a threshold of {:.2f}.".format(
                score, settings.sure_threshold
            ),
            started,
            score=round(score, 3),
            threshold=settings.sure_threshold,
            sufficient=sufficient,
            out_of_scope=state.out_of_scope,
            # What the verifier itself said about findability, as opposed to
            # what the loop concluded from the score. A wrong out-of-scope
            # verdict is unreadable from the trace without both.
            in_scope=in_scope,
            covered=covered,
            # Gaps are only actionable while the loop can still act on them.
            missing=[] if sufficient else state.missing_aspects,
            # What the verifier was told the corpus holds. Surfaced because a
            # wrong verdict is usually a wrong premise, and there is no other
            # way to see which one it had.
            corpus=corpus,
        )
    except Exception as exc:  # noqa: BLE001
        # A failed verification must not block the answer. Assume sufficient and
        # let the synthesis prompt's grounding rules carry the load.
        logger.warning("Sufficiency verification failed: %s", str(exc)[:200])
        state.sufficient = True
        state.sufficiency_score = 0.5
        state.add_trace(
            "verify",
            "Evidence check unavailable",
            "The verifier did not respond; answering with the retrieved evidence.",
            started,
            score=0.5,
            sufficient=True,
            failed=True,
        )

    return state
