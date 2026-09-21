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

logger = get_logger(__name__)

# Chunks fed to the verifier. Fewer than synthesis uses: the judgement is about
# coverage, not detail, and a shorter prompt keeps the call cheap.
VERIFY_CHUNK_LIMIT = 8
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

    context = _condensed_context(state.chunks)
    prompt = VERIFIER_TEMPLATE.format(question=state.query, context=context)

    try:
        from app.llm.factory import get_llm

        llm = get_llm()
        payload, usage = await llm.complete_json(
            VERIFIER_SYSTEM, prompt, temperature=0.0, max_tokens=600
        )
        state.spend_call(usage)

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

        state.sufficiency_score = score
        state.sufficient = sufficient
        state.missing_aspects = _string_list(payload.get("missing"))
        state.expansion_queries = _string_list(payload.get("suggested_queries"), limit=3)
        reason = str(payload.get("reason") or "").strip()[:300]

        covered = _string_list(payload.get("covered"))

        state.add_trace(
            "verify",
            "Evidence check: {}".format("sufficient" if sufficient else "insufficient"),
            reason
            or "Scored {:.2f} against a threshold of {:.2f}.".format(
                score, settings.sure_threshold
            ),
            started,
            score=round(score, 3),
            threshold=settings.sure_threshold,
            sufficient=sufficient,
            covered=covered,
            missing=state.missing_aspects,
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
