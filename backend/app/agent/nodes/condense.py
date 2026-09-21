"""Node 1: rewrite a follow-up into a standalone question.

"What about the second one?" is meaningless to a vector index. Rewriting it to
"What is the warranty period for the Model B pump?" is the difference between
retrieving the right passage and retrieving noise.
"""
from __future__ import annotations

import time

from app.agent.prompts import CONDENSE_SYSTEM, CONDENSE_TEMPLATE, format_history
from app.agent.state import AgentState
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

# Below this length a question is usually already standalone.
MIN_REWRITE_LENGTH = 12
# Signals that a question depends on earlier turns.
REFERENCE_MARKERS = (
    " it", "it ", "its ", "they", "them", "their", "this", "that", "these",
    "those", "he ", "she ", "his ", "her ", "the same", "above", "previous",
    "earlier", "instead", "also", "what about", "how about", "and the",
)


def needs_rewrite(question: str, history: list[ChatMessage]) -> bool:
    if not history:
        return False
    lowered = " " + question.strip().casefold()
    if len(question.strip()) < MIN_REWRITE_LENGTH:
        return True
    return any(marker in lowered for marker in REFERENCE_MARKERS)


async def condense(state: AgentState) -> AgentState:
    started = time.perf_counter()

    if not needs_rewrite(state.question, state.history):
        state.condensed_question = state.question
        state.add_trace(
            "condense",
            "Question understood",
            "Already standalone; used as-is.",
            started,
            rewritten=False,
        )
        return state

    history_text = format_history(state.history)
    if state.history_summary:
        history_text = "Earlier context: {}\n\n{}".format(state.history_summary, history_text)

    prompt = CONDENSE_TEMPLATE.format(history=history_text, question=state.question)

    try:
        llm = get_llm()
        result = await llm.complete(
            CONDENSE_SYSTEM,
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=200,
        )
        state.spend_call(result.usage)
        rewritten = result.text.strip().strip('"').strip()

        # Guard against a model that returns an explanation instead of a question.
        if rewritten and 5 <= len(rewritten) <= 500 and "\n" not in rewritten.strip():
            state.condensed_question = rewritten
            state.add_trace(
                "condense",
                "Question rewritten",
                rewritten,
                started,
                rewritten=True,
                original=state.question,
            )
        else:
            state.condensed_question = state.question
            state.add_trace(
                "condense",
                "Question understood",
                "Rewrite rejected; used the original.",
                started,
                rewritten=False,
            )
    except Exception as exc:  # noqa: BLE001 - never fail a turn on the rewrite
        logger.warning("Condense failed, using the original question: %s", str(exc)[:160])
        state.condensed_question = state.question
        state.add_trace(
            "condense", "Question understood", "Used the original.", started, rewritten=False
        )

    return state
