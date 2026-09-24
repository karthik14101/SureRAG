"""Node 1: rewrite a follow-up into a standalone question.

"What about the second one?" is meaningless to a vector index. Rewriting it to
"What is the warranty period for the Model B pump?" is the difference between
retrieving the right passage and retrieving noise.
"""
from __future__ import annotations

import re
import time

from app.agent.prompts import CONDENSE_SYSTEM, CONDENSE_TEMPLATE, format_history
from app.agent.state import AgentState
from app.llm.base import ChatMessage
from app.llm.factory import get_fast_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

# Below this length a question is usually already standalone.
MIN_REWRITE_LENGTH = 12

# Words that only make sense against an earlier turn. Matched on word
# boundaries, which is the whole point: this list used to be tested with
# substring containment, and "he " matched inside "t-h-e ". Every question
# containing the word "the" therefore looked like a follow-up, so 44 of 47
# second-and-later turns paid a 2.7s rewrite of a question that already stood
# alone -- and a needless rewrite can only damage a good question.
REFERENCE_WORDS = (
    "it", "its", "they", "them", "their", "this", "that", "these", "those",
    "he", "she", "him", "his", "her", "hers", "one", "ones", "both",
)
REFERENCE_PHRASES = (
    "the same", "the above", "the previous", "the former", "the latter",
    "what about", "how about", "as well", "instead", "earlier", "previously",
    "mentioned", "you said", "your answer",
)
# "the first"/"the second" are deliberately absent: they point at an earlier
# turn about as often as they name an ordinal in the question itself ("the
# first owner"), and a false positive here costs a model call.

_REFERENCE_RE = re.compile(
    r"\b(?:{})\b".format("|".join(REFERENCE_WORDS)), re.IGNORECASE
)
_PHRASE_RE = re.compile(
    r"(?:{})".format("|".join(re.escape(phrase) for phrase in REFERENCE_PHRASES)),
    re.IGNORECASE,
)


def needs_rewrite(question: str, history: list[ChatMessage]) -> bool:
    """Whether this question can only be understood against the conversation.

    Over-triggering is the safe direction -- rewriting a standalone question
    usually returns it near-verbatim -- but it is not free: every false positive
    is a model call, two to three seconds of the reader's time, and one more
    chance for the rewrite to lose a detail the original had.
    """
    if not history:
        return False
    text = question.strip()
    if len(text) < MIN_REWRITE_LENGTH:
        return True
    return bool(_REFERENCE_RE.search(text) or _PHRASE_RE.search(text))


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
        llm = get_fast_llm()
        result = await llm.complete(
            CONDENSE_SYSTEM,
            [ChatMessage(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=200,
        )
        state.spend_call(result.usage, fast=True)
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
