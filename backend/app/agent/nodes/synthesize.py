"""Node 7: answer synthesis, citation resolution and image selection.

Two things happen here that most RAG implementations skip:

  1. Citation validation. Models cite [7] when only 5 sources exist, or cite a
     source they did not use. Every marker is checked against the source list;
     invalid ones are stripped from the text rather than shown as dead chips.
  2. Image selection. Figures attached to a chunk the model actually cited are
     surfaced alongside the answer, so the diagram appears with the paragraph
     that discusses it.
"""
from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator

from app.agent.prompts import (
    DIRECT_SYSTEM,
    NO_CONTEXT_SYSTEM,
    SCOPE_NOTE,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_TEMPLATE,
    THIN_EVIDENCE_NOTE,
    format_context,
)
from app.agent.state import AgentState, Citation
from app.config import settings
from app.llm.base import ChatMessage
from app.llm.factory import get_llm
from app.logging_conf import get_logger

logger = get_logger(__name__)

CITATION_RE = re.compile(r"\[(\d{1,2})\]")
MAX_IMAGES = 4
# End of a sentence, with any citation markers that trail it.
#
# Two details, both learned from real answers rather than guessed at. Citation
# markers belong to the sentence they follow, whether written flush against the
# stop ("...x.[1]") or after a space ("...x. [1]"); split them off and the claim
# they support looks uncited. And a full stop only ends a sentence when it
# stands alone and the next thing starts like a sentence -- quoted statute is
# dense with ellipses ('"...any change in ... any of the Lists..."'), and
# treating those as three sentence ends scored a fully cited answer at 57%.
_BOUNDARY = re.compile(
    r"""(?<![.!?])        # not the tail of an ellipsis
        [.!?]             # the stop itself
        (?![.!?])         # nor the head of one
        (?:\s*\[\d{1,2}\])*   # citations belong to the sentence they follow
        (?=\s+["'(A-Z]|\s*$)  # next sentence, or end of line
    """,
    re.VERBOSE,
)
# Below this a fragment is a heading, a list lead-in or a connective, not a
# claim anyone needs a source for.
MIN_CLAIM_CHARS = 60
# Chunks shown to the synthesiser. More context is not always better: a long
# prompt dilutes attention and burns free-tier tokens.
SYNTHESIS_CHUNK_LIMIT = 10


# Every answer is written by the main model, including the out-of-scope ones.
# That was tried the other way and reverted, so the reasoning is recorded here
# rather than rediscovered. Routing absence answers to the fast tier looked
# free: the verdict is settled before generation starts, the scope note caps the
# reply at 150 words, and it saved about nine seconds of reasoning tokens billed
# as output. But naming what a corpus holds *instead* of the answer is a
# relevance judgement, and that is the one thing the cheap tier is worst at.
# Asked whether the Constitution covers MRP, it offered Article 273, "Grants in
# lieu of export duty on jute and jute products", as an illustrative example and
# quoted two clauses of it; the main model, on the same question, had picked the
# nearest genuinely related provision and said plainly that it was tangential.
# Nine seconds on the rarest path is not worth an answer that pads with jute.


def build_prompt(state: AgentState) -> tuple[str, str, list]:
    """Returns (system_prompt, user_prompt, the chunks that were numbered)."""
    chunks = state.context_chunks(SYNTHESIS_CHUNK_LIMIT)

    if not chunks and not state.graph_text:
        return (
            NO_CONTEXT_SYSTEM,
            "Question: {}\n\nThe knowledge base returned no relevant passages.".format(
                state.question
            ),
            [],
        )

    graph_section = ""
    if state.graph_text:
        graph_section = "\n{}\n".format(state.graph_text)

    user_prompt = SYNTHESIS_TEMPLATE.format(
        question=state.question,
        graph_section=graph_section,
        context=format_context(chunks),
    )
    # Weak evidence invites padding: the model compensates for what it cannot
    # support with length. Say so explicitly rather than hoping.
    if state.out_of_scope:
        user_prompt = SCOPE_NOTE + user_prompt
    elif not state.sufficient:
        user_prompt = THIN_EVIDENCE_NOTE + user_prompt
    return SYNTHESIS_SYSTEM, user_prompt, chunks


def resolve_citations(answer: str, chunks: list) -> tuple[str, list[Citation]]:
    """Validate [n] markers, renumber them densely, and build Citation records.

    A model that cites sources 1, 4 and 9 produces chips numbered 1, 2, 3 so the
    reader sees a clean sequence, with each chip pointing at the right chunk.
    """
    if not chunks:
        return CITATION_RE.sub("", answer).strip(), []

    used_order: list[int] = []
    for match in CITATION_RE.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= len(chunks) and index not in used_order:
            used_order.append(index)

    renumber = {original: position for position, original in enumerate(used_order, start=1)}

    def _replace(match: re.Match) -> str:
        index = int(match.group(1))
        if index in renumber:
            return "[{}]".format(renumber[index])
        # Out-of-range marker: the model invented it. Drop it silently.
        return ""

    cleaned = CITATION_RE.sub(_replace, answer)
    # Tidy the whitespace left behind by removed markers.
    cleaned = re.sub(r" +([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    citations = [
        Citation(marker_index=position, chunk=chunks[original - 1])
        for original, position in renumber.items()
    ]
    citations.sort(key=lambda c: c.marker_index)
    return cleaned, citations


# A sentence about the evidence itself -- "the sources do not explain X", "what
# is missing from these extracts" -- makes no claim about the world and needs no
# citation. THIN_EVIDENCE_NOTE explicitly ASKS for one of these when evidence is
# thin, so counting it as an uncited claim docked the answer for following its
# instructions: a complete, fully sourced answer about Article 368 scored 75% on
# the strength of its closing gap sentence alone.
#
# Both halves must match. A reference to the evidence on its own is too loose
# ("in the context of Article 21, no person shall be deprived...") -- it is the
# combination with an absence that marks a sentence as being about what the
# corpus lacks rather than about the law.
_EVIDENCE_REF = re.compile(
    r"\b(?:this|the)\s+knowledge\s+base\b"
    r"|\b(?:these|those|the|available|provided|retrieved|supplied|given)\s+"
    r"(?:\w+\s+){0,2}(?:sources?|extracts?|passages?|excerpts?|corpus|materials?)\b",
    re.IGNORECASE,
)
_ABSENCE = re.compile(
    r"\b(?:no|not|none|nothing|never|missing|absent|lacks?|lacking|silent|beyond)\b",
    re.IGNORECASE,
)


def _is_about_the_evidence(sentence: str) -> bool:
    return bool(_EVIDENCE_REF.search(sentence) and _ABSENCE.search(sentence))


_LIST_ITEM = re.compile(r"^\s*(?:[-*•–]|\d+[.)])\s+")


def _claim_units(answer: str) -> list[tuple[bool, str]]:
    """Group an answer into units a citation can reasonably cover.

    Returns (is_list, text) pairs. A list and the sentence introducing it are
    one unit, because that is how a list drawn from a single source is actually
    cited -- once, on the lead-in or the last item. Scoring each bullet
    separately misread the convention as three unsourced assertions and rated a
    correct, fully cited answer about Article 368 at 50%.
    """
    units: list[tuple[bool, list[str]]] = []
    open_list: list[str] | None = None

    for raw in answer.splitlines():
        line = raw.strip()
        if not line:
            # A blank line does not end a list. Markdown nearly always puts one
            # between a lead-in and the bullets it introduces, and treating it
            # as a break split them into two units -- the lead-in stranded
            # without the citation that sat on the final bullet.
            continue
        if line.startswith("#"):
            open_list = None
            continue

        if _LIST_ITEM.search(line):
            if open_list is None:
                open_list = [line]
                units.append((True, open_list))
            else:
                open_list.append(line)
            continue

        open_list = None
        if line.endswith(":"):
            # Introduces the list below; they stand or fall together.
            open_list = [line]
            units.append((True, open_list))
        else:
            units.append((False, [line]))

    return [(is_list, " ".join(lines)) for is_list, lines in units]


def _split_sentences(line: str) -> list[str]:
    """Cut a line at sentence ends, keeping each sentence's trailing citations."""
    sentences: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(line):
        sentences.append(line[start : match.end()].strip())
        start = match.end()
    tail = line[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def measure_grounding(answer: str, citations: list[Citation]) -> float | None:
    """What share of the answer's substantive sentences cite a source.

    The sufficiency score answers a different question from the one a reader is
    actually asking. It measures how much of the question the retrieved pool
    could have covered, judged before a word was written, and on questions whose
    answer must be assembled from several provisions the verifier is reliably
    pessimistic: it wants a passage stating the conclusion, which no corpus of
    primary texts contains. That produced a complete, correctly cited answer
    about Article 368 and the Seventh Schedule labelled "50% partly supported".

    This measures the answer instead. It is a weaker claim than correctness --
    a cited sentence can still misread its source -- but it is an honest one,
    and it is the question a reader means by "can I trust this": is what I am
    reading drawn from my documents, or is it the model talking?
    """
    if not answer.strip():
        return None
    if not citations:
        return 0.0

    claims = cited = 0
    for is_list, text in _claim_units(answer):
        # A list is one evidential unit; prose is counted sentence by sentence.
        pieces = [text] if is_list else _split_sentences(text)
        for piece in pieces:
            if len(piece) < MIN_CLAIM_CHARS or _is_about_the_evidence(piece):
                continue
            claims += 1
            if CITATION_RE.search(piece):
                cited += 1

    # An answer whose every sentence is about what the corpus lacks has nothing
    # to ground. Reporting 100% there would be a boast about saying nothing; the
    # scope banner already tells the reader what happened.
    if not claims:
        return None
    return round(cited / claims, 3)


def select_images(citations: list[Citation], all_chunks: list) -> list[str]:
    """Images from cited chunks first, then from other retrieved chunks."""
    ordered: list[str] = []
    seen: set[str] = set()

    for citation in citations:
        for media_id in citation.chunk.media_ids or []:
            if media_id not in seen:
                seen.add(media_id)
                ordered.append(media_id)

    if len(ordered) < MAX_IMAGES:
        for chunk in all_chunks:
            for media_id in chunk.media_ids or []:
                if media_id not in seen:
                    seen.add(media_id)
                    ordered.append(media_id)
                if len(ordered) >= MAX_IMAGES:
                    break
            if len(ordered) >= MAX_IMAGES:
                break

    return ordered[:MAX_IMAGES]


async def synthesize(state: AgentState) -> AgentState:
    """Non-streaming synthesis. Used by the plain JSON /ask endpoint."""
    started = time.perf_counter()
    llm = get_llm()

    if state.route.value == "DIRECT":
        result = await llm.complete(
            DIRECT_SYSTEM,
            [ChatMessage(role="user", content=state.question)],
            temperature=0.5,
            max_tokens=300,
        )
        state.spend_call(result.usage)
        state.answer = result.text.strip()
        state.add_trace("synthesize", "Replied directly", "No document lookup needed.", started)
        return state

    system, prompt, chunks = build_prompt(state)

    messages: list[ChatMessage] = []
    if state.history:
        messages.extend(state.history[-4:])
    messages.append(ChatMessage(role="user", content=prompt))

    result = await llm.complete(
        system, messages, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens
    )
    state.spend_call(result.usage)

    answer, citations = resolve_citations(result.text.strip(), chunks)
    state.answer = answer
    state.citations = citations
    state.media_ids = select_images(citations, state.chunks)
    state.grounding = measure_grounding(answer, citations)

    detail = "Cited {} of {} source(s); attached {} image(s).".format(
        len(citations), len(chunks), len(state.media_ids)
    )
    if state.grounding is not None:
        detail += " {:.0%} of its claims carry a citation.".format(state.grounding)

    state.add_trace(
        "synthesize",
        "Answer composed",
        detail,
        started,
        cited=len(citations),
        sources=len(chunks),
        images=len(state.media_ids),
        # Carried in the trace rather than a new column: create_all() never
        # alters an existing table, so adding one would break live databases.
        grounding=state.grounding,
        out_of_scope=state.out_of_scope,
    )
    return state


async def synthesize_stream(state: AgentState) -> AsyncIterator[str]:
    """Streaming synthesis.

    Markers are emitted as-is while streaming; `finalise_stream` then rewrites
    the accumulated text so the stored message has validated, densely numbered
    citations. Renumbering mid-stream would make the visible text jump around.
    """
    started = time.perf_counter()
    llm = get_llm()

    if state.route.value == "DIRECT":
        collected: list[str] = []
        async for piece in llm.stream(
            DIRECT_SYSTEM,
            [ChatMessage(role="user", content=state.question)],
            temperature=0.5,
            max_tokens=300,
        ):
            collected.append(piece)
            yield piece
        state.spend_call()
        state.answer = "".join(collected).strip()
        state.add_trace("synthesize", "Replied directly", "No document lookup needed.", started)
        return

    system, prompt, chunks = build_prompt(state)

    messages: list[ChatMessage] = []
    if state.history:
        messages.extend(state.history[-4:])
    messages.append(ChatMessage(role="user", content=prompt))

    collected = []
    async for piece in llm.stream(
        system, messages, temperature=settings.llm_temperature, max_tokens=settings.llm_max_tokens
    ):
        collected.append(piece)
        yield piece

    state.spend_call()
    raw_answer = "".join(collected).strip()

    answer, citations = resolve_citations(raw_answer, chunks)
    state.answer = answer
    state.citations = citations
    state.media_ids = select_images(citations, state.chunks)
    state.grounding = measure_grounding(answer, citations)

    detail = "Cited {} of {} source(s); attached {} image(s).".format(
        len(citations), len(chunks), len(state.media_ids)
    )
    if state.grounding is not None:
        detail += " {:.0%} of its claims carry a citation.".format(state.grounding)

    state.add_trace(
        "synthesize",
        "Answer composed",
        detail,
        started,
        cited=len(citations),
        sources=len(chunks),
        images=len(state.media_ids),
        # Carried in the trace rather than a new column: create_all() never
        # alters an existing table, so adding one would break live databases.
        grounding=state.grounding,
        out_of_scope=state.out_of_scope,
    )
