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
    SYNTHESIS_SYSTEM,
    SYNTHESIS_TEMPLATE,
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
# Chunks shown to the synthesiser. More context is not always better: a long
# prompt dilutes attention and burns free-tier tokens.
SYNTHESIS_CHUNK_LIMIT = 10


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

    state.add_trace(
        "synthesize",
        "Answer composed",
        "Cited {} of {} source(s); attached {} image(s).".format(
            len(citations), len(chunks), len(state.media_ids)
        ),
        started,
        cited=len(citations),
        sources=len(chunks),
        images=len(state.media_ids),
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

    state.add_trace(
        "synthesize",
        "Answer composed",
        "Cited {} of {} source(s); attached {} image(s).".format(
            len(citations), len(chunks), len(state.media_ids)
        ),
        started,
        cited=len(citations),
        sources=len(chunks),
        images=len(state.media_ids),
    )
