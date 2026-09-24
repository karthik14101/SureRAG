"""Chat session handling: history assembly, persistence and titling."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent.state import AgentState, Citation
from app.db import models
from app.llm.base import ChatMessage
from app.logging_conf import get_logger
from app.schemas.chat import MessageOut
from app.schemas.citation import CitationOut

logger = get_logger(__name__)

# Turns kept verbatim. Older ones are folded into history_summary so a long
# thread cannot grow past the context window or the free-tier token limit.
HISTORY_TURN_LIMIT = 8
SUMMARY_TRIGGER = 12


# Filenames listed in the corpus profile. Enough to characterise the corpus
# without turning a verifier prompt into a directory listing.
PROFILE_FILE_LIMIT = 6


def corpus_profile(db: Session, kb_id: str) -> str:
    """One line describing what this knowledge base holds.

    The verifier uses it to judge sufficiency against what the corpus could
    plausibly contain. Without it, asking a corpus of statutes about case law
    produces "insufficient" forever: the verifier compares the context to an
    ideal answer, and no amount of searching closes a gap the corpus never had.
    """
    kb = db.get(models.KnowledgeBase, kb_id)
    if kb is None:
        return ""

    filenames = db.execute(
        select(models.Document.filename)
        .where(models.Document.kb_id == kb_id, models.Document.status == "ready")
        .order_by(models.Document.created_at.asc())
        .limit(PROFILE_FILE_LIMIT + 1)
    ).scalars().all()

    parts = ['Knowledge base: "{}"'.format(kb.name)]
    if kb.description:
        parts.append(kb.description.strip()[:300])

    if filenames:
        shown = ", ".join(filenames[:PROFILE_FILE_LIMIT])
        if len(filenames) > PROFILE_FILE_LIMIT:
            shown += " and {} more".format(kb.doc_count - PROFILE_FILE_LIMIT)
        parts.append(
            "Contains {} document(s) ({} passages): {}".format(
                kb.doc_count, kb.chunk_count, shown
            )
        )
    return ". ".join(parts)


def load_history(db: Session, session_id: str) -> tuple[list[ChatMessage], str | None]:
    chat = db.get(models.ChatSession, session_id)
    summary = chat.history_summary if chat else None

    rows = db.execute(
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.desc())
        .limit(HISTORY_TURN_LIMIT)
    ).scalars().all()

    messages = [
        ChatMessage(role=row.role, content=row.content)
        for row in reversed(rows)
        if row.content and not row.error
    ]
    return messages, summary


def save_user_message(db: Session, session_id: str, content: str) -> models.Message:
    message = models.Message(session_id=session_id, role="user", content=content)
    db.add(message)

    chat = db.get(models.ChatSession, session_id)
    if chat is not None:
        chat.message_count += 1
    db.commit()
    db.refresh(message)
    return message


DEFAULT_TITLE = "New chat"


def _fallback_title(text: str) -> str:
    """A thread name taken straight from a question, with no model call."""
    return " ".join((text or "").split())[:60].rstrip() or DEFAULT_TITLE


def delete_message(db: Session, session_id: str, message_id: str) -> int:
    """Remove one turn from a thread. Returns how many messages went.

    A question takes the answer it produced with it; the rest of the thread is
    left alone. Deleting an answer on its own leaves the question standing,
    which is what someone clearing a bad response means by it.

    Ordering is by position rather than timestamp. A question and its answer can
    land in the same clock tick, and comparing timestamps would then delete
    either both or neither depending on which way the tie fell.
    """
    rows = db.execute(
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc(), models.Message.id.asc())
    ).scalars().all()

    index = next((i for i, row in enumerate(rows) if row.id == message_id), None)
    if index is None:
        return 0

    doomed = [rows[index]]
    # An answer belongs to the question above it; take it along.
    if rows[index].role == "user":
        for row in rows[index + 1 :]:
            if row.role == "user":
                break
            doomed.append(row)

    doomed_ids = {row.id for row in doomed}
    for row in doomed:
        db.delete(row)

    chat = db.get(models.ChatSession, session_id)
    if chat is not None:
        chat.message_count = max(0, chat.message_count - len(doomed))

        # A thread is named after its first question, and only ever once --
        # `maybe_title_session` refuses to rename a thread that already has a
        # name. Delete that first question and the sidebar goes on advertising
        # something no longer in the thread.
        if index == 0:
            survivor = next(
                (row.content for row in rows if row.id not in doomed_ids and row.role == "user"),
                "",
            )
            # A question still standing renames the thread immediately, with no
            # model call. An emptied thread goes back to the default, which is
            # the one state `maybe_title_session` will act on again.
            chat.title = _fallback_title(survivor) if survivor else DEFAULT_TITLE
        # The rolling summary describes turns that may no longer exist. Drop it
        # rather than let the next answer cite a question the user removed; the
        # summariser rebuilds it once the thread is long enough again.
        chat.history_summary = None

    db.commit()
    return len(doomed)


def save_assistant_message(db: Session, session_id: str, state: AgentState) -> models.Message:
    """Persist the answer with its citations, images and reasoning trace."""
    message = models.Message(
        session_id=session_id,
        role="assistant",
        content=state.answer or "",
        route_used=state.route.value,
        sufficiency_score=state.sufficiency_score,
        iterations=state.iterations,
        latency_ms=state.elapsed_ms,
        token_usage=state.usage.as_dict(),
        trace=state.trace_dicts(),
        error=state.error,
    )
    db.add(message)
    db.flush()

    for citation in state.citations:
        chunk = citation.chunk
        db.add(
            models.MessageCitation(
                message_id=message.id,
                marker_index=citation.marker_index,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page_no=chunk.page_no,
                section=chunk.section,
                snippet=chunk.snippet(),
                score=float(chunk.score),
                retrieval_source=chunk.source,
            )
        )

    for position, media_id in enumerate(state.media_ids):
        db.add(
            models.MessageMedia(
                message_id=message.id,
                media_asset_id=media_id,
                relevance=1.0 - position * 0.1,
            )
        )

    chat = db.get(models.ChatSession, session_id)
    if chat is not None:
        chat.message_count += 1

    db.commit()
    db.refresh(message)
    return message


async def maybe_title_session(db: Session, session_id: str, first_message: str) -> None:
    """Name a new thread after its first question."""
    chat = db.get(models.ChatSession, session_id)
    if chat is None or chat.title not in ("New chat", "", None):
        return

    fallback = _fallback_title(first_message)
    try:
        from app.agent.prompts import TITLE_SYSTEM
        from app.llm.factory import get_fast_llm

        llm = get_fast_llm()
        result = await llm.complete(
            TITLE_SYSTEM,
            [ChatMessage(role="user", content=first_message[:500])],
            temperature=0.3,
            max_tokens=30,
        )
        title = result.text.strip().strip('"').strip()
        chat.title = title[:200] if 2 < len(title) <= 100 else (fallback or "New chat")
    except Exception as exc:  # noqa: BLE001 - titling is cosmetic
        logger.debug("Could not generate a chat title: %s", str(exc)[:160])
        chat.title = fallback or "New chat"

    db.commit()


async def maybe_summarise_history(db: Session, session_id: str) -> None:
    """Fold older turns into a summary once a thread gets long."""
    chat = db.get(models.ChatSession, session_id)
    if chat is None or chat.message_count < SUMMARY_TRIGGER:
        return
    # Re-summarise every 6 messages rather than on every turn.
    if chat.message_count % 6 != 0:
        return

    rows = db.execute(
        select(models.Message)
        .where(models.Message.session_id == session_id)
        .order_by(models.Message.created_at.asc())
    ).scalars().all()

    older = rows[:-HISTORY_TURN_LIMIT]
    if len(older) < 2:
        return

    transcript = "\n\n".join(
        "{}: {}".format("User" if m.role == "user" else "Assistant", m.content[:800])
        for m in older
        if m.content
    )

    try:
        from app.agent.prompts import SUMMARY_SYSTEM
        from app.llm.factory import get_fast_llm

        llm = get_fast_llm()
        result = await llm.complete(
            SUMMARY_SYSTEM,
            [ChatMessage(role="user", content=transcript[:12000])],
            temperature=0.2,
            max_tokens=400,
        )
        summary = result.text.strip()
        if summary:
            chat.history_summary = summary[:4000]
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("History summarisation skipped: %s", str(exc)[:160])


def serialise_message(db: Session, message: models.Message, media_url_fn) -> MessageOut:
    """Build the API representation, resolving citations and image URLs."""
    citations = [
        CitationOut(
            marker_index=c.marker_index,
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            filename=c.filename,
            page_no=c.page_no,
            section=c.section,
            snippet=c.snippet,
            score=c.score,
            retrieval_source=c.retrieval_source,
        )
        for c in message.citations
    ]

    media = []
    if message.media_links:
        asset_ids = [link.media_asset_id for link in message.media_links]
        assets = db.execute(
            select(models.MediaAsset).where(models.MediaAsset.id.in_(asset_ids))
        ).scalars().all()
        by_id = {a.id: a for a in assets}
        for link in message.media_links:
            asset = by_id.get(link.media_asset_id)
            if asset is None:
                continue  # the source document was deleted since
            media.append(media_url_fn(asset))

    return MessageOut(
        id=message.id,
        session_id=message.session_id,
        role=message.role,
        content=message.content,
        route_used=message.route_used,
        sufficiency_score=message.sufficiency_score,
        iterations=message.iterations,
        latency_ms=message.latency_ms,
        error=message.error,
        created_at=message.created_at,
        citations=citations,
        media=media,
        trace=message.trace or [],
    )


def citations_payload(citations: list[Citation]) -> list[dict]:
    return [c.as_dict() for c in citations]
