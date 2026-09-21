"""Chat endpoints: sessions, history, and the streaming ask endpoint."""
from __future__ import annotations

import json
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.agent import orchestrator
from app.agent.state import AgentState, Route
from app.core.deps import CurrentUser, DbSession, client_key, get_owned_chat, get_owned_kb
from app.core.errors import ValidationError
from app.core.rate_limit import chat_limiter
from app.db import models
from app.db.base import SessionLocal
from app.logging_conf import get_logger
from app.schemas.chat import (
    AskRequest,
    AskResponse,
    ChatSessionCreate,
    ChatSessionOut,
    ChatSessionUpdate,
    MessageOut,
)
from app.services import chat_service, document_service

logger = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _session_out(db, chat: models.ChatSession) -> ChatSessionOut:
    kb = db.get(models.KnowledgeBase, chat.kb_id)
    return ChatSessionOut(
        id=chat.id,
        kb_id=chat.kb_id,
        kb_name=kb.name if kb else None,
        title=chat.title,
        message_count=chat.message_count,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(
    db: DbSession, user: CurrentUser, kb_id: str | None = None
) -> list[ChatSessionOut]:
    query = select(models.ChatSession).where(models.ChatSession.user_id == user.id)
    if kb_id:
        query = query.where(models.ChatSession.kb_id == kb_id)
    rows = db.execute(query.order_by(models.ChatSession.updated_at.desc())).scalars().all()
    return [_session_out(db, row) for row in rows]


@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
def create_session(
    payload: ChatSessionCreate, db: DbSession, user: CurrentUser
) -> ChatSessionOut:
    kb = get_owned_kb(payload.kb_id, db, user)
    chat = models.ChatSession(
        user_id=user.id, kb_id=kb.id, title=(payload.title or "New chat").strip()[:200]
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return _session_out(db, chat)


@router.patch("/sessions/{session_id}", response_model=ChatSessionOut)
def rename_session(
    session_id: str, payload: ChatSessionUpdate, db: DbSession, user: CurrentUser
) -> ChatSessionOut:
    chat = get_owned_chat(session_id, db, user)
    title = payload.title.strip()
    if not title:
        raise ValidationError("The title cannot be empty.")
    chat.title = title[:200]
    db.commit()
    db.refresh(chat)
    return _session_out(db, chat)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: DbSession, user: CurrentUser) -> None:
    chat = get_owned_chat(session_id, db, user)
    db.delete(chat)
    db.commit()


@router.get("/sessions/{session_id}/messages", response_model=list[MessageOut])
def list_messages(session_id: str, db: DbSession, user: CurrentUser) -> list[MessageOut]:
    chat = get_owned_chat(session_id, db, user)
    rows = db.execute(
        select(models.Message)
        .where(models.Message.session_id == chat.id)
        .order_by(models.Message.created_at.asc())
    ).scalars().all()
    return [
        chat_service.serialise_message(db, row, document_service.media_url) for row in rows
    ]


# ---------------------------------------------------------------------------
# Asking
# ---------------------------------------------------------------------------
def _build_state(db, chat: models.ChatSession, user, question: str, force_route: str | None):
    history, summary = chat_service.load_history(db, chat.id)
    # The question has already been persisted by the caller, so it comes back as
    # the last history entry. Drop it, or the model sees it twice.
    if history and history[-1].role == "user":
        history = history[:-1]
    return AgentState(
        question=question,
        user_id=user.id,
        kb_id=chat.kb_id,
        session_id=chat.id,
        history=history,
        history_summary=summary,
        forced_route=Route.parse(force_route),
    )


@router.post("/sessions/{session_id}/ask", response_model=AskResponse)
async def ask(
    session_id: str,
    payload: AskRequest,
    db: DbSession,
    user: CurrentUser,
    key: Annotated[str, Depends(client_key)],
) -> AskResponse:
    """Non-streaming ask. Returns the full answer with citations in one response."""
    chat_limiter.check(key)
    chat = get_owned_chat(session_id, db, user)

    question = payload.question.strip()
    if not question:
        raise ValidationError("Type a question first.")

    user_message = chat_service.save_user_message(db, chat.id, question)
    state = _build_state(db, chat, user, question, payload.force_route)

    await orchestrator.run(state)

    assistant_message = chat_service.save_assistant_message(db, chat.id, state)
    await chat_service.maybe_title_session(db, chat.id, question)
    await chat_service.maybe_summarise_history(db, chat.id)

    return AskResponse(
        user_message=chat_service.serialise_message(
            db, user_message, document_service.media_url
        ),
        assistant_message=chat_service.serialise_message(
            db, assistant_message, document_service.media_url
        ),
    )


def _sse(event: str, data) -> str:
    """Format one Server-Sent Event frame."""
    return "event: {}\ndata: {}\n\n".format(event, json.dumps(data, default=str))


@router.post("/sessions/{session_id}/ask/stream")
async def ask_stream(
    session_id: str,
    payload: AskRequest,
    db: DbSession,
    user: CurrentUser,
    key: Annotated[str, Depends(client_key)],
) -> StreamingResponse:
    """Streaming ask over SSE.

    The request-scoped session is used for validation only; the generator opens
    its own session because the dependency's session closes when this function
    returns, well before the stream finishes.
    """
    chat_limiter.check(key)
    chat = get_owned_chat(session_id, db, user)

    question = payload.question.strip()
    if not question:
        raise ValidationError("Type a question first.")

    chat_id = chat.id
    kb_id = chat.kb_id
    user_id = user.id
    force_route = payload.force_route

    async def event_stream():
        db_stream = SessionLocal()
        try:
            user_message = chat_service.save_user_message(db_stream, chat_id, question)
            yield _sse(
                "user_message",
                {"id": user_message.id, "content": question, "role": "user"},
            )

            history, summary = chat_service.load_history(db_stream, chat_id)
            # load_history includes the message just saved; drop it so the
            # question is not duplicated as both history and query.
            if history and history[-1].role == "user":
                history = history[:-1]

            state = AgentState(
                question=question,
                user_id=user_id,
                kb_id=kb_id,
                session_id=chat_id,
                history=history,
                history_summary=summary,
                forced_route=Route.parse(force_route),
            )

            async for event_name, data in orchestrator.run_stream(state):
                if event_name == "token":
                    yield _sse("token", {"text": data})
                else:
                    yield _sse(event_name, data)

            assistant_message = chat_service.save_assistant_message(db_stream, chat_id, state)
            await chat_service.maybe_title_session(db_stream, chat_id, question)
            await chat_service.maybe_summarise_history(db_stream, chat_id)

            refreshed = db_stream.get(models.ChatSession, chat_id)
            serialised = chat_service.serialise_message(
                db_stream, assistant_message, document_service.media_url
            )

            yield _sse(
                "saved",
                {
                    "message": json.loads(serialised.model_dump_json()),
                    "session_title": refreshed.title if refreshed else None,
                },
            )
        except Exception as exc:  # noqa: BLE001 - the stream owns its errors
            logger.exception("Streaming ask failed")
            yield _sse("error", {"message": "Something went wrong: {}".format(str(exc)[:200])})
        finally:
            db_stream.close()
            yield _sse("end", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            # Prevents buffering if the app is ever put behind nginx.
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/citations/{chunk_id}")
def expand_citation(chunk_id: str, db: DbSession, user: CurrentUser) -> dict:
    """Full text and surrounding context for one citation."""
    from app.core.errors import NotFoundError
    from app.services import citation_service

    result = citation_service.expand_citation(db, chunk_id, user.id)
    if result is None:
        raise NotFoundError("That source is no longer available.")
    return result
