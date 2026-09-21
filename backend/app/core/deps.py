"""FastAPI dependencies: authentication and the ownership guards.

Every resource lookup in the API goes through one of the guards here. A route
must never fetch a KnowledgeBase / ChatSession / Document by id alone -- the
guard re-checks user_id on the row itself, so a forged id yields 404, not data.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import AuthError, NotFoundError
from app.core.security import hash_token, is_expired
from app.db import models
from app.db.base import get_db

DbSession = Annotated[Session, Depends(get_db)]


def _extract_token(authorization: str | None, x_session_token: str | None) -> str:
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
    if x_session_token:
        return x_session_token.strip()
    raise AuthError("Not signed in.")


def get_current_user(
    db: DbSession,
    authorization: Annotated[str | None, Header()] = None,
    x_session_token: Annotated[str | None, Header()] = None,
) -> models.User:
    token = _extract_token(authorization, x_session_token)
    token_hash = hash_token(token)

    auth_session = db.execute(
        select(models.AuthSession).where(models.AuthSession.token_hash == token_hash)
    ).scalar_one_or_none()

    if auth_session is None:
        raise AuthError("Session is invalid. Please sign in again.")

    if is_expired(auth_session.expires_at):
        db.delete(auth_session)
        db.commit()
        raise AuthError("Session expired. Please sign in again.")

    user = db.get(models.User, auth_session.user_id)
    if user is None or not user.is_active:
        raise AuthError("Account is unavailable.")

    # Cheap last-seen tracking; avoids a write on every single request.
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    last_seen = auth_session.last_seen_at
    if last_seen.tzinfo is None:
        last_seen = last_seen.replace(tzinfo=timezone.utc)
    if (now - last_seen).total_seconds() > 300:
        auth_session.last_seen_at = now
        db.commit()

    return user


CurrentUser = Annotated[models.User, Depends(get_current_user)]


def get_owned_kb(kb_id: str, db: Session, user: models.User) -> models.KnowledgeBase:
    kb = db.execute(
        select(models.KnowledgeBase).where(
            models.KnowledgeBase.id == kb_id,
            models.KnowledgeBase.user_id == user.id,
        )
    ).scalar_one_or_none()
    if kb is None:
        raise NotFoundError("Knowledge base not found.")
    return kb


def get_owned_document(doc_id: str, db: Session, user: models.User) -> models.Document:
    doc = db.execute(
        select(models.Document).where(
            models.Document.id == doc_id,
            models.Document.user_id == user.id,
        )
    ).scalar_one_or_none()
    if doc is None:
        raise NotFoundError("Document not found.")
    return doc


def get_owned_chat(session_id: str, db: Session, user: models.User) -> models.ChatSession:
    chat = db.execute(
        select(models.ChatSession).where(
            models.ChatSession.id == session_id,
            models.ChatSession.user_id == user.id,
        )
    ).scalar_one_or_none()
    if chat is None:
        raise NotFoundError("Chat session not found.")
    return chat


def get_owned_media(asset_id: str, db: Session, user: models.User) -> models.MediaAsset:
    asset = db.execute(
        select(models.MediaAsset).where(
            models.MediaAsset.id == asset_id,
            models.MediaAsset.user_id == user.id,
        )
    ).scalar_one_or_none()
    if asset is None:
        raise NotFoundError("Image not found.")
    return asset


def get_owned_job(job_id: str, db: Session, user: models.User) -> models.IngestionJob:
    job = db.execute(
        select(models.IngestionJob).where(
            models.IngestionJob.id == job_id,
            models.IngestionJob.user_id == user.id,
        )
    ).scalar_one_or_none()
    if job is None:
        raise NotFoundError("Ingestion job not found.")
    return job


def client_key(
    authorization: Annotated[str | None, Header()] = None,
    x_forwarded_for: Annotated[str | None, Header()] = None,
) -> str:
    """Rate-limit bucket key: the session when present, else the client address."""
    if authorization:
        return hash_token(authorization)[:32]
    return (x_forwarded_for or "local").split(",")[0].strip()


def settings_dep():
    return settings
