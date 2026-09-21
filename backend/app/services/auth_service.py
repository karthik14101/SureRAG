"""Sign-up, sign-in and session lifecycle."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.core.errors import AuthError, ConflictError, ValidationError
from app.core.security import (
    generate_token,
    hash_password,
    hash_token,
    session_expiry,
    validate_password,
    validate_username,
    verify_password,
)
from app.db import models
from app.logging_conf import get_logger

logger = get_logger(__name__)


def create_user(db: Session, username: str, password: str, display_name: str | None) -> models.User:
    username = (username or "").strip()

    error = validate_username(username)
    if error:
        raise ValidationError(error)
    error = validate_password(password)
    if error:
        raise ValidationError(error)

    existing = db.execute(
        select(models.User).where(models.User.username == username)
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError("That username is already taken.")

    user = models.User(
        username=username,
        password_hash=hash_password(password),
        display_name=(display_name or "").strip() or None,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info("New user registered: %s", username)
    return user


def authenticate(db: Session, username: str, password: str) -> models.User:
    user = db.execute(
        select(models.User).where(models.User.username == (username or "").strip())
    ).scalar_one_or_none()

    # Same message either way, so the endpoint cannot be used to enumerate users.
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Incorrect username or password.")
    if not user.is_active:
        raise AuthError("This account has been deactivated.")

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    return user


def issue_session(db: Session, user: models.User) -> tuple[str, datetime]:
    """Create a session and return the raw token (shown to the client once)."""
    token = generate_token()
    expires_at = session_expiry(settings.session_ttl_hours)

    db.add(
        models.AuthSession(
            token_hash=hash_token(token),
            user_id=user.id,
            expires_at=expires_at,
        )
    )
    db.commit()
    return token, expires_at


def revoke_session(db: Session, token: str) -> None:
    session = db.execute(
        select(models.AuthSession).where(models.AuthSession.token_hash == hash_token(token))
    ).scalar_one_or_none()
    if session is not None:
        db.delete(session)
        db.commit()


def purge_expired_sessions(db: Session) -> int:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    expired = db.execute(
        select(models.AuthSession).where(models.AuthSession.expires_at < now)
    ).scalars().all()
    for session in expired:
        db.delete(session)
    if expired:
        db.commit()
    return len(expired)
