"""Password hashing and opaque session tokens.

bcrypt is used directly rather than through passlib: passlib 1.7.4 reads
bcrypt.__about__, which bcrypt 4.x removed, producing the well-known
"AttributeError: module 'bcrypt' has no attribute '__about__'" crash.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt

# bcrypt hashes at most 72 bytes and raises on longer input in 4.x.
BCRYPT_MAX_BYTES = 72
TOKEN_BYTES = 32


def hash_password(password: str) -> str:
    payload = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(payload, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        payload = password.encode("utf-8")[:BCRYPT_MAX_BYTES]
        return bcrypt.checkpw(payload, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed stored hash: treat as a failed login, never a 500.
        return False


def generate_token() -> str:
    """The raw bearer token. Returned to the client exactly once."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    """What actually gets stored. A DB leak must not yield usable tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def tokens_match(candidate: str, stored_hash: str) -> bool:
    return hmac.compare_digest(hash_token(candidate), stored_hash)


def session_expiry(hours: int) -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=hours)


def is_expired(expires_at: datetime) -> bool:
    # Rows read back from SQLite are naive; assume UTC, which is how we wrote them.
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) >= expires_at


def validate_username(username: str) -> str | None:
    """Returns an error message, or None when the username is acceptable."""
    name = (username or "").strip()
    if len(name) < 3:
        return "Username must be at least 3 characters."
    if len(name) > 64:
        return "Username must be 64 characters or fewer."
    if not all(c.isalnum() or c in "_-." for c in name):
        return "Username may only contain letters, numbers, and _ - ."
    return None


def validate_password(password: str) -> str | None:
    if len(password or "") < 8:
        return "Password must be at least 8 characters."
    if len(password) > 256:
        return "Password must be 256 characters or fewer."
    return None
