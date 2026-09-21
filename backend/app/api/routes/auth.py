"""Authentication endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Header

from app.core.deps import CurrentUser, DbSession, client_key
from app.core.rate_limit import login_limiter, signup_limiter
from app.schemas.auth import AuthResponse, SignInRequest, SignUpRequest, UserOut
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/signup", response_model=AuthResponse, status_code=201)
def sign_up(
    payload: SignUpRequest,
    db: DbSession,
    key: Annotated[str, Depends(client_key)],
) -> AuthResponse:
    signup_limiter.check(key)
    user = auth_service.create_user(
        db, payload.username, payload.password, payload.display_name
    )
    token, expires_at = auth_service.issue_session(db, user)
    return AuthResponse(
        token=token, expires_at=expires_at, user=UserOut.model_validate(user)
    )


@router.post("/signin", response_model=AuthResponse)
def sign_in(
    payload: SignInRequest,
    db: DbSession,
    key: Annotated[str, Depends(client_key)],
) -> AuthResponse:
    login_limiter.check(key)
    user = auth_service.authenticate(db, payload.username, payload.password)
    token, expires_at = auth_service.issue_session(db, user)
    return AuthResponse(
        token=token, expires_at=expires_at, user=UserOut.model_validate(user)
    )


@router.post("/signout", status_code=204)
def sign_out(
    db: DbSession,
    _user: CurrentUser,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    if authorization:
        _, _, token = authorization.partition(" ")
        if token.strip():
            auth_service.revoke_session(db, token.strip())


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
