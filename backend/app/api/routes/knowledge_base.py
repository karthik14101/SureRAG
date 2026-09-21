"""Knowledge base endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.deps import CurrentUser, DbSession, get_owned_kb
from app.schemas.kb import KBCreate, KBOut, KBStats, KBUpdate
from app.services import cleanup_service, kb_service

router = APIRouter(prefix="/kb", tags=["knowledge-base"])


@router.get("", response_model=list[KBOut])
def list_knowledge_bases(db: DbSession, user: CurrentUser) -> list[KBOut]:
    return [KBOut.model_validate(kb) for kb in kb_service.list_kbs(db, user.id)]


@router.post("", response_model=KBOut, status_code=201)
def create_knowledge_base(payload: KBCreate, db: DbSession, user: CurrentUser) -> KBOut:
    kb = kb_service.create_kb(db, user.id, payload.name, payload.description)
    return KBOut.model_validate(kb)


@router.get("/{kb_id}", response_model=KBOut)
def get_knowledge_base(kb_id: str, db: DbSession, user: CurrentUser) -> KBOut:
    return KBOut.model_validate(get_owned_kb(kb_id, db, user))


@router.patch("/{kb_id}", response_model=KBOut)
def update_knowledge_base(
    kb_id: str, payload: KBUpdate, db: DbSession, user: CurrentUser
) -> KBOut:
    kb = get_owned_kb(kb_id, db, user)
    return KBOut.model_validate(
        kb_service.update_kb(db, kb, payload.name, payload.description)
    )


@router.delete("/{kb_id}", status_code=204)
async def delete_knowledge_base(kb_id: str, db: DbSession, user: CurrentUser) -> None:
    kb = get_owned_kb(kb_id, db, user)
    await cleanup_service.delete_knowledge_base(db, kb)


@router.get("/{kb_id}/stats", response_model=KBStats)
async def knowledge_base_stats(kb_id: str, db: DbSession, user: CurrentUser) -> KBStats:
    kb = get_owned_kb(kb_id, db, user)
    return await kb_service.kb_statistics(db, kb)
