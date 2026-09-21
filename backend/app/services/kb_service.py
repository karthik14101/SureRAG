"""Knowledge base CRUD and statistics."""
from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.errors import ConflictError, ValidationError
from app.db import models
from app.graph.writer import graph_stats
from app.logging_conf import get_logger
from app.schemas.kb import KBStats

logger = get_logger(__name__)


def create_kb(db: Session, user_id: str, name: str, description: str | None) -> models.KnowledgeBase:
    name = (name or "").strip()
    if not name:
        raise ValidationError("Give the knowledge base a name.")
    if len(name) > 120:
        raise ValidationError("The name must be 120 characters or fewer.")

    existing = db.execute(
        select(models.KnowledgeBase).where(
            models.KnowledgeBase.user_id == user_id,
            models.KnowledgeBase.name == name,
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ConflictError("You already have a knowledge base called '{}'.".format(name))

    kb = models.KnowledgeBase(
        user_id=user_id,
        name=name,
        description=(description or "").strip() or None,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info("Created knowledge base '%s' for user %s", name, user_id)
    return kb


def list_kbs(db: Session, user_id: str) -> list[models.KnowledgeBase]:
    return list(
        db.execute(
            select(models.KnowledgeBase)
            .where(models.KnowledgeBase.user_id == user_id)
            .order_by(models.KnowledgeBase.updated_at.desc())
        ).scalars().all()
    )


def update_kb(
    db: Session, kb: models.KnowledgeBase, name: str | None, description: str | None
) -> models.KnowledgeBase:
    if name is not None:
        cleaned = name.strip()
        if not cleaned:
            raise ValidationError("The name cannot be empty.")
        if cleaned != kb.name:
            clash = db.execute(
                select(models.KnowledgeBase).where(
                    models.KnowledgeBase.user_id == kb.user_id,
                    models.KnowledgeBase.name == cleaned,
                    models.KnowledgeBase.id != kb.id,
                )
            ).scalar_one_or_none()
            if clash is not None:
                raise ConflictError("You already have a knowledge base with that name.")
        kb.name = cleaned

    if description is not None:
        kb.description = description.strip() or None

    db.commit()
    db.refresh(kb)
    return kb


async def kb_statistics(db: Session, kb: models.KnowledgeBase) -> KBStats:
    counts = db.execute(
        select(models.Document.status, func.count(models.Document.id))
        .where(models.Document.kb_id == kb.id)
        .group_by(models.Document.status)
    ).all()
    by_status = {status: count for status, count in counts}

    total_bytes = db.execute(
        select(func.coalesce(func.sum(models.Document.size_bytes), 0)).where(
            models.Document.kb_id == kb.id
        )
    ).scalar_one()

    pending = sum(
        by_status.get(state, 0) for state in ("pending", "parsing", "embedding", "graphing")
    )

    stats = await graph_stats(kb.id)

    return KBStats(
        id=kb.id,
        name=kb.name,
        doc_count=kb.doc_count,
        chunk_count=kb.chunk_count,
        image_count=kb.image_count,
        entity_count=int(stats.get("entities") or 0),
        relation_count=int(stats.get("relations") or 0),
        ready_docs=by_status.get("ready", 0),
        failed_docs=by_status.get("failed", 0),
        pending_docs=pending,
        total_bytes=int(total_bytes or 0),
        graph_available=bool(stats.get("available")),
    )
