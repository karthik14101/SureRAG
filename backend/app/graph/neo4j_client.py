"""Async Neo4j driver, schema constraints, and a global availability flag.

The graph is an enhancement, not a hard dependency: if Neo4j is unreachable the
engine logs it once, marks the graph unavailable, and every GRAPH-routed query
falls back to vector search rather than erroring the user's chat.
"""
from __future__ import annotations

import threading

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import settings
from app.logging_conf import get_logger

logger = get_logger(__name__)

_driver: AsyncDriver | None = None
_lock = threading.Lock()
_available = False

CONSTRAINTS = [
    # Entities are unique per knowledge base, so two users' "Apple" never merge.
    "CREATE CONSTRAINT entity_key IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.kb_id, e.norm_name) IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX entity_kb IF NOT EXISTS FOR (e:Entity) ON (e.kb_id)",
    "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.norm_name)",
    "CREATE INDEX chunk_kb IF NOT EXISTS FOR (c:Chunk) ON (c.kb_id)",
    "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE INDEX document_kb IF NOT EXISTS FOR (d:Document) ON (d.kb_id)",
    # Full-text index powers fuzzy entity seeding from a natural-language query.
    "CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.name, e.description]",
]


def get_driver() -> AsyncDriver:
    global _driver
    if _driver is not None:
        return _driver
    with _lock:
        if _driver is None:
            _driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
                max_connection_pool_size=20,
                connection_acquisition_timeout=30,
            )
    return _driver


def is_available() -> bool:
    return _available


def set_available(value: bool) -> None:
    global _available
    _available = value


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        try:
            await _driver.close()
        except Exception:  # noqa: BLE001 - shutdown must not raise
            pass
        _driver = None


async def ensure_schema() -> bool:
    """Apply constraints/indexes. Returns whether the graph is usable."""
    try:
        driver = get_driver()
        async with driver.session(database=settings.neo4j_database) as session:
            for statement in CONSTRAINTS + INDEXES:
                try:
                    await session.run(statement)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Neo4j schema statement skipped: %s", str(exc)[:160])
        set_available(True)
        logger.info("Neo4j ready at %s", settings.neo4j_uri)
        return True
    except Exception as exc:  # noqa: BLE001
        set_available(False)
        logger.warning(
            "Neo4j unavailable at %s -- graph features are disabled, vector search "
            "still works. (%s)",
            settings.neo4j_uri,
            str(exc)[:200],
        )
        return False


async def run_read(cypher: str, params: dict | None = None) -> list[dict]:
    if not _available:
        return []
    try:
        driver = get_driver()
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j read failed: %s", str(exc)[:200])
        return []


async def run_write(cypher: str, params: dict | None = None) -> bool:
    if not _available:
        return False
    try:
        driver = get_driver()
        async with driver.session(database=settings.neo4j_database) as session:
            await session.run(cypher, params or {})
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Neo4j write failed: %s", str(exc)[:200])
        return False


async def health() -> tuple[bool, str]:
    try:
        driver = get_driver()
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run("RETURN 1 AS ok")
            record = await result.single()
            ok = bool(record and record.get("ok") == 1)
        set_available(ok)
        return ok, "bolt connection ok" if ok else "unexpected response"
    except Exception as exc:  # noqa: BLE001
        set_available(False)
        return False, str(exc)[:200]
