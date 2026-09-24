"""Isolation helpers shared by the pytest fixtures and the standalone runner.

The suite has to work without pytest installed, so the per-test setup lives here
as plain functions rather than only as fixtures. `conftest.py` wraps these for
pytest; `run.py` calls them directly.
"""
from __future__ import annotations

import pathlib
import sys

BACKEND_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def snapshot_settings() -> dict:
    """Copy the settings singleton so a test can be undone.

    Most of these tests poke at `settings` -- switching provider, toggling
    reranking, moving a threshold. Without restoring it, the suite's result
    would depend on the order the tests happened to run in.
    """
    from app.config import settings

    return settings.model_dump()


def restore_settings(saved: dict) -> None:
    from app.config import settings

    for field, value in saved.items():
        setattr(settings, field, value)


def reset_providers() -> None:
    """Drop cached provider instances.

    The factory memoises by tier, so a provider built under one configuration
    would otherwise be handed to the next test unchanged.
    """
    from app.llm import factory

    factory.reset_llm()
