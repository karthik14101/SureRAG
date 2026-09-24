"""Pytest wiring for the test suite.

These tests run offline. Nothing here reaches Azure, Qdrant or Neo4j: providers
are stubbed, the vector store is faked, and anything needing a database gets a
throwaway SQLite file. That is deliberate -- a suite you hesitate to run because
it costs quota is a suite that stops being run.

Everything substantive lives in `_support.py`, so the same isolation applies
when the suite is run through `run.py` without pytest installed.
"""
from __future__ import annotations

import pytest

from tests._support import reset_providers, restore_settings, snapshot_settings


@pytest.fixture(autouse=True)
def isolate():
    saved = snapshot_settings()
    reset_providers()
    yield
    restore_settings(saved)
    reset_providers()
