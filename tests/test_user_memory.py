"""Tests for UserMemory — SQLite-backed user interaction history and preferences."""

from __future__ import annotations

from pathlib import Path

import pytest

from prellm.context.user_memory import UserMemory


@pytest.fixture
async def memory(tmp_path: Path) -> UserMemory:
    db_path = tmp_path / "test_memory.db"
    mem = UserMemory(backend="sqlite", path=db_path)
    yield mem
    mem.close()


@pytest.mark.asyncio
async def test_user_memory_add_interaction(memory: UserMemory):
    """UserMemory stores and retrieves interactions."""
    await memory.add_interaction("Deploy app", "Deployment plan created", {"intent": "deploy"})
    recent = await memory.get_recent_context("Deploy", limit=5)
    assert len(recent) == 1
    assert recent[0]["query"] == "Deploy app"
    assert recent[0]["response_summary"] == "Deployment plan created"
    assert recent[0]["metadata"]["intent"] == "deploy"


@pytest.mark.asyncio
async def test_user_memory_get_recent(memory: UserMemory):
    """UserMemory returns most recent interactions up to limit."""
    for i in range(10):
        await memory.add_interaction(f"Query {i}", f"Response {i}", {"i": i})

    recent = await memory.get_recent_context("Query", limit=3)
    assert len(recent) == 3
    # Most recent first
    assert recent[0]["query"] == "Query 9"
    assert recent[1]["query"] == "Query 8"
    assert recent[2]["query"] == "Query 7"


@pytest.mark.asyncio
async def test_user_memory_preferences(memory: UserMemory):
    """UserMemory stores and retrieves preferences."""
    await memory.set_preference("language", "pl")
    await memory.set_preference("output_format", "json")

    prefs = await memory.get_user_preferences()
    assert prefs["language"] == "pl"
    assert prefs["output_format"] == "json"


@pytest.mark.asyncio
async def test_user_memory_preference_update(memory: UserMemory):
    """UserMemory overwrites preferences on update."""
    await memory.set_preference("language", "pl")
    await memory.set_preference("language", "en")

    prefs = await memory.get_user_preferences()
    assert prefs["language"] == "en"


@pytest.mark.asyncio
async def test_user_memory_sqlite_backend(tmp_path: Path):
    """UserMemory creates SQLite file on disk."""
    db_path = tmp_path / "subdir" / "memory.db"
    mem = UserMemory(backend="sqlite", path=db_path)
    await mem.add_interaction("test", "test response", {})
    mem.close()
    assert db_path.exists()


@pytest.mark.asyncio
async def test_user_memory_empty_history(memory: UserMemory):
    """UserMemory returns empty list for no interactions."""
    recent = await memory.get_recent_context("anything")
    assert recent == []

    prefs = await memory.get_user_preferences()
    assert prefs == {}


@pytest.mark.asyncio
async def test_user_memory_clear(memory: UserMemory):
    """UserMemory.clear() removes all data."""
    await memory.add_interaction("Deploy app", "Plan", {})
    await memory.set_preference("lang", "pl")
    await memory.clear()

    assert await memory.get_recent_context("Deploy") == []
    assert await memory.get_user_preferences() == {}
