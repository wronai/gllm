"""Tests for v0.4 session persistence — UserMemory export/import/RAG/auto-inject."""

from __future__ import annotations

from pathlib import Path

import pytest

from prellm.context.user_memory import UserMemory
from prellm.models import SessionSnapshot


@pytest.fixture
async def memory(tmp_path: Path) -> UserMemory:
    db_path = tmp_path / "test_session.db"
    mem = UserMemory(backend="sqlite", path=db_path)
    yield mem
    mem.close()


@pytest.mark.asyncio
async def test_export_session_creates_snapshot(memory: UserMemory):
    """export_session returns a SessionSnapshot with interactions."""
    await memory.add_interaction("Deploy app", "Plan created", {"intent": "deploy"})
    await memory.add_interaction("Check status", "Status OK", {"intent": "query"})

    snapshot = await memory.export_session(session_id="test-export")
    assert isinstance(snapshot, SessionSnapshot)
    assert snapshot.session_id == "test-export"
    assert len(snapshot.interactions) == 2


@pytest.mark.asyncio
async def test_export_session_includes_preferences(memory: UserMemory):
    """export_session includes user preferences."""
    await memory.set_preference("lang", "pl")
    await memory.set_preference("format", "json")

    snapshot = await memory.export_session()
    assert snapshot.preferences["lang"] == "pl"
    assert snapshot.preferences["format"] == "json"


@pytest.mark.asyncio
async def test_import_session_restores_state(tmp_path: Path):
    """import_session restores interactions and preferences."""
    # Export from source
    source = UserMemory(path=tmp_path / "source.db")
    await source.add_interaction("Q1", "A1", {"k": "v"})
    await source.set_preference("lang", "pl")
    snapshot = await source.export_session(session_id="roundtrip")
    source.close()

    # Import to target
    target = UserMemory(path=tmp_path / "target.db")
    await target.import_session(snapshot)

    recent = await target.get_recent_context("Q1", limit=5)
    assert len(recent) == 1
    assert recent[0]["query"] == "Q1"

    prefs = await target.get_user_preferences()
    assert prefs["lang"] == "pl"
    target.close()


@pytest.mark.asyncio
async def test_get_relevant_context_uses_history(memory: UserMemory):
    """get_relevant_context returns formatted history text."""
    await memory.add_interaction("Deploy to prod", "Deployed via CI/CD", {})
    await memory.add_interaction("Check logs", "Logs are clean", {})

    context = await memory.get_relevant_context("Deploy")
    assert "Deploy to prod" in context or "Check logs" in context
    assert context != ""


@pytest.mark.asyncio
async def test_get_relevant_context_respects_token_limit(memory: UserMemory):
    """get_relevant_context stops before exceeding max_tokens."""
    for i in range(50):
        await memory.add_interaction(f"Long query {i} " * 20, f"Long response {i} " * 20, {})

    context = await memory.get_relevant_context("query", max_tokens=100)
    # Should be roughly within token budget
    est_tokens = len(context) // 4
    assert est_tokens <= 200  # generous margin


@pytest.mark.asyncio
async def test_get_relevant_context_empty_history(memory: UserMemory):
    """get_relevant_context returns empty string for no history."""
    context = await memory.get_relevant_context("anything")
    assert context == ""


@pytest.mark.asyncio
async def test_auto_inject_context_builds_system_prompt(memory: UserMemory):
    """auto_inject_context combines preferences + history into prompt."""
    await memory.set_preference("lang", "pl")
    await memory.add_interaction("Deploy app", "Deployment plan", {})

    result = await memory.auto_inject_context("Deploy")
    assert "lang=pl" in result
    assert "Deploy app" in result or "Deployment plan" in result


@pytest.mark.asyncio
async def test_auto_inject_with_empty_history(memory: UserMemory):
    """auto_inject_context works with no history."""
    result = await memory.auto_inject_context("Test query")
    assert result == ""  # no prefs, no history


@pytest.mark.asyncio
async def test_auto_inject_with_system_prompt(memory: UserMemory):
    """auto_inject_context prepends existing system prompt."""
    await memory.set_preference("format", "yaml")
    result = await memory.auto_inject_context("Test", system_prompt="You are a helpful assistant.")
    assert result.startswith("You are a helpful assistant.")
    assert "format=yaml" in result


@pytest.mark.asyncio
async def test_learn_preference_from_interaction(memory: UserMemory):
    """learn_preference_from_interaction detects preference patterns."""
    await memory.learn_preference_from_interaction(
        "I always use Python for scripts",
        "Noted, using Python."
    )
    # Pattern: "always use <tool>" → preferred_tool
    prefs = await memory.get_user_preferences()
    assert "preferred_tool" in prefs


@pytest.mark.asyncio
async def test_session_roundtrip_file_export_import(tmp_path: Path):
    """Full roundtrip: add data → export to file → import from file."""
    mem1 = UserMemory(path=tmp_path / "mem1.db")
    await mem1.add_interaction("Query 1", "Response 1", {"k": "v"})
    await mem1.set_preference("lang", "pl")

    snapshot = await mem1.export_session(session_id="file-rt")
    json_path = tmp_path / "session_export.json"
    snapshot.to_file(json_path)
    mem1.close()

    # Import from file
    loaded = SessionSnapshot.from_file(json_path)
    assert loaded.session_id == "file-rt"

    mem2 = UserMemory(path=tmp_path / "mem2.db")
    await mem2.import_session(loaded)

    recent = await mem2.get_recent_context("Query 1", limit=5)
    assert len(recent) == 1
    prefs = await mem2.get_user_preferences()
    assert prefs["lang"] == "pl"
    mem2.close()
