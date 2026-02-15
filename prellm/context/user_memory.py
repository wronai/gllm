"""UserMemory — stores user interaction history and learned preferences.

Backend: SQLite (MVP) → ChromaDB (production) → Redis (enterprise).
Uses synchronous SQLite for MVP simplicity; async wrappers provided for pipeline integration.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("prellm.context.user_memory")

_DEFAULT_DB_PATH = ".prellm/user_memory.db"


class UserMemory:
    """Stores user query history and learned preferences.

    Usage:
        memory = UserMemory(backend="sqlite", path=".prellm/user_memory.db")
        await memory.add_interaction("Deploy app", "Deployment plan...", {"intent": "deploy"})
        recent = await memory.get_recent_context("Deploy", limit=5)
        prefs = await memory.get_user_preferences()
    """

    def __init__(self, backend: str = "sqlite", path: str | Path = _DEFAULT_DB_PATH):
        self.backend = backend
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None

        if backend == "sqlite":
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initialize SQLite database and tables."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                response_summary TEXT NOT NULL DEFAULT '',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS preferences (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    async def add_interaction(
        self, query: str, response_summary: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Record a user interaction."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT INTO interactions (query, response_summary, metadata_json, timestamp) VALUES (?, ?, ?, ?)",
            (query, response_summary, json.dumps(metadata or {}), time.time()),
        )
        self._conn.commit()
        logger.debug(f"Stored interaction: {query[:80]}...")

    async def get_recent_context(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Get the most recent interactions, optionally filtered by relevance to query.

        MVP: returns last N interactions ordered by recency.
        Production: would use vector similarity search.
        """
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            "SELECT query, response_summary, metadata_json, timestamp "
            "FROM interactions ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        results = []
        for row in cursor.fetchall():
            results.append({
                "query": row["query"],
                "response_summary": row["response_summary"],
                "metadata": json.loads(row["metadata_json"]),
                "timestamp": row["timestamp"],
            })
        return results

    async def get_user_preferences(self) -> dict[str, str]:
        """Get all learned user preferences."""
        if self._conn is None:
            return {}
        cursor = self._conn.execute("SELECT key, value FROM preferences")
        return {row["key"]: row["value"] for row in cursor.fetchall()}

    async def set_preference(self, key: str, value: str) -> None:
        """Set a user preference."""
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        self._conn.commit()

    async def clear(self) -> None:
        """Clear all stored data (for testing)."""
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM interactions")
        self._conn.execute("DELETE FROM preferences")
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
