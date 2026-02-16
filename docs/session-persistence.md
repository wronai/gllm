# Session Persistence

> **v0.4.0** — Export, import, and auto-inject session context for small LLMs.

## Problem

Small LLMs lose context between sessions. After restarting the pipeline, Bielik/Qwen have no memory of previous interactions, preferences, or decisions. Users must re-explain context every time.

## Solution

`UserMemory` now supports full session lifecycle:

- **Export** session to JSON (like LM Studio "save session")
- **Import** session from JSON (resume with full context)
- **RAG retrieval** — similarity-based context injection from history
- **Auto-inject** — build enriched system prompt from preferences + history
- **Auto-learn** — extract preferences from interactions (like Oobabooga Dynamic Context)

## Quick Start

### Enable Session Persistence

```python
from prellm import preprocess_and_execute

result = await preprocess_and_execute(
    query="Deploy ESP32 firmware",
    small_llm="ollama/bielik:7b",
    large_llm="openrouter/google/gemini-3-flash-preview",
    session_path=".prellm/sessions.db",   # enables persistence
)
# Interaction is automatically saved to sessions.db
# Preferences are auto-learned from the exchange
```

### Export and Import Sessions

```python
from prellm.context.user_memory import UserMemory

# Export current session
memory = UserMemory(path=".prellm/sessions.db")
snapshot = await memory.export_session(session_id="esp32-project")
snapshot.to_file("session_backup.json")

# Import on another machine or after restart
from prellm.models import SessionSnapshot
loaded = SessionSnapshot.from_file("session_backup.json")
new_memory = UserMemory(path=".prellm/new_sessions.db")
await new_memory.import_session(loaded)
```

### RAG-Style Context Retrieval

```python
memory = UserMemory(path=".prellm/sessions.db")

# Get relevant history fragments (token-budgeted)
context = await memory.get_relevant_context("Deploy firmware", max_tokens=1024)
# Returns: "Q: Deploy ESP32 firmware\nA: Used OTA update via..."

# Build enriched system prompt automatically
system_prompt = await memory.auto_inject_context("Deploy firmware")
# Returns: "User preferences: lang=pl, preferred_tool=platformio\n\nRelevant history:\n..."
```

## CLI Commands

```bash
# List recent interactions
prellm session list
prellm session list --memory .prellm/custom.db

# Export session to JSON
prellm session export session_backup.json
prellm session export backup.json --id my-project --memory .prellm/sessions.db

# Import session from JSON
prellm session import session_backup.json
prellm session import backup.json --memory .prellm/new.db

# Clear all session data
prellm session clear
prellm session clear --force   # skip confirmation
```

## SessionSnapshot Format

The exported JSON contains:

```json
{
  "session_id": "esp32-project",
  "interactions": [
    {
      "query": "Deploy ESP32 firmware",
      "response_summary": "Used OTA update via PlatformIO...",
      "metadata": {"intent": "deploy"},
      "timestamp": 1708000000.0
    }
  ],
  "preferences": {
    "preferred_tool": "platformio",
    "preferred_language": "cpp",
    "lang": "pl"
  },
  "runtime_context": null,
  "codebase_summary": null,
  "created_at": "1708000000.0",
  "exported_at": "1708100000.0"
}
```

## Auto-Learn Preferences

`learn_preference_from_interaction()` detects patterns in exchanges:

| Pattern | Detected preference |
| --- | --- |
| "I always use Python" | `preferred_tool=python` |
| "preferuję Rust" | `preferred_tool=rust` |
| "language: Polish" | `preferred_language=polish` |
| "format as YAML" | `preferred_format=yaml` |

Preferences persist across sessions and are auto-injected into future prompts.

## How It Integrates

When `session_path` is set in `preprocess_and_execute()`:

1. **Before preprocessing** — `auto_inject_context()` enriches the small-LLM prompt with relevant history + preferences
2. **After execution** — interaction is saved to `UserMemory`
3. **After execution** — `learn_preference_from_interaction()` extracts new preferences

This happens automatically — no manual code needed.

## Related Docs

- [Persistent Context](persistent-context.md) — full context layer overview
- [Sensitive Data Filtering](sensitive-data.md) — what gets filtered
- [CHANGELOG](../CHANGELOG.md) — v0.4.0 details
