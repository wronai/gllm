# Persistent Context for Small LLMs

> **v0.4.0** — preLLM becomes an automatic persistent context layer for small LLMs.

## Problem

Small LLMs (Bielik 7B/11B, Qwen 3B, Phi3) lose context after 5–10 exchanges, hallucinate without pre-prompts, and don't know the execution environment. Users must manually craft system prompts with env info, project structure, and session history.

## Solution

preLLM automatically:

1. **Collects** runtime context (env, process, locale, network, git, system)
2. **Compresses** codebase into token-efficient `.toon` format
3. **Persists** session history across restarts (SQLite)
4. **Retrieves** relevant context via RAG-style similarity search
5. **Filters** sensitive data (API keys, tokens) before the large-LLM

All of this happens with **zero manual pre-prompts**.

## Architecture

```text
User Query
    │
    ▼
┌─────────────────────────────────────────────────┐
│  CONTEXT LAYER (automatic)                      │
│                                                 │
│  RuntimeContext      → env, process, locale,    │
│                        network, git, system     │
│  SessionPersistence  → SQLite history + prefs   │
│  CodebaseCompressor  → .toon project summary    │
│  SensitiveFilter     → block API keys/tokens    │
└──────────────┬──────────────────────────────────┘
               │ enriched context
               ▼
┌─────────────────────────────────────────────────┐
│  PREPROCESSOR AGENT (small LLM ≤24B)            │
│  auto-strategy selection + pipeline execution   │
│  receives: query + runtime + history + codebase │
└──────────────┬──────────────────────────────────┘
               │ structured executor_input
               ▼
┌─────────────────────────────────────────────────┐
│  EXECUTOR AGENT (large LLM)                     │
│  receives: sanitized prompt (no secrets)        │
└─────────────────────────────────────────────────┘
```

## Quick Start

```python
from prellm import preprocess_and_execute

# Zero-config — everything is automatic in v0.4
result = await preprocess_and_execute(
    query="Zoptymalizuj monitoring ESP32",
    small_llm="ollama/bielik:7b",
    large_llm="openrouter/google/gemini-3-flash-preview",
)
# Bielik receives: RuntimeContext + auto-selected strategy
# Gemini receives: sanitized prompt (no API keys)
```

### Full Persistent Context

```python
result = await preprocess_and_execute(
    query="Zoptymalizuj monitoring ESP32",
    small_llm="ollama/bielik:7b",
    large_llm="openrouter/google/gemini-3-flash-preview",
    strategy="auto",                        # auto-select best strategy
    collect_runtime=True,                   # full env/shell snapshot
    session_path=".prellm/sessions.db",     # persistent history
    codebase_path=".",                      # compress project → .toon
    sanitize=True,                          # filter secrets
)
```

## RuntimeContext

The `RuntimeContext` model captures the full execution environment:

```python
from prellm.analyzers.context_engine import ContextEngine

engine = ContextEngine()
runtime = engine.gather_runtime()

print(runtime.env_safe)               # filtered env vars (no secrets)
print(runtime.process)                # {"pid": 1234, "cwd": "/project", ...}
print(runtime.locale)                 # {"lang": "pl_PL.UTF-8", "timezone": "CET", ...}
print(runtime.network)                # {"hostname": "nvidia", "local_ip": "192.168.1.10"}
print(runtime.git)                    # {"branch": "main", "short_sha": "abc1234"}
print(runtime.system)                 # {"os": "Linux", "arch": "x86_64", "python": "3.13"}
print(runtime.sensitive_blocked_count) # 7 (env vars blocked)
print(runtime.token_estimate)          # 350 tokens
```

### CLI Inspection

```bash
prellm context show                   # formatted runtime context
prellm context show --json            # as JSON
prellm context show --blocked         # show env vars + what was blocked
prellm context show --codebase .      # include compressed project
```

## New API Parameters

| Parameter | Default | Description |
|---|---|---|
| `strategy` | `"auto"` | Small-LLM auto-selects strategy (was: `"classify"`) |
| `collect_runtime` | `True` | Collect full env/process/locale/network/git/system |
| `session_path` | `None` | Path to session persistence SQLite DB |
| `sanitize` | `True` | Filter sensitive data before large-LLM |
| `sensitive_rules` | `None` | Custom YAML rules for sensitive data |
| `codebase_path` | `None` | Folder to compress for context |

## Context-Aware Pipeline

The new `context_aware` pipeline in `configs/pipelines.yaml` runs 6 steps:

1. **collect_runtime** — gather `RuntimeContext`
2. **inject_session** — RAG-retrieve relevant history from `UserMemory`
3. **classify_with_context** — auto-select strategy using runtime context
4. **decompose** — classify/structure/split/enrich the query
5. **compose** — build optimized prompt for large-LLM
6. **sanitize** — filter sensitive data before output

```python
result = await preprocess_and_execute(
    query="Deploy to production",
    pipeline="context_aware",           # use the full context pipeline
)
```

## Related Docs

- [Session Persistence](session-persistence.md) — export/import/RAG
- [Sensitive Data Filtering](sensitive-data.md) — rules and config
- [Flow Graphs](flow-graphs.md) — Mermaid diagrams
- [CHANGELOG](../CHANGELOG.md) — v0.4.0 details
- [ROADMAP](../ROADMAP.md) — future plans
