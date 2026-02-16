# Sensitive Data Filtering

> **v0.4.0** — Automatic classification and filtering of sensitive data before LLM calls.

## Problem

When using cloud-based large LLMs, environment variables containing API keys, tokens, passwords, and database URLs must never be sent to the model. At the same time, safe context (locale, shell, hostname) is critical for accurate responses.

## Solution

`SensitiveDataFilter` classifies every piece of data into three levels:

| Level | Action | Examples |
| --- | --- | --- |
| **SAFE** | Pass through | `LANG`, `SHELL`, `HOME`, `USER`, `PWD`, `HOSTNAME` |
| **MASKED** | Partially shown (first/last 2 chars) | `DATABASE_URL`, `REDIS_URL`, `SMTP_*` |
| **BLOCKED** | Completely removed | `*API_KEY`, `*SECRET`, `*TOKEN`, `*PASSWORD` |

Value-based detection catches tokens even in non-standard variable names:

| Pattern | Provider |
| --- | --- |
| `sk-[a-zA-Z0-9]{20,}` | OpenAI |
| `sk-ant-[a-zA-Z0-9]{20,}` | Anthropic |
| `ghp_[a-zA-Z0-9]{36}` | GitHub PAT |
| `gsk_[a-zA-Z0-9]{20,}` | Groq |
| `sk-or-v1-[a-zA-Z0-9]{20,}` | OpenRouter |
| `xox[bpsa]-[a-zA-Z0-9-]{20,}` | Slack |

## Usage

### Automatic (default in v0.4)

```python
from prellm import preprocess_and_execute

# sanitize=True is the default — secrets are filtered automatically
result = await preprocess_and_execute(
    query="Deploy to production",
    small_llm="ollama/bielik:7b",
    large_llm="openrouter/google/gemini-3-flash-preview",
    # sanitize=True,  # default
)
```

### Disable for Development

```python
result = await preprocess_and_execute(
    query="Debug local issue",
    sanitize=False,  # dev mode — nothing filtered
)
```

### Custom Rules

```python
result = await preprocess_and_execute(
    query="Deploy",
    sensitive_rules="my_rules.yaml",  # custom YAML
)
```

### Direct API

```python
from prellm.context.sensitive_filter import SensitiveDataFilter
from prellm.models import SensitivityLevel

filt = SensitiveDataFilter()

# Classify individual keys
filt.classify_key("OPENAI_API_KEY")    # → SensitivityLevel.BLOCKED
filt.classify_key("DATABASE_URL")      # → SensitivityLevel.MASKED
filt.classify_key("LANG")              # → SensitivityLevel.SAFE

# Classify values
filt.classify_value("sk-abc123def456ghi789")  # → BLOCKED (OpenAI pattern)
filt.classify_value("hello world")             # → SAFE

# Filter a dict
data = {"LANG": "pl_PL", "OPENAI_API_KEY": "sk-secret", "HOME": "/home/user"}
safe = filt.filter_dict(data)
# → {"LANG": "pl_PL", "HOME": "/home/user"}

# Sanitize free text
text = "Use key sk-1234567890abcdefghijklmnop for auth"
clean = filt.sanitize_text(text)
# → "Use key [REDACTED] for auth"

# Get report
report = filt.get_filter_report()
print(report.blocked_keys)   # ["OPENAI_API_KEY"]
print(report.safe_keys)      # ["LANG", "HOME"]
```

## Configuration YAML

Default rules are in `configs/sensitive_rules.yaml`:

```yaml
sensitive_keys:
  blocked:
    - "API_KEY"
    - "SECRET"
    - "TOKEN"
    - "PASSWORD"
    - "PRIVATE_KEY"
    - "CREDENTIAL"
    - "AUTH_KEY"
  masked:
    - "DATABASE_URL"
    - "REDIS_URL"
    - "SMTP_"
    - "MONGO_URI"
  safe:
    - "LANG"
    - "TERM"
    - "SHELL"
    - "HOME"
    - "USER"
    - "PWD"
    - "PATH"
    - "EDITOR"
    - "HOSTNAME"

sensitive_value_patterns:
  - "sk-[a-zA-Z0-9]{20,}"        # OpenAI
  - "sk-ant-[a-zA-Z0-9]{20,}"    # Anthropic
  - "ghp_[a-zA-Z0-9]{36}"        # GitHub PAT
  - "gsk_[a-zA-Z0-9]{20,}"       # Groq
  - "sk-or-v1-[a-zA-Z0-9]{20,}"  # OpenRouter
```

### Custom Rules File

Create your own YAML to extend the defaults:

```yaml
# my_rules.yaml
sensitive_keys:
  blocked:
    - "INTERNAL_SECRET"
    - "CORP_TOKEN"
  masked:
    - "LDAP_URL"
  safe:
    - "MY_SAFE_VAR"

sensitive_value_patterns:
  - "corp-[a-zA-Z0-9]{32}"  # corporate token format
```

```python
result = await preprocess_and_execute(
    query="Deploy",
    sensitive_rules="my_rules.yaml",
)
```

## Integration Points

The filter is applied at two points in the pipeline:

1. **Context preparation** (`_prepare_context` in `core.py`) — filters `extra_context` dict before it reaches the preprocessor
2. **Executor input** (`ExecutorAgent.execute`) — `sanitize_text()` on the final prompt before the large-LLM call

This means the small LLM (local) sees more context than the large LLM (cloud).

## Related Docs

- [Persistent Context](persistent-context.md) — full context layer
- [Session Persistence](session-persistence.md) — export/import
- [CHANGELOG](../CHANGELOG.md) — v0.4.0 details
