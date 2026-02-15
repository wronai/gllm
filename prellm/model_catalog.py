"""Model catalog — curated model pairs and provider examples.

Pure data + pure functions, no IO. Extracted from cli.py to reduce cyclomatic complexity.
"""

from __future__ import annotations

from typing import Any


# (name, small_llm, large_llm, cost_hint)
MODEL_PAIRS: list[tuple[str, str, str, str]] = [
    ("Ollama local (free)", "ollama/qwen2.5:3b", "ollama/llama3:8b", "$0.00"),
    ("Ollama + OpenAI", "ollama/qwen2.5:3b", "gpt-4o-mini", "~$0.15"),
    ("Ollama + Claude", "ollama/qwen2.5:3b", "anthropic/claude-sonnet-4-20250514", "~$0.30"),
    ("Ollama + Kimi K2.5", "ollama/qwen2.5:3b", "openrouter/moonshotai/kimi-k2.5", "~$0.10"),
    ("OpenAI only", "gpt-4o-mini", "gpt-4o", "~$0.20"),
    ("Anthropic only", "anthropic/claude-haiku", "anthropic/claude-sonnet-4-20250514", "~$0.35"),
    ("Groq fast", "groq/llama-3.1-8b-instant", "groq/llama-3.3-70b-versatile", "~$0.05"),
    ("Mistral", "mistral/mistral-small-latest", "mistral/mistral-large-latest", "~$0.20"),
    ("DeepSeek", "deepseek/deepseek-chat", "deepseek/deepseek-reasoner", "~$0.15"),
    ("Google Gemini", "gemini/gemini-2.0-flash", "gemini/gemini-2.5-pro-preview-06-05", "~$0.20"),
    ("Together AI", "together_ai/Qwen/Qwen2.5-7B-Instruct-Turbo", "together_ai/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "~$0.10"),
    ("Azure OpenAI", "azure/gpt-4o-mini-deployment", "azure/gpt-4o-deployment", "~$0.20"),
    ("AWS Bedrock", "bedrock/anthropic.claude-3-haiku-20240307-v1:0", "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0", "~$0.30"),
]

# (model_id, description)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    ("openrouter/moonshotai/kimi-k2.5", "Moonshot Kimi K2.5 — strong reasoning, competitive pricing"),
    ("openrouter/google/gemini-2.5-pro-preview-06-05", "Google Gemini 2.5 Pro"),
    ("openrouter/anthropic/claude-sonnet-4-20250514", "Claude Sonnet 4"),
    ("openrouter/openai/gpt-4o", "OpenAI GPT-4o"),
    ("openrouter/meta-llama/llama-3.3-70b-instruct", "Meta Llama 3.3 70B"),
    ("openrouter/deepseek/deepseek-r1", "DeepSeek R1 reasoning"),
    ("openrouter/qwen/qwen-2.5-72b-instruct", "Qwen 2.5 72B"),
    ("openrouter/mistralai/mistral-large-2411", "Mistral Large"),
]


def list_model_pairs(
    provider: str | None = None,
    search: str | None = None,
) -> list[dict[str, str]]:
    """Filter model pairs by provider and/or search term. Pure function — no IO."""
    results: list[dict[str, str]] = []
    provider_lower = provider.lower() if provider else None
    search_lower = search.lower() if search else None

    for name, small, large, cost in MODEL_PAIRS:
        haystack = f"{name} {small} {large}".lower()
        if provider_lower and provider_lower not in haystack:
            continue
        if search_lower and search_lower not in haystack:
            continue
        results.append({"name": name, "small": small, "large": large, "cost": cost})

    return results


def list_openrouter_models(
    provider: str | None = None,
    search: str | None = None,
) -> list[dict[str, str]]:
    """Filter OpenRouter models by provider and/or search term. Pure function — no IO."""
    results: list[dict[str, str]] = []
    provider_lower = provider.lower() if provider else None
    search_lower = search.lower() if search else None

    for model_id, desc in OPENROUTER_MODELS:
        haystack = f"{model_id} {desc}".lower()
        if provider_lower and provider_lower not in haystack:
            continue
        if search_lower and search_lower not in haystack:
            continue
        results.append({"model_id": model_id, "description": desc})

    return results
