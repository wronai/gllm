#!/usr/bin/env python3
"""preLLM Quick Start — runnable examples for all major use cases.

Usage:
    # Requires: pip install prellm
    # Requires: ollama serve (for local models)

    python examples/quick_start.py
"""

from __future__ import annotations

import asyncio


async def example_zero_config():
    """Simplest possible usage — one line, default models."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute("Refaktoryzuj kod")
    print(f"[zero-config] {result.content[:100]}...")
    print(f"  model: {result.model_used}")
    print(f"  small: {result.small_model_used}")


async def example_strategy_based():
    """v0.2 — strategy-based preprocessing (classify, structure, split, enrich)."""
    from prellm import preprocess_and_execute

    # Strategy: STRUCTURE — extracts action, target, parameters
    result = await preprocess_and_execute(
        query="Deploy backend v2.3 to production with rollback",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        user_context={"team": "backend", "env": "production"},
    )
    print(f"[strategy=structure] {result.content[:100]}...")
    if result.decomposition:
        print(f"  classification: {result.decomposition.classification}")
        print(f"  composed_prompt: {result.decomposition.composed_prompt[:80]}...")


async def example_pipeline_based():
    """v0.3 — pipeline-based two-agent architecture."""
    from prellm import preprocess_and_execute

    # Pipeline: DUAL_AGENT_FULL — 4-step preprocessing pipeline
    result = await preprocess_and_execute(
        query="Deploy backend v2.3 to production with rollback",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        pipeline="dual_agent_full",
    )
    print(f"[pipeline=dual_agent_full] {result.content[:100]}...")


async def example_ollama_local():
    """Both models run locally via Ollama — $0.00 cost."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Explain Python decorators with examples",
        small_llm="ollama/qwen2.5:3b",
        large_llm="ollama/llama3:8b",
        strategy="classify",
    )
    print(f"[ollama local] {result.content[:100]}...")


async def example_hybrid_ollama_openai():
    """Hybrid: local preprocessing + cloud execution."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Review this Python function for security issues",
        small_llm="ollama/qwen2.5:3b",       # free, local
        large_llm="gpt-4o-mini",              # ~$0.15
        strategy="enrich",
    )
    print(f"[hybrid ollama→openai] {result.content[:100]}...")


async def example_hybrid_ollama_anthropic():
    """Hybrid: local preprocessing + Anthropic execution."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Write Kubernetes deployment manifest for a Python app",
        small_llm="ollama/phi3:mini",
        large_llm="anthropic/claude-sonnet-4-20250514",
        pipeline="structure",
    )
    print(f"[hybrid ollama→anthropic] {result.content[:100]}...")


async def example_domain_rules():
    """Domain rules catch missing safety-critical fields."""
    from prellm import preprocess_and_execute

    result = await preprocess_and_execute(
        query="Usuń bazę danych klientów",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
        strategy="structure",
        domain_rules=[{
            "name": "destructive_db",
            "keywords": ["delete", "drop", "usuń", "remove"],
            "required_fields": ["target_database", "backup_confirmed"],
            "severity": "critical",
        }],
    )
    print(f"[domain-rules] {result.content[:100]}...")
    if result.decomposition and result.decomposition.missing_fields:
        print(f"  missing fields: {result.decomposition.missing_fields}")


async def example_sync():
    """Synchronous version — for scripts, notebooks, non-async code."""
    from prellm import preprocess_and_execute_sync

    result = preprocess_and_execute_sync(
        "Explain Docker networking",
        small_llm="ollama/qwen2.5:3b",
        large_llm="gpt-4o-mini",
    )
    print(f"[sync] {result.content[:100]}...")


async def example_custom_pipeline():
    """Use PromptPipeline directly for maximum control."""
    from prellm import PromptRegistry, PromptPipeline, PreprocessorAgent, ExecutorAgent
    from prellm import LLMProvider, LLMProviderConfig

    # Build components
    registry = PromptRegistry()  # loads configs/prompts.yaml
    small = LLMProvider(LLMProviderConfig(model="ollama/qwen2.5:3b", max_tokens=512))
    large = LLMProvider(LLMProviderConfig(model="gpt-4o-mini", max_tokens=2048))

    # Load pipeline from YAML
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=None,        # default: configs/pipelines.yaml
        pipeline_name="structure",
        registry=registry,
        small_llm=small,
    )

    # Build agents
    preprocessor = PreprocessorAgent(small_llm=small, registry=registry, pipeline=pipeline)
    executor = ExecutorAgent(large_llm=large)

    # Execute
    prep = await preprocessor.preprocess("Deploy app to production")
    result = await executor.execute(prep.executor_input)

    print(f"[custom pipeline] {result.content[:100]}...")
    print(f"  pipeline: {prep.pipeline_name}")
    print(f"  confidence: {prep.confidence}")
    print(f"  steps: {len(prep.decomposition.steps_executed)}")


async def example_openai_sdk_client():
    """Use preLLM server from any OpenAI SDK client."""
    print("[openai-sdk] Start preLLM server first:")
    print("  prellm serve --port 8080 --small ollama/qwen2.5:3b --large gpt-4o-mini")
    print()
    print("  Then use OpenAI SDK:")
    print("  import openai")
    print("  client = openai.OpenAI(base_url='http://localhost:8080/v1', api_key='any')")
    print("  response = client.chat.completions.create(")
    print("      model='prellm:default',")
    print("      messages=[{'role': 'user', 'content': 'Deploy app'}],")
    print("  )")


async def main():
    """Run all examples (requires LLM providers to be configured)."""
    print("=" * 60)
    print("preLLM Quick Start Examples")
    print("=" * 60)
    print()
    print("NOTE: These examples require LLM providers to be running.")
    print("      For Ollama: ollama serve && ollama pull qwen2.5:3b")
    print("      For OpenAI: export OPENAI_API_KEY=sk-...")
    print("      For Anthropic: export ANTHROPIC_API_KEY=sk-ant-...")
    print()

    examples = [
        ("Zero Config", example_zero_config),
        ("Strategy-Based (v0.2)", example_strategy_based),
        ("Pipeline-Based (v0.3)", example_pipeline_based),
        ("Ollama Local", example_ollama_local),
        ("Hybrid Ollama→OpenAI", example_hybrid_ollama_openai),
        ("Hybrid Ollama→Anthropic", example_hybrid_ollama_anthropic),
        ("Domain Rules", example_domain_rules),
        ("Custom Pipeline", example_custom_pipeline),
        ("OpenAI SDK Client", example_openai_sdk_client),
    ]

    for name, fn in examples:
        print(f"\n--- {name} ---")
        try:
            await fn()
        except Exception as e:
            print(f"  (skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    asyncio.run(main())
