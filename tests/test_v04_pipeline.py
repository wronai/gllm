"""Tests for v0.4 pipeline features — new algo handlers and context_aware pipeline."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from prellm.pipeline import PromptPipeline, PipelineConfig, PipelineStep, PipelineResult


class TestRuntimeCollectorHandler:
    """Tests for _algo_runtime_collector pipeline handler."""

    def test_runtime_collector_returns_dict(self):
        handler = PromptPipeline._algo_runtime_collector
        result = handler({}, {}, {})
        assert isinstance(result, dict)
        assert "process" in result
        assert "locale" in result
        assert "network" in result
        assert "system" in result
        assert "collected_at" in result

    def test_runtime_collector_has_pid(self):
        result = PromptPipeline._algo_runtime_collector({}, {}, {})
        assert result["process"]["pid"] > 0


class TestSensitiveFilterHandler:
    """Tests for _algo_sensitive_filter pipeline handler."""

    def test_sensitive_filter_redacts_tokens(self):
        handler = PromptPipeline._algo_sensitive_filter
        inputs = {"composed_prompt": "Use key sk-1234567890abcdefghijklmnop for auth"}
        result = handler(inputs, {}, {})
        assert "sk-1234567890abcdefghijklmnop" not in result
        assert "[REDACTED]" in result

    def test_sensitive_filter_passes_safe_text(self):
        handler = PromptPipeline._algo_sensitive_filter
        inputs = {"composed_prompt": "Deploy the application to production"}
        result = handler(inputs, {}, {})
        assert "Deploy the application to production" == result

    def test_sensitive_filter_fallback_to_query(self):
        handler = PromptPipeline._algo_sensitive_filter
        result = handler({}, {"query": "test query"}, {})
        assert "test query" in result


class TestSessionInjectorHandler:
    """Tests for _algo_session_injector pipeline handler."""

    def test_session_injector_no_memory_path(self):
        handler = PromptPipeline._algo_session_injector
        result = handler({}, {"context": {}}, {})
        assert result == ""

    def test_session_injector_with_memory_path(self, tmp_path):
        from prellm.context.user_memory import UserMemory
        import asyncio

        # Set up memory with some data
        db_path = tmp_path / "test.db"
        mem = UserMemory(path=str(db_path))
        asyncio.run(mem.add_interaction("Deploy app", "OK deployed", {}))
        asyncio.run(mem.set_preference("lang", "pl"))
        mem.close()

        handler = PromptPipeline._algo_session_injector
        result = handler(
            {},
            {"query": "Deploy", "context": {"memory_path": str(db_path)}},
            {},
        )
        # Should contain preference or history
        assert "lang=pl" in result or "Deploy" in result


class TestContextAwarePipelineYAML:
    """Tests for loading the context_aware pipeline from YAML."""

    def test_context_aware_pipeline_loads(self):
        """context_aware pipeline is defined in pipelines.yaml."""
        from prellm.prompt_registry import PromptRegistry
        from prellm.llm_provider import LLMProvider
        from prellm.models import LLMProviderConfig

        registry = PromptRegistry()
        provider = LLMProvider(LLMProviderConfig(model="test"))

        pipeline = PromptPipeline.from_yaml(
            pipelines_path=None,  # default
            pipeline_name="context_aware",
            registry=registry,
            small_llm=provider,
        )
        assert pipeline.config.name == "context_aware"
        step_names = [s.name for s in pipeline.config.steps]
        assert "collect_runtime" in step_names
        assert "inject_session" in step_names
        assert "sanitize" in step_names

    def test_all_handlers_registered(self):
        """All new handlers are registered in the pipeline."""
        from prellm.prompt_registry import PromptRegistry
        from prellm.llm_provider import LLMProvider
        from prellm.models import LLMProviderConfig

        pipeline = PromptPipeline(
            config=PipelineConfig(name="test", steps=[]),
            registry=PromptRegistry(),
            small_llm=LLMProvider(LLMProviderConfig(model="test")),
        )
        handlers = pipeline._algo_handlers
        assert "runtime_collector" in handlers
        assert "sensitive_filter" in handlers
        assert "session_injector" in handlers
        assert "domain_rule_matcher" in handlers
        assert "field_validator" in handlers
        assert "yaml_formatter" in handlers
