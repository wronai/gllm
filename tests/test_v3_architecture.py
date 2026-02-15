"""Tests for v0.3 two-agent architecture — integration tests across all new modules.

Covers:
- PreprocessorAgent + ExecutorAgent full pipeline
- preprocess_and_execute_v3 function
- Pipeline ↔ Registry ↔ Validator integration
- Agent independence (test each in isolation)
- Cost tracking readiness
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prellm.agents.executor import ExecutorAgent, ExecutorResult
from prellm.agents.preprocessor import PreprocessorAgent, PreprocessResult
from prellm.analyzers.context_engine import ContextEngine
from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig, PreLLMResponse
from prellm.pipeline import PipelineConfig, PipelineStep, PromptPipeline
from prellm.prompt_registry import PromptRegistry
from prellm.validators import ResponseValidator


# ============================================================
# Shared fixtures
# ============================================================

@pytest.fixture
def tmp_configs(tmp_path: Path) -> dict[str, Path]:
    """Create all config files needed for v0.3 integration tests."""
    prompts = {
        "prompts": {
            "classify": {"system": "Classify. JSON: {intent, confidence, domain}.", "max_tokens": 256, "temperature": 0.1},
            "structure": {"system": "Extract fields. JSON.", "max_tokens": 512, "temperature": 0.1},
            "split": {"system": "Split into {{ max_subtasks | default(3) }} sub-questions.", "max_tokens": 256},
            "enrich": {"system": "Enrich query.", "max_tokens": 512, "temperature": 0.2},
            "compose": {"system": "Compose final prompt.", "max_tokens": 512, "temperature": 0.2},
            "context_analyze": {"system": "Analyze context.", "max_tokens": 256, "temperature": 0.1},
        }
    }
    pipelines = {
        "pipelines": {
            "classify": {
                "description": "Classify intent",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                ],
            },
            "structure": {
                "description": "Full structural decomposition",
                "steps": [
                    {"name": "classify", "prompt": "classify", "output": "classification"},
                    {"name": "extract_fields", "prompt": "structure", "output": "fields"},
                    {"name": "compose", "prompt": "compose", "input": ["query", "classification", "fields"], "output": "composed_prompt"},
                ],
            },
            "dual_agent_full": {
                "description": "Full two-agent pipeline",
                "steps": [
                    {"name": "context_analyze", "prompt": "context_analyze", "output": "context_enrichment"},
                    {"name": "intent_decompose", "prompt": "split", "output": "subtasks", "config": {"max_subtasks": 3}},
                    {"name": "quality_optimize", "prompt": "compose", "output": "meta_prompt"},
                    {"name": "format_structure", "type": "yaml_formatter", "input": "meta_prompt", "output": "executor_input"},
                ],
            },
            "passthrough": {
                "description": "No steps",
                "steps": [],
            },
        }
    }
    schemas = {
        "schemas": {
            "classification": {
                "required_fields": ["intent", "confidence"],
                "types": {"intent": "string", "confidence": "float"},
                "constraints": {"confidence": {"min": 0.0, "max": 1.0}},
            },
            "final_response": {
                "required_fields": ["content"],
                "types": {"content": "string"},
            },
        }
    }

    paths = {}
    for name, data in [("prompts", prompts), ("pipelines", pipelines), ("schemas", schemas)]:
        p = tmp_path / f"{name}.yaml"
        with open(p, "w") as f:
            yaml.dump(data, f)
        paths[name] = p
    return paths


@pytest.fixture
def mock_small_llm() -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.config = LLMProviderConfig(model="ollama/qwen2.5:3b", max_tokens=512, temperature=0.0)
    provider.complete_json = AsyncMock(return_value={
        "intent": "deploy", "confidence": 0.9, "domain": "devops"
    })
    provider.complete = AsyncMock(return_value="Optimized prompt: deploy app to prod with rollback plan")
    return provider


@pytest.fixture
def mock_large_llm() -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.config = LLMProviderConfig(model="anthropic/claude-sonnet-4-20250514", max_tokens=2048)
    provider.complete = AsyncMock(return_value="Here is the full deployment plan with rollback...")
    return provider


# ============================================================
# Integration: Full two-agent pipeline
# ============================================================

@pytest.mark.asyncio
async def test_agents_integration_full_pipeline(tmp_configs, mock_small_llm, mock_large_llm):
    """Full integration: PreprocessorAgent → ExecutorAgent with real configs."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="classify",
        registry=registry,
        small_llm=mock_small_llm,
    )

    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm,
        registry=registry,
        pipeline=pipeline,
    )
    executor = ExecutorAgent(large_llm=mock_large_llm)

    # Step 1: Preprocess
    prep = await preprocessor.preprocess("Deploy app to production")
    assert isinstance(prep, PreprocessResult)
    assert prep.original_query == "Deploy app to production"
    assert prep.pipeline_name == "classify"

    # Step 2: Execute
    result = await executor.execute(prep.executor_input)
    assert isinstance(result, ExecutorResult)
    assert result.content != ""
    assert result.model_used == "anthropic/claude-sonnet-4-20250514"


@pytest.mark.asyncio
async def test_agents_integration_structure_pipeline(tmp_configs, mock_small_llm, mock_large_llm):
    """Integration with multi-step structure pipeline."""
    call_count = 0

    async def llm_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"intent": "deploy", "confidence": 0.95, "domain": "devops"}
        elif call_count == 2:
            return {"action": "deploy", "target": "production", "parameters": {}}
        else:
            return {"composed_prompt": "Deploy app to production with monitoring and rollback"}

    mock_small_llm.complete_json = AsyncMock(side_effect=llm_side_effect)

    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="structure",
        registry=registry,
        small_llm=mock_small_llm,
    )

    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm, registry=registry, pipeline=pipeline,
    )
    executor = ExecutorAgent(large_llm=mock_large_llm)

    prep = await preprocessor.preprocess("Deploy app to production")
    assert prep.decomposition is not None
    assert prep.decomposition.success is True
    assert len(prep.decomposition.steps_executed) == 3

    result = await executor.execute(prep.executor_input)
    assert result.content != ""


@pytest.mark.asyncio
async def test_agents_integration_dual_agent_full(tmp_configs, mock_small_llm, mock_large_llm):
    """Integration with full dual-agent pipeline (4 steps including algo)."""
    mock_small_llm.complete_json = AsyncMock(return_value={
        "context": "production", "entities": ["app", "deploy"]
    })

    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="dual_agent_full",
        registry=registry,
        small_llm=mock_small_llm,
    )

    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm, registry=registry, pipeline=pipeline,
    )

    prep = await preprocessor.preprocess("Deploy app to production")
    assert prep.decomposition is not None
    assert prep.decomposition.success is True

    # Should have 4 steps: 3 LLM + 1 algo (yaml_formatter)
    executed = [s for s in prep.decomposition.steps_executed if not s.skipped]
    assert len(executed) == 4


# ============================================================
# Agent independence tests
# ============================================================

@pytest.mark.asyncio
async def test_preprocessor_independent_of_executor(tmp_configs, mock_small_llm):
    """PreprocessorAgent works without any ExecutorAgent."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="classify",
        registry=registry,
        small_llm=mock_small_llm,
    )
    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm, registry=registry, pipeline=pipeline,
    )

    prep = await preprocessor.preprocess("Test query")
    assert prep.executor_input != ""
    # Large LLM should NOT be called
    mock_small_llm.complete_json.assert_called()


@pytest.mark.asyncio
async def test_executor_independent_of_preprocessor(mock_large_llm):
    """ExecutorAgent works with a raw string, no preprocessing needed."""
    executor = ExecutorAgent(large_llm=mock_large_llm)
    result = await executor.execute("Raw prompt without preprocessing")
    assert result.content == "Here is the full deployment plan with rollback..."
    mock_large_llm.complete.assert_called_once()


# ============================================================
# Validator integration
# ============================================================

@pytest.mark.asyncio
async def test_executor_with_schema_validation(tmp_configs, mock_large_llm):
    """ExecutorAgent validates JSON responses against schema."""
    mock_large_llm.complete = AsyncMock(return_value='{"content": "deployment plan"}')
    validator = ResponseValidator(schemas_path=tmp_configs["schemas"])
    executor = ExecutorAgent(
        large_llm=mock_large_llm,
        response_validator=validator,
        response_schema_name="final_response",
    )
    result = await executor.execute("Execute task")
    assert result.schema_valid is True


@pytest.mark.asyncio
async def test_pipeline_with_validator_integration(tmp_configs, mock_small_llm):
    """Pipeline steps produce data that passes schema validation."""
    mock_small_llm.complete_json = AsyncMock(return_value={
        "intent": "deploy", "confidence": 0.85, "domain": "devops"
    })

    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="classify",
        registry=registry,
        small_llm=mock_small_llm,
    )
    result = await pipeline.execute("Deploy app")

    # Validate the classification output against schema
    validator = ResponseValidator(schemas_path=tmp_configs["schemas"])
    classification = result.state.get("classification", {})
    validation = validator.validate(classification, "classification")
    assert validation.valid is True


# ============================================================
# Passthrough pipeline
# ============================================================

@pytest.mark.asyncio
async def test_passthrough_pipeline_integration(tmp_configs, mock_small_llm, mock_large_llm):
    """Passthrough pipeline forwards query directly to executor."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="passthrough",
        registry=registry,
        small_llm=mock_small_llm,
    )

    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm, registry=registry, pipeline=pipeline,
    )
    executor = ExecutorAgent(large_llm=mock_large_llm)

    prep = await preprocessor.preprocess("Just pass me through")
    assert prep.executor_input == "Just pass me through"
    assert len(prep.decomposition.steps_executed) == 0

    result = await executor.execute(prep.executor_input)
    assert result.content != ""


# ============================================================
# Registry + Pipeline config validation
# ============================================================

def test_registry_validates_against_pipeline_needs(tmp_configs):
    """All prompts referenced by pipelines exist in the registry."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    errors = registry.validate()
    assert errors == [], f"Registry validation errors: {errors}"


def test_all_configs_load_without_error(tmp_configs):
    """All YAML configs load without parsing errors."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    assert len(registry.list_prompts()) >= 5

    validator = ResponseValidator(schemas_path=tmp_configs["schemas"])
    assert len(validator.list_schemas()) >= 2


# ============================================================
# Cost tracking readiness
# ============================================================

@pytest.mark.asyncio
async def test_preprocessor_tracks_pipeline_steps(tmp_configs, mock_small_llm):
    """PreprocessorAgent decomposition tracks which steps executed (for cost attribution)."""
    registry = PromptRegistry(prompts_path=tmp_configs["prompts"])
    pipeline = PromptPipeline.from_yaml(
        pipelines_path=tmp_configs["pipelines"],
        pipeline_name="structure",
        registry=registry,
        small_llm=mock_small_llm,
    )
    preprocessor = PreprocessorAgent(
        small_llm=mock_small_llm, registry=registry, pipeline=pipeline,
    )

    prep = await preprocessor.preprocess("Deploy app")
    assert prep.decomposition is not None

    # Each step should have a name and type recorded
    for step in prep.decomposition.steps_executed:
        assert step.step_name != ""
        assert step.step_type in ("llm", "algo")


@pytest.mark.asyncio
async def test_executor_tracks_model_used(mock_large_llm):
    """ExecutorAgent records which model was used (for cost attribution)."""
    executor = ExecutorAgent(large_llm=mock_large_llm)
    result = await executor.execute("Test")
    assert result.model_used == "anthropic/claude-sonnet-4-20250514"
