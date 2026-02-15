"""Tests for PreprocessorAgent — small LLM preprocessing agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from prellm.agents.preprocessor import PreprocessorAgent, PreprocessResult
from prellm.analyzers.context_engine import ContextEngine
from prellm.llm_provider import LLMProvider
from prellm.pipeline import PipelineConfig, PipelineStep, PromptPipeline
from prellm.prompt_registry import PromptRegistry


@pytest.fixture
def sample_prompts_yaml(tmp_path: Path) -> Path:
    data = {
        "prompts": {
            "classify": {"system": "Classify the query.", "max_tokens": 256, "temperature": 0.1},
            "compose": {"system": "Compose a final prompt.", "max_tokens": 512, "temperature": 0.2},
        }
    }
    path = tmp_path / "prompts.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def registry(sample_prompts_yaml: Path) -> PromptRegistry:
    return PromptRegistry(prompts_path=sample_prompts_yaml)


@pytest.fixture
def mock_llm() -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.complete_json = AsyncMock(return_value={
        "intent": "deploy", "confidence": 0.85, "domain": "devops"
    })
    provider.complete = AsyncMock(return_value="composed prompt for executor")
    return provider


@pytest.fixture
def pipeline(registry: PromptRegistry, mock_llm: LLMProvider) -> PromptPipeline:
    config = PipelineConfig(
        name="classify",
        steps=[
            PipelineStep(name="classify", prompt="classify", output="classification"),
            PipelineStep(name="compose", prompt="compose", output="composed_prompt"),
        ],
    )
    return PromptPipeline(config=config, registry=registry, small_llm=mock_llm)


@pytest.fixture
def agent(mock_llm: LLMProvider, registry: PromptRegistry, pipeline: PromptPipeline) -> PreprocessorAgent:
    return PreprocessorAgent(
        small_llm=mock_llm,
        registry=registry,
        pipeline=pipeline,
        context_engine=ContextEngine(),
    )


@pytest.mark.asyncio
async def test_preprocessor_agent_returns_structured_input(agent: PreprocessorAgent):
    """PreprocessorAgent returns a PreprocessResult with executor_input."""
    result = await agent.preprocess("Deploy app to production")
    assert isinstance(result, PreprocessResult)
    assert result.original_query == "Deploy app to production"
    assert result.executor_input != ""
    assert result.pipeline_name == "classify"


@pytest.mark.asyncio
async def test_preprocessor_agent_context_gathering(
    mock_llm: LLMProvider, registry: PromptRegistry, pipeline: PromptPipeline
):
    """PreprocessorAgent gathers and merges context."""
    agent = PreprocessorAgent(
        small_llm=mock_llm,
        registry=registry,
        pipeline=pipeline,
        context_engine=ContextEngine(),
    )
    result = await agent.preprocess(
        "Deploy app",
        user_context={"env": "production", "team": "backend"},
    )
    assert "env" in result.context_used
    assert result.context_used["env"] == "production"


@pytest.mark.asyncio
async def test_preprocessor_agent_pipeline_execution(agent: PreprocessorAgent):
    """PreprocessorAgent executes the pipeline and stores decomposition."""
    result = await agent.preprocess("Scale database cluster")
    assert result.decomposition is not None
    assert result.decomposition.success is True
    assert len(result.decomposition.steps_executed) == 2


@pytest.mark.asyncio
async def test_preprocessor_agent_confidence_extraction(agent: PreprocessorAgent):
    """PreprocessorAgent extracts confidence from classification step."""
    result = await agent.preprocess("Deploy app")
    assert result.confidence == 0.85


@pytest.mark.asyncio
async def test_preprocessor_agent_pipeline_name_override(agent: PreprocessorAgent):
    """PreprocessorAgent allows pipeline_name override."""
    result = await agent.preprocess("Deploy app", pipeline_name="custom_pipeline")
    assert result.pipeline_name == "custom_pipeline"
