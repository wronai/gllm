"""Tests for ExecutorAgent — large LLM execution agent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from prellm.agents.executor import ExecutorAgent, ExecutorResult
from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig
from prellm.validators import ResponseValidator


@pytest.fixture
def mock_large_llm() -> LLMProvider:
    provider = MagicMock(spec=LLMProvider)
    provider.config = LLMProviderConfig(model="anthropic/claude-sonnet-4-20250514")
    provider.complete = AsyncMock(return_value="Here is the deployment plan...")
    return provider


@pytest.fixture
def sample_schemas_yaml(tmp_path: Path) -> Path:
    data = {
        "schemas": {
            "final_response": {
                "required_fields": ["content"],
                "types": {"content": "string"},
            },
        }
    }
    path = tmp_path / "schemas.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


@pytest.fixture
def validator(sample_schemas_yaml: Path) -> ResponseValidator:
    return ResponseValidator(schemas_path=sample_schemas_yaml)


@pytest.fixture
def agent(mock_large_llm: LLMProvider) -> ExecutorAgent:
    return ExecutorAgent(large_llm=mock_large_llm)


@pytest.mark.asyncio
async def test_executor_agent_calls_large_llm(agent: ExecutorAgent, mock_large_llm: LLMProvider):
    """ExecutorAgent calls the large LLM with the structured prompt."""
    result = await agent.execute("Deploy application to production cluster")
    assert isinstance(result, ExecutorResult)
    assert result.content == "Here is the deployment plan..."
    assert result.model_used == "anthropic/claude-sonnet-4-20250514"
    mock_large_llm.complete.assert_called_once()


@pytest.mark.asyncio
async def test_executor_agent_retry_fallback(mock_large_llm: LLMProvider):
    """ExecutorAgent handles LLM failures gracefully."""
    mock_large_llm.complete = AsyncMock(side_effect=RuntimeError("API timeout"))
    agent = ExecutorAgent(large_llm=mock_large_llm)

    result = await agent.execute("Deploy app")
    assert result.content == ""
    assert result.retries == 1


@pytest.mark.asyncio
async def test_executor_agent_schema_validation(
    mock_large_llm: LLMProvider, validator: ResponseValidator
):
    """ExecutorAgent validates response against schema when configured."""
    mock_large_llm.complete = AsyncMock(return_value='{"content": "deployment plan"}')
    agent = ExecutorAgent(
        large_llm=mock_large_llm,
        response_validator=validator,
        response_schema_name="final_response",
    )
    result = await agent.execute("Deploy app")
    assert result.schema_valid is True
    assert result.validation is not None


@pytest.mark.asyncio
async def test_executor_agent_no_validation_by_default(agent: ExecutorAgent):
    """ExecutorAgent skips validation when no validator configured."""
    result = await agent.execute("Deploy app")
    assert result.schema_valid is None
    assert result.validation is None


@pytest.mark.asyncio
async def test_executor_agent_system_prompt(agent: ExecutorAgent, mock_large_llm: LLMProvider):
    """ExecutorAgent passes system_prompt to the LLM call."""
    await agent.execute("Deploy app", system_prompt="You are a DevOps expert.")
    call_kwargs = mock_large_llm.complete.call_args
    assert call_kwargs.kwargs.get("system_prompt") == "You are a DevOps expert."
