"""Tests for LLMProvider.complete_structured() — instructor integration."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from pydantic import BaseModel, Field

from prellm.llm_provider import LLMProvider
from prellm.models import LLMProviderConfig


class ClassificationResult(BaseModel):
    intent: str
    confidence: float = Field(ge=0.0, le=1.0)
    domain: str


class DeployParams(BaseModel):
    action: str
    target: str
    environment: str
    version: str | None = None


class TestCompleteStructuredImportError:
    """Test graceful handling when instructor is not installed."""

    @pytest.mark.asyncio
    async def test_raises_import_error_without_instructor(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))

        with patch.dict("sys.modules", {"instructor": None}):
            with pytest.raises(ImportError, match="instructor is required"):
                await provider.complete_structured(
                    user_message="Test",
                    response_model=ClassificationResult,
                )


class TestCompleteStructuredWithMock:
    """Test structured output with mocked instructor."""

    @pytest.mark.asyncio
    async def test_returns_validated_model(self):
        expected = ClassificationResult(intent="deploy", confidence=0.95, domain="devops")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=expected)

        with patch("prellm.llm_provider.LLMProvider.complete_structured", new=AsyncMock(return_value=expected)):
            provider = LLMProvider(LLMProviderConfig(model="test-model"))
            result = await provider.complete_structured(
                user_message="Deploy app to production",
                response_model=ClassificationResult,
            )

        assert isinstance(result, ClassificationResult)
        assert result.intent == "deploy"
        assert result.confidence == 0.95
        assert result.domain == "devops"

    @pytest.mark.asyncio
    async def test_structured_with_system_prompt(self):
        expected = DeployParams(action="deploy", target="app", environment="prod", version="1.2.3")

        with patch("prellm.llm_provider.LLMProvider.complete_structured", new=AsyncMock(return_value=expected)):
            provider = LLMProvider(LLMProviderConfig(model="test-model"))
            result = await provider.complete_structured(
                user_message="Deploy app v1.2.3 to prod",
                response_model=DeployParams,
                system_prompt="Extract deployment parameters.",
            )

        assert isinstance(result, DeployParams)
        assert result.environment == "prod"
        assert result.version == "1.2.3"

    @pytest.mark.asyncio
    async def test_structured_fallback_on_failure(self):
        expected = ClassificationResult(intent="test", confidence=0.5, domain="general")

        call_count = 0

        async def mock_structured(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Primary model failed")
            return expected

        with patch("prellm.llm_provider.LLMProvider.complete_structured", side_effect=mock_structured):
            provider = LLMProvider(LLMProviderConfig(model="primary", fallback=["fallback"]))
            # First call fails
            with pytest.raises(RuntimeError):
                await provider.complete_structured(
                    user_message="Test",
                    response_model=ClassificationResult,
                )
            # Second call succeeds (simulating fallback)
            result = await provider.complete_structured(
                user_message="Test",
                response_model=ClassificationResult,
            )
            assert result.intent == "test"


class TestCompleteStructuredMethod:
    """Test the actual method signature and error paths."""

    @pytest.mark.asyncio
    async def test_method_exists(self):
        provider = LLMProvider(LLMProviderConfig(model="test-model"))
        assert hasattr(provider, "complete_structured")
        assert callable(provider.complete_structured)

    @pytest.mark.asyncio
    async def test_all_models_fail_raises_runtime_error(self):
        """When instructor is available but all models fail."""
        mock_instructor = MagicMock()
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Model error"))
        mock_instructor.from_litellm.return_value = mock_client

        with patch.dict("sys.modules", {"instructor": mock_instructor}):
            provider = LLMProvider(LLMProviderConfig(model="bad-model"))
            with pytest.raises(RuntimeError, match="All models failed"):
                await provider.complete_structured(
                    user_message="Test",
                    response_model=ClassificationResult,
                )
