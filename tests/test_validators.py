"""Tests for ResponseValidator — YAML schema validation for LLM outputs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prellm.validators import ResponseValidator, ValidationResult


@pytest.fixture
def sample_schemas_yaml(tmp_path: Path) -> Path:
    data = {
        "schemas": {
            "classification": {
                "required_fields": ["intent", "confidence"],
                "types": {
                    "intent": "string",
                    "confidence": "float",
                    "domain": "string",
                },
                "constraints": {
                    "confidence": {"min": 0.0, "max": 1.0},
                    "intent": {
                        "enum": ["deploy", "query", "create", "delete", "other"],
                    },
                },
            },
            "split_result": {
                "required_fields": ["sub_queries"],
                "types": {
                    "sub_queries": "list[string]",
                },
                "constraints": {
                    "sub_queries": {"min_length": 1, "max_length": 5},
                },
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


def test_validator_loads_schemas(validator: ResponseValidator):
    """Validator loads schemas from YAML."""
    schemas = validator.list_schemas()
    assert "classification" in schemas
    assert "split_result" in schemas


def test_validator_passes_valid_classification(validator: ResponseValidator):
    """Valid classification data passes validation."""
    result = validator.validate(
        {"intent": "deploy", "confidence": 0.9, "domain": "devops"},
        "classification",
    )
    assert result.valid is True
    assert result.errors == []


def test_validator_rejects_invalid_confidence(validator: ResponseValidator):
    """Confidence outside [0.0, 1.0] range fails validation."""
    result = validator.validate(
        {"intent": "deploy", "confidence": 1.5},
        "classification",
    )
    assert result.valid is False
    assert any("above maximum" in e for e in result.errors)


def test_validator_rejects_negative_confidence(validator: ResponseValidator):
    """Negative confidence fails validation."""
    result = validator.validate(
        {"intent": "deploy", "confidence": -0.1},
        "classification",
    )
    assert result.valid is False
    assert any("below minimum" in e for e in result.errors)


def test_validator_rejects_missing_required(validator: ResponseValidator):
    """Missing required fields fail validation."""
    result = validator.validate({"domain": "devops"}, "classification")
    assert result.valid is False
    assert any("intent" in e for e in result.errors)
    assert any("confidence" in e for e in result.errors)


def test_validator_enum_constraint(validator: ResponseValidator):
    """Invalid enum value fails validation."""
    result = validator.validate(
        {"intent": "fly_to_moon", "confidence": 0.5},
        "classification",
    )
    assert result.valid is False
    assert any("not in allowed" in e for e in result.errors)


def test_validator_list_length_constraint(validator: ResponseValidator):
    """List exceeding max_length fails validation."""
    result = validator.validate(
        {"sub_queries": ["q1", "q2", "q3", "q4", "q5", "q6"]},
        "split_result",
    )
    assert result.valid is False
    assert any("above maximum" in e for e in result.errors)


def test_validator_list_min_length_constraint(validator: ResponseValidator):
    """Empty list below min_length fails validation."""
    result = validator.validate(
        {"sub_queries": []},
        "split_result",
    )
    assert result.valid is False
    assert any("below minimum" in e for e in result.errors)


def test_validator_valid_split(validator: ResponseValidator):
    """Valid split result passes validation."""
    result = validator.validate(
        {"sub_queries": ["What is X?", "How to Y?"]},
        "split_result",
    )
    assert result.valid is True


def test_validate_or_retry_succeeds_on_second(validator: ResponseValidator):
    """validate_or_retry retries and succeeds when retry_fn returns valid data."""
    call_count = 0

    def retry_fn():
        nonlocal call_count
        call_count += 1
        return {"intent": "deploy", "confidence": 0.8}

    bad_data = {"intent": "fly_to_moon", "confidence": 1.5}
    result_data, retries = validator.validate_or_retry(bad_data, "classification", retry_fn)

    assert retries >= 1
    assert result_data["intent"] == "deploy"
    assert result_data["confidence"] == 0.8


def test_validate_or_retry_exhausts_retries(validator: ResponseValidator):
    """validate_or_retry stops after max_retries even if still invalid."""
    def retry_fn():
        return {"intent": "invalid_forever", "confidence": 99.0}

    bad_data = {"intent": "invalid", "confidence": 5.0}
    result_data, retries = validator.validate_or_retry(
        bad_data, "classification", retry_fn, max_retries=2
    )
    assert retries == 2


def test_validator_unknown_schema(validator: ResponseValidator):
    """Validating against non-existent schema returns invalid."""
    result = validator.validate({"foo": "bar"}, "nonexistent")
    assert result.valid is False
    assert any("not found" in e for e in result.errors)


def test_validator_nonexistent_file():
    """Validator handles missing YAML file gracefully."""
    v = ResponseValidator(schemas_path="/nonexistent/schemas.yaml")
    assert v.list_schemas() == []
