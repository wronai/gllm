"""Tests for shipped YAML configs — prompts, pipelines, response_schemas.

Ensures that after `pip install prellm`, the default configs are valid and consistent.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from prellm.pipeline import PromptPipeline
from prellm.prompt_registry import PromptRegistry
from prellm.validators import ResponseValidator

CONFIGS_DIR = Path(__file__).parent.parent / "configs"
PROMPTS_PATH = CONFIGS_DIR / "prompts.yaml"
PIPELINES_PATH = CONFIGS_DIR / "pipelines.yaml"
SCHEMAS_PATH = CONFIGS_DIR / "response_schemas.yaml"


# ============================================================
# YAML loading tests
# ============================================================

def test_shipped_prompts_yaml_loads():
    """configs/prompts.yaml loads without error and contains required prompts."""
    assert PROMPTS_PATH.exists(), f"Missing shipped config: {PROMPTS_PATH}"
    with open(PROMPTS_PATH) as f:
        data = yaml.safe_load(f)
    assert "prompts" in data
    prompts = data["prompts"]
    required = {"classify", "structure", "split", "enrich", "compose"}
    assert required.issubset(prompts.keys()), f"Missing prompts: {required - prompts.keys()}"
    for name, entry in prompts.items():
        assert isinstance(entry, dict), f"Prompt '{name}' should be a dict"
        assert "system" in entry, f"Prompt '{name}' missing 'system' key"
        assert entry["system"].strip(), f"Prompt '{name}' has empty system template"


def test_shipped_pipelines_yaml_loads():
    """configs/pipelines.yaml loads without error and contains required pipelines."""
    assert PIPELINES_PATH.exists(), f"Missing shipped config: {PIPELINES_PATH}"
    with open(PIPELINES_PATH) as f:
        data = yaml.safe_load(f)
    assert "pipelines" in data
    pipelines = data["pipelines"]
    required = {"classify", "structure", "split", "enrich", "passthrough"}
    assert required.issubset(pipelines.keys()), f"Missing pipelines: {required - pipelines.keys()}"
    # passthrough must have empty steps
    assert pipelines["passthrough"]["steps"] == [] or pipelines["passthrough"].get("steps") is None or len(pipelines["passthrough"]["steps"]) == 0


def test_shipped_schemas_yaml_loads():
    """configs/response_schemas.yaml loads without error and contains required schemas."""
    assert SCHEMAS_PATH.exists(), f"Missing shipped config: {SCHEMAS_PATH}"
    with open(SCHEMAS_PATH) as f:
        data = yaml.safe_load(f)
    assert "schemas" in data
    schemas = data["schemas"]
    required = {"classification", "structure_extraction", "split_result", "enrichment"}
    assert required.issubset(schemas.keys()), f"Missing schemas: {required - schemas.keys()}"


# ============================================================
# Cross-reference tests
# ============================================================

def test_all_pipeline_prompts_exist_in_registry():
    """Every prompt referenced in pipelines.yaml exists in prompts.yaml."""
    with open(PIPELINES_PATH) as f:
        pipelines_data = yaml.safe_load(f)
    with open(PROMPTS_PATH) as f:
        prompts_data = yaml.safe_load(f)

    available_prompts = set(prompts_data.get("prompts", {}).keys())
    missing = []

    for pipe_name, pipe_def in pipelines_data.get("pipelines", {}).items():
        for step in pipe_def.get("steps", []):
            prompt_name = step.get("prompt")
            if prompt_name and prompt_name not in available_prompts:
                missing.append(f"Pipeline '{pipe_name}' step '{step['name']}' references missing prompt '{prompt_name}'")

    assert not missing, "\n".join(missing)


# ============================================================
# Component integration tests
# ============================================================

def test_prompt_registry_loads_shipped_prompts():
    """PromptRegistry loads shipped prompts.yaml and validates all required prompts."""
    registry = PromptRegistry(prompts_path=PROMPTS_PATH)
    errors = registry.validate()
    assert errors == [], f"Validation errors: {errors}"
    assert len(registry.list_prompts()) >= 5


def test_prompt_registry_renders_shipped_prompts():
    """All shipped prompts render without error (using defaults)."""
    registry = PromptRegistry(prompts_path=PROMPTS_PATH)
    for name in registry.list_prompts():
        rendered = registry.get(name)
        assert rendered.strip(), f"Prompt '{name}' rendered to empty string"


def test_response_validator_loads_shipped_schemas():
    """ResponseValidator loads shipped response_schemas.yaml."""
    validator = ResponseValidator(schemas_path=SCHEMAS_PATH)
    schemas = validator.list_schemas()
    assert "classification" in schemas
    assert "structure_extraction" in schemas
    assert "split_result" in schemas
    assert "enrichment" in schemas


def test_response_validator_validates_classification():
    """ResponseValidator correctly validates a classification result."""
    validator = ResponseValidator(schemas_path=SCHEMAS_PATH)
    result = validator.validate(
        {"intent": "deploy", "confidence": 0.9, "domain": "devops"},
        "classification",
    )
    assert result.valid is True
    assert result.errors == []


def test_response_validator_rejects_invalid_classification():
    """ResponseValidator rejects classification with missing required fields."""
    validator = ResponseValidator(schemas_path=SCHEMAS_PATH)
    result = validator.validate({"domain": "devops"}, "classification")
    assert result.valid is False
    assert any("intent" in e for e in result.errors)


# ============================================================
# Domain config tests
# ============================================================

def test_domain_configs_exist():
    """Domain config files exist in configs/domains/."""
    domains_dir = CONFIGS_DIR / "domains"
    assert domains_dir.exists(), f"Missing domains directory: {domains_dir}"
    yaml_files = list(domains_dir.glob("*.yaml"))
    assert len(yaml_files) >= 1, "No domain config files found"


def test_domain_configs_load():
    """All domain config YAML files load without error."""
    domains_dir = CONFIGS_DIR / "domains"
    if not domains_dir.exists():
        pytest.skip("No domains directory")
    for yaml_file in domains_dir.glob("*.yaml"):
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        assert isinstance(data, dict), f"{yaml_file.name} did not load as dict"
        assert "domain_rules" in data, f"{yaml_file.name} missing 'domain_rules'"
