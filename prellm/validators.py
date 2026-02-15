"""ResponseValidator — validates LLM outputs against YAML-defined schemas.

Supports required fields, type checking, enum constraints, and numeric range constraints.
Used by both pipeline intermediate steps and final executor output validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger("prellm.validators")

_DEFAULT_SCHEMAS_PATH = Path(__file__).parent.parent / "configs" / "response_schemas.yaml"


class ValidationResult(BaseModel):
    """Result of validating data against a schema."""
    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    schema_name: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


class SchemaDefinition(BaseModel):
    """Parsed schema definition from YAML."""
    name: str
    required_fields: list[str] = Field(default_factory=list)
    types: dict[str, str] = Field(default_factory=dict)
    constraints: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ResponseValidator:
    """Validates LLM responses against YAML-defined schemas.

    Usage:
        validator = ResponseValidator()
        result = validator.validate({"intent": "deploy", "confidence": 0.9}, "classification")
        assert result.valid
    """

    def __init__(self, schemas_path: Path | str | None = None):
        self._path = Path(schemas_path) if schemas_path else _DEFAULT_SCHEMAS_PATH
        self._schemas: dict[str, SchemaDefinition] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        """Load schemas from YAML file."""
        if not self._path.exists():
            logger.warning(f"Schemas file not found: {self._path}, using empty validator")
            self._loaded = True
            return

        with open(self._path) as f:
            raw = yaml.safe_load(f) or {}

        for name, schema_raw in raw.get("schemas", {}).items():
            if isinstance(schema_raw, dict):
                self._schemas[name] = SchemaDefinition(
                    name=name,
                    required_fields=schema_raw.get("required_fields", []),
                    types=schema_raw.get("types", {}),
                    constraints=schema_raw.get("constraints", {}),
                )

        self._loaded = True
        logger.debug(f"Loaded {len(self._schemas)} schemas from {self._path}")

    def list_schemas(self) -> list[str]:
        """List available schema names."""
        self._ensure_loaded()
        return sorted(self._schemas.keys())

    def validate(self, data: dict[str, Any], schema_name: str) -> ValidationResult:
        """Validate a dict against a named schema.

        Args:
            data: The data dict to validate (typically parsed JSON from LLM).
            schema_name: Name of the schema to validate against.

        Returns:
            ValidationResult with valid flag and list of errors.
        """
        self._ensure_loaded()

        schema = self._schemas.get(schema_name)
        if schema is None:
            return ValidationResult(
                valid=False,
                errors=[f"Schema '{schema_name}' not found. Available: {self.list_schemas()}"],
                schema_name=schema_name,
                data=data,
            )

        errors: list[str] = []

        # Check required fields
        for field in schema.required_fields:
            if field not in data:
                errors.append(f"Missing required field: '{field}'")

        # Check types
        for field, expected_type in schema.types.items():
            if field in data:
                type_error = self._check_type(data[field], expected_type, field)
                if type_error:
                    errors.append(type_error)

        # Check constraints
        for field, constraint in schema.constraints.items():
            if field in data:
                constraint_errors = self._check_constraints(data[field], constraint, field)
                errors.extend(constraint_errors)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            schema_name=schema_name,
            data=data,
        )

    def validate_or_retry(
        self,
        data: dict[str, Any],
        schema_name: str,
        retry_fn: Callable[[], dict[str, Any]],
        max_retries: int = 2,
    ) -> tuple[dict[str, Any], int]:
        """Validate, and if invalid, call retry_fn and try again.

        Args:
            data: Initial data to validate.
            schema_name: Schema name to validate against.
            retry_fn: Callable that returns new data on retry.
            max_retries: Maximum number of retries.

        Returns:
            Tuple of (validated_data, number_of_retries_used).
        """
        result = self.validate(data, schema_name)
        retries = 0

        while not result.valid and retries < max_retries:
            retries += 1
            logger.info(f"Validation failed for '{schema_name}', retry {retries}/{max_retries}: {result.errors}")
            data = retry_fn()
            result = self.validate(data, schema_name)

        return data, retries

    @staticmethod
    def _check_type(value: Any, expected_type: str, field: str) -> str | None:
        """Check if a value matches the expected type string."""
        type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "str": str,
            "float": (int, float),
            "int": int,
            "bool": bool,
            "list": list,
            "dict": dict,
        }

        # Handle parameterized types like list[string]
        if expected_type.startswith("list["):
            if not isinstance(value, list):
                return f"Field '{field}': expected list, got {type(value).__name__}"
            return None

        expected = type_map.get(expected_type)
        if expected is None:
            return None  # Unknown type, skip validation

        if not isinstance(value, expected):
            return f"Field '{field}': expected {expected_type}, got {type(value).__name__}"
        return None

    @staticmethod
    def _check_constraints(value: Any, constraint: dict[str, Any], field: str) -> list[str]:
        """Check constraints on a value."""
        errors: list[str] = []

        # Numeric range: min, max
        if "min" in constraint and isinstance(value, (int, float)):
            if value < constraint["min"]:
                errors.append(f"Field '{field}': value {value} below minimum {constraint['min']}")

        if "max" in constraint and isinstance(value, (int, float)):
            if value > constraint["max"]:
                errors.append(f"Field '{field}': value {value} above maximum {constraint['max']}")

        # Enum constraint
        if "enum" in constraint:
            allowed = constraint["enum"]
            if value not in allowed:
                errors.append(f"Field '{field}': value '{value}' not in allowed values {allowed}")

        # List length constraints
        if isinstance(value, list):
            if "min_length" in constraint and len(value) < constraint["min_length"]:
                errors.append(f"Field '{field}': list length {len(value)} below minimum {constraint['min_length']}")
            if "max_length" in constraint and len(value) > constraint["max_length"]:
                errors.append(f"Field '{field}': list length {len(value)} above maximum {constraint['max_length']}")

        return errors
