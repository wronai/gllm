"""ContextSchemaGenerator — generates a structured context schema for small-LLM decision making.

Combines shell context, folder compression, user memory, and domain rules
into a single schema that the small LLM uses to choose strategy and enrich prompts.
"""

from __future__ import annotations

import logging
import platform
import shutil
from typing import Any

from prellm.models import CompressedFolder, ContextSchema, DomainRule, ShellContext

logger = logging.getLogger("prellm.context.schema_generator")

# Common tools to detect
_TOOL_COMMANDS = [
    "git", "docker", "kubectl", "helm", "terraform", "ansible",
    "npm", "yarn", "pnpm", "pip", "poetry", "cargo", "go",
    "make", "cmake", "gcc", "rustc", "node", "python3", "java",
    "curl", "wget", "jq", "yq", "ssh", "rsync",
]

# Project type detection by file presence (checked in folder compression)
_PROJECT_TYPE_MARKERS = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "requirements.txt": "python",
    "package.json": "node",
    "Cargo.toml": "rust",
    "go.mod": "go",
    "pom.xml": "java",
    "build.gradle": "java",
    "Makefile": None,  # generic
}


class ContextSchemaGenerator:
    """Generates a structured context schema from available context sources."""

    def generate(
        self,
        shell_context: ShellContext | None = None,
        folder_compressed: CompressedFolder | None = None,
        user_memory: list[dict[str, Any]] | None = None,
        domain_rules: list[DomainRule] | None = None,
        sensitive_blocked: int = 0,
    ) -> ContextSchema:
        """Compose schema from available context sources.

        Args:
            shell_context: Shell environment snapshot.
            folder_compressed: Compressed folder representation.
            user_memory: Recent user interaction history.
            domain_rules: Active domain rules.
            sensitive_blocked: Count of sensitive fields that were blocked.

        Returns:
            ContextSchema ready for small-LLM consumption.
        """
        schema = ContextSchema(
            platform=platform.system().lower(),
            sensitive_fields_blocked=sensitive_blocked,
        )

        # Shell context enrichment
        if shell_context:
            schema.execution_env = self._detect_execution_env(shell_context)
            if shell_context.locale.lang:
                schema.locale = shell_context.locale.lang
            if shell_context.locale.timezone:
                schema.timezone = shell_context.locale.timezone

        # Detect available tools
        schema.available_tools = self._detect_tools()

        # Folder compression enrichment
        if folder_compressed:
            schema.project_type = self._detect_project_type(folder_compressed)
            schema.project_summary = self._build_project_summary(folder_compressed)

        # User memory enrichment
        if user_memory:
            schema.user_history_summary = self._summarize_history(user_memory)

        # Estimate schema token cost
        schema.schema_token_cost = self._estimate_token_cost(schema)

        return schema

    def to_prompt_section(self, schema: ContextSchema) -> str:
        """Format schema as a prompt section for small-LLM."""
        lines = ["[Environment Context]"]
        lines.append(f"Platform: {schema.platform}")
        lines.append(f"Env: {schema.execution_env}")

        if schema.project_type:
            lines.append(f"Project: {schema.project_type}")
        if schema.project_summary:
            lines.append(f"Summary: {schema.project_summary}")
        if schema.available_tools:
            lines.append(f"Tools: {', '.join(schema.available_tools[:10])}")
        if schema.locale:
            lines.append(f"Locale: {schema.locale}")
        if schema.timezone:
            lines.append(f"Timezone: {schema.timezone}")
        if schema.user_history_summary:
            lines.append(f"History: {schema.user_history_summary}")
        if schema.sensitive_fields_blocked > 0:
            lines.append(f"Filtered: {schema.sensitive_fields_blocked} sensitive fields blocked")

        return "\n".join(lines)

    def estimate_relevance(
        self, schema: ContextSchema, query: str
    ) -> dict[str, float]:
        """Score which context parts are relevant for the query (0-1)."""
        query_lower = query.lower()
        scores: dict[str, float] = {}

        # Platform relevance
        if any(w in query_lower for w in ("linux", "windows", "macos", "os", "system")):
            scores["platform"] = 0.9
        else:
            scores["platform"] = 0.3

        # Project relevance
        if any(w in query_lower for w in ("code", "refactor", "function", "class", "module", "project")):
            scores["project"] = 0.9
        else:
            scores["project"] = 0.2

        # Tools relevance
        tool_mentions = sum(1 for t in (schema.available_tools or []) if t in query_lower)
        scores["tools"] = min(1.0, tool_mentions * 0.3 + 0.1)

        # History relevance
        if schema.user_history_summary:
            scores["history"] = 0.5
        else:
            scores["history"] = 0.0

        # Locale relevance
        if any(w in query_lower for w in ("polish", "polski", "pl_", "locale", "language", "timezone")):
            scores["locale"] = 0.8
        else:
            scores["locale"] = 0.1

        return scores

    @staticmethod
    def _detect_execution_env(shell_context: ShellContext) -> str:
        """Detect execution environment from shell context."""
        env_vars = shell_context.env_vars
        if "KUBERNETES_SERVICE_HOST" in env_vars:
            return "kubernetes"
        if "DOCKER_CONTAINER" in env_vars or "/.dockerenv" in env_vars.get("HOME", ""):
            return "docker"
        if shell_context.shell.term:
            return "shell"
        return "cli"

    @staticmethod
    def _detect_tools() -> list[str]:
        """Detect which common tools are available on the system."""
        found: list[str] = []
        for tool in _TOOL_COMMANDS:
            if shutil.which(tool):
                found.append(tool)
        return found

    @staticmethod
    def _detect_project_type(compressed: CompressedFolder) -> str | None:
        """Detect project type from compressed folder data."""
        toon = compressed.toon_output.lower()
        if ".py," in toon:
            return "python"
        if ".js," in toon or ".ts," in toon:
            return "node"
        if ".rs," in toon:
            return "rust"
        if ".go," in toon:
            return "go"
        return None

    @staticmethod
    def _build_project_summary(compressed: CompressedFolder) -> str:
        """Build a 1-line project summary from compressed data."""
        return (
            f"{compressed.total_modules} modules, "
            f"{compressed.total_functions} functions, "
            f"~{compressed.estimated_tokens} tokens"
        )

    @staticmethod
    def _summarize_history(history: list[dict[str, Any]]) -> str:
        """Summarize recent user interaction history."""
        if not history:
            return ""
        queries = [h.get("query", "") for h in history[:3] if h.get("query")]
        if not queries:
            return ""
        return f"Recent: {'; '.join(q[:50] for q in queries)}"

    @staticmethod
    def _estimate_token_cost(schema: ContextSchema) -> int:
        """Estimate how many tokens the schema will use in a prompt."""
        text = schema.model_dump_json()
        return len(text) // 4
