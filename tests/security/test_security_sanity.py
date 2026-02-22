"""Security sanity tests.

Validates that the repository does not contain hardcoded secrets,
credentials, or other sensitive data that should be externalized
via environment variables or secret managers.

Uses dynamic file scanning — no hardcoded file lists.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (externalized, not hardcoded)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Directories to skip during scanning
_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "node_modules",
        ".venv",
        "venv",
        "htmlcov",
        ".tox",
        ".agent",
        ".gemini",
        "wandb",
    }
)

# Extensions to scan
_SCAN_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py",
        ".yaml",
        ".yml",
        ".toml",
        ".cfg",
        ".ini",
        ".json",
        ".sh",
        ".ps1",
    }
)

# Known placeholder values that are safe to commit
_SAFE_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "YOUR_WANDB_API_KEY_HERE",
        "YOUR_API_KEY_HERE",
        "test-key",
        "placeholder",
        "changeme",
        "REPLACE_ME",
    }
)

# Sensitive environment variable names to scan for real values
_SENSITIVE_ENV_NAMES: frozenset[str] = frozenset(
    {
        "WANDB_API_KEY",
        "API_KEY",
        "SECRET_KEY",
        "DB_PASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "GCP_SERVICE_ACCOUNT_KEY",
        "PRIVATE_KEY",
        "AUTH_TOKEN",
    }
)

# Patterns that indicate potential hardcoded secrets
_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS Access Key", re.compile(r"AKIA[0-9A-Z]{16}", re.ASCII)),
    ("GCP Service Account Key", re.compile(r'"type"\s*:\s*"service_account"')),
    ("Private Key Block", re.compile(r"-----BEGIN (?:RSA |EC )?PRIVATE KEY-----")),
    (
        "Generic API Key Assignment",
        re.compile(
            r"""(?:api[_-]?key|secret[_-]?key|password|token)\s*[=:]\s*["'][A-Za-z0-9+/=]{16,}["']""",
            re.IGNORECASE,
        ),
    ),
    (
        "YAML Env Secret Value",
        re.compile(
            r"""value:\s*["'][A-Za-z0-9+/=]{20,}["']""",
            re.IGNORECASE,
        ),
    ),
]

# Files that are exempt from scanning (templates, test fixtures, etc.)
_EXEMPT_PATTERNS: frozenset[str] = frozenset(
    {
        ".env.template",
        ".env.example",
        "test_security_sanity.py",  # This file itself contains patterns
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iter_source_files(
    root: Path,
    extensions: frozenset[str],
    skip_dirs: frozenset[str],
) -> Generator[Path, None, None]:
    """Walk the project tree yielding files with target extensions.

    Skips directories in *skip_dirs* for efficiency and safety.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place (modifies os.walk behavior)
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                yield Path(dirpath) / filename


def _is_exempt(filepath: Path) -> bool:
    """Check if a file is exempt from secret scanning."""
    return filepath.name in _EXEMPT_PATTERNS


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSecuritySanity:
    """Repository-wide security sanity checks."""

    def test_no_hardcoded_secrets(self) -> None:
        """Scan source files for patterns that look like hardcoded secrets."""
        violations: list[str] = []

        for filepath in _iter_source_files(_PROJECT_ROOT, _SCAN_EXTENSIONS, _SKIP_DIRS):
            if _is_exempt(filepath):
                continue

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for pattern_name, pattern in _SECRET_PATTERNS:
                matches = pattern.findall(content)
                if matches:
                    rel_path = filepath.relative_to(_PROJECT_ROOT)
                    violations.append(f"{rel_path}: {pattern_name} ({len(matches)} match(es))")

        if violations:
            msg = "Potential hardcoded secrets found:\n" + "\n".join(f"  - {v}" for v in violations)
            pytest.fail(msg)
        else:
            logger.info("No hardcoded secrets detected in %s", _PROJECT_ROOT)

    def test_env_file_gitignored(self) -> None:
        """Verify that .env is listed in .gitignore."""
        gitignore_path = _PROJECT_ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.skip(".gitignore not found")

        content = gitignore_path.read_text(encoding="utf-8")
        # Check for .env pattern (not .env.template or .env.example)
        assert re.search(r"^\.env$", content, re.MULTILINE), ".env must be listed in .gitignore"

    def test_env_template_exists(self) -> None:
        """Verify that a .env.template or .env.example exists."""
        has_template = (_PROJECT_ROOT / ".env.template").exists() or (
            _PROJECT_ROOT / ".env.example"
        ).exists()
        assert has_template, "Project should have .env.template or .env.example"

    def test_service_account_json_gitignored(self) -> None:
        """Verify service account JSON files are gitignored."""
        gitignore_path = _PROJECT_ROOT / ".gitignore"
        if not gitignore_path.exists():
            pytest.skip(".gitignore not found")

        content = gitignore_path.read_text(encoding="utf-8")
        assert re.search(
            r"service_account.*\.json", content, re.IGNORECASE
        ), "service_account*.json should be in .gitignore"

    def test_no_credentials_json_committed(self) -> None:
        """Verify no credentials.json or service_account.json in the source tree."""
        dangerous_names = {"credentials.json", "service_account.json"}
        found: list[str] = []

        for filepath in _iter_source_files(_PROJECT_ROOT, frozenset({".json"}), _SKIP_DIRS):
            if filepath.name in dangerous_names:
                rel_path = filepath.relative_to(_PROJECT_ROOT)
                found.append(str(rel_path))

        assert not found, f"Credential files found in source tree: {found}"

    def test_dockerignore_excludes_secrets(self) -> None:
        """Verify .dockerignore excludes sensitive files."""
        dockerignore_path = _PROJECT_ROOT / ".dockerignore"
        if not dockerignore_path.exists():
            pytest.skip(".dockerignore not found")

        content = dockerignore_path.read_text(encoding="utf-8")
        assert ".env" in content, ".dockerignore should exclude .env files"

    def test_no_debug_flags_in_production_code(self) -> None:
        """Check that DEBUG=True or similar flags aren't hardcoded in source."""
        violations: list[str] = []

        for filepath in _iter_source_files(_PROJECT_ROOT, frozenset({".py"}), _SKIP_DIRS):
            if "test" in filepath.name.lower():
                continue  # Skip test files

            try:
                content = filepath.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            # Look for hardcoded debug assignments (not in comments)
            for line_num, line in enumerate(content.splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if re.match(r"^DEBUG\s*=\s*True", stripped):
                    rel_path = filepath.relative_to(_PROJECT_ROOT)
                    violations.append(f"{rel_path}:{line_num}")

        assert not violations, f"Hardcoded DEBUG=True found: {violations}"

    def test_yaml_configs_no_real_secrets(self) -> None:
        """Scan YAML config files for non-placeholder secret values.

        Catches secrets in nested YAML structures like Vertex AI env blocks
        that regex-only scanning may miss.
        """
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")

        violations: list[str] = []
        config_dir = _PROJECT_ROOT / "configs"

        if not config_dir.exists():
            return

        for filepath in config_dir.rglob("*.yaml"):
            if _is_exempt(filepath):
                continue
            try:
                content = filepath.read_text(encoding="utf-8")
                docs = yaml.safe_load_all(content)
                for doc in docs:
                    if doc is not None:
                        self._scan_yaml_node(
                            doc,
                            filepath.relative_to(_PROJECT_ROOT),
                            violations,
                        )
            except (OSError, yaml.YAMLError):
                continue

        if violations:
            msg = "Potential real secrets in YAML configs:\n" + "\n".join(
                f"  - {v}" for v in violations
            )
            pytest.fail(msg)
        else:
            logger.info("No real secrets found in YAML configs")

    @staticmethod
    def _scan_yaml_node(
        node: Any,
        filepath: Path,
        violations: list[str],
        *,
        parent_key: str = "",
    ) -> None:
        """Recursively scan a YAML node for sensitive env values."""
        if isinstance(node, dict):
            name = node.get("name", "")
            value = node.get("value", "")
            # Check for env-block pattern: {name: SENSITIVE, value: real_key}
            if (
                isinstance(name, str)
                and isinstance(value, str)
                and name.upper() in _SENSITIVE_ENV_NAMES
                and value not in _SAFE_PLACEHOLDERS
                and len(value) > 8
            ):
                violations.append(f"{filepath}: {name} has non-placeholder value")
            for key, child in node.items():
                TestSecuritySanity._scan_yaml_node(
                    child,
                    filepath,
                    violations,
                    parent_key=str(key),
                )
        elif isinstance(node, list):
            for item in node:
                TestSecuritySanity._scan_yaml_node(
                    item,
                    filepath,
                    violations,
                    parent_key=parent_key,
                )
