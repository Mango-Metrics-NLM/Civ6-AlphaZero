"""
Tests for alpha_zero/utils/env_loader.py.

Covers loading from .env files, overriding with os.environ,
required variable warnings, and error handling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero.utils.env_loader import (
    _parse_dotenv,
    load_civ6az_env,
    require_civ6az_env,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env() -> None:
    """Remove CIV6AZ_* vars from environment before/after each test."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("CIV6AZ_")}
    for k in saved:
        del os.environ[k]
    yield
    for k in list(os.environ):
        if k.startswith("CIV6AZ_"):
            del os.environ[k]
    os.environ.update(saved)


def _write_dotenv(content: str) -> Path:
    """Write a .env file in a temp directory and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".env", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    return Path(tmp.name)


# ---------------------------------------------------------------------------
# TestParseDotenv
# ---------------------------------------------------------------------------


class TestParseDotenv:
    """Tests for the internal _parse_dotenv() function."""

    def test_basic_key_value(self) -> None:
        """Should parse basic KEY=value lines."""
        path = _write_dotenv("FOO=bar\nBAZ=qux\n")
        result = _parse_dotenv(path)
        assert result["FOO"] == "bar"
        assert result["BAZ"] == "qux"

    def test_ignores_comments(self) -> None:
        """Lines starting with # should be ignored."""
        path = _write_dotenv("# comment\nFOO=bar\n")
        result = _parse_dotenv(path)
        assert "# comment" not in result
        assert result["FOO"] == "bar"

    def test_ignores_blank_lines(self) -> None:
        """Blank lines should be ignored."""
        path = _write_dotenv("\n\nFOO=bar\n\n")
        result = _parse_dotenv(path)
        assert result == {"FOO": "bar"}

    def test_strips_quotes(self) -> None:
        """Quoted values should have their quotes stripped."""
        path = _write_dotenv('FOO="quoted"\nBAR=\'single\'\n')
        result = _parse_dotenv(path)
        assert result["FOO"] == "quoted"
        assert result["BAR"] == "single"

    def test_strips_whitespace(self) -> None:
        """Keys and values should have surrounding whitespace stripped."""
        path = _write_dotenv("  FOO = bar  \n")
        result = _parse_dotenv(path)
        assert result["FOO"] == "bar"

    def test_ignores_lines_without_equals(self) -> None:
        """Lines without '=' should be silently skipped."""
        path = _write_dotenv("NO_EQUALS_HERE\nFOO=bar\n")
        result = _parse_dotenv(path)
        assert "NO_EQUALS_HERE" not in result
        assert result["FOO"] == "bar"

    def test_value_with_equals_sign(self) -> None:
        """Values containing '=' should be handled correctly."""
        path = _write_dotenv("URL=http://example.com?a=b&c=d\n")
        result = _parse_dotenv(path)
        # Only first '=' splits key from value
        assert result["URL"] == "http://example.com?a=b&c=d"


# ---------------------------------------------------------------------------
# TestLoadCiv6azEnv
# ---------------------------------------------------------------------------


class TestLoadCiv6azEnv:
    """Tests for load_civ6az_env()."""

    def test_loads_from_explicit_path(self) -> None:
        """Should load variables from an explicit .env path."""
        path = _write_dotenv(
            "CIV6AZ_GCS_BUCKET=civ6az-training\n"
            "CIV6AZ_VERTEX_PROJECT=hrmtrain\n"
        )
        result = load_civ6az_env(dotenv_path=path, warn_missing=False)
        assert result["CIV6AZ_GCS_BUCKET"] == "civ6az-training"
        assert result["CIV6AZ_VERTEX_PROJECT"] == "hrmtrain"
        assert os.environ["CIV6AZ_GCS_BUCKET"] == "civ6az-training"

    def test_does_not_override_existing_env_by_default(self) -> None:
        """Existing env vars should not be overridden when override=False."""
        os.environ["CIV6AZ_GCS_BUCKET"] = "original-bucket"
        path = _write_dotenv("CIV6AZ_GCS_BUCKET=new-bucket\n")
        load_civ6az_env(dotenv_path=path, override=False, warn_missing=False)
        assert os.environ["CIV6AZ_GCS_BUCKET"] == "original-bucket"

    def test_overrides_existing_env_when_override_true(self) -> None:
        """With override=True, .env values should replace existing env vars."""
        os.environ["CIV6AZ_GCS_BUCKET"] = "original-bucket"
        path = _write_dotenv("CIV6AZ_GCS_BUCKET=new-bucket\n")
        load_civ6az_env(dotenv_path=path, override=True, warn_missing=False)
        assert os.environ["CIV6AZ_GCS_BUCKET"] == "new-bucket"

    def test_warns_on_missing_required_vars(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log a warning when required variables are missing."""
        import logging

        with caplog.at_level(logging.WARNING):
            load_civ6az_env(dotenv_path=None, warn_missing=True)

        warning_messages = " ".join(r.message for r in caplog.records)
        assert "CIV6AZ_GCS_BUCKET" in warning_messages or "Missing" in warning_messages

    def test_no_warning_when_warn_missing_false(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should not warn when warn_missing=False."""
        import logging

        with caplog.at_level(logging.WARNING):
            load_civ6az_env(dotenv_path=None, warn_missing=False)

        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 0

    def test_no_dotenv_file_no_error(self) -> None:
        """Should not raise when no .env file is found."""
        # Point to a nonexistent directory
        result = load_civ6az_env(
            dotenv_path="/nonexistent/path/.env", warn_missing=False
        )
        assert isinstance(result, dict)

    def test_returns_resolved_dict(self) -> None:
        """Should return a dict containing required and optional vars."""
        path = _write_dotenv(
            "CIV6AZ_GCS_BUCKET=mybucket\n"
            "CIV6AZ_VERTEX_PROJECT=myproject\n"
        )
        result = load_civ6az_env(dotenv_path=path, warn_missing=False)
        assert "CIV6AZ_GCS_BUCKET" in result
        assert "CIV6AZ_VERTEX_PROJECT" in result

    def test_optional_vars_included_with_defaults(self) -> None:
        """Optional variables should appear in returned dict with defaults."""
        result = load_civ6az_env(dotenv_path=None, warn_missing=False)
        # CIV6AZ_VERTEX_REGION has a default of us-central1
        assert "CIV6AZ_VERTEX_REGION" in result

    def test_invalid_dotenv_file_does_not_raise(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OSError reading .env should be logged as warning, not raised."""
        import logging

        with caplog.at_level(logging.WARNING):
            # Point to a directory, not a file
            load_civ6az_env(dotenv_path="/tmp", warn_missing=False)
        # No assertion error = it didn't raise


# ---------------------------------------------------------------------------
# TestRequireCiv6azEnv
# ---------------------------------------------------------------------------


class TestRequireCiv6azEnv:
    """Tests for require_civ6az_env()."""

    def test_raises_when_required_vars_missing(self) -> None:
        """Should raise RuntimeError when required vars are not set."""
        with pytest.raises(RuntimeError, match="Missing required"):
            require_civ6az_env(dotenv_path=None)

    def test_does_not_raise_when_all_required_set(self) -> None:
        """Should not raise when all required vars are set."""
        path = _write_dotenv(
            "CIV6AZ_GCS_BUCKET=civ6az-training\n"
            "CIV6AZ_VERTEX_PROJECT=hrmtrain\n"
        )
        result = require_civ6az_env(dotenv_path=path)
        assert result["CIV6AZ_GCS_BUCKET"] == "civ6az-training"

    def test_raises_with_descriptive_message(self) -> None:
        """Error message should include the missing variable name."""
        with pytest.raises(RuntimeError) as exc_info:
            require_civ6az_env(dotenv_path=None)
        assert "CIV6AZ_GCS_BUCKET" in str(exc_info.value) or \
               "CIV6AZ_VERTEX_PROJECT" in str(exc_info.value)

    def test_loads_from_env_file(self) -> None:
        """Should load from .env file and return without raising."""
        path = _write_dotenv(
            "CIV6AZ_GCS_BUCKET=testbucket\n"
            "CIV6AZ_VERTEX_PROJECT=testproject\n"
        )
        result = require_civ6az_env(dotenv_path=path)
        assert result["CIV6AZ_GCS_BUCKET"] == "testbucket"
        assert result["CIV6AZ_VERTEX_PROJECT"] == "testproject"
