"""Unit tests for container_validator module.

Tests the pre-flight container image validation used by the orchestrator
to verify images exist before dispatching Vertex AI jobs.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.cloud.gcp.container_validator import (
    ValidationResult,
    _build_describe_command,
    _parse_digest,
    extract_image_tag,
    image_matches_version,
    validate_container_image,
)

# =============================================================================
# extract_image_tag
# =============================================================================


class TestExtractImageTag:
    """Tests for URI tag extraction."""

    def test_gcr_uri_with_tag(self) -> None:
        """Extracts tag from GCR URI."""
        assert extract_image_tag("gcr.io/hrmtrain/civ6az-training:v1.0.0") == "v1.0.0"

    def test_artifact_registry_uri_with_tag(self) -> None:
        """Extracts tag from Artifact Registry URI."""
        assert (
            extract_image_tag("us-central1-docker.pkg.dev/hrmtrain/civ6az/training:v2.1") == "v2.1"
        )

    def test_uri_without_tag_returns_latest(self) -> None:
        """URI without a tag defaults to 'latest'."""
        assert extract_image_tag("gcr.io/hrmtrain/civ6az-training") == "latest"

    def test_tag_with_sha(self) -> None:
        """Handles tag that looks like a sha."""
        assert extract_image_tag("gcr.io/proj/img:abc123") == "abc123"

    def test_complex_path_with_tag(self) -> None:
        """Handles multi-segment paths."""
        uri = "us-central1-docker.pkg.dev/project/repo/sub/image:v3.0.0-beta"
        assert extract_image_tag(uri) == "v3.0.0-beta"


# =============================================================================
# image_matches_version
# =============================================================================


class TestImageMatchesVersion:
    """Tests for version tag comparison."""

    def test_matching_tags(self) -> None:
        """Exact match returns True."""
        assert image_matches_version("gcr.io/proj/img:v1.0.0", "v1.0.0") is True

    def test_case_insensitive(self) -> None:
        """Comparison is case-insensitive."""
        assert image_matches_version("gcr.io/proj/img:V1.0.0", "v1.0.0") is True

    def test_non_matching_tags(self) -> None:
        """Different tags return False."""
        assert image_matches_version("gcr.io/proj/img:v1.0.0", "v2.0.0") is False

    def test_no_tag_vs_latest(self) -> None:
        """URI without tag treated as 'latest'."""
        assert image_matches_version("gcr.io/proj/img", "latest") is True


# =============================================================================
# _build_describe_command
# =============================================================================


class TestBuildDescribeCommand:
    """Tests for gcloud command construction."""

    def test_gcr_command(self) -> None:
        """GCR URIs use 'gcloud container images describe'."""
        cmd = _build_describe_command("gcr.io/hrmtrain/civ6az-training:v1.0.0")
        assert cmd[:3] == ["gcloud", "container", "images"]

    def test_artifact_registry_command(self) -> None:
        """Artifact Registry URIs use 'gcloud artifacts docker images describe'."""
        cmd = _build_describe_command("us-central1-docker.pkg.dev/hrmtrain/civ6az/training:v2")
        assert cmd[:4] == ["gcloud", "artifacts", "docker", "images"]


# =============================================================================
# _parse_digest
# =============================================================================


class TestParseDigest:
    """Tests for digest extraction from gcloud output."""

    def test_direct_sha_line(self) -> None:
        """Parses a line starting with sha256:."""
        output = "sha256:abc123def456\n"
        assert _parse_digest(output) == "sha256:abc123def456"

    def test_colon_prefix_format(self) -> None:
        """Parses 'digest: sha256:...' format from Artifact Registry."""
        output = "digest: sha256:xyz789\nother: stuff\n"
        assert _parse_digest(output) == "sha256:xyz789"

    def test_no_digest(self) -> None:
        """Returns None when no digest found."""
        assert _parse_digest("some random output") is None


# =============================================================================
# validate_container_image — mocked
# =============================================================================


class TestValidateContainerImage:
    """Tests for the main validation function with mocked subprocess."""

    @patch("alpha_zero.cloud.gcp.container_validator.shutil.which", return_value=None)
    def test_no_gcloud_graceful_fallback(self, mock_which: MagicMock) -> None:
        """When gcloud is missing, returns exists=True with warning."""
        result = validate_container_image("gcr.io/proj/img:v1")
        assert result.exists is True
        assert result.error == "gcloud not found"

    @patch("alpha_zero.cloud.gcp.container_validator.shutil.which", return_value="/usr/bin/gcloud")
    @patch("alpha_zero.cloud.gcp.container_validator.subprocess.run")
    def test_image_exists(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Successful describe → exists=True with digest."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="sha256:abc123def456\n",
            stderr="",
        )
        result = validate_container_image("gcr.io/proj/img:v1")
        assert result.exists is True
        assert result.digest == "sha256:abc123def456"

    @patch("alpha_zero.cloud.gcp.container_validator.shutil.which", return_value="/usr/bin/gcloud")
    @patch("alpha_zero.cloud.gcp.container_validator.subprocess.run")
    def test_image_not_found(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """Non-zero exit → exists=False."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="NOT_FOUND: image not found",
        )
        result = validate_container_image("gcr.io/proj/img:v999")
        assert result.exists is False
        assert "NOT_FOUND" in (result.error or "")

    @patch("alpha_zero.cloud.gcp.container_validator.shutil.which", return_value="/usr/bin/gcloud")
    @patch("alpha_zero.cloud.gcp.container_validator.subprocess.run")
    def test_timeout_graceful_fallback(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """gcloud timeout → exists=True (don't block)."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="gcloud", timeout=5)
        result = validate_container_image("gcr.io/proj/img:v1")
        assert result.exists is True
        assert result.error == "gcloud timeout"

    @patch("alpha_zero.cloud.gcp.container_validator.shutil.which", return_value="/usr/bin/gcloud")
    @patch("alpha_zero.cloud.gcp.container_validator.subprocess.run")
    def test_os_error_graceful_fallback(self, mock_run: MagicMock, mock_which: MagicMock) -> None:
        """OS error during gcloud execution → exists=True."""
        mock_run.side_effect = OSError("Permission denied")
        result = validate_container_image("gcr.io/proj/img:v1")
        assert result.exists is True
        assert "Permission denied" in (result.error or "")


# =============================================================================
# ValidationResult dataclass
# =============================================================================


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_defaults(self) -> None:
        """Default digest and error are None."""
        result = ValidationResult(exists=True)
        assert result.digest is None
        assert result.error is None

    def test_frozen(self) -> None:
        """Dataclass is frozen/immutable."""
        result = ValidationResult(exists=True, digest="sha256:abc")
        with pytest.raises(AttributeError):
            result.exists = False  # type: ignore[misc]

    def test_equality(self) -> None:
        """Two identical results are equal."""
        a = ValidationResult(exists=True, digest="sha256:abc")
        b = ValidationResult(exists=True, digest="sha256:abc")
        assert a == b
