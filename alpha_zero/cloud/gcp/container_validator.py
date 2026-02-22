"""Container image pre-flight validation for Vertex AI orchestrator.

Validates that the specified container image exists in the registry before
dispatching Vertex AI jobs.  Supports both GCR (gcr.io) and Artifact Registry
(``<region>-docker.pkg.dev``) URIs.

Key design decisions (see ``docs/architecture/ADR-pipeline-hardening.md``):

- Uses ``gcloud`` subprocess with a 5-second timeout to prevent Windows hangs.
- Graceful fallback: on any error (network, missing gcloud, timeout) the
  validator returns ``exists=True`` with a logged warning, so local dev is
  never blocked.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import shutil
import subprocess
from typing import Final

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_SUBPROCESS_TIMEOUT_S: Final[int] = 5
_GCR_PATTERN: Final[re.Pattern[str]] = re.compile(r"^(gcr\.io|[a-z]+-docker\.pkg\.dev)/")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ValidationResult:
    """Result of a container image validation check."""

    exists: bool
    digest: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_image_tag(uri: str) -> str:
    """Extract the tag from a container image URI.

    Args:
        uri: Full container image reference, e.g.
             ``gcr.io/project/image:v1.0.0`` or
             ``us-central1-docker.pkg.dev/proj/repo/img:v2``.

    Returns:
        The tag portion (e.g. ``"v1.0.0"``), or ``"latest"`` if no tag is
        present.

    >>> extract_image_tag("gcr.io/project/image:v1.0.0")
    'v1.0.0'
    >>> extract_image_tag("gcr.io/project/image")
    'latest'
    """
    if ":" in uri.split("/")[-1]:
        return uri.rsplit(":", 1)[1]
    return "latest"


def image_matches_version(uri: str, expected_tag: str) -> bool:
    """Check whether the image URI's tag matches *expected_tag*.

    Comparison is case-insensitive.

    Args:
        uri: Full container image reference.
        expected_tag: The version tag to match against.

    Returns:
        ``True`` if the tags match.
    """
    return extract_image_tag(uri).lower() == expected_tag.lower()


def validate_container_image(uri: str) -> ValidationResult:
    """Validate that a container image exists in the remote registry.

    On any failure (no ``gcloud``, network error, timeout, non-zero exit),
    a warning is logged and ``ValidationResult(exists=True)`` is returned
    so that the orchestrator is *not* blocked for local development or
    CI environments without registry access.

    Args:
        uri: Full container image URI (GCR or Artifact Registry).

    Returns:
        A :class:`ValidationResult` with ``exists`` indicating whether the
        image was confirmed, and ``digest`` if available.
    """
    # Check gcloud is available
    if not shutil.which("gcloud"):
        logger.warning("gcloud CLI not found; skipping container validation for %s", uri)
        return ValidationResult(exists=True, error="gcloud not found")

    # Build the describe command depending on registry type
    cmd = _build_describe_command(uri)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )

        if result.returncode == 0:
            digest = _parse_digest(result.stdout)
            logger.info("Container image validated: %s (digest=%s)", uri, digest or "n/a")
            return ValidationResult(exists=True, digest=digest)

        # Non-zero exit → image not found or permissions issue
        err_msg = result.stderr.strip() or result.stdout.strip()
        logger.warning("Container image not found: %s — %s", uri, err_msg)
        return ValidationResult(exists=False, error=err_msg)

    except subprocess.TimeoutExpired:
        logger.warning(
            "gcloud timed out after %ds validating %s; assuming image exists",
            _SUBPROCESS_TIMEOUT_S,
            uri,
        )
        return ValidationResult(exists=True, error="gcloud timeout")

    except OSError as exc:
        logger.warning("gcloud execution error for %s: %s; assuming image exists", uri, exc)
        return ValidationResult(exists=True, error=str(exc))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_describe_command(uri: str) -> list[str]:
    """Build the appropriate ``gcloud`` describe command for the URI format.

    - **Artifact Registry**: ``gcloud artifacts docker images describe <uri>``
    - **GCR / other**: ``gcloud container images describe <uri>``
    """
    if "docker.pkg.dev" in uri:
        return ["gcloud", "artifacts", "docker", "images", "describe", uri]
    return [
        "gcloud",
        "container",
        "images",
        "describe",
        uri,
        "--format=value(image_summary.digest)",
    ]


def _parse_digest(stdout: str) -> str | None:
    """Extract a ``sha256:`` digest from command output, if present."""
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("sha256:"):
            return line
        # Artifact Registry format may use 'digest: sha256:...'
        if "sha256:" in line:
            idx = line.index("sha256:")
            return line[idx:].strip()
    return None
