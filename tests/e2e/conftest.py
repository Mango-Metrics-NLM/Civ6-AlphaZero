"""E2E test fixtures and configuration.

Provides specialized fixtures for end-to-end testing with real
GCP infrastructure. These fixtures handle:
- Environment validation
- Automatic cleanup of cloud resources
- Timeout handling for cloud operations
- Diagnostic logging
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.cloud.config import CloudConfig

# Configure E2E test logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("e2e")


# =============================================================================
# Environment Validation Fixtures
# =============================================================================


def _get_gcp_environment_info() -> dict[str, Any]:
    """Get GCP environment information for diagnostics."""
    info: dict[str, Any] = {
        "GCP_PROJECT_ID": os.environ.get("GCP_PROJECT_ID", "NOT SET"),
        "GCS_BUCKET_NAME": os.environ.get("GCS_BUCKET_NAME", "NOT SET"),
        "GCP_REGION": os.environ.get("GCP_REGION", "us-central1 (default)"),
    }

    # Check for credentials
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        info["credentials"] = str(creds_path) if creds_path.exists() else "FILE NOT FOUND"
    else:
        # Check for ADC
        if os.name == "nt":
            adc_path = Path(os.environ.get("APPDATA", "")) / "gcloud" / "application_default_credentials.json"
        else:
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        info["credentials"] = "ADC" if adc_path.exists() else "NONE"

    return info


@pytest.fixture(scope="session")
def gcp_environment_info() -> dict[str, Any]:
    """Get GCP environment information for all E2E tests.

    Returns:
        Dictionary with environment configuration details.
    """
    info = _get_gcp_environment_info()
    logger.info("GCP Environment Configuration:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    return info


@pytest.fixture(scope="session")
def e2e_cloud_config() -> CloudConfig:
    """Create cloud configuration from environment for E2E testing.

    This fixture relies on environment variables being set:
    - GCP_PROJECT_ID
    - GCS_BUCKET_NAME
    - GCP_REGION (optional, defaults to us-central1)

    Returns:
        Validated CloudConfig instance.

    Raises:
        pytest.skip: If required environment variables are not set.
    """
    project_id = os.environ.get("GCP_PROJECT_ID", "")
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")

    if not project_id or not bucket_name:
        pytest.skip(
            "E2E tests require GCP_PROJECT_ID and GCS_BUCKET_NAME environment variables"
        )

    config = CloudConfig.from_env()
    config.validate()

    logger.info(f"E2E Cloud Config: project={config.project_id}, bucket={config.bucket_name}")
    return config


# =============================================================================
# Artifact and Cleanup Fixtures
# =============================================================================


@pytest.fixture
def unique_test_id() -> str:
    """Generate a unique ID for test artifacts.

    Returns:
        Unique 8-character hex string.
    """
    return uuid.uuid4().hex[:8]


@pytest.fixture
def test_artifact_path(unique_test_id: str) -> str:
    """Generate unique path for test artifacts to avoid conflicts.

    Args:
        unique_test_id: Unique test identifier.

    Returns:
        GCS path prefix for test artifacts.
    """
    return f"e2e-tests/{unique_test_id}"


@pytest.fixture
def cleanup_gcs_artifacts(
    e2e_cloud_config: CloudConfig,
    test_artifact_path: str,
) -> Generator[str, None, None]:
    """Fixture that yields path and cleans up after test.

    Automatically deletes all GCS objects created during the test.

    Args:
        e2e_cloud_config: Cloud configuration.
        test_artifact_path: Path prefix for test artifacts.

    Yields:
        The test artifact path for use in tests.
    """
    yield test_artifact_path

    # Cleanup: Delete test artifacts from GCS
    try:
        from alpha_zero.cloud.gcp import GCS_AVAILABLE, GCSCheckpointManager

        if not GCS_AVAILABLE:
            return

        manager = GCSCheckpointManager(e2e_cloud_config)
        checkpoints = manager.list_checkpoints(prefix=test_artifact_path)

        deleted_count = 0
        for cp in checkpoints:
            try:
                manager._client.bucket(e2e_cloud_config.bucket_name).blob(cp).delete()
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to cleanup {cp}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} test artifacts")

    except Exception as e:
        logger.warning(f"Cleanup failed: {e}")


# =============================================================================
# Timeout and Retry Fixtures
# =============================================================================


@pytest.fixture
def cloud_operation_timeout() -> int:
    """Get timeout for cloud operations.

    Returns:
        Timeout in seconds (from environment or default).
    """
    return int(os.environ.get("E2E_CLOUD_TIMEOUT", "60"))


class RetryHelper:
    """Helper for retrying flaky cloud operations."""

    def __init__(
        self,
        max_attempts: int = 3,
        delay_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        """Initialize retry helper.

        Args:
            max_attempts: Maximum number of retry attempts.
            delay_seconds: Initial delay between retries.
            backoff_multiplier: Multiplier for exponential backoff.
        """
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.backoff_multiplier = backoff_multiplier

    def execute(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        """Execute operation with retries.

        Args:
            operation: Callable to execute.
            *args: Positional arguments for operation.
            **kwargs: Keyword arguments for operation.

        Returns:
            Result of successful operation.

        Raises:
            Last exception if all retries fail.
        """
        last_exception: Exception | None = None
        delay = self.delay_seconds

        for attempt in range(self.max_attempts):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_attempts - 1:
                    logger.warning(
                        f"Attempt {attempt + 1}/{self.max_attempts} failed: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= self.backoff_multiplier

        raise last_exception  # type: ignore[misc]


@pytest.fixture
def retry_helper() -> RetryHelper:
    """Get retry helper for cloud operations.

    Returns:
        Configured RetryHelper instance.
    """
    return RetryHelper()


# =============================================================================
# SDK Availability Markers
# =============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for E2E tests."""
    config.addinivalue_line(
        "markers",
        "requires_gcs: marks tests requiring GCS access"
    )
    config.addinivalue_line(
        "markers",
        "requires_vertex: marks tests requiring Vertex AI access"
    )
    config.addinivalue_line(
        "markers",
        "requires_logging: marks tests requiring Cloud Logging access"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (>30s)"
    )


# =============================================================================
# Diagnostic Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def log_test_name(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Log test name at start and end."""
    test_name = request.node.name
    logger.info(f"{'=' * 40}")
    logger.info(f"Starting: {test_name}")
    logger.info(f"{'=' * 40}")

    start_time = time.time()
    yield

    duration = time.time() - start_time
    logger.info(f"Completed: {test_name} ({duration:.2f}s)")
