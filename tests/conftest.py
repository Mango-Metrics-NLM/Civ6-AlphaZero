"""Shared pytest fixtures and configuration for all tests.

Provides:
- Temporary directory fixtures for checkpoint testing
- DDP mock fixtures for distributed training tests
- Bridge configuration fixtures
- Skip markers for conditional test execution
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch

    _HAS_TORCH = True
except ModuleNotFoundError:
    _HAS_TORCH = False


def _has_gcp_credentials() -> bool:
    """Check if GCP credentials are available.

    Checks for:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Application default credentials in standard location
    """
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        if creds_path.exists():
            return True

    home = Path.home()
    if os.name == "nt":
        adc_path = (
            Path(os.environ.get("APPDATA", "")) / "gcloud" / "application_default_credentials.json"
        )
    else:
        adc_path = home / ".config" / "gcloud" / "application_default_credentials.json"

    return adc_path.exists()


def _has_gcp_config() -> bool:
    """Check if required GCP configuration is set."""
    return bool(os.environ.get("GCP_PROJECT_ID") and os.environ.get("GCS_BUCKET_NAME"))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_gcp: marks tests requiring GCP credentials (skip if unavailable)"
    )
    config.addinivalue_line(
        "markers", "requires_distributed: marks tests requiring distributed training setup"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_env: marks tests requiring external environment (Docker, etc.)"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow-running")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked with requires_gcp if credentials unavailable."""
    skip_gcp = pytest.mark.skip(
        reason="GCP credentials not available. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'"
    )
    skip_config = pytest.mark.skip(
        reason="GCP configuration incomplete. Set GCP_PROJECT_ID and GCS_BUCKET_NAME"
    )

    for item in items:
        if "requires_gcp" in item.keywords:
            if not _has_gcp_credentials():
                item.add_marker(skip_gcp)
            elif not _has_gcp_config():
                item.add_marker(skip_config)


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for checkpoint testing.

    Properly handles Windows file locking by using TemporaryDirectory.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# DDP / Distributed Training Fixtures
# =============================================================================


@pytest.fixture
def ddp_config():
    """Create test DDPLearnerConfig.

    Returns:
        DDPLearnerConfig with default test values.
    """
    from alpha_zero.distributed.ddp_learner import DDPLearnerConfig

    return DDPLearnerConfig(
        world_size=4,
        rank=0,
        local_rank=0,
        backend="gloo",
        master_addr="localhost",
        master_port="29500",
    )


if _HAS_TORCH:

    @pytest.fixture
    def mock_shared_storage() -> MagicMock:
        """Create mock shared storage for DDP testing.

        Returns:
            MagicMock configured as SharedStorage.
        """
        storage = MagicMock()
        storage.get_weights.return_value = {}
        storage.set_weights.return_value = None
        return storage

    @pytest.fixture
    def mock_replay_buffer() -> MagicMock:
        """Create mock replay buffer for DDP testing.

        Returns:
            MagicMock configured with sample method.
        """
        buffer = MagicMock()
        buffer.sample.return_value = {
            "states": torch.randn(8, 10),
            "policies": torch.randn(8, 5),
            "values": torch.randn(8, 1),
        }
        return buffer


# =============================================================================
# Bridge Fixtures
# =============================================================================


@pytest.fixture
def bridge_config() -> Generator[Path, None, None]:
    """Create bridge configuration for testing.

    Yields:
        BridgeConfig with test-appropriate values.
    """
    from alpha_zero.bridge.config import BridgeConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        yield BridgeConfig(
            comm_dir=Path(tmpdir),
            poll_interval_ms=10,
            command_timeout_ms=500,
            heartbeat_timeout_s=1.0,
            max_retries=3,
            retry_delay_s=0.1,
            board_size=16,
        )
