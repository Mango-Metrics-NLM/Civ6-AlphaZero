"""Shared pytest fixtures and configuration for all tests.

Provides:
- GCP credential detection fixtures
- Cloud configuration fixtures for testing
- DDP mock fixtures for distributed training tests
- Skip markers for conditional test execution
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from tests.fixtures.mock_distributed import MockDistributedBackend


def _has_gcp_credentials() -> bool:
    """Check if GCP credentials are available.

    Checks for:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable
    2. Application default credentials in standard location
    3. GCP_PROJECT_ID set (indicates intent to use GCP)
    """
    # Check explicit credentials file
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds_path = Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
        if creds_path.exists():
            return True

    # Check for application default credentials
    # Windows: %APPDATA%\gcloud\application_default_credentials.json
    # Linux/Mac: ~/.config/gcloud/application_default_credentials.json
    home = Path.home()
    if os.name == "nt":  # Windows
        adc_path = (
            Path(os.environ.get("APPDATA", "")) / "gcloud" / "application_default_credentials.json"
        )
    else:  # Linux/Mac
        adc_path = home / ".config" / "gcloud" / "application_default_credentials.json"

    return adc_path.exists()


def _has_gcp_config() -> bool:
    """Check if required GCP configuration is set."""
    return bool(os.environ.get("GCP_PROJECT_ID") and os.environ.get("GCS_BUCKET_NAME"))


# Pytest markers for conditional execution
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


# Fixtures for cloud testing
@pytest.fixture
def gcp_credentials_available() -> bool:
    """Check if GCP credentials are available.

    Returns:
        True if credentials are available, False otherwise.
    """
    return _has_gcp_credentials() and _has_gcp_config()


@pytest.fixture
def temp_checkpoint_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for checkpoint testing.

    Properly handles Windows file locking by using TemporaryDirectory.

    Yields:
        Path to temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_cloud_config():
    """Create a test cloud configuration.

    Uses mock values that work with force_mock=True for offline testing.

    Returns:
        CloudConfig instance for testing.
    """
    from alpha_zero.cloud.config import CloudConfig

    return CloudConfig(
        project_id="test-project-id",
        region="us-central1",
        zone="us-central1-a",
        bucket_name="test-bucket",
        checkpoint_prefix="test-checkpoints/",
        artifact_prefix="test-artifacts/",
        enable_cloud_logging=True,
        enable_cloud_monitoring=True,
        use_spot=True,
        preemption_check_interval=1.0,  # Faster for tests
    )


@pytest.fixture
def env_cloud_config():
    """Load cloud configuration from environment.

    Only use this fixture for E2E tests with real credentials.

    Returns:
        CloudConfig instance from environment, or skip if not configured.
    """
    from alpha_zero.cloud.config import CloudConfig

    if not _has_gcp_config():
        pytest.skip("GCP environment not configured")

    config = CloudConfig.from_env()
    config.validate()
    return config


# =============================================================================
# DDP / Distributed Training Fixtures
# =============================================================================


@pytest.fixture
def mock_distributed() -> Generator[MockDistributedBackend, None, None]:
    """Create and activate mock distributed backend.

    Patches torch.distributed with mock implementation.

    Yields:
        MockDistributedBackend instance with patches applied.
    """
    from tests.fixtures.mock_distributed import MockDistributedBackend

    backend = MockDistributedBackend(world_size=4, rank=0)
    with backend.patch():
        yield backend


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


class SimpleTestNetwork(nn.Module):
    """Simple network for testing DDP learners."""

    def __init__(self, input_size: int = 10, output_size: int = 5) -> None:
        """Initialize simple network."""
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass."""
        out = self.linear(x)
        return {
            "policy": torch.softmax(out, dim=-1),
            "value": out.mean(dim=-1, keepdim=True),
        }


@pytest.fixture
def simple_test_network() -> SimpleTestNetwork:
    """Create simple network for DDP testing.

    Returns:
        SimpleTestNetwork instance.
    """
    return SimpleTestNetwork()


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
# Bridge / Docker Fixtures
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


@pytest.fixture
def docker_connection_config():
    """Create Docker connection configuration for testing.

    Returns:
        DockerConnectionConfig with test-appropriate values.
    """
    from alpha_zero.envs.docker_config import DockerConnectionConfig

    return DockerConnectionConfig(
        host="localhost",
        port=6300,
        connection_timeout_s=5.0,
        retry_count=3,
        retry_delay_s=0.5,
    )


# =============================================================================
# Game State Fixtures
# =============================================================================


@pytest.fixture
def sample_game_state_dict() -> dict:
    """Create sample game state dictionary for testing.

    Returns:
        Dictionary representing a minimal game state.
    """
    return {
        "msg_id": 1,
        "turn_id": 10,
        "player_id": 0,
        "era": 0,
        "map": {
            "width": 84,
            "height": 54,
            "tiles": [],
        },
        "units": [
            {
                "id": 1001,
                "type": "UNIT_WARRIOR",
                "owner": 0,
                "x": 42,
                "y": 27,
                "health": 100,
                "max_health": 100,
                "moves": 2,
                "max_moves": 2,
                "combat_strength": 20,
            },
        ],
        "cities": [
            {
                "id": 2001,
                "name": "Capital",
                "owner": 0,
                "x": 40,
                "y": 25,
                "population": 5,
                "is_capital": True,
            }
        ],
        "players": [
            {
                "id": 0,
                "civilization": "CIVILIZATION_AMERICA",
                "gold": 100,
                "science_per_turn": 10,
                "culture_per_turn": 5,
                "faith": 0,
                "score": 50,
                "techs_researched": ["TECH_POTTERY", "TECH_MINING"],
            }
        ],
        "legal_actions": [
            {"action_type": "unit_move", "unit_id": 1001, "targets": [{"x": 43, "y": 27}]},
            {"action_type": "end_turn"},
        ],
    }


@pytest.fixture
def sample_trajectory_data() -> list[dict]:
    """Create sample trajectory data for testing.

    Returns:
        List of trajectory step dictionaries.
    """
    import numpy as np

    return [
        {
            "state": np.random.randn(13, 16, 16).astype(
                np.float32
            ),  # MicroCivEncoder.TOTAL_CHANNELS
            "action": np.array([5, 8]),
            "policy": np.random.dirichlet(np.ones(376)).astype(np.float32),
            "reward": 0.0,
            "done": False,
            "turn": 1,
        },
        {
            "state": np.random.randn(13, 16, 16).astype(
                np.float32
            ),  # MicroCivEncoder.TOTAL_CHANNELS
            "action": np.array([6, 8]),
            "policy": np.random.dirichlet(np.ones(376)).astype(np.float32),
            "reward": 0.1,
            "done": False,
            "turn": 2,
        },
        {
            "state": np.random.randn(13, 16, 16).astype(
                np.float32
            ),  # MicroCivEncoder.TOTAL_CHANNELS
            "action": np.array([6, 9]),
            "policy": np.random.dirichlet(np.ones(376)).astype(np.float32),
            "reward": 1.0,
            "done": True,
            "turn": 3,
        },
    ]
