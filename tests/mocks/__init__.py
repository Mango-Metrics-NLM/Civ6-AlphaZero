"""Test mocks for cloud and distributed components."""

from __future__ import annotations

from tests.mocks.cloud_mocks import (
    MockAIPlatform,
    MockBucket,
    MockGCSBlob,
    MockGCSClient,
)
from tests.mocks.distributed_mocks import (
    MockDDP,
    MockDistributedEnvironment,
    MockGradScaler,
    MockProcessGroup,
    MockProcessGroupConfig,
    MockTensorboardWriter,
    create_mock_network,
    patch_distributed_for_testing,
)

__all__ = [
    # Cloud mocks
    "MockAIPlatform",
    "MockBucket",
    # Distributed mocks
    "MockDDP",
    "MockDistributedEnvironment",
    "MockGCSBlob",
    "MockGCSClient",
    "MockGradScaler",
    "MockProcessGroup",
    "MockProcessGroupConfig",
    "MockTensorboardWriter",
    "create_mock_network",
    "patch_distributed_for_testing",
]
