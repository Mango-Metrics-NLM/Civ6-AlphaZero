"""Cloud infrastructure module for distributed AlphaZero training.

Provides:
- CloudConfig: Configuration for cloud backends (GCP, AWS, etc.)
- Storage protocols for checkpoint management
- Monitoring and logging integration
- Preemption handling for spot/preemptible instances

All cloud SDK dependencies are optional. Use availability flags to check
if a specific cloud provider is available.
"""

from __future__ import annotations

from alpha_zero.cloud.config import CloudConfig
from alpha_zero.cloud.exceptions import (
    CloudConfigError,
    CloudConnectionError,
    CloudStorageError,
    PreemptionError,
)
from alpha_zero.cloud.protocols import (
    CloudMonitoringProtocol,
    CloudStorageProtocol,
    PreemptionHandlerProtocol,
)

__all__ = [
    # Configuration
    "CloudConfig",
    # Protocols
    "CloudStorageProtocol",
    "CloudMonitoringProtocol",
    "PreemptionHandlerProtocol",
    # Exceptions
    "CloudConfigError",
    "CloudStorageError",
    "CloudConnectionError",
    "PreemptionError",
]
