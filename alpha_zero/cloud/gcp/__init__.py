"""GCP cloud infrastructure implementation.

Provides GCP-specific implementations of cloud protocols:
- GCSCheckpointManager: Google Cloud Storage checkpoint management
- VertexAITrainer: Vertex AI Custom Training job submission
- PreemptionHandler: GCE metadata-based preemption detection
- CloudLogger: Cloud Logging integration
- CloudMetricsExporter: Cloud Monitoring integration

All implementations follow the force_mock pattern for testing:
- Pass force_mock=True to use mock implementations
- Allows testing without GCP SDK installed
- Maintains consistent API regardless of backend
"""

from __future__ import annotations

import logging

# SDK availability flags
GCS_AVAILABLE = False
VERTEX_AVAILABLE = False
LOGGING_AVAILABLE = False
MONITORING_AVAILABLE = False

try:
    from google.cloud import storage  # noqa: F401  # type: ignore[import-untyped]

    GCS_AVAILABLE = True
except ImportError:
    pass

try:
    from google.cloud import aiplatform  # noqa: F401  # type: ignore[import-untyped]

    VERTEX_AVAILABLE = True
except ImportError:
    pass

try:
    from google.cloud import logging as cloud_logging  # noqa: F401  # type: ignore[import-untyped]

    LOGGING_AVAILABLE = True
except ImportError:
    pass

try:
    from google.cloud import monitoring_v3  # noqa: F401  # type: ignore[import-untyped]

    MONITORING_AVAILABLE = True
except ImportError:
    pass

# Import implementations (these handle missing SDKs internally)
try:
    from alpha_zero.cloud.gcp.monitoring import CloudLogger, CloudMetricsExporter  # noqa: E402
    from alpha_zero.cloud.gcp.preemption import (  # noqa: E402
        PreemptionHandler,
        register_preemption_handler,
    )
    from alpha_zero.cloud.gcp.storage import GCSCheckpointManager  # noqa: E402
    from alpha_zero.cloud.gcp.vertex import VertexAITrainer, VertexJobConfig  # noqa: E402

    _IMPLEMENTATIONS_AVAILABLE = True
except ImportError:
    _IMPLEMENTATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)

__all__ = [
    # Availability flags
    "GCS_AVAILABLE",
    "VERTEX_AVAILABLE",
    "LOGGING_AVAILABLE",
    "MONITORING_AVAILABLE",
]

if _IMPLEMENTATIONS_AVAILABLE:
    __all__.extend([
        # Storage
        "GCSCheckpointManager",
        # Vertex AI
        "VertexAITrainer",
        "VertexJobConfig",
        # Preemption
        "PreemptionHandler",
        "register_preemption_handler",
        # Monitoring
        "CloudLogger",
        "CloudMetricsExporter",
    ])
