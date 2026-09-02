"""
GCP Cloud Integration Module for Civ6 AlphaZero

Provides cloud infrastructure components for distributed training:
- GCS storage for models, trajectories, and checkpoints
- Vertex AI job management for self-play and training
- Configuration with environment variable support
- Cost optimization (pipeline caching, spot availability, worker autoscaling)
- Alerting (Slack, email, W&B)
- Data lifecycle management

Usage:
    from alpha_zero.cloud.gcp import GCPConfig, GCSModelStorage

    config = GCPConfig.from_env()
    storage = GCSModelStorage(config)
    storage.save_model(model, version=1, metadata={"elo": 1000})
"""

# Phase 6: Alert Manager
from alpha_zero.cloud.gcp.alert_manager import (
    AlertChannel,
    AlertEvent,
    AlertManager,
    AlertManagerConfig,
    AlertSeverity,
    EmailConfig,
    SlackConfig,
    get_alert_manager,
)
from alpha_zero.cloud.gcp.config import (
    GCPConfig,
    SelfPlayWorkerConfig,
    TrainingWorkerConfig,
    VertexTier,
)
from alpha_zero.cloud.gcp.exceptions import (
    ConfigurationError,
    GCPError,
    GCSStorageError,
    PreemptionError,
    QuotaExceededError,
    VertexJobError,
)

# Phase 6: Data Lifecycle
# Model version selection utilities
from alpha_zero.cloud.gcp.model_selection import subsample_model_versions

# Phase 6: Pipeline Caching
# Phase 6: Scaling Configuration
# Phase 6: Spot Availability
from alpha_zero.cloud.gcp.spot_availability import (
    AvailabilityLevel,
    PreemptionEvent,
    SpotAvailabilityConfig,
    SpotAvailabilityTracker,
    SpotPattern,
    get_spot_tracker,
)
from alpha_zero.cloud.gcp.storage import (
    GCSBackend,
    GCSCheckpointStorage,
    GCSModelStorage,
    GCSTrajectoryStorage,
    ModelMetadata,
)
from alpha_zero.cloud.gcp.vertex_jobs import (
    JobResult,
    JobSpec,
    JobState,
    VertexAIClient,
    VertexSelfPlayJob,
    VertexTournamentJob,
    VertexTrainingJob,
)

# Phase 6: Worker Autoscaling

__all__ = [
    "AlertChannel",
    "AlertEvent",
    "AlertManager",
    "AlertManagerConfig",
    # Phase 6: Alert Manager
    "AlertSeverity",
    # Phase 6: Spot Availability
    "AvailabilityLevel",
    # Phase 6: Scaling Configuration
    # Phase 6: Pipeline Caching
    "ConfigurationError",
    "EmailConfig",
    # Config
    "GCPConfig",
    # Exceptions
    "GCPError",
    # Storage
    "GCSBackend",
    "GCSCheckpointStorage",
    "GCSModelStorage",
    "GCSStorageError",
    "GCSTrajectoryStorage",
    "JobResult",
    # Vertex AI Jobs
    "JobSpec",
    "JobState",
    "ModelMetadata",
    "PreemptionError",
    "PreemptionEvent",
    "QuotaExceededError",
    # Phase 6: Worker Autoscaling
    "SelfPlayWorkerConfig",
    "SlackConfig",
    "SpotAvailabilityConfig",
    "SpotAvailabilityTracker",
    "SpotPattern",
    # Phase 6: Data Lifecycle
    "TrainingWorkerConfig",
    "VertexAIClient",
    "VertexJobError",
    "VertexSelfPlayJob",
    "VertexTier",
    "VertexTournamentJob",
    "VertexTrainingJob",
    "get_alert_manager",
    "get_spot_tracker",
    "subsample_model_versions",
]
