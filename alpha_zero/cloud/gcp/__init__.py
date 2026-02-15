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
from alpha_zero.cloud.gcp.lifecycle import (
    CleanupResult,
    GCSLifecycleManager,
    LifecycleConfig,
    LifecycleRule,
    RetentionPolicyConfig,
    StorageClass,
    get_lifecycle_manager,
)

# Model version selection utilities
from alpha_zero.cloud.gcp.model_selection import subsample_model_versions

# Phase 6: Pipeline Caching
from alpha_zero.cloud.gcp.pipeline_cache import (
    CacheEntry,
    ComponentSignature,
    ComponentType,
    PipelineCache,
    PipelineCacheConfig,
    get_pipeline_cache,
)

# Phase 6: Scaling Configuration
from alpha_zero.cloud.gcp.scaling_config import (
    BudgetTier,
    CycleTierConfig,
    EvaluationTierConfig,
    GPUSpec,
    ScalingTierConfig,
    SelfPlayTierConfig,
    TrainingTierConfig,
)

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
from alpha_zero.cloud.gcp.worker_autoscaler import (
    ScalingAction,
    ScalingDecision,
    WorkerAutoscaler,
    WorkerAutoscalerConfig,
    WorkerMetrics,
    get_worker_autoscaler,
)

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
    "BudgetTier",
    "CacheEntry",
    "CleanupResult",
    "ComponentSignature",
    # Phase 6: Pipeline Caching
    "ComponentType",
    "ConfigurationError",
    "CycleTierConfig",
    "EmailConfig",
    "EvaluationTierConfig",
    # Config
    "GCPConfig",
    # Exceptions
    "GCPError",
    # Storage
    "GCSBackend",
    "GCSCheckpointStorage",
    "GCSLifecycleManager",
    "GCSModelStorage",
    "GCSStorageError",
    "GCSTrajectoryStorage",
    "GPUSpec",
    "JobResult",
    # Vertex AI Jobs
    "JobSpec",
    "JobState",
    "LifecycleConfig",
    "LifecycleRule",
    "ModelMetadata",
    "PipelineCache",
    "PipelineCacheConfig",
    "PreemptionError",
    "PreemptionEvent",
    "QuotaExceededError",
    "RetentionPolicyConfig",
    # Phase 6: Worker Autoscaling
    "ScalingAction",
    "ScalingDecision",
    "ScalingTierConfig",
    "SelfPlayTierConfig",
    "SelfPlayWorkerConfig",
    "SlackConfig",
    "SpotAvailabilityConfig",
    "SpotAvailabilityTracker",
    "SpotPattern",
    # Phase 6: Data Lifecycle
    "StorageClass",
    "TrainingTierConfig",
    "TrainingWorkerConfig",
    "VertexAIClient",
    "VertexJobError",
    "VertexSelfPlayJob",
    "VertexTier",
    "VertexTournamentJob",
    "VertexTrainingJob",
    "WorkerAutoscaler",
    "WorkerAutoscalerConfig",
    "WorkerMetrics",
    "get_alert_manager",
    "get_lifecycle_manager",
    "get_pipeline_cache",
    "get_spot_tracker",
    "get_worker_autoscaler",
    "subsample_model_versions",
]
