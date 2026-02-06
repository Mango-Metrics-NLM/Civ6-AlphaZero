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
    VertexTrainingJob,
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

# Phase 6: Pipeline Caching
from alpha_zero.cloud.gcp.pipeline_cache import (
    CacheEntry,
    ComponentSignature,
    ComponentType,
    PipelineCache,
    PipelineCacheConfig,
    get_pipeline_cache,
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

# Phase 6: Worker Autoscaling
from alpha_zero.cloud.gcp.worker_autoscaler import (
    ScalingAction,
    ScalingDecision,
    WorkerAutoscaler,
    WorkerAutoscalerConfig,
    WorkerMetrics,
    get_worker_autoscaler,
)

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
__all__ = [
    # Config
    "GCPConfig",
    "SelfPlayWorkerConfig",
    "TrainingWorkerConfig",
    "VertexTier",
    # Exceptions
    "GCPError",
    "GCSStorageError",
    "VertexJobError",
    "ConfigurationError",
    "PreemptionError",
    "QuotaExceededError",
    # Storage
    "GCSBackend",
    "GCSModelStorage",
    "GCSTrajectoryStorage",
    "GCSCheckpointStorage",
    "ModelMetadata",
    # Vertex AI Jobs
    "JobSpec",
    "JobResult",
    "JobState",
    "VertexAIClient",
    "VertexSelfPlayJob",
    "VertexTrainingJob",
    # Phase 6: Scaling Configuration
    "BudgetTier",
    "GPUSpec",
    "SelfPlayTierConfig",
    "TrainingTierConfig",
    "EvaluationTierConfig",
    "CycleTierConfig",
    "ScalingTierConfig",
    # Phase 6: Data Lifecycle
    "StorageClass",
    "RetentionPolicyConfig",
    "LifecycleRule",
    "CleanupResult",
    "LifecycleConfig",
    "GCSLifecycleManager",
    "get_lifecycle_manager",
    # Phase 6: Pipeline Caching
    "ComponentType",
    "ComponentSignature",
    "CacheEntry",
    "PipelineCacheConfig",
    "PipelineCache",
    "get_pipeline_cache",
    # Phase 6: Spot Availability
    "AvailabilityLevel",
    "SpotPattern",
    "PreemptionEvent",
    "SpotAvailabilityConfig",
    "SpotAvailabilityTracker",
    "get_spot_tracker",
    # Phase 6: Worker Autoscaling
    "ScalingAction",
    "ScalingDecision",
    "WorkerMetrics",
    "WorkerAutoscalerConfig",
    "WorkerAutoscaler",
    "get_worker_autoscaler",
    # Phase 6: Alert Manager
    "AlertSeverity",
    "AlertChannel",
    "AlertEvent",
    "SlackConfig",
    "EmailConfig",
    "AlertManagerConfig",
    "AlertManager",
    "get_alert_manager",
]
