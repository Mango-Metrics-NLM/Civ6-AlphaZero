"""Training modules for Civ6 AlphaZero.

This package contains training infrastructure including:
- MuZero trainer with trajectory-based training
- Curriculum learning and transfer learning
- Replay buffers for experience storage
- Training profiling for diminishing returns detection
- Elo plateau detection for training stagnation
"""

from alpha_zero.training.muzero_trainer import (
    GameTrajectory,
    MuZeroTrainer,
    MuZeroTrainingConfig,
    TrajectoryBuffer,
)

from .curriculum_pipeline import (
    CurriculumPipeline,
    CurriculumPipelineConfig,
    CurriculumState,
    DifficultyStage,
    EnvironmentFactory,
    EnvironmentTier,
    EnvironmentTierConfig,
    TransferLearningConfig,
    WeightTransferManager,
    create_curriculum_pipeline,
    create_environment_factory,
)

# Phase 6: Elo Plateau Detector
from .elo_plateau_detector import (
    EloPlateauDetector,
    EloPlateauDetectorConfig,
    EloSnapshot,
    EloStatus,
    get_elo_detector,
)

# Phase 9: MuZero Integration
from .muzero_integration import (
    MuZeroIntegrationConfig,
    MuZeroTrainingBridge,
)

# Phase 6: Training Profiler
from .training_profiler import (
    DiminishingReturnsAnalysis,
    DiminishingReturnsLevel,
    TrainingProfiler,
    TrainingProfilerConfig,
    TrainingStepMetrics,
    get_training_profiler,
)

# Phase 8: Training Safeguards & Reports
from .training_report import (
    TrainingReport,
    TrainingReportConfig,
    TrainingReportData,
)
from .training_safeguards import (
    CheckpointValidation,
    GradientStatus,
    Intervention,
    InterventionAction,
    SafeguardReport,
    TrainingSafeguard,
    TrainingSafeguardConfig,
)

__all__ = [
    # Phase 8: Training Safeguards
    "CheckpointValidation",
    # Main pipeline
    "CurriculumPipeline",
    "CurriculumPipelineConfig",
    "CurriculumState",
    # Configuration
    "DifficultyStage",
    "DiminishingReturnsAnalysis",
    # Phase 6: Training Profiler
    "DiminishingReturnsLevel",
    "EloPlateauDetector",
    "EloPlateauDetectorConfig",
    "EloSnapshot",
    # Phase 6: Elo Plateau Detector
    "EloStatus",
    "EnvironmentFactory",
    # Environment management
    "EnvironmentTier",
    "EnvironmentTierConfig",
    # MuZero components
    "GameTrajectory",
    "GradientStatus",
    "Intervention",
    "InterventionAction",
    "MuZeroIntegrationConfig",
    "MuZeroTrainer",
    "MuZeroTrainingBridge",
    "MuZeroTrainingConfig",
    "SafeguardReport",
    "TrainingProfiler",
    "TrainingProfilerConfig",
    "TrainingReport",
    "TrainingReportConfig",
    "TrainingReportData",
    "TrainingSafeguard",
    "TrainingSafeguardConfig",
    "TrainingStepMetrics",
    "TrajectoryBuffer",
    "TransferLearningConfig",
    # Weight transfer
    "WeightTransferManager",
    # Factory functions
    "create_curriculum_pipeline",
    "create_environment_factory",
    "get_elo_detector",
    "get_training_profiler",
]
