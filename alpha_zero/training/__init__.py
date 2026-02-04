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

# Phase 6: Training Profiler
from .training_profiler import (
    DiminishingReturnsAnalysis,
    DiminishingReturnsLevel,
    TrainingProfiler,
    TrainingProfilerConfig,
    TrainingStepMetrics,
    get_training_profiler,
)

# Phase 6: Elo Plateau Detector
from .elo_plateau_detector import (
    EloPlateauDetector,
    EloPlateauDetectorConfig,
    EloSnapshot,
    EloStatus,
    get_elo_detector,
)

__all__ = [
    # MuZero components
    "GameTrajectory",
    "MuZeroTrainer",
    "MuZeroTrainingConfig",
    "TrajectoryBuffer",
    # Main pipeline
    "CurriculumPipeline",
    "CurriculumPipelineConfig",
    "CurriculumState",
    # Configuration
    "DifficultyStage",
    "EnvironmentTierConfig",
    "TransferLearningConfig",
    # Environment management
    "EnvironmentTier",
    "EnvironmentFactory",
    # Weight transfer
    "WeightTransferManager",
    # Factory functions
    "create_curriculum_pipeline",
    "create_environment_factory",
    # Phase 6: Training Profiler
    "DiminishingReturnsLevel",
    "TrainingStepMetrics",
    "DiminishingReturnsAnalysis",
    "TrainingProfilerConfig",
    "TrainingProfiler",
    "get_training_profiler",
    # Phase 6: Elo Plateau Detector
    "EloStatus",
    "EloSnapshot",
    "EloPlateauDetectorConfig",
    "EloPlateauDetector",
    "get_elo_detector",
]
