"""Training modules for Civ6 AlphaZero.

This package contains training infrastructure including:
- MuZero trainer with trajectory-based training
- Curriculum learning and transfer learning
- Replay buffers for experience storage
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
]
