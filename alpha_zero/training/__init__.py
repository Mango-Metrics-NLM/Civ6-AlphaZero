"""Training infrastructure for Civ6-AlphaZero.

This module provides training utilities including:
- Multi-environment curriculum learning pipeline
- Transfer learning between environments
- Training orchestration and scheduling
"""

from __future__ import annotations

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
