"""Distributed training components for Civ6 AlphaZero.

This module provides:
- SharedStorage: Process-safe model weight sharing
- DistributedReplayBuffer: Process-safe replay buffer
- SelfPlayActor: Self-play worker for generating training data
- Learner: Central network training component
- TrainingCoordinator: Orchestrates the distributed training process

Usage:
    # Quick start with coordinator
    from alpha_zero.distributed import TrainingCoordinator, CoordinatorConfig

    config = CoordinatorConfig(
        num_actors=4,
        batch_size=128,
        max_games=1000,
    )
    coordinator = TrainingCoordinator(config)
    coordinator.run()

    # Or run as script:
    # python -m alpha_zero.distributed.coordinator --num_actors=4 --dry_run
"""

from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
    TrainingExample,
    WeightSynchronizer,
)
from alpha_zero.distributed.actor import (
    ActorConfig,
    ActorPool,
    SelfPlayActor,
    actor_process,
)
from alpha_zero.distributed.learner import (
    GradientAccumulator,
    Learner,
    LearnerConfig,
    LearningRateScheduler,
    learner_process,
)
from alpha_zero.distributed.coordinator import (
    CoordinatorConfig,
    TrainingCoordinator,
)

__all__ = [
    # Shared storage
    "SharedStorage",
    "SharedStorageConfig",
    "DistributedReplayBuffer",
    "ReplayBufferConfig",
    "TrainingExample",
    "GameTrajectory",
    "WeightSynchronizer",
    # Actor
    "SelfPlayActor",
    "ActorConfig",
    "ActorPool",
    "actor_process",
    # Learner
    "Learner",
    "LearnerConfig",
    "LearningRateScheduler",
    "GradientAccumulator",
    "learner_process",
    # Coordinator
    "TrainingCoordinator",
    "CoordinatorConfig",
]
