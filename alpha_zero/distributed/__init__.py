"""Distributed training components for Civ6 AlphaZero.

This module provides:
- SharedStorage: Process-safe model weight sharing
- DistributedReplayBuffer: Process-safe replay buffer
- SelfPlayActor: Self-play worker for generating training data
- Learner: Central network training component
- DDPLearner: Distributed Data Parallel learner for multi-GPU training
- ParameterServer: Central weight management for multi-node training
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

    # Multi-GPU training with DDP
    from alpha_zero.distributed import DDPLearner, DDPConfig

    ddp_config = DDPConfig.from_env()  # Detect from environment
    learner = create_ddp_learner(learner_config, ddp_config)
    learner.run()

    # Or run as script:
    # python -m alpha_zero.distributed.coordinator --num_actors=4 --dry_run
"""

from alpha_zero.distributed.actor import (
    ActorConfig,
    ActorPool,
    SelfPlayActor,
    actor_process,
)
from alpha_zero.distributed.coordinator import (
    CoordinatorConfig,
    TrainingCoordinator,
)
from alpha_zero.distributed.ddp_config import (
    DDPBackend,
    DDPConfig,
    detect_ddp_environment,
    setup_ddp_environment,
)
from alpha_zero.distributed.ddp_learner import (
    DDPLearner,
    DDPLearnerConfig,
    create_ddp_learner,
)
from alpha_zero.distributed.learner import (
    GradientAccumulator,
    Learner,
    LearnerConfig,
    LearningRateScheduler,
    learner_process,
)
from alpha_zero.distributed.parameter_server import (
    DistributedParameterServer,
    MockParameterServer,
    ParameterServer,
    ParameterServerConfig,
    WeightUpdateMode,
    create_parameter_server,
)
from alpha_zero.distributed.shared_storage import (
    DistributedReplayBuffer,
    GameTrajectory,
    ReplayBufferConfig,
    SharedStorage,
    SharedStorageConfig,
    TrainingExample,
    WeightSynchronizer,
)

__all__ = [
    # Actor
    "ActorConfig",
    "ActorPool",
    # Coordinator
    "CoordinatorConfig",
    # DDP Learner
    "DDPBackend",
    "DDPConfig",
    "DDPLearner",
    "DDPLearnerConfig",
    # Parameter Server
    "DistributedParameterServer",
    # Shared storage
    "DistributedReplayBuffer",
    "GameTrajectory",
    # Learner
    "GradientAccumulator",
    "Learner",
    "LearnerConfig",
    "LearningRateScheduler",
    "MockParameterServer",
    "ParameterServer",
    "ParameterServerConfig",
    "ReplayBufferConfig",
    "SelfPlayActor",
    "SharedStorage",
    "SharedStorageConfig",
    "TrainingCoordinator",
    "TrainingExample",
    "WeightSynchronizer",
    "WeightUpdateMode",
    "actor_process",
    "create_ddp_learner",
    "create_parameter_server",
    "detect_ddp_environment",
    "learner_process",
    "setup_ddp_environment",
]
