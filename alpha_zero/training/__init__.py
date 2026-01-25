"""Training modules for Civ6 AlphaZero.

This package contains training infrastructure including:
- MuZero trainer with trajectory-based training
- Replay buffers for experience storage
- Training utilities and helpers
"""

from alpha_zero.training.muzero_trainer import (
    GameTrajectory,
    MuZeroTrainer,
    MuZeroTrainingConfig,
    TrajectoryBuffer,
)

__all__ = [
    "GameTrajectory",
    "MuZeroTrainer",
    "MuZeroTrainingConfig",
    "TrajectoryBuffer",
]
