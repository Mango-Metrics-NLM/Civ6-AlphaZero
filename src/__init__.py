"""Strategic Planning SDK -- domain-agnostic AlphaZero framework.

Provides pluggable MCTS search, neural network training, and
environment protocols for strategic planning across domains.
"""
from src.core.config import SDKConfig, load_config
from src.core.types import (
    DeviceType,
    GameOutcome,
    GameTrajectory,
    ObservationSpec,
    TrainingExample,
)

__all__ = [
    "DeviceType",
    "GameOutcome",
    "GameTrajectory",
    "ObservationSpec",
    "SDKConfig",
    "TrainingExample",
    "load_config",
]
