"""
Civ6 AlphaZero Inference Service.

Production-ready ML inference for real-time MCTS-based gameplay.
"""

from alpha_zero.inference.api_schemas import (
    ActionDistribution,
    GameState,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    MCTSType,
    MetricsResponse,
    ModelSwapRequest,
    ModelSwapResponse,
)
from alpha_zero.inference.client import InferenceClient, create_random_game_state
from alpha_zero.inference.inference_engine import (
    InferenceConfig,
    InferenceEngine,
    ModelCache,
)

__all__ = [
    "ActionDistribution",
    # API Schemas
    "GameState",
    "HealthResponse",
    # Client
    "InferenceClient",
    "InferenceConfig",
    # Inference Engine
    "InferenceEngine",
    "InferenceRequest",
    "InferenceResponse",
    "MCTSType",
    "MetricsResponse",
    "ModelCache",
    "ModelSwapRequest",
    "ModelSwapResponse",
    "create_random_game_state",
]
