"""
API Schemas for Civ6 AlphaZero Inference Service.

Request/response schemas using Pydantic for validation and FastAPI integration.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MCTSType(str, Enum):
    """MCTS algorithm variant."""

    GUMBEL = "gumbel"
    BATCH_GUMBEL = "batch_gumbel"
    CIV6 = "civ6"


class GameState(BaseModel):
    """Game state representation for inference request.

    Attributes:
        spatial: Spatial features as flattened array [channels * height * width].
        vector: Vector features [vector_dim].
        action_mask: Valid action mask [action_space_size].
        board_size: Size of the hex board (default 8).
        turn_number: Current turn number (for temperature scheduling).
    """

    spatial: list[float] = Field(..., description="Flattened spatial features (13*8*8=832)")
    vector: list[float] = Field(..., description="Vector features (64 dims)")
    action_mask: list[int] = Field(..., description="Valid action mask (376 actions)")
    board_size: int = Field(default=8, ge=4, le=16)
    turn_number: int = Field(default=0, ge=0)

    @field_validator("spatial")
    @classmethod
    def validate_spatial_size(cls, v: list[float], info: Any) -> list[float]:
        """Validate spatial features match expected dimensions."""
        board_size = info.data.get("board_size", 8)
        expected = 13 * board_size * board_size
        if len(v) != expected:
            raise ValueError(f"Spatial must have {expected} elements, got {len(v)}")
        return v

    @field_validator("vector")
    @classmethod
    def validate_vector_size(cls, v: list[float]) -> list[float]:
        """Validate vector features match expected dimensions."""
        if len(v) != 64:
            raise ValueError(f"Vector must have 64 elements, got {len(v)}")
        return v

    @field_validator("action_mask")
    @classmethod
    def validate_action_mask_size(cls, v: list[int]) -> list[int]:
        """Validate action mask matches action space size."""
        if len(v) != 376:
            raise ValueError(f"Action mask must have 376 elements, got {len(v)}")
        return v


class InferenceRequest(BaseModel):
    """Request for model inference.

    Attributes:
        game_state: Current game state.
        mcts_simulations: Number of MCTS simulations (default 25).
        mcts_type: MCTS algorithm to use (default gumbel).
        temperature: Action selection temperature (default 1.0).
        model_version: Specific model version (None = latest).
        session_id: Optional session ID for caching/tracking.
    """

    game_state: GameState
    mcts_simulations: int = Field(default=25, ge=1, le=800)
    mcts_type: MCTSType = Field(default=MCTSType.GUMBEL)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    model_version: int | None = Field(default=None, description="None = latest champion")
    session_id: str | None = Field(default=None, max_length=128)


class BatchInferenceRequest(BaseModel):
    """Batch inference request for multiple game states.

    Attributes:
        requests: List of individual inference requests.
        max_batch_size: Maximum batch size for parallel processing.
    """

    requests: list[InferenceRequest] = Field(..., min_length=1, max_length=100)
    max_batch_size: int = Field(default=32, ge=1, le=64)


class ActionDistribution(BaseModel):
    """Action probability distribution.

    Attributes:
        action_probs: Probability for each action [action_space_size].
        action_id: Recommended action (argmax of probs).
        value_estimate: Expected value [-1, 1].
        top_k_actions: Top K actions with probabilities.
    """

    action_probs: list[float] = Field(..., description="Full action distribution (376)")
    action_id: int = Field(..., ge=0, lt=376)
    value_estimate: float = Field(..., ge=-1.0, le=1.0)
    top_k_actions: list[tuple[int, float]] = Field(
        default_factory=list, description="Top 5 (action_id, prob) pairs"
    )


class InferenceResponse(BaseModel):
    """Response from model inference.

    Attributes:
        action_distribution: Recommended action distribution.
        model_version: Model version used for inference.
        inference_time_ms: Time taken for inference (milliseconds).
        mcts_time_ms: Time spent in MCTS search.
        session_id: Session ID if provided in request.
        metadata: Additional metadata (num visits, etc.).
    """

    action_distribution: ActionDistribution
    model_version: int
    inference_time_ms: float
    mcts_time_ms: float
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchInferenceResponse(BaseModel):
    """Batch inference response.

    Attributes:
        responses: Individual inference responses.
        total_time_ms: Total batch processing time.
        batch_size: Number of requests processed.
    """

    responses: list[InferenceResponse]
    total_time_ms: float
    batch_size: int


class HealthResponse(BaseModel):
    """Health check response.

    Attributes:
        status: Service status (healthy/degraded/unhealthy).
        model_loaded: Whether model is loaded.
        model_version: Current loaded model version.
        gpu_available: Whether GPU is available.
        memory_usage_mb: Current memory usage.
        uptime_seconds: Service uptime.
    """

    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    model_loaded: bool
    model_version: int | None = None
    gpu_available: bool
    memory_usage_mb: float
    uptime_seconds: float


class MetricsResponse(BaseModel):
    """Service metrics response.

    Attributes:
        total_requests: Total requests served.
        requests_per_second: Average RPS over last minute.
        average_latency_ms: P50 latency.
        p95_latency_ms: P95 latency.
        p99_latency_ms: P99 latency.
        active_sessions: Number of active sessions.
        cache_hit_rate: Model cache hit rate.
    """

    total_requests: int
    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    active_sessions: int
    cache_hit_rate: float


class ModelSwapRequest(BaseModel):
    """Request to load a new model version.

    Attributes:
        model_version: Model version to load.
        force: Force reload even if already loaded.
    """

    model_version: int = Field(..., ge=0)
    force: bool = Field(default=False)


class ModelSwapResponse(BaseModel):
    """Response from model swap operation.

    Attributes:
        success: Whether swap succeeded.
        previous_version: Previous model version.
        new_version: New model version.
        load_time_ms: Time taken to load model.
        message: Status message.
    """

    success: bool
    previous_version: int | None
    new_version: int
    load_time_ms: float
    message: str
