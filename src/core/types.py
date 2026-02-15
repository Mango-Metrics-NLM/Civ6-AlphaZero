"""Shared types for the Strategic Planning SDK.

This module defines all domain-agnostic types used across the SDK:
enums for categorical choices, constants for algorithm defaults,
lightweight data containers, and type aliases for array types.

No game-specific or domain-specific references should appear here.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PolicyArray = NDArray[np.float32]
"""Probability distribution over actions, shape ``(num_actions,)``."""

ValueEstimate = float
"""Scalar value estimate in ``[-1, 1]``."""

ActionMask = NDArray[np.bool_]
"""Boolean mask of legal actions, shape ``(num_actions,)``."""

SpatialTensor = NDArray[np.float32]
"""Spatial feature tensor, shape ``(channels, height, width)``."""

VectorFeatures = NDArray[np.float32]
"""Flat vector of non-spatial features, shape ``(vector_dim,)``."""

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DeviceType(StrEnum):
    """Hardware device for tensor computation.

    Attributes:
        cpu: CPU computation.
        cuda: NVIDIA GPU via CUDA.
        mps: Apple Silicon GPU via Metal Performance Shaders.
        auto: Automatically select the best available device.
    """

    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"
    auto = "auto"


class SelectionPolicyType(StrEnum):
    """MCTS child-selection policy variant.

    Attributes:
        puct: Polynomial Upper Confidence Trees (standard AlphaZero).
        gumbel: Gumbel AlphaZero selection with sequential halving.
        alphazero: Classic AlphaZero PUCT with Dirichlet noise at root.
    """

    puct = "puct"
    gumbel = "gumbel"
    alphazero = "alphazero"


class BackboneType(StrEnum):
    """Neural network backbone architecture.

    Attributes:
        resnet: Residual convolutional network (AlphaZero default).
        transformer: Vision-transformer-style backbone.
        gnn: Graph neural network for non-grid state spaces.
    """

    resnet = "resnet"
    transformer = "transformer"
    gnn = "gnn"


class ActionSpaceType(StrEnum):
    """How the action space is structured.

    Attributes:
        flat: Single discrete action index.
        hierarchical: Multi-level action decomposition.
    """

    flat = "flat"
    hierarchical = "hierarchical"


class ReplayBufferType(StrEnum):
    """Replay buffer sampling strategy.

    Attributes:
        uniform: Uniform random sampling.
        prioritized: Prioritized experience replay (weighted by TD error).
    """

    uniform = "uniform"
    prioritized = "prioritized"


class GameOutcome(StrEnum):
    """Terminal result of a single game episode.

    Attributes:
        win: The evaluated player won.
        loss: The evaluated player lost.
        draw: The game ended in a draw.
        timeout: The game exceeded the maximum number of steps.
    """

    win = "win"
    loss = "loss"
    draw = "draw"
    timeout = "timeout"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MCTS_INF: Final[float] = float("inf")
"""Positive infinity sentinel used for unvisited MCTS nodes."""

DEFAULT_C_PUCT: Final[float] = 1.5
"""Default PUCT exploration constant."""

MIN_VISIT_COUNT: Final[int] = 1
"""Minimum visit count before a node can be selected."""

DEFAULT_DIRICHLET_ALPHA: Final[float] = 0.3
"""Default Dirichlet noise alpha for root exploration."""

DEFAULT_DIRICHLET_EPSILON: Final[float] = 0.25
"""Default mixing weight for Dirichlet noise at the root."""

SDK_ENV_PREFIX: Final[str] = "SPSDK_"
"""Prefix for all SDK-specific environment variables."""

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObservationSpec:
    """Describes the shape of observations produced by an environment.

    Attributes:
        spatial_shape: Dimensions of the spatial feature tensor,
            e.g. ``(channels, height, width)``.
        vector_dim: Length of the non-spatial feature vector.
        temporal_depth: Number of stacked temporal frames.
            Defaults to ``1`` (single frame, no history).
    """

    spatial_shape: tuple[int, ...]
    vector_dim: int
    temporal_depth: int = 1

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Return the full observation shape including temporal stacking.

        For a spatial shape of ``(C, H, W)`` with ``temporal_depth=T``,
        returns ``(C * T, H, W)``.

        Returns:
            The fully expanded spatial shape.
        """
        if len(self.spatial_shape) == 0:
            return (self.vector_dim * self.temporal_depth,)
        channels, *spatial_dims = self.spatial_shape
        return (channels * self.temporal_depth, *spatial_dims)


@dataclass(frozen=True, slots=True)
class HierarchicalActionSpec:
    """Specification for a hierarchical (multi-level) action space.

    In a hierarchical action space, a single logical action is decomposed
    into multiple levels.  For example: ``(action_type, target_unit, direction)``
    where each level has its own discrete size.

    Attributes:
        num_levels: Number of hierarchy levels.
        level_sizes: Discrete size of each level.
        category_names: Optional human-readable names for each level.
        total_actions: Total number of unique actions across all
            combinations (product of level sizes, or a custom count
            when not all combinations are valid).
    """

    num_levels: int
    level_sizes: tuple[int, ...]
    category_names: tuple[str, ...] = ()
    total_actions: int = 0

    def __post_init__(self) -> None:
        """Validate invariants after initialisation."""
        if len(self.level_sizes) != self.num_levels:
            raise ValueError(
                f"level_sizes length ({len(self.level_sizes)}) "
                f"must equal num_levels ({self.num_levels})"
            )
        if self.category_names and len(self.category_names) != self.num_levels:
            raise ValueError(
                f"category_names length ({len(self.category_names)}) "
                f"must equal num_levels ({self.num_levels})"
            )


@dataclass(frozen=True, slots=True)
class HierarchicalAction:
    """A concrete action selected within a hierarchical action space.

    Each entry in ``levels`` is the chosen index at the corresponding
    hierarchy level.

    Attributes:
        levels: Selected index at each hierarchy level.
    """

    levels: tuple[int, ...]


@dataclass(slots=True)
class TrainingExample:
    """A single training sample collected during self-play.

    Attributes:
        spatial: Spatial observation features.
        vector: Non-spatial (vector) observation features.
        action_mask: Boolean mask of legal actions.
        policy: Target policy distribution from MCTS.
        value: Target value label (game outcome from this player's
            perspective).
    """

    spatial: NDArray[np.float32]
    vector: NDArray[np.float32]
    action_mask: NDArray[np.bool_]
    policy: NDArray[np.float32]
    value: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise this example to a plain dictionary.

        Numpy arrays are converted to lists so the result is
        JSON-serialisable.

        Returns:
            Dictionary representation of the training example.
        """
        return {
            "spatial": self.spatial.tolist(),
            "vector": self.vector.tolist(),
            "action_mask": self.action_mask.tolist(),
            "policy": self.policy.tolist(),
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingExample:
        """Deserialise a training example from a dictionary.

        Args:
            data: Dictionary previously produced by ``to_dict()``.

        Returns:
            Reconstructed ``TrainingExample``.
        """
        return cls(
            spatial=np.asarray(data["spatial"], dtype=np.float32),
            vector=np.asarray(data["vector"], dtype=np.float32),
            action_mask=np.asarray(data["action_mask"], dtype=np.bool_),
            policy=np.asarray(data["policy"], dtype=np.float32),
            value=float(data["value"]),
        )


@dataclass(slots=True)
class GameTrajectory:
    """A complete game episode of training examples.

    Attributes:
        examples: Ordered list of training examples from the episode.
        outcome: Terminal outcome of the game.
        episode_id: Unique identifier for this episode.
        iteration: Training iteration when this episode was generated.
        metadata: Arbitrary key-value metadata (e.g. opponent id,
            environment settings).
    """

    examples: list[TrainingExample]
    outcome: GameOutcome
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Output of a single MCTS search from a root state.

    Attributes:
        action: Best action selected by the search.
        policy: Improved policy distribution over all actions.
        root_value: Value estimate at the root node.
        visit_counts: Raw visit counts for each action.
        num_simulations: Total number of simulations performed.
    """

    action: int
    policy: NDArray[np.float32]
    root_value: float
    visit_counts: NDArray[np.float32]
    num_simulations: int


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregated evaluation result over a set of games.

    Attributes:
        win_rate: Fraction of games won in ``[0, 1]``.
        draw_rate: Fraction of games drawn in ``[0, 1]``.
        elo_rating: Estimated Elo rating.
        num_games: Total number of games played.
    """

    win_rate: float
    draw_rate: float
    elo_rating: float
    num_games: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "DEFAULT_C_PUCT",
    "DEFAULT_DIRICHLET_ALPHA",
    "DEFAULT_DIRICHLET_EPSILON",
    # Constants
    "MCTS_INF",
    "MIN_VISIT_COUNT",
    "SDK_ENV_PREFIX",
    "ActionMask",
    "ActionSpaceType",
    "BackboneType",
    # Enums
    "DeviceType",
    "EvalResult",
    "GameOutcome",
    "GameTrajectory",
    "HierarchicalAction",
    "HierarchicalActionSpec",
    # Data containers
    "ObservationSpec",
    # Type aliases
    "PolicyArray",
    "ReplayBufferType",
    "SearchResult",
    "SelectionPolicyType",
    "SpatialTensor",
    "TrainingExample",
    "ValueEstimate",
    "VectorFeatures",
]
