"""Core protocol definitions for the Strategic Planning SDK.

This module defines the foundational protocols (structural typing interfaces)
that all SDK components use. Domain experts implement StrategicEnvironment
to plug their domain into the planning engine.

Protocols use Python's structural subtyping - any class implementing the
required methods is compatible without explicit inheritance.

Example:
    >>> class MyDomainEnv:
    ...     # Just implement the required methods
    ...     @property
    ...     def num_objectives(self) -> int: return 3
    ...     def reset(self, **kwargs): ...
    ...     def step(self, action): ...
    ...     # etc.
    >>>
    >>> isinstance(MyDomainEnv(), StrategicEnvironment)  # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StateSpec:
    """Specification for environment state tensors.

    Describes the shape and semantics of the observation tensors
    that the environment produces. Used by the network to auto-configure
    its input layers.

    Attributes:
        spatial_shape: Shape of spatial observations (C, H, W) or None if non-spatial.
        vector_dim: Dimension of the vector observation.
        action_space_size: Total number of discrete actions.
        spatial_channel_names: Optional names for spatial channels.
        vector_feature_names: Optional names for vector features.
    """

    spatial_shape: tuple[int, ...] | None = None
    vector_dim: int = 64
    action_space_size: int = 100
    spatial_channel_names: list[str] = field(default_factory=list)
    vector_feature_names: list[str] = field(default_factory=list)

    @property
    def has_spatial(self) -> bool:
        """Whether the environment has spatial observations."""
        return self.spatial_shape is not None and len(self.spatial_shape) == 3

    @property
    def spatial_channels(self) -> int:
        """Number of spatial channels, or 0 if non-spatial."""
        if self.spatial_shape is not None and len(self.spatial_shape) >= 1:
            return self.spatial_shape[0]
        return 0

    @property
    def spatial_size(self) -> tuple[int, int]:
        """Spatial dimensions (H, W), or (0, 0) if non-spatial."""
        if self.spatial_shape is not None and len(self.spatial_shape) == 3:
            return (self.spatial_shape[1], self.spatial_shape[2])
        return (0, 0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "spatial_shape": list(self.spatial_shape) if self.spatial_shape else None,
            "vector_dim": self.vector_dim,
            "action_space_size": self.action_space_size,
            "spatial_channel_names": self.spatial_channel_names,
            "vector_feature_names": self.vector_feature_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateSpec:
        """Deserialize from dictionary."""
        spatial = data.get("spatial_shape")
        return cls(
            spatial_shape=tuple(spatial) if spatial else None,
            vector_dim=int(data.get("vector_dim", 64)),
            action_space_size=int(data.get("action_space_size", 100)),
            spatial_channel_names=data.get("spatial_channel_names", []),
            vector_feature_names=data.get("vector_feature_names", []),
        )


@dataclass
class ActionSpec:
    """Specification for the action space.

    Attributes:
        total_size: Total number of discrete actions.
        hierarchy_levels: Names of hierarchy levels (if hierarchical).
        level_sizes: Size of each hierarchy level.
        action_names: Optional mapping from action index to human-readable name.
    """

    total_size: int = 100
    hierarchy_levels: list[str] = field(default_factory=list)
    level_sizes: list[int] = field(default_factory=list)
    action_names: dict[int, str] = field(default_factory=dict)

    @property
    def is_hierarchical(self) -> bool:
        """Whether the action space is hierarchical."""
        return len(self.hierarchy_levels) > 1

    def get_action_name(self, action_idx: int) -> str:
        """Get human-readable name for an action index."""
        return self.action_names.get(action_idx, f"action_{action_idx}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "total_size": self.total_size,
            "hierarchy_levels": self.hierarchy_levels,
            "level_sizes": self.level_sizes,
            "action_names": {str(k): v for k, v in self.action_names.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ActionSpec:
        """Deserialize from dictionary."""
        action_names = {int(k): v for k, v in data.get("action_names", {}).items()}
        return cls(
            total_size=int(data.get("total_size", 100)),
            hierarchy_levels=data.get("hierarchy_levels", []),
            level_sizes=data.get("level_sizes", []),
            action_names=action_names,
        )


@dataclass
class StrategicObservation:
    """Observation from a strategic environment.

    Extends the base observation with multi-objective and multi-agent support.

    Attributes:
        spatial: Spatial observation tensor (C, H, W) or None.
        vector: Vector observation (global features).
        action_mask: Boolean mask of valid actions.
        agent_id: Index of the observing agent.
        step_number: Current step in the episode.
        done: Whether the episode is over.
        info: Additional domain-specific information.
    """

    spatial: np.ndarray | None
    vector: np.ndarray
    action_mask: np.ndarray
    agent_id: int = 0
    step_number: int = 0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "spatial": self.spatial.tolist() if self.spatial is not None else None,
            "vector": self.vector.tolist(),
            "action_mask": self.action_mask.tolist(),
            "agent_id": self.agent_id,
            "step_number": self.step_number,
            "done": self.done,
            "info": self.info,
        }


@dataclass
class StrategicStepResult:
    """Result of taking an action in a strategic environment.

    Attributes:
        observation: New observation after action.
        rewards: Per-objective reward vector.
        done: Whether the episode is over.
        truncated: Whether the episode was truncated (time limit).
        info: Additional domain-specific information.
    """

    observation: StrategicObservation
    rewards: np.ndarray  # Shape: (num_objectives,)
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StrategicEnvironment(Protocol):
    """Protocol for strategic planning environments.

    Domain experts implement this protocol to plug their domain into
    the planning engine. The environment defines the state space,
    action space, objectives, and transition dynamics.

    Key differences from standard RL environments:
    - Multi-objective: rewards is a vector, not a scalar
    - Multi-agent: multiple agents with different perspectives
    - Scenario-aware: can be parameterized with uncertainty scenarios
    - Cloneable: MCTS requires cloning for tree search

    Example:
        >>> class ClimateEnv:
        ...     @property
        ...     def num_objectives(self) -> int:
        ...         return 3  # warming, GDP, equity
        ...
        ...     @property
        ...     def objective_names(self) -> list[str]:
        ...         return ["minimize_warming", "maximize_gdp", "maximize_equity"]
        ...
        ...     def reset(self, seed=None, scenario=None):
        ...         # Initialize climate-economy model
        ...         return StrategicObservation(...)
        ...
        ...     def step(self, action):
        ...         # Apply policy action, simulate one period
        ...         return StrategicStepResult(rewards=np.array([...]))
    """

    @property
    def num_objectives(self) -> int:
        """Number of optimization objectives."""
        ...

    @property
    def objective_names(self) -> list[str]:
        """Human-readable names for each objective."""
        ...

    @property
    def num_agents(self) -> int:
        """Number of agents in the environment."""
        ...

    @property
    def current_agent(self) -> int:
        """Index of the agent whose turn it is."""
        ...

    @property
    def state_spec(self) -> StateSpec:
        """Specification of the observation space."""
        ...

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the action space."""
        ...

    def reset(
        self,
        seed: int | None = None,
        scenario: dict[str, Any] | None = None,
    ) -> StrategicObservation:
        """Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility.
            scenario: Optional scenario parameters for uncertainty modeling.

        Returns:
            Initial observation.
        """
        ...

    def step(self, action: int) -> StrategicStepResult:
        """Take an action in the environment.

        Args:
            action: Discrete action index.

        Returns:
            Step result with per-objective rewards.
        """
        ...

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean array where True/1 indicates valid action.
        """
        ...

    def clone(self) -> StrategicEnvironment:
        """Create a deep copy of the environment state.

        Required for MCTS tree search simulation.

        Returns:
            Independent copy with identical state.
        """
        ...

    def get_info(self) -> dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with domain-specific state information.
        """
        ...


@runtime_checkable
class ObjectiveFunction(Protocol):
    """Protocol for multi-objective optimization functions.

    Defines what to optimize, supporting multiple competing objectives
    with configurable scalarization strategies.

    Example:
        >>> class ClimateObjective:
        ...     @property
        ...     def num_objectives(self) -> int:
        ...         return 3
        ...
        ...     def evaluate(self, state, action, next_state, info):
        ...         warming = -info["temperature_increase"]
        ...         gdp = info["gdp_growth"]
        ...         equity = -info["inequality_index"]
        ...         return np.array([warming, gdp, equity])
    """

    @property
    def num_objectives(self) -> int:
        """Number of objectives."""
        ...

    @property
    def names(self) -> list[str]:
        """Names of each objective."""
        ...

    def evaluate(
        self,
        state: dict[str, Any],
        action: int,
        next_state: dict[str, Any],
        info: dict[str, Any],
    ) -> np.ndarray:
        """Evaluate objectives for a state transition.

        Args:
            state: State before action.
            action: Action taken.
            next_state: State after action.
            info: Additional transition information.

        Returns:
            Array of objective values, shape (num_objectives,).
        """
        ...

    def scalarize(
        self,
        objective_values: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> float:
        """Convert multi-objective values to a single scalar.

        Args:
            objective_values: Per-objective values.
            weights: Optional objective weights.

        Returns:
            Scalarized value.
        """
        ...


@runtime_checkable
class ScenarioProtocol(Protocol):
    """Protocol for uncertainty scenarios.

    A scenario represents a specific set of assumptions about
    uncertain parameters (e.g., climate sensitivity, pandemic R0,
    disruption duration).

    Example:
        >>> class HighClimateScenario:
        ...     @property
        ...     def name(self) -> str:
        ...         return "high_sensitivity"
        ...
        ...     @property
        ...     def parameters(self) -> dict:
        ...         return {"climate_sensitivity": 4.5}
        ...
        ...     @property
        ...     def probability(self) -> float:
        ...         return 0.2
    """

    @property
    def name(self) -> str:
        """Human-readable scenario name."""
        ...

    @property
    def parameters(self) -> dict[str, Any]:
        """Scenario-specific parameter overrides."""
        ...

    @property
    def probability(self) -> float:
        """Prior probability of this scenario (0-1)."""
        ...


__all__ = [
    "ActionSpec",
    "ObjectiveFunction",
    "ScenarioProtocol",
    "StateSpec",
    "StrategicEnvironment",
    "StrategicObservation",
    "StrategicStepResult",
]
