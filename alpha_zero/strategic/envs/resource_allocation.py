"""Generic resource allocation environment for the Strategic Planning SDK.

Models the problem of distributing limited resources (medical supplies, food,
equipment) across multiple regions over time, balancing efficiency, equity,
and cost objectives.

Example:
    >>> config = ResourceAllocationConfig(num_regions=4, num_resource_types=3)
    >>> env = ResourceAllocationEnv(config)
    >>> obs = env.reset(seed=42)
    >>> result = env.step(0)
    >>> print(result.rewards)  # [shortage_reward, equity_reward, cost_reward]
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from alpha_zero.strategic.protocols import (
    ActionSpec,
    StateSpec,
    StrategicObservation,
    StrategicStepResult,
)

logger = logging.getLogger(__name__)

# Number of discrete allocation levels per (resource, region) pair.
_NUM_ALLOCATION_LEVELS = 5  # 0%, 25%, 50%, 75%, 100% of remaining budget per step

# Fraction of remaining budget corresponding to each level.
_LEVEL_FRACTIONS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)


@dataclass
class ResourceAllocationConfig:
    """Configuration for the resource allocation environment.

    Attributes:
        num_regions: Number of geographic regions to allocate across.
        num_resource_types: Number of distinct resource types.
        num_time_steps: Episode length in time steps.
        total_budget: Total budget available for the entire episode.
        resource_names: Human-readable names for each resource type.
        region_names: Human-readable names for each region.
        seed: Optional random seed for reproducibility.
    """

    num_regions: int = 4
    num_resource_types: int = 3
    num_time_steps: int = 10
    total_budget: float = 100.0
    resource_names: list[str] = field(
        default_factory=lambda: ["medical", "food", "equipment"]
    )
    region_names: list[str] = field(
        default_factory=lambda: ["North", "South", "East", "West"]
    )
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_regions < 1:
            raise ValueError(f"num_regions must be >= 1, got {self.num_regions}")
        if self.num_resource_types < 1:
            raise ValueError(
                f"num_resource_types must be >= 1, got {self.num_resource_types}"
            )
        if self.num_time_steps < 1:
            raise ValueError(
                f"num_time_steps must be >= 1, got {self.num_time_steps}"
            )
        if self.total_budget <= 0.0:
            raise ValueError(
                f"total_budget must be > 0, got {self.total_budget}"
            )
        if len(self.resource_names) != self.num_resource_types:
            raise ValueError(
                f"resource_names length ({len(self.resource_names)}) must match "
                f"num_resource_types ({self.num_resource_types})"
            )
        if len(self.region_names) != self.num_regions:
            raise ValueError(
                f"region_names length ({len(self.region_names)}) must match "
                f"num_regions ({self.num_regions})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResourceAllocationConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            New ResourceAllocationConfig instance.
        """
        return cls(
            num_regions=int(data.get("num_regions", 4)),
            num_resource_types=int(data.get("num_resource_types", 3)),
            num_time_steps=int(data.get("num_time_steps", 10)),
            total_budget=float(data.get("total_budget", 100.0)),
            resource_names=list(data.get("resource_names", ["medical", "food", "equipment"])),
            region_names=list(data.get("region_names", ["North", "South", "East", "West"])),
            seed=data.get("seed"),
        )


class ResourceAllocationEnv:
    """Generic resource allocation environment.

    Simulates the allocation of limited resources across regions over time.
    Regions have time-varying demands that follow a random walk. The agent
    decides how much of the remaining budget to allocate to each
    (resource_type, region) pair at each time step.

    Objectives:
        1. minimize_shortage: Minimize the total unmet demand across all regions.
        2. maximize_equity: Maximize fairness of allocation (minimize variance
           in fulfillment ratios across regions).
        3. minimize_cost: Minimize total expenditure (conserve budget).

    Action Space:
        For each (resource_type, region) pair there are 5 discrete allocation
        levels (0%, 25%, 50%, 75%, 100% of the remaining per-step budget).
        The total action space size is ``5 ^ (num_resource_types * num_regions)``
        encoded as a single integer.  Alternatively, the flat action index is
        decoded into per-pair levels.
    """

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    def __init__(self, config: ResourceAllocationConfig | None = None) -> None:
        """Initialize the resource allocation environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self._config = config or ResourceAllocationConfig()
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)

        # Derived constants
        self._num_pairs = self._config.num_resource_types * self._config.num_regions
        # Action space: each pair picks one of 5 levels independently
        self._action_space_size = _NUM_ALLOCATION_LEVELS ** self._num_pairs
        # Clamp the action space to a tractable size.  If the combinatorial
        # space is too large we fall back to a *flat* representation where each
        # action encodes *one* (pair, level) choice and the agent selects one
        # pair to allocate per step.
        self._use_flat_actions = self._action_space_size > 10_000
        if self._use_flat_actions:
            self._action_space_size = self._num_pairs * _NUM_ALLOCATION_LEVELS

        # Pre-compute vector dimension
        # Per-region: demand per resource + capacity per resource + allocation per resource
        # Global: remaining_budget + time_step_normalized
        self._vector_dim = (
            self._config.num_regions * self._config.num_resource_types * 3
            + 2
        )

        # Build action names
        self._action_names = self._build_action_names()

        # Mutable episode state (initialized in reset)
        self._demands: np.ndarray = np.array([])
        self._capacities: np.ndarray = np.array([])
        self._allocations: np.ndarray = np.array([])
        self._remaining_budget: float = 0.0
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_shortage: float = 0.0
        self._cumulative_cost: float = 0.0

    # ------------------------------------------------------------------
    # Action-name helpers
    # ------------------------------------------------------------------

    def _build_action_names(self) -> dict[int, str]:
        """Build a mapping from action index to human-readable name."""
        names: dict[int, str] = {}
        if self._use_flat_actions:
            for pair_idx in range(self._num_pairs):
                r_type = pair_idx // self._config.num_regions
                region = pair_idx % self._config.num_regions
                r_name = self._config.resource_names[r_type]
                reg_name = self._config.region_names[region]
                for lvl in range(_NUM_ALLOCATION_LEVELS):
                    act = pair_idx * _NUM_ALLOCATION_LEVELS + lvl
                    pct = int(_LEVEL_FRACTIONS[lvl] * 100)
                    names[act] = f"{r_name}->{reg_name}@{pct}%"
        else:
            # For small action spaces enumerate all combinations
            for act in range(self._action_space_size):
                parts: list[str] = []
                remainder = act
                for pair_idx in range(self._num_pairs):
                    lvl = remainder % _NUM_ALLOCATION_LEVELS
                    remainder //= _NUM_ALLOCATION_LEVELS
                    pct = int(_LEVEL_FRACTIONS[lvl] * 100)
                    r_type = pair_idx // self._config.num_regions
                    region = pair_idx % self._config.num_regions
                    r_name = self._config.resource_names[r_type]
                    reg_name = self._config.region_names[region]
                    parts.append(f"{r_name}->{reg_name}@{pct}%")
                names[act] = "; ".join(parts)
        return names

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def num_objectives(self) -> int:
        """Number of optimization objectives."""
        return 3

    @property
    def objective_names(self) -> list[str]:
        """Human-readable names for each objective."""
        return ["minimize_shortage", "maximize_equity", "minimize_cost"]

    @property
    def num_agents(self) -> int:
        """Number of agents (single-agent environment)."""
        return 1

    @property
    def current_agent(self) -> int:
        """Index of the current agent (always 0)."""
        return 0

    @property
    def state_spec(self) -> StateSpec:
        """Specification of the observation space."""
        feature_names: list[str] = []
        for reg in self._config.region_names:
            for res in self._config.resource_names:
                feature_names.append(f"demand_{reg}_{res}")
        for reg in self._config.region_names:
            for res in self._config.resource_names:
                feature_names.append(f"capacity_{reg}_{res}")
        for reg in self._config.region_names:
            for res in self._config.resource_names:
                feature_names.append(f"alloc_{reg}_{res}")
        feature_names.extend(["remaining_budget", "time_step"])

        return StateSpec(
            spatial_shape=None,
            vector_dim=self._vector_dim,
            action_space_size=self._action_space_size,
            spatial_channel_names=[],
            vector_feature_names=feature_names,
        )

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the action space."""
        return ActionSpec(
            total_size=self._action_space_size,
            hierarchy_levels=["allocation"],
            level_sizes=[self._action_space_size],
            action_names=self._action_names,
        )

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        scenario: dict[str, Any] | None = None,
    ) -> StrategicObservation:
        """Reset the environment to an initial state.

        Args:
            seed: Optional random seed.
            scenario: Optional scenario parameters.  Supported keys:
                - ``initial_demands``: array of shape (num_regions, num_resource_types)
                - ``demand_volatility``: float controlling random-walk step size

        Returns:
            Initial observation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        nr = self._config.num_regions
        nrt = self._config.num_resource_types

        # Initialize demands -- uniform random in [5, 15] per (region, resource)
        if scenario and "initial_demands" in scenario:
            self._demands = np.asarray(scenario["initial_demands"], dtype=np.float64)
        else:
            self._demands = self._rng.uniform(5.0, 15.0, size=(nr, nrt))

        # Capacities start at zero (nothing allocated yet)
        self._capacities = np.zeros((nr, nrt), dtype=np.float64)
        self._allocations = np.zeros((nr, nrt), dtype=np.float64)
        self._remaining_budget = self._config.total_budget
        self._step_count = 0
        self._done = False
        self._cumulative_shortage = 0.0
        self._cumulative_cost = 0.0

        # Store scenario params for demand evolution
        self._demand_volatility: float = 2.0
        if scenario and "demand_volatility" in scenario:
            self._demand_volatility = float(scenario["demand_volatility"])

        logger.debug(
            "ResourceAllocationEnv reset: regions=%d, resources=%d, budget=%.1f",
            nr,
            nrt,
            self._remaining_budget,
        )

        return self._make_observation()

    def step(self, action: int) -> StrategicStepResult:
        """Take an allocation action.

        Args:
            action: Discrete action index encoding allocation levels.

        Returns:
            Step result with per-objective rewards.

        Raises:
            RuntimeError: If the environment has not been reset or is done.
            ValueError: If the action is out of range.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action < 0 or action >= self._action_space_size:
            raise ValueError(
                f"Action {action} out of range [0, {self._action_space_size})"
            )

        # Decode action into per-pair allocation levels
        levels = self._decode_action(action)

        # Per-step budget share (evenly divided across remaining steps)
        remaining_steps = self._config.num_time_steps - self._step_count
        step_budget = self._remaining_budget / max(remaining_steps, 1)

        # Apply allocations
        total_spent = 0.0
        for pair_idx in range(self._num_pairs):
            r_type = pair_idx // self._config.num_regions
            region = pair_idx % self._config.num_regions
            fraction = _LEVEL_FRACTIONS[levels[pair_idx]]
            amount = fraction * step_budget / max(self._num_pairs, 1)
            amount = min(amount, self._remaining_budget - total_spent)
            amount = max(amount, 0.0)
            self._allocations[region, r_type] += amount
            self._capacities[region, r_type] += amount
            total_spent += amount

        self._remaining_budget -= total_spent
        self._remaining_budget = max(self._remaining_budget, 0.0)

        # Compute per-region shortages
        shortages = np.maximum(self._demands - self._capacities, 0.0)
        total_shortage = float(shortages.sum())
        self._cumulative_shortage += total_shortage

        # Compute fulfillment ratios per region (for equity)
        demand_totals = self._demands.sum(axis=1)
        capacity_totals = self._capacities.sum(axis=1)
        fulfillment = np.where(
            demand_totals > 0,
            np.minimum(capacity_totals / demand_totals, 1.0),
            1.0,
        )
        equity = 1.0 - float(np.std(fulfillment))  # higher is better

        # Evolve demands (random walk)
        noise = self._rng.normal(0.0, self._demand_volatility, size=self._demands.shape)
        self._demands = np.maximum(self._demands + noise, 1.0)

        # Decay capacities slightly (resources get consumed)
        self._capacities *= 0.8

        self._step_count += 1
        self._cumulative_cost += total_spent

        # Check termination
        truncated = self._step_count >= self._config.num_time_steps
        self._done = truncated

        # Compute rewards (all normalized to roughly [-1, 1])
        max_possible_shortage = float(self._demands.sum()) * self._config.num_time_steps
        shortage_reward = -total_shortage / max(max_possible_shortage / self._config.num_time_steps, 1.0)
        equity_reward = equity  # already in [0, 1]
        cost_reward = -total_spent / max(self._config.total_budget / self._config.num_time_steps, 1.0)

        rewards = np.array(
            [shortage_reward, equity_reward, cost_reward], dtype=np.float64
        )

        obs = self._make_observation()
        info = self.get_info()
        info["total_spent_this_step"] = total_spent
        info["total_shortage_this_step"] = total_shortage
        info["fulfillment_ratios"] = fulfillment.tolist()

        return StrategicStepResult(
            observation=obs,
            rewards=rewards,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        All actions are valid as long as the episode is not done.

        Returns:
            Boolean array of shape (action_space_size,).
        """
        mask = np.zeros(self._action_space_size, dtype=np.float32)
        if not self._done:
            mask[:] = 1.0
        return mask

    def clone(self) -> ResourceAllocationEnv:
        """Create a deep copy of this environment.

        Returns:
            Independent copy with identical state.
        """
        return copy.deepcopy(self)

    def get_info(self) -> dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with domain-specific state information.
        """
        return {
            "step": self._step_count,
            "remaining_budget": self._remaining_budget,
            "cumulative_shortage": self._cumulative_shortage,
            "cumulative_cost": self._cumulative_cost,
            "demands": self._demands.tolist() if self._demands.size > 0 else [],
            "capacities": self._capacities.tolist() if self._capacities.size > 0 else [],
            "allocations": self._allocations.tolist() if self._allocations.size > 0 else [],
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_action(self, action: int) -> list[int]:
        """Decode a flat action index into per-pair allocation levels.

        Args:
            action: Flat action index.

        Returns:
            List of allocation levels, one per (resource, region) pair.
        """
        if self._use_flat_actions:
            # Flat: action = pair_idx * 5 + level.  Only one pair is allocated.
            pair_idx = action // _NUM_ALLOCATION_LEVELS
            level = action % _NUM_ALLOCATION_LEVELS
            levels = [0] * self._num_pairs
            levels[pair_idx] = level
            return levels
        else:
            # Combinatorial: mixed-radix decoding
            combo_levels: list[int] = []
            remainder = action
            for _ in range(self._num_pairs):
                combo_levels.append(remainder % _NUM_ALLOCATION_LEVELS)
                remainder //= _NUM_ALLOCATION_LEVELS
            return combo_levels

    def _make_observation(self) -> StrategicObservation:
        """Build a StrategicObservation from the current state.

        Returns:
            Current observation.
        """
        # Flatten demands, capacities, allocations per (region, resource)
        demand_flat = self._demands.flatten()
        capacity_flat = self._capacities.flatten()
        alloc_flat = self._allocations.flatten()

        budget_norm = self._remaining_budget / max(self._config.total_budget, 1.0)
        time_norm = self._step_count / max(self._config.num_time_steps, 1)

        vector = np.concatenate([
            demand_flat / 20.0,  # rough normalization
            capacity_flat / 20.0,
            alloc_flat / 20.0,
            [budget_norm, time_norm],
        ]).astype(np.float32)

        return StrategicObservation(
            spatial=None,
            vector=vector,
            action_mask=self.get_valid_actions(),
            agent_id=0,
            step_number=self._step_count,
            done=self._done,
            info={"remaining_budget": self._remaining_budget},
        )
