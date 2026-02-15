"""Pandemic response environment for the Strategic Planning SDK.

Models a simplified pandemic across multiple regions with an SIR (Susceptible-
Infected-Recovered-Dead) epidemiological model. The agent allocates limited
resources (vaccines, PPE, ventilators) across regions to minimize deaths,
economic impact, and inequity.

Example:
    >>> config = PandemicConfig(num_regions=4, num_time_steps=30)
    >>> env = PandemicResponseEnv(config)
    >>> obs = env.reset(seed=42)
    >>> result = env.step(0)
    >>> print(result.rewards)  # [deaths_reward, economic_reward, equity_reward]
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

# Allocation levels (fraction of per-step resource budget)
_NUM_ALLOCATION_LEVELS = 5
_LEVEL_FRACTIONS = np.array([0.0, 0.25, 0.50, 0.75, 1.0], dtype=np.float64)


@dataclass
class PandemicConfig:
    """Configuration for the pandemic response environment.

    Attributes:
        num_regions: Number of geographic regions.
        num_time_steps: Episode length (days or weeks).
        initial_infected: Initial number of infected individuals per region.
        r0: Basic reproduction number (average secondary infections).
        recovery_rate: Fraction of infected who recover per time step.
        total_resources: Total resource units available for entire episode.
        resource_types: Names of resource types.
        region_populations: Population per region.  If None, defaults to
            uniform 1,000,000 per region.
        base_mortality_rate: Baseline mortality rate for infected without
            treatment.
        seed: Optional random seed.
    """

    num_regions: int = 4
    num_time_steps: int = 30
    initial_infected: int = 100
    r0: float = 2.5
    recovery_rate: float = 0.1
    total_resources: float = 1000.0
    resource_types: list[str] = field(
        default_factory=lambda: ["vaccines", "ppe", "ventilators"]
    )
    region_populations: list[int] | None = None
    base_mortality_rate: float = 0.02
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_regions < 1:
            raise ValueError(f"num_regions must be >= 1, got {self.num_regions}")
        if self.num_time_steps < 1:
            raise ValueError(f"num_time_steps must be >= 1, got {self.num_time_steps}")
        if self.r0 <= 0:
            raise ValueError(f"r0 must be > 0, got {self.r0}")
        if self.recovery_rate <= 0 or self.recovery_rate > 1:
            raise ValueError(
                f"recovery_rate must be in (0, 1], got {self.recovery_rate}"
            )
        if self.total_resources <= 0:
            raise ValueError(
                f"total_resources must be > 0, got {self.total_resources}"
            )
        if self.base_mortality_rate < 0 or self.base_mortality_rate > 1:
            raise ValueError(
                f"base_mortality_rate must be in [0, 1], got {self.base_mortality_rate}"
            )
        if len(self.resource_types) < 1:
            raise ValueError("At least one resource type is required")
        if self.region_populations is not None:
            if len(self.region_populations) != self.num_regions:
                raise ValueError(
                    f"region_populations length ({len(self.region_populations)}) "
                    f"must match num_regions ({self.num_regions})"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PandemicConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            New PandemicConfig instance.
        """
        return cls(
            num_regions=int(data.get("num_regions", 4)),
            num_time_steps=int(data.get("num_time_steps", 30)),
            initial_infected=int(data.get("initial_infected", 100)),
            r0=float(data.get("r0", 2.5)),
            recovery_rate=float(data.get("recovery_rate", 0.1)),
            total_resources=float(data.get("total_resources", 1000.0)),
            resource_types=list(data.get("resource_types", ["vaccines", "ppe", "ventilators"])),
            region_populations=data.get("region_populations"),
            base_mortality_rate=float(data.get("base_mortality_rate", 0.02)),
            seed=data.get("seed"),
        )


class PandemicResponseEnv:
    """Pandemic resource allocation environment with SIR dynamics.

    Each region has an independent SIR-D (Susceptible, Infected, Recovered,
    Dead) compartmental model.  The agent allocates three types of resources
    across regions each time step:

    - **Vaccines**: Reduce the transmission rate (move susceptible to recovered).
    - **PPE**: Reduce the effective R0 in a region.
    - **Ventilators**: Reduce the mortality rate for infected individuals.

    Objectives:
        1. minimize_deaths: Minimize cumulative deaths across all regions.
        2. minimize_economic_impact: Economic damage is proportional to the
           number of active infections (lockdown proxy).
        3. maximize_equity: Minimize variance in per-capita death rates across
           regions.

    Action Space:
        For each (resource_type, region) pair, choose an allocation level
        (0-4). Uses a flat encoding: action = pair_index * 5 + level.
        Total actions = num_resource_types * num_regions * 5.
    """

    def __init__(self, config: PandemicConfig | None = None) -> None:
        """Initialize the pandemic response environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self._config = config or PandemicConfig()
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)

        nr = self._config.num_regions
        nrt = len(self._config.resource_types)

        # Populations per region
        if self._config.region_populations is not None:
            self._populations = np.array(
                self._config.region_populations, dtype=np.float64
            )
        else:
            self._populations = np.full(nr, 1_000_000.0, dtype=np.float64)

        # Action space: flat = pair_index * 5 + level
        self._num_pairs = nrt * nr
        self._action_space_size = self._num_pairs * _NUM_ALLOCATION_LEVELS

        # Vector observation dimension
        # Per-region: S, I, R, D, effective_r0 (5 features)
        # Per-resource: remaining amount (nrt features)
        # Global: remaining_budget, time_step_norm, total_infected_norm (3)
        self._vector_dim = nr * 5 + nrt + 3

        # Build action names
        self._action_names = self._build_action_names()

        # Mutable state (initialized in reset)
        self._susceptible: np.ndarray = np.array([])
        self._infected: np.ndarray = np.array([])
        self._recovered: np.ndarray = np.array([])
        self._dead: np.ndarray = np.array([])
        self._effective_r0: np.ndarray = np.array([])
        self._resources_remaining: np.ndarray = np.array([])
        self._remaining_budget: float = 0.0
        self._step_count: int = 0
        self._done: bool = True
        self._cumulative_deaths: float = 0.0
        self._cumulative_economic_impact: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_action_names(self) -> dict[int, str]:
        """Build action index to human-readable name mapping."""
        names: dict[int, str] = {}
        nr = self._config.num_regions
        for pair_idx in range(self._num_pairs):
            r_type = pair_idx // nr
            region = pair_idx % nr
            rname = self._config.resource_types[r_type]
            for lvl in range(_NUM_ALLOCATION_LEVELS):
                act = pair_idx * _NUM_ALLOCATION_LEVELS + lvl
                pct = int(_LEVEL_FRACTIONS[lvl] * 100)
                names[act] = f"{rname}->R{region}@{pct}%"
        return names

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def num_objectives(self) -> int:
        return 3

    @property
    def objective_names(self) -> list[str]:
        return ["minimize_deaths", "minimize_economic_impact", "maximize_equity"]

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def current_agent(self) -> int:
        return 0

    @property
    def state_spec(self) -> StateSpec:
        nr = self._config.num_regions
        feature_names: list[str] = []
        for i in range(nr):
            feature_names.extend([
                f"region{i}_susceptible",
                f"region{i}_infected",
                f"region{i}_recovered",
                f"region{i}_dead",
                f"region{i}_effective_r0",
            ])
        for rname in self._config.resource_types:
            feature_names.append(f"remaining_{rname}")
        feature_names.extend(["remaining_budget", "time_step", "total_infected"])
        return StateSpec(
            spatial_shape=None,
            vector_dim=self._vector_dim,
            action_space_size=self._action_space_size,
            spatial_channel_names=[],
            vector_feature_names=feature_names,
        )

    @property
    def action_spec(self) -> ActionSpec:
        return ActionSpec(
            total_size=self._action_space_size,
            hierarchy_levels=["resource_region_allocation"],
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
        """Reset the environment to an initial pandemic state.

        Args:
            seed: Optional random seed.
            scenario: Optional overrides.  Supported keys:
                - ``r0``: override base reproduction number
                - ``initial_infected``: override initial infected count
                - ``mortality_rate``: override base mortality rate

        Returns:
            Initial observation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        nr = self._config.num_regions
        nrt = len(self._config.resource_types)

        r0 = self._config.r0
        initial_infected = float(self._config.initial_infected)
        mortality = self._config.base_mortality_rate
        if scenario:
            r0 = float(scenario.get("r0", r0))
            initial_infected = float(scenario.get("initial_infected", initial_infected))
            mortality = float(scenario.get("mortality_rate", mortality))

        self._base_r0 = r0
        self._base_mortality = mortality

        # SIR compartments per region
        self._infected = np.full(nr, initial_infected, dtype=np.float64)
        self._susceptible = self._populations - self._infected
        self._recovered = np.zeros(nr, dtype=np.float64)
        self._dead = np.zeros(nr, dtype=np.float64)
        self._effective_r0 = np.full(nr, r0, dtype=np.float64)

        # Resource tracking: split budget equally among resource types initially
        per_resource = self._config.total_resources / max(nrt, 1)
        self._resources_remaining = np.full(nrt, per_resource, dtype=np.float64)
        self._remaining_budget = self._config.total_resources

        self._step_count = 0
        self._done = False
        self._cumulative_deaths = 0.0
        self._cumulative_economic_impact = 0.0

        logger.debug(
            "PandemicResponseEnv reset: regions=%d, R0=%.1f, infected=%d",
            nr,
            r0,
            int(initial_infected),
        )

        return self._make_observation()

    def step(self, action: int) -> StrategicStepResult:
        """Take a resource allocation action and advance SIR dynamics.

        Args:
            action: Flat action index = pair_index * 5 + level.

        Returns:
            Step result with per-objective rewards.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action < 0 or action >= self._action_space_size:
            raise ValueError(
                f"Action {action} out of range [0, {self._action_space_size})"
            )

        nr = self._config.num_regions

        # Decode action: which (resource, region) pair and what level
        pair_idx = action // _NUM_ALLOCATION_LEVELS
        level = action % _NUM_ALLOCATION_LEVELS
        r_type = pair_idx // nr
        region = pair_idx % nr

        # Compute allocation amount
        remaining_steps = max(self._config.num_time_steps - self._step_count, 1)
        step_budget = self._resources_remaining[r_type] / remaining_steps
        amount = _LEVEL_FRACTIONS[level] * step_budget
        amount = min(amount, self._resources_remaining[r_type])
        amount = max(amount, 0.0)
        self._resources_remaining[r_type] -= amount
        self._remaining_budget = float(self._resources_remaining.sum())

        # Apply resource effects before SIR update
        resource_name = self._config.resource_types[r_type]
        if resource_name == "vaccines" or r_type == 0:
            # Vaccines: move susceptible to recovered (reduce transmission pool)
            vaccinated = min(amount * 100.0, self._susceptible[region])
            self._susceptible[region] -= vaccinated
            self._recovered[region] += vaccinated
        elif resource_name == "ppe" or r_type == 1:
            # PPE: reduce effective R0 in the target region
            r0_reduction = amount * 0.005  # each unit reduces R0 by 0.005
            self._effective_r0[region] = max(
                self._effective_r0[region] - r0_reduction, 0.1
            )
        elif resource_name == "ventilators" or r_type == 2:
            # Ventilators: reduce mortality (effect applied during SIR step)
            # Store the ventilator allocation; it reduces mortality this step
            pass  # Effect handled in _sir_step via effective mortality
        else:
            # Generic resource: small R0 reduction
            r0_reduction = amount * 0.002
            self._effective_r0[region] = max(
                self._effective_r0[region] - r0_reduction, 0.1
            )

        # Compute effective mortality per region (ventilators help)
        # Ventilator effect: amount allocated this step reduces mortality
        ventilator_amount = amount if (resource_name == "ventilators" or r_type == 2) else 0.0
        effective_mortality = np.full(nr, self._base_mortality, dtype=np.float64)
        if ventilator_amount > 0:
            # Ventilators reduce mortality in the targeted region
            mortality_reduction = min(ventilator_amount * 0.001, self._base_mortality * 0.9)
            effective_mortality[region] = max(
                self._base_mortality - mortality_reduction, 0.001
            )

        # SIR dynamics for all regions
        new_deaths_total = 0.0
        for i in range(nr):
            pop = self._populations[i]
            if pop <= 0:
                continue

            s = self._susceptible[i]
            inf = self._infected[i]
            r = self._recovered[i]

            # Transmission: new infections
            beta = self._effective_r0[i] * self._config.recovery_rate
            new_infections = beta * s * inf / pop
            new_infections = min(new_infections, s)
            # Add stochastic noise
            noise = self._rng.normal(0, max(new_infections * 0.1, 1.0))
            new_infections = max(new_infections + noise, 0.0)
            new_infections = min(new_infections, s)

            # Recovery
            new_recovered = self._config.recovery_rate * inf
            new_recovered = min(new_recovered, inf)

            # Deaths
            new_deaths = effective_mortality[i] * inf
            new_deaths = min(new_deaths, inf - new_recovered)
            new_deaths = max(new_deaths, 0.0)

            # Update compartments
            self._susceptible[i] = s - new_infections
            self._infected[i] = inf + new_infections - new_recovered - new_deaths
            self._recovered[i] = r + new_recovered
            self._dead[i] += new_deaths

            # Ensure non-negative
            self._susceptible[i] = max(self._susceptible[i], 0.0)
            self._infected[i] = max(self._infected[i], 0.0)

            new_deaths_total += new_deaths

        # R0 mean-reversion (PPE effects fade)
        self._effective_r0 = (
            0.95 * self._effective_r0 + 0.05 * self._base_r0
        )

        self._cumulative_deaths += new_deaths_total

        # Economic impact: proportional to active infections
        total_infected = float(self._infected.sum())
        total_pop = float(self._populations.sum())
        economic_impact = total_infected / max(total_pop, 1.0)
        self._cumulative_economic_impact += economic_impact

        self._step_count += 1
        truncated = self._step_count >= self._config.num_time_steps

        # Check if pandemic is over (no more infected)
        all_clear = float(self._infected.sum()) < 1.0
        self._done = truncated or all_clear

        # Compute rewards
        # 1. Deaths: penalize new deaths
        death_reward = -new_deaths_total / max(total_pop * 0.001, 1.0)

        # 2. Economic: penalize infection rate
        economic_reward = -economic_impact * 10.0

        # 3. Equity: penalize variance in per-capita death rates
        per_capita_deaths = self._dead / np.maximum(self._populations, 1.0)
        equity_reward = -float(np.std(per_capita_deaths)) * 1000.0

        rewards = np.array(
            [death_reward, economic_reward, equity_reward], dtype=np.float64
        )

        obs = self._make_observation()
        info = self.get_info()
        info["new_deaths"] = new_deaths_total
        info["total_infected_current"] = total_infected

        return StrategicStepResult(
            observation=obs,
            rewards=rewards,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Actions are invalid if their resource type is exhausted.

        Returns:
            Boolean array of shape (action_space_size,).
        """
        nr = self._config.num_regions
        mask = np.zeros(self._action_space_size, dtype=np.float32)

        if self._done:
            return mask

        for pair_idx in range(self._num_pairs):
            r_type = pair_idx // nr
            for lvl in range(_NUM_ALLOCATION_LEVELS):
                act = pair_idx * _NUM_ALLOCATION_LEVELS + lvl
                if lvl == 0:
                    # Zero allocation is always valid
                    mask[act] = 1.0
                elif self._resources_remaining.size > 0 and self._resources_remaining[r_type] > 0.01:
                    mask[act] = 1.0
        return mask

    def clone(self) -> PandemicResponseEnv:
        """Create a deep copy of this environment.

        Returns:
            Independent copy with identical state.
        """
        return copy.deepcopy(self)

    def get_info(self) -> dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with epidemiological state.
        """
        return {
            "step": self._step_count,
            "susceptible": self._susceptible.tolist() if self._susceptible.size > 0 else [],
            "infected": self._infected.tolist() if self._infected.size > 0 else [],
            "recovered": self._recovered.tolist() if self._recovered.size > 0 else [],
            "dead": self._dead.tolist() if self._dead.size > 0 else [],
            "effective_r0": self._effective_r0.tolist() if self._effective_r0.size > 0 else [],
            "resources_remaining": self._resources_remaining.tolist() if self._resources_remaining.size > 0 else [],
            "cumulative_deaths": self._cumulative_deaths,
            "cumulative_economic_impact": self._cumulative_economic_impact,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> StrategicObservation:
        """Build observation from current SIR state.

        Returns:
            Current observation.
        """
        nr = self._config.num_regions

        region_feats: list[float] = []
        for i in range(nr):
            pop = self._populations[i]
            s_norm = self._susceptible[i] / pop if self._susceptible.size > 0 and pop > 0 else 1.0
            i_norm = self._infected[i] / pop if self._infected.size > 0 and pop > 0 else 0.0
            r_norm = self._recovered[i] / pop if self._recovered.size > 0 and pop > 0 else 0.0
            d_norm = self._dead[i] / pop if self._dead.size > 0 and pop > 0 else 0.0
            r0_norm = self._effective_r0[i] / 5.0 if self._effective_r0.size > 0 else 0.5
            region_feats.extend([s_norm, i_norm, r_norm, d_norm, r0_norm])

        resource_feats = []
        if self._resources_remaining.size > 0:
            per_resource_init = self._config.total_resources / max(len(self._config.resource_types), 1)
            for val in self._resources_remaining:
                resource_feats.append(val / max(per_resource_init, 1.0))
        else:
            resource_feats = [0.0] * len(self._config.resource_types)

        budget_norm = self._remaining_budget / max(self._config.total_resources, 1.0)
        time_norm = self._step_count / max(self._config.num_time_steps, 1)
        total_inf = float(self._infected.sum()) if self._infected.size > 0 else 0.0
        total_pop = float(self._populations.sum())
        inf_norm = total_inf / max(total_pop, 1.0)

        vector = np.array(
            region_feats + resource_feats + [budget_norm, time_norm, inf_norm],
            dtype=np.float32,
        )

        return StrategicObservation(
            spatial=None,
            vector=vector,
            action_mask=self.get_valid_actions(),
            agent_id=0,
            step_number=self._step_count,
            done=self._done,
            info={"cumulative_deaths": self._cumulative_deaths},
        )
