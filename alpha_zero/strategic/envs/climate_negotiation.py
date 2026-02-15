"""Simplified climate negotiation environment for the Strategic Planning SDK.

Models multi-nation climate negotiations where each nation sets carbon pricing
and technology investment levels over multiple periods. The environment tracks
global temperature, emissions trajectories, and economic impacts.

Example:
    >>> config = ClimateConfig(num_nations=3, num_periods=20)
    >>> env = ClimateNegotiationEnv(config)
    >>> obs = env.reset(seed=42)
    >>> result = env.step(0)  # Nation 0's policy action
    >>> print(result.rewards)  # [warming_reward, gdp_reward, equity_reward]
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

# Action encoding constants
_NUM_CARBON_PRICE_LEVELS = 5   # 0-4
_NUM_TECH_INVESTMENT_LEVELS = 5  # 0-4
_ACTIONS_PER_NATION = _NUM_CARBON_PRICE_LEVELS * _NUM_TECH_INVESTMENT_LEVELS  # 25

# Carbon price levels in $/tonne CO2
_CARBON_PRICES = np.array([0.0, 25.0, 50.0, 100.0, 200.0], dtype=np.float64)

# Tech investment as fraction of GDP
_TECH_FRACTIONS = np.array([0.0, 0.005, 0.01, 0.02, 0.04], dtype=np.float64)

# Baseline cumulative emissions reference (GtCO2) for temperature calculation
_BASELINE_CUMULATIVE = 2000.0


@dataclass
class ClimateConfig:
    """Configuration for the climate negotiation environment.

    Attributes:
        num_nations: Number of negotiating nations/blocs.
        num_periods: Number of negotiation periods (episode length).
        initial_temperature: Initial global temperature anomaly (deg C above
            pre-industrial).
        climate_sensitivity: Equilibrium climate sensitivity -- degrees C
            warming per doubling of cumulative emissions.
        baseline_emissions: Annual baseline emissions per nation (GtCO2/yr)
            before any policy.
        gdp: GDP per nation (trillion USD).
        vulnerability: Climate vulnerability index per nation (0-1, higher
            means more affected by warming).
        seed: Optional random seed.
    """

    num_nations: int = 3
    num_periods: int = 20
    initial_temperature: float = 1.1
    climate_sensitivity: float = 3.0
    baseline_emissions: list[float] = field(
        default_factory=lambda: [10.0, 8.0, 5.0]
    )
    gdp: list[float] = field(default_factory=lambda: [20.0, 15.0, 8.0])
    vulnerability: list[float] = field(default_factory=lambda: [0.2, 0.3, 0.7])
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_nations < 1:
            raise ValueError(f"num_nations must be >= 1, got {self.num_nations}")
        if self.num_periods < 1:
            raise ValueError(f"num_periods must be >= 1, got {self.num_periods}")
        if self.climate_sensitivity <= 0:
            raise ValueError(
                f"climate_sensitivity must be > 0, got {self.climate_sensitivity}"
            )
        if len(self.baseline_emissions) != self.num_nations:
            raise ValueError(
                f"baseline_emissions length ({len(self.baseline_emissions)}) "
                f"must match num_nations ({self.num_nations})"
            )
        if len(self.gdp) != self.num_nations:
            raise ValueError(
                f"gdp length ({len(self.gdp)}) must match num_nations ({self.num_nations})"
            )
        if len(self.vulnerability) != self.num_nations:
            raise ValueError(
                f"vulnerability length ({len(self.vulnerability)}) "
                f"must match num_nations ({self.num_nations})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClimateConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            New ClimateConfig instance.
        """
        return cls(
            num_nations=int(data.get("num_nations", 3)),
            num_periods=int(data.get("num_periods", 20)),
            initial_temperature=float(data.get("initial_temperature", 1.1)),
            climate_sensitivity=float(data.get("climate_sensitivity", 3.0)),
            baseline_emissions=list(data.get("baseline_emissions", [10.0, 8.0, 5.0])),
            gdp=list(data.get("gdp", [20.0, 15.0, 8.0])),
            vulnerability=list(data.get("vulnerability", [0.2, 0.3, 0.7])),
            seed=data.get("seed"),
        )


class ClimateNegotiationEnv:
    """Simplified climate negotiation environment.

    Multi-agent turn-based environment where nations set carbon pricing
    and technology investment policies. Within each period, nations take
    turns. After all nations have acted, the climate-economy model advances.

    Objectives:
        1. minimize_warming: Minimize global temperature anomaly at the end.
        2. maximize_total_gdp: Maximize cumulative global GDP.
        3. maximize_equity: Minimize inequality in climate impacts (GDP-weighted
           vulnerability).

    Action Space:
        Each nation chooses a (carbon_price_level, tech_investment_level) pair.
        5 * 5 = 25 actions per nation turn.

    Climate Physics:
        Uses a simplified log-linear temperature model:
            T = T_initial + sensitivity * log2(cumulative / baseline)
        where cumulative is total emissions to date.
    """

    def __init__(self, config: ClimateConfig | None = None) -> None:
        """Initialize the climate negotiation environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self._config = config or ClimateConfig()
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)

        nn = self._config.num_nations

        # Vector observation dimension:
        # Global: temperature, cumulative_emissions, period_normalized
        # Per-nation: current_emissions, gdp, tech_level, vulnerability, carbon_price
        self._vector_dim = 3 + nn * 5

        # Pre-compute action names
        self._action_names = self._build_action_names()

        # Mutable episode state
        self._temperature: float = 0.0
        self._cumulative_emissions: float = 0.0
        self._emissions: np.ndarray = np.array([])      # per-nation current emissions
        self._gdp_current: np.ndarray = np.array([])     # per-nation current GDP
        self._tech_levels: np.ndarray = np.array([])      # per-nation tech level [0, 1]
        self._carbon_prices: np.ndarray = np.array([])    # per-nation carbon price
        self._period: int = 0
        self._current_nation: int = 0
        self._done: bool = True
        self._nations_acted: list[bool] = []
        self._cumulative_gdp: float = 0.0
        self._pending_actions: dict[int, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Action-name helpers
    # ------------------------------------------------------------------

    def _build_action_names(self) -> dict[int, str]:
        """Build action index to human-readable name mapping."""
        names: dict[int, str] = {}
        for cp_level in range(_NUM_CARBON_PRICE_LEVELS):
            for ti_level in range(_NUM_TECH_INVESTMENT_LEVELS):
                act = cp_level * _NUM_TECH_INVESTMENT_LEVELS + ti_level
                cp_val = int(_CARBON_PRICES[cp_level])
                ti_pct = _TECH_FRACTIONS[ti_level] * 100
                names[act] = f"carbon${cp_val}/t,tech{ti_pct:.1f}%GDP"
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
        return ["minimize_warming", "maximize_total_gdp", "maximize_equity"]

    @property
    def num_agents(self) -> int:
        """Number of agents (one per nation)."""
        return self._config.num_nations

    @property
    def current_agent(self) -> int:
        """Index of the nation whose turn it is."""
        return self._current_nation

    @property
    def state_spec(self) -> StateSpec:
        """Specification of the observation space."""
        nn = self._config.num_nations
        feature_names = ["temperature", "cumulative_emissions", "period"]
        for i in range(nn):
            feature_names.extend([
                f"nation{i}_emissions",
                f"nation{i}_gdp",
                f"nation{i}_tech_level",
                f"nation{i}_vulnerability",
                f"nation{i}_carbon_price",
            ])
        return StateSpec(
            spatial_shape=None,
            vector_dim=self._vector_dim,
            action_space_size=_ACTIONS_PER_NATION,
            spatial_channel_names=[],
            vector_feature_names=feature_names,
        )

    @property
    def action_spec(self) -> ActionSpec:
        """Specification of the action space."""
        return ActionSpec(
            total_size=_ACTIONS_PER_NATION,
            hierarchy_levels=["carbon_price", "tech_investment"],
            level_sizes=[_NUM_CARBON_PRICE_LEVELS, _NUM_TECH_INVESTMENT_LEVELS],
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
            scenario: Optional scenario overrides.  Supported keys:
                - ``initial_temperature``: override starting temperature
                - ``climate_sensitivity``: override sensitivity

        Returns:
            Initial observation for the first nation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        nn = self._config.num_nations

        temp_init = self._config.initial_temperature
        sensitivity = self._config.climate_sensitivity
        if scenario:
            temp_init = float(scenario.get("initial_temperature", temp_init))
            sensitivity = float(scenario.get("climate_sensitivity", sensitivity))

        self._temperature = temp_init
        self._climate_sensitivity = sensitivity
        self._cumulative_emissions = _BASELINE_CUMULATIVE
        self._emissions = np.array(self._config.baseline_emissions, dtype=np.float64)
        self._gdp_current = np.array(self._config.gdp, dtype=np.float64)
        self._tech_levels = np.zeros(nn, dtype=np.float64)
        self._carbon_prices = np.zeros(nn, dtype=np.float64)
        self._period = 0
        self._current_nation = 0
        self._done = False
        self._nations_acted = [False] * nn
        self._cumulative_gdp = 0.0
        self._pending_actions = {}

        logger.debug(
            "ClimateNegotiationEnv reset: nations=%d, periods=%d, T0=%.1f",
            nn,
            self._config.num_periods,
            self._temperature,
        )

        return self._make_observation()

    def step(self, action: int) -> StrategicStepResult:
        """Apply the current nation's policy action.

        Args:
            action: Action index encoding (carbon_price_level, tech_investment_level).

        Returns:
            Step result.  If all nations have acted, the climate-economy model
            advances one period and rewards are computed.

        Raises:
            RuntimeError: If the environment is done or not reset.
            ValueError: If the action is out of range.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action < 0 or action >= _ACTIONS_PER_NATION:
            raise ValueError(
                f"Action {action} out of range [0, {_ACTIONS_PER_NATION})"
            )

        # Decode action
        cp_level = action // _NUM_TECH_INVESTMENT_LEVELS
        ti_level = action % _NUM_TECH_INVESTMENT_LEVELS

        nation = self._current_nation
        self._carbon_prices[nation] = _CARBON_PRICES[cp_level]
        self._pending_actions[nation] = (cp_level, ti_level)
        self._nations_acted[nation] = True

        # Advance current nation pointer
        self._current_nation = (self._current_nation + 1) % self._config.num_nations

        # Check if all nations have acted this period
        all_acted = all(self._nations_acted)
        rewards = np.zeros(3, dtype=np.float64)
        truncated = False

        if all_acted:
            rewards, truncated = self._advance_period()

        obs = self._make_observation()
        info = self.get_info()

        return StrategicStepResult(
            observation=obs,
            rewards=rewards,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def _advance_period(self) -> tuple[np.ndarray, bool]:
        """Advance the climate-economy model by one period.

        Returns:
            Tuple of (rewards, truncated).
        """
        nn = self._config.num_nations

        # Apply policy effects
        for i in range(nn):
            cp_level, ti_level = self._pending_actions.get(i, (0, 0))

            # Carbon pricing reduces emissions
            cp = _CARBON_PRICES[cp_level]
            emission_reduction = cp / 500.0  # rough: $200/t -> 40% reduction
            self._emissions[i] = self._config.baseline_emissions[i] * (
                1.0 - emission_reduction
            ) * (1.0 - self._tech_levels[i] * 0.5)
            self._emissions[i] = max(self._emissions[i], 0.0)

            # Tech investment accumulates
            ti_frac = _TECH_FRACTIONS[ti_level]
            tech_investment = ti_frac * self._gdp_current[i]
            self._tech_levels[i] = min(self._tech_levels[i] + tech_investment * 0.01, 1.0)

            # GDP impact: carbon pricing has a cost, tech investment too
            gdp_loss = cp * self._emissions[i] * 0.001  # cost of carbon pricing
            gdp_loss += ti_frac * self._gdp_current[i] * 0.5  # investment cost
            # Tech provides long-term GDP boost
            gdp_gain = self._tech_levels[i] * self._config.gdp[i] * 0.02
            self._gdp_current[i] = max(
                self._gdp_current[i] - gdp_loss + gdp_gain, 0.1
            )

        # Global emissions and temperature
        annual_emissions = float(self._emissions.sum())
        self._cumulative_emissions += annual_emissions

        # Temperature model: T = T_init + sensitivity * log2(cumulative / baseline)
        if self._cumulative_emissions > _BASELINE_CUMULATIVE:
            ratio = self._cumulative_emissions / _BASELINE_CUMULATIVE
            self._temperature = self._config.initial_temperature + (
                self._climate_sensitivity * np.log2(ratio)
            )
        else:
            self._temperature = self._config.initial_temperature

        # Track cumulative GDP
        period_gdp = float(self._gdp_current.sum())
        self._cumulative_gdp += period_gdp

        # Reset nation turn tracking
        self._period += 1
        self._current_nation = 0
        self._nations_acted = [False] * nn
        self._pending_actions = {}

        # Check termination
        truncated = self._period >= self._config.num_periods
        self._done = truncated

        # Compute rewards
        # 1. Warming: penalize temperature anomaly
        warming_reward = -(self._temperature - self._config.initial_temperature) / 3.0

        # 2. GDP: reward cumulative GDP (normalized by baseline)
        baseline_gdp = float(sum(self._config.gdp))
        gdp_reward = period_gdp / max(baseline_gdp, 1.0) - 1.0  # deviation from baseline

        # 3. Equity: penalize vulnerability-weighted GDP loss disparity
        vulnerability = np.array(self._config.vulnerability, dtype=np.float64)
        baseline_arr = np.array(self._config.gdp, dtype=np.float64)
        gdp_ratios = self._gdp_current / np.maximum(baseline_arr, 0.1)
        # Vulnerability-weighted impact
        impacts = vulnerability * (1.0 - gdp_ratios)
        equity_reward = -float(np.std(impacts))

        rewards = np.array(
            [warming_reward, gdp_reward, equity_reward], dtype=np.float64
        )

        return rewards, truncated

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        All 25 actions are valid for the current nation.

        Returns:
            Boolean array of shape (25,).
        """
        mask = np.zeros(_ACTIONS_PER_NATION, dtype=np.float32)
        if not self._done:
            mask[:] = 1.0
        return mask

    def clone(self) -> ClimateNegotiationEnv:
        """Create a deep copy of this environment.

        Returns:
            Independent copy with identical state.
        """
        return copy.deepcopy(self)

    def get_info(self) -> dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with climate-economy state.
        """
        return {
            "period": self._period,
            "current_nation": self._current_nation,
            "temperature": self._temperature,
            "cumulative_emissions": self._cumulative_emissions,
            "emissions": self._emissions.tolist() if self._emissions.size > 0 else [],
            "gdp": self._gdp_current.tolist() if self._gdp_current.size > 0 else [],
            "tech_levels": self._tech_levels.tolist() if self._tech_levels.size > 0 else [],
            "carbon_prices": self._carbon_prices.tolist() if self._carbon_prices.size > 0 else [],
            "cumulative_gdp": self._cumulative_gdp,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> StrategicObservation:
        """Build observation from current state.

        Returns:
            Current observation for the acting nation.
        """
        nn = self._config.num_nations

        # Normalize values for the vector
        temp_norm = self._temperature / 5.0  # ~5C is extreme
        cum_em_norm = self._cumulative_emissions / (_BASELINE_CUMULATIVE * 3.0)
        period_norm = self._period / max(self._config.num_periods, 1)

        global_feats = [temp_norm, cum_em_norm, period_norm]

        nation_feats: list[float] = []
        baseline_em = np.array(self._config.baseline_emissions, dtype=np.float64)
        baseline_gdp = np.array(self._config.gdp, dtype=np.float64)
        for i in range(nn):
            em_norm = self._emissions[i] / max(float(baseline_em[i]), 1.0) if self._emissions.size > 0 else 0.0
            gdp_norm = self._gdp_current[i] / max(float(baseline_gdp[i]), 1.0) if self._gdp_current.size > 0 else 0.0
            tech = self._tech_levels[i] if self._tech_levels.size > 0 else 0.0
            vuln = self._config.vulnerability[i]
            cp_norm = self._carbon_prices[i] / 200.0 if self._carbon_prices.size > 0 else 0.0
            nation_feats.extend([em_norm, gdp_norm, tech, vuln, cp_norm])

        vector = np.array(global_feats + nation_feats, dtype=np.float32)

        return StrategicObservation(
            spatial=None,
            vector=vector,
            action_mask=self.get_valid_actions(),
            agent_id=self._current_nation,
            step_number=self._period * nn + self._current_nation,
            done=self._done,
            info={
                "current_nation": self._current_nation,
                "period": self._period,
            },
        )
