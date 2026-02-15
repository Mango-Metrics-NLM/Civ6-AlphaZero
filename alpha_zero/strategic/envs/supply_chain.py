"""Supply chain disruption environment for the Strategic Planning SDK.

Models a multi-product supply chain with multiple suppliers subject to
random disruptions. The agent manages supplier selection and inventory
levels to balance cost, fill rate (demand fulfillment), and lead time.

Example:
    >>> config = SupplyChainConfig(num_suppliers=3, num_products=2)
    >>> env = SupplyChainEnv(config)
    >>> obs = env.reset(seed=42)
    >>> result = env.step(0)
    >>> print(result.rewards)  # [cost_reward, fill_rate_reward, lead_time_reward]
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from alpha_zero.strategic.protocols import (
    ActionSpec,
    StateSpec,
    StrategicObservation,
    StrategicStepResult,
)

logger = logging.getLogger(__name__)

# Inventory level choices
_INVENTORY_LEVELS = 3  # low=0, medium=1, high=2
_INVENTORY_MULTIPLIERS = np.array([0.5, 1.0, 2.0], dtype=np.float64)
_INVENTORY_NAMES = ["low", "medium", "high"]


@dataclass
class SupplyChainConfig:
    """Configuration for the supply chain environment.

    Attributes:
        num_suppliers: Number of available suppliers.
        num_products: Number of distinct products.
        num_periods: Episode length (planning periods).
        disruption_probability: Per-supplier per-period disruption probability.
        base_demand: Baseline demand per product per period.
        holding_cost_rate: Cost per unit of inventory held per period.
        stockout_penalty: Penalty per unit of unmet demand.
        supplier_costs: Cost per unit from each supplier.  If None, uses
            default increasing costs.
        supplier_lead_times: Lead time (periods) for each supplier.  If None,
            uses defaults.
        supplier_capacities: Max units per period per supplier.  If None,
            uses defaults.
        seed: Optional random seed.
    """

    num_suppliers: int = 3
    num_products: int = 2
    num_periods: int = 15
    disruption_probability: float = 0.1
    base_demand: float = 100.0
    holding_cost_rate: float = 0.5
    stockout_penalty: float = 10.0
    supplier_costs: list[float] | None = None
    supplier_lead_times: list[int] | None = None
    supplier_capacities: list[float] | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.num_suppliers < 1:
            raise ValueError(f"num_suppliers must be >= 1, got {self.num_suppliers}")
        if self.num_products < 1:
            raise ValueError(f"num_products must be >= 1, got {self.num_products}")
        if self.num_periods < 1:
            raise ValueError(f"num_periods must be >= 1, got {self.num_periods}")
        if self.disruption_probability < 0 or self.disruption_probability > 1:
            raise ValueError(
                f"disruption_probability must be in [0, 1], got {self.disruption_probability}"
            )
        if self.base_demand <= 0:
            raise ValueError(f"base_demand must be > 0, got {self.base_demand}")
        if self.supplier_costs is not None and len(self.supplier_costs) != self.num_suppliers:
            raise ValueError(
                f"supplier_costs length ({len(self.supplier_costs)}) "
                f"must match num_suppliers ({self.num_suppliers})"
            )
        if self.supplier_lead_times is not None and len(self.supplier_lead_times) != self.num_suppliers:
            raise ValueError(
                f"supplier_lead_times length ({len(self.supplier_lead_times)}) "
                f"must match num_suppliers ({self.num_suppliers})"
            )
        if self.supplier_capacities is not None and len(self.supplier_capacities) != self.num_suppliers:
            raise ValueError(
                f"supplier_capacities length ({len(self.supplier_capacities)}) "
                f"must match num_suppliers ({self.num_suppliers})"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupplyChainConfig:
        """Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration parameters.

        Returns:
            New SupplyChainConfig instance.
        """
        return cls(
            num_suppliers=int(data.get("num_suppliers", 3)),
            num_products=int(data.get("num_products", 2)),
            num_periods=int(data.get("num_periods", 15)),
            disruption_probability=float(data.get("disruption_probability", 0.1)),
            base_demand=float(data.get("base_demand", 100.0)),
            holding_cost_rate=float(data.get("holding_cost_rate", 0.5)),
            stockout_penalty=float(data.get("stockout_penalty", 10.0)),
            supplier_costs=data.get("supplier_costs"),
            supplier_lead_times=data.get("supplier_lead_times"),
            supplier_capacities=data.get("supplier_capacities"),
            seed=data.get("seed"),
        )


class SupplyChainEnv:
    """Supply chain disruption management environment.

    The agent manages a multi-product supply chain by selecting which
    suppliers to activate and what inventory level to maintain for each
    product.  Suppliers may be randomly disrupted, reducing their
    capacity to zero for a period.

    Action Space:
        For each product, the agent can:
        - Toggle each supplier active/inactive (num_suppliers binary choices)
        - Set inventory level to low/medium/high (3 choices)
        Total = num_products * (num_suppliers + 3) flat actions, where each
        action activates one specific (product, decision) pair.

    Objectives:
        1. minimize_cost: Procurement cost + holding cost.
        2. maximize_fill_rate: Fraction of demand fulfilled.
        3. minimize_lead_time: Weighted average lead time of active suppliers.
    """

    def __init__(self, config: SupplyChainConfig | None = None) -> None:
        """Initialize the supply chain environment.

        Args:
            config: Environment configuration. Uses defaults if None.
        """
        self._config = config or SupplyChainConfig()
        self._rng: np.random.Generator = np.random.default_rng(self._config.seed)

        ns = self._config.num_suppliers
        np_ = self._config.num_products

        # Default supplier parameters (cost increases with capability)
        if self._config.supplier_costs is not None:
            self._supplier_costs = np.array(self._config.supplier_costs, dtype=np.float64)
        else:
            self._supplier_costs = np.array(
                [5.0 + i * 3.0 for i in range(ns)], dtype=np.float64
            )

        if self._config.supplier_lead_times is not None:
            self._supplier_lead_times = np.array(
                self._config.supplier_lead_times, dtype=np.int64
            )
        else:
            # Cheaper suppliers have longer lead times
            self._supplier_lead_times = np.array(
                [max(3 - i, 1) for i in range(ns)], dtype=np.int64
            )

        if self._config.supplier_capacities is not None:
            self._supplier_capacities_base = np.array(
                self._config.supplier_capacities, dtype=np.float64
            )
        else:
            self._supplier_capacities_base = np.array(
                [self._config.base_demand * (0.8 + 0.4 * i / max(ns - 1, 1)) for i in range(ns)],
                dtype=np.float64,
            )

        # Action space: for each product, (num_suppliers toggle actions + 3 inventory levels)
        self._actions_per_product = ns + _INVENTORY_LEVELS
        self._action_space_size = np_ * self._actions_per_product

        # Vector observation dimension
        # Per-product: inventory_level_norm, demand_norm, fill_rate (3)
        # Per-supplier: active (0/1), disrupted (0/1), capacity_norm (3)
        # Per-(product, supplier): pending_orders_norm (1)
        # Global: period_norm, total_cost_norm, avg_lead_time_norm (3)
        self._vector_dim = (
            np_ * 3
            + ns * 3
            + np_ * ns
            + 3
        )

        # Build action names
        self._action_names = self._build_action_names()

        # Mutable state
        self._inventory: np.ndarray = np.array([])       # (num_products,)
        self._active_suppliers: np.ndarray = np.array([]) # (num_suppliers,) bool
        self._disrupted: np.ndarray = np.array([])        # (num_suppliers,) bool
        self._current_capacities: np.ndarray = np.array([])  # (num_suppliers,)
        self._pending_orders: np.ndarray = np.array([])   # (num_products, num_suppliers)
        self._inventory_targets: np.ndarray = np.array([])  # (num_products,) target level index
        self._demands: np.ndarray = np.array([])           # (num_products,) current period demand
        self._period: int = 0
        self._done: bool = True
        self._cumulative_cost: float = 0.0
        self._cumulative_fill: float = 0.0
        self._cumulative_lead: float = 0.0
        self._total_demand: float = 0.0

    def _build_action_names(self) -> dict[int, str]:
        """Build action index to name mapping."""
        names: dict[int, str] = {}
        ns = self._config.num_suppliers
        for p in range(self._config.num_products):
            base = p * self._actions_per_product
            for s in range(ns):
                names[base + s] = f"P{p}_toggle_S{s}"
            for lvl in range(_INVENTORY_LEVELS):
                names[base + ns + lvl] = f"P{p}_inv_{_INVENTORY_NAMES[lvl]}"
        return names

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def num_objectives(self) -> int:
        return 3

    @property
    def objective_names(self) -> list[str]:
        return ["minimize_cost", "maximize_fill_rate", "minimize_lead_time"]

    @property
    def num_agents(self) -> int:
        return 1

    @property
    def current_agent(self) -> int:
        return 0

    @property
    def state_spec(self) -> StateSpec:
        np_ = self._config.num_products
        ns = self._config.num_suppliers
        feature_names: list[str] = []
        for p in range(np_):
            feature_names.extend([
                f"product{p}_inventory",
                f"product{p}_demand",
                f"product{p}_fill_rate",
            ])
        for s in range(ns):
            feature_names.extend([
                f"supplier{s}_active",
                f"supplier{s}_disrupted",
                f"supplier{s}_capacity",
            ])
        for p in range(np_):
            for s in range(ns):
                feature_names.append(f"pending_P{p}_S{s}")
        feature_names.extend(["period", "total_cost", "avg_lead_time"])
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
            hierarchy_levels=["product_decision"],
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
        """Reset the supply chain to initial state.

        Args:
            seed: Optional random seed.
            scenario: Optional overrides.  Supported keys:
                - ``disruption_probability``: override disruption rate
                - ``base_demand``: override base demand

        Returns:
            Initial observation.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ns = self._config.num_suppliers
        np_ = self._config.num_products

        self._disruption_prob = self._config.disruption_probability
        self._base_demand = self._config.base_demand
        if scenario:
            self._disruption_prob = float(
                scenario.get("disruption_probability", self._disruption_prob)
            )
            self._base_demand = float(
                scenario.get("base_demand", self._base_demand)
            )

        # Initial inventory = 1x base demand per product
        self._inventory = np.full(np_, self._base_demand, dtype=np.float64)
        # All suppliers active initially
        self._active_suppliers = np.ones(ns, dtype=bool)
        self._disrupted = np.zeros(ns, dtype=bool)
        self._current_capacities = self._supplier_capacities_base.copy()
        # No pending orders
        self._pending_orders = np.zeros((np_, ns), dtype=np.float64)
        # Default to medium inventory target
        self._inventory_targets = np.ones(np_, dtype=np.int64)
        # Initial demand
        self._demands = np.full(np_, self._base_demand, dtype=np.float64)

        self._period = 0
        self._done = False
        self._cumulative_cost = 0.0
        self._cumulative_fill = 0.0
        self._cumulative_lead = 0.0
        self._total_demand = 0.0
        self._periods_counted = 0

        logger.debug(
            "SupplyChainEnv reset: suppliers=%d, products=%d, periods=%d",
            ns,
            np_,
            self._config.num_periods,
        )

        return self._make_observation()

    def step(self, action: int) -> StrategicStepResult:
        """Apply a supply chain decision and advance one period.

        The agent takes one action per step.  Multiple actions in a period
        can be achieved by calling step multiple times before the period
        auto-advances (it advances every ``actions_per_product`` steps for
        simplicity -- but in this implementation each step IS one period,
        and the single action toggles one supplier or sets one inventory
        level for one product).

        Args:
            action: Flat action index.

        Returns:
            Step result with per-objective rewards.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if action < 0 or action >= self._action_space_size:
            raise ValueError(
                f"Action {action} out of range [0, {self._action_space_size})"
            )

        ns = self._config.num_suppliers
        np_ = self._config.num_products

        # Decode action
        product = action // self._actions_per_product
        sub_action = action % self._actions_per_product

        if sub_action < ns:
            # Toggle supplier active/inactive for this product
            supplier = sub_action
            self._active_suppliers[supplier] = not self._active_suppliers[supplier]
        else:
            # Set inventory level for this product
            inv_level = sub_action - ns
            self._inventory_targets[product] = inv_level

        # -- Simulate one period --

        # 1. Random disruptions
        for s in range(ns):
            if self._rng.random() < self._disruption_prob:
                self._disrupted[s] = True
                self._current_capacities[s] = 0.0
            elif self._disrupted[s]:
                # Disruptions last one period, then recover
                if self._rng.random() < 0.5:
                    self._disrupted[s] = False
                    self._current_capacities[s] = self._supplier_capacities_base[s]

        # 2. Demand realization (random variation around base)
        demand_noise = self._rng.normal(0, self._base_demand * 0.1, size=np_)
        self._demands = np.maximum(self._base_demand + demand_noise, 1.0)

        # 3. Place orders based on inventory targets and active suppliers
        period_cost = 0.0
        total_fulfilled = 0.0
        total_demanded = float(self._demands.sum())
        self._total_demand += total_demanded

        active_lead_times: list[float] = []

        for p in range(np_):
            target_mult = _INVENTORY_MULTIPLIERS[int(self._inventory_targets[p])]
            target_inventory = self._base_demand * target_mult
            deficit = max(target_inventory - self._inventory[p] + self._demands[p], 0.0)

            # Order from active, non-disrupted suppliers
            ordered = 0.0
            for s in range(ns):
                if not self._active_suppliers[s] or self._disrupted[s]:
                    continue
                order_amount = min(
                    deficit - ordered,
                    self._current_capacities[s],
                )
                order_amount = max(order_amount, 0.0)
                ordered += order_amount
                period_cost += order_amount * self._supplier_costs[s]
                active_lead_times.append(float(self._supplier_lead_times[s]))

            # Receive ordered goods (simplified: immediate for lead_time=1,
            # otherwise partial delivery)
            received = 0.0
            for s in range(ns):
                if not self._active_suppliers[s] or self._disrupted[s]:
                    continue
                lt = max(int(self._supplier_lead_times[s]), 1)
                # Simplified: receive 1/lead_time fraction each period
                order_for_supplier = min(
                    deficit / max(ns, 1),
                    self._current_capacities[s],
                )
                received += order_for_supplier / lt

            self._inventory[p] += received

            # 4. Fulfill demand
            fulfilled = min(self._inventory[p], self._demands[p])
            self._inventory[p] -= fulfilled
            total_fulfilled += fulfilled

            # Holding cost
            period_cost += self._inventory[p] * self._config.holding_cost_rate

            # Stockout penalty
            unmet = self._demands[p] - fulfilled
            period_cost += unmet * self._config.stockout_penalty

        # Fill rate for this period
        fill_rate = total_fulfilled / max(total_demanded, 1.0)

        # Average lead time of active suppliers
        avg_lead = float(np.mean(active_lead_times)) if active_lead_times else 0.0

        self._cumulative_cost += period_cost
        self._cumulative_fill += fill_rate
        self._cumulative_lead += avg_lead
        self._periods_counted += 1

        self._period += 1
        truncated = self._period >= self._config.num_periods
        self._done = truncated

        # Compute rewards
        # 1. Cost: penalize cost (normalized by base cost estimate)
        base_cost_estimate = self._base_demand * np_ * float(self._supplier_costs.mean())
        cost_reward = -period_cost / max(base_cost_estimate, 1.0)

        # 2. Fill rate: reward fulfillment
        fill_rate_reward = fill_rate  # [0, 1]

        # 3. Lead time: penalize long lead times
        max_lead = float(self._supplier_lead_times.max()) if self._supplier_lead_times.size > 0 else 3.0
        lead_time_reward = -(avg_lead / max(max_lead, 1.0))

        rewards = np.array(
            [cost_reward, fill_rate_reward, lead_time_reward], dtype=np.float64
        )

        obs = self._make_observation()
        info = self.get_info()
        info["period_cost"] = period_cost
        info["period_fill_rate"] = fill_rate
        info["period_avg_lead_time"] = avg_lead

        return StrategicStepResult(
            observation=obs,
            rewards=rewards,
            done=self._done,
            truncated=truncated,
            info=info,
        )

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        All actions are valid (toggling or setting inventory).

        Returns:
            Boolean array of shape (action_space_size,).
        """
        mask = np.zeros(self._action_space_size, dtype=np.float32)
        if not self._done:
            mask[:] = 1.0
        return mask

    def clone(self) -> SupplyChainEnv:
        """Create a deep copy of this environment.

        Returns:
            Independent copy with identical state.
        """
        return copy.deepcopy(self)

    def get_info(self) -> dict[str, Any]:
        """Get current environment information.

        Returns:
            Dictionary with supply chain state.
        """
        avg_fill = self._cumulative_fill / max(self._periods_counted, 1)
        avg_lead = self._cumulative_lead / max(self._periods_counted, 1)
        return {
            "period": self._period,
            "inventory": self._inventory.tolist() if self._inventory.size > 0 else [],
            "active_suppliers": self._active_suppliers.tolist() if self._active_suppliers.size > 0 else [],
            "disrupted": self._disrupted.tolist() if self._disrupted.size > 0 else [],
            "demands": self._demands.tolist() if self._demands.size > 0 else [],
            "cumulative_cost": self._cumulative_cost,
            "average_fill_rate": avg_fill,
            "average_lead_time": avg_lead,
            "done": self._done,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> StrategicObservation:
        """Build observation from current state.

        Returns:
            Current observation.
        """
        np_ = self._config.num_products
        ns = self._config.num_suppliers

        product_feats: list[float] = []
        for p in range(np_):
            inv_norm = self._inventory[p] / max(self._base_demand * 2.0, 1.0) if self._inventory.size > 0 else 0.0
            dem_norm = self._demands[p] / max(self._base_demand * 2.0, 1.0) if self._demands.size > 0 else 0.5
            # Period fill rate (approximate)
            fill = min(self._inventory[p], self._demands[p]) / max(self._demands[p], 1.0) if (
                self._inventory.size > 0 and self._demands.size > 0 and self._demands[p] > 0
            ) else 0.0
            product_feats.extend([inv_norm, dem_norm, fill])

        supplier_feats: list[float] = []
        for s in range(ns):
            active = 1.0 if (self._active_suppliers.size > 0 and self._active_suppliers[s]) else 0.0
            disrupted = 1.0 if (self._disrupted.size > 0 and self._disrupted[s]) else 0.0
            cap_norm = self._current_capacities[s] / max(float(self._supplier_capacities_base[s]), 1.0) if self._current_capacities.size > 0 else 0.0
            supplier_feats.extend([active, disrupted, cap_norm])

        pending_feats: list[float] = []
        for p in range(np_):
            for s in range(ns):
                val = self._pending_orders[p, s] / max(self._base_demand, 1.0) if self._pending_orders.size > 0 else 0.0
                pending_feats.append(val)

        period_norm = self._period / max(self._config.num_periods, 1)
        base_cost_est = self._base_demand * np_ * float(self._supplier_costs.mean())
        cost_norm = self._cumulative_cost / max(base_cost_est * self._config.num_periods, 1.0)
        max_lead = float(self._supplier_lead_times.max()) if self._supplier_lead_times.size > 0 else 3.0
        avg_lead = self._cumulative_lead / max(self._periods_counted, 1)
        lead_norm = avg_lead / max(max_lead, 1.0)

        vector = np.array(
            product_feats + supplier_feats + pending_feats + [period_norm, cost_norm, lead_norm],
            dtype=np.float32,
        )

        return StrategicObservation(
            spatial=None,
            vector=vector,
            action_mask=self.get_valid_actions(),
            agent_id=0,
            step_number=self._period,
            done=self._done,
            info={"period": self._period},
        )
