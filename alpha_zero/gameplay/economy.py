"""Composable economy system for Civ6 AlphaZero.

Handles gold income, maintenance, purchasing, and bankruptcy with fully
configurable parameters -- no hardcoded values.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EconomyConfig:
    """All tunable knobs for the economy system."""

    starting_gold: int = 0
    gold_per_city_base: int = 3
    gold_per_population: int = 1
    unit_maintenance: dict[str, int] = field(
        default_factory=lambda: {"warrior": 1, "archer": 1, "settler": 0}
    )
    default_unit_maintenance: int = 1
    purchase_cost_multiplier: float = 4.0
    bankruptcy_penalty: float = -0.5  # reward penalty per turn when bankrupt
    gold_reward_scaling: float = 0.001  # small reward for gold income
    gold_normalization_max: float = 500.0  # divisor for gold in encode_for_vector
    income_normalization_max: float = 50.0  # divisor for income in encode_for_vector


class EconomySystem:
    """Composable economy system that tracks gold, income, and purchases.

    Each player has a gold balance that is updated each turn based on city
    income and unit maintenance.  The system supports purchasing items,
    bankruptcy detection, and neural-network-friendly vector encoding.
    """

    def __init__(
        self,
        num_players: int = 2,
        config: EconomyConfig | None = None,
    ) -> None:
        self.config = config or EconomyConfig()
        self._num_players = num_players
        self.gold: dict[int, int] = dict.fromkeys(
            range(num_players), self.config.starting_gold
        )
        self.gold_per_turn: dict[int, int] = dict.fromkeys(range(num_players), 0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self, num_players: int | None = None) -> None:
        """Reset all gold to starting amounts.

        Args:
            num_players: Optionally change the player count on reset.
        """
        if num_players is not None:
            self._num_players = num_players
        self.gold = dict.fromkeys(
            range(self._num_players), self.config.starting_gold
        )
        self.gold_per_turn = dict.fromkeys(range(self._num_players), 0)

    # ------------------------------------------------------------------
    # Income helpers
    # ------------------------------------------------------------------

    def calculate_income(
        self,
        player: int,
        cities: list[dict[str, Any]],
        units: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Calculate per-turn income breakdown for *player*.

        Args:
            player: The player id.
            cities: List of city dicts.  Each must contain a ``"population"``
                key (int).
            units: List of unit dicts.  Each must contain a ``"type"`` key
                (str).

        Returns:
            A dict with keys ``"city_income"``, ``"maintenance"``, and
            ``"net"``.
        """
        cfg = self.config

        city_income = sum(
            cfg.gold_per_city_base + city.get("population", 0) * cfg.gold_per_population
            for city in cities
        )

        maintenance = sum(
            cfg.unit_maintenance.get(unit.get("type", ""), cfg.default_unit_maintenance)
            for unit in units
        )

        net = city_income - maintenance
        return {"city_income": city_income, "maintenance": maintenance, "net": net}

    # ------------------------------------------------------------------
    # Turn processing
    # ------------------------------------------------------------------

    def process_turn(
        self,
        player: int,
        cities: list[dict[str, Any]],
        units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Process a full economy turn for *player*.

        Calculates income, updates the gold balance, and caches the net
        income value for later queries.

        Returns:
            A dict with keys ``"income"``, ``"maintenance"``, ``"net"``,
            ``"new_balance"``, and ``"is_bankrupt"``.
        """
        breakdown = self.calculate_income(player, cities, units)

        self.gold[player] = self.gold.get(player, 0) + breakdown["net"]
        self.gold_per_turn[player] = breakdown["net"]

        bankrupt = self.is_bankrupt(player)
        if bankrupt:
            logger.debug("Player %d is bankrupt (gold=%d)", player, self.gold[player])

        return {
            "income": breakdown["city_income"],
            "maintenance": breakdown["maintenance"],
            "net": breakdown["net"],
            "new_balance": self.gold[player],
            "is_bankrupt": bankrupt,
        }

    # ------------------------------------------------------------------
    # Purchasing
    # ------------------------------------------------------------------

    def can_purchase(self, player: int, item_type: str, base_cost: int) -> bool:
        """Return whether *player* can afford to purchase *item_type*.

        The effective cost is ``base_cost * purchase_cost_multiplier``.
        """
        effective_cost = int(base_cost * self.config.purchase_cost_multiplier)
        return self.get_gold(player) >= effective_cost

    def purchase(self, player: int, item_type: str, base_cost: int) -> bool:
        """Attempt to purchase *item_type* for *player*.

        Deducts ``base_cost * purchase_cost_multiplier`` from the player's
        gold if they can afford it.

        Returns:
            ``True`` if the purchase succeeded, ``False`` otherwise.
        """
        effective_cost = int(base_cost * self.config.purchase_cost_multiplier)
        if self.get_gold(player) < effective_cost:
            return False

        self.gold[player] -= effective_cost
        logger.debug(
            "Player %d purchased %s for %d gold (base %d)",
            player,
            item_type,
            effective_cost,
            base_cost,
        )
        return True

    # ------------------------------------------------------------------
    # Gold accessors
    # ------------------------------------------------------------------

    def get_gold(self, player: int) -> int:
        """Return the current gold balance for *player*."""
        return self.gold.get(player, 0)

    def add_gold(self, player: int, amount: int) -> None:
        """Add *amount* gold to *player*'s balance (may be negative)."""
        self.gold[player] = self.gold.get(player, 0) + amount

    def is_bankrupt(self, player: int) -> bool:
        """Return ``True`` if *player*'s gold is below zero."""
        return self.get_gold(player) < 0

    # ------------------------------------------------------------------
    # Neural-network encoding
    # ------------------------------------------------------------------

    def encode_for_vector(self, player: int) -> dict[str, float]:
        """Return normalized values suitable for neural network input.

        Keys:
            gold_normalized: gold / gold_normalization_max
            income_normalized: gold_per_turn / income_normalization_max
            is_bankrupt: 0.0 or 1.0
        """
        return {
            "gold_normalized": self.get_gold(player) / self.config.gold_normalization_max,
            "income_normalized": self.gold_per_turn.get(player, 0) / self.config.income_normalization_max,
            "is_bankrupt": 1.0 if self.is_bankrupt(player) else 0.0,
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the economy state to a plain dict."""
        return {
            "gold": dict(self.gold),
            "gold_per_turn": dict(self.gold_per_turn),
            "num_players": self._num_players,
            "config": {
                "starting_gold": self.config.starting_gold,
                "gold_per_city_base": self.config.gold_per_city_base,
                "gold_per_population": self.config.gold_per_population,
                "unit_maintenance": dict(self.config.unit_maintenance),
                "default_unit_maintenance": self.config.default_unit_maintenance,
                "purchase_cost_multiplier": self.config.purchase_cost_multiplier,
                "bankruptcy_penalty": self.config.bankruptcy_penalty,
                "gold_reward_scaling": self.config.gold_reward_scaling,
                "gold_normalization_max": self.config.gold_normalization_max,
                "income_normalization_max": self.config.income_normalization_max,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EconomySystem:
        """Deserialize an ``EconomySystem`` from a dict produced by
        :meth:`to_dict`."""
        cfg_data = data.get("config", {})
        config = EconomyConfig(
            starting_gold=cfg_data.get("starting_gold", 0),
            gold_per_city_base=cfg_data.get("gold_per_city_base", 3),
            gold_per_population=cfg_data.get("gold_per_population", 1),
            unit_maintenance=cfg_data.get(
                "unit_maintenance", {"warrior": 1, "archer": 1, "settler": 0}
            ),
            default_unit_maintenance=cfg_data.get("default_unit_maintenance", 1),
            purchase_cost_multiplier=cfg_data.get("purchase_cost_multiplier", 4.0),
            bankruptcy_penalty=cfg_data.get("bankruptcy_penalty", -0.5),
            gold_reward_scaling=cfg_data.get("gold_reward_scaling", 0.001),
            gold_normalization_max=cfg_data.get("gold_normalization_max", 500.0),
            income_normalization_max=cfg_data.get("income_normalization_max", 50.0),
        )

        num_players = data.get("num_players", 2)
        system = cls(num_players=num_players, config=config)

        # Restore mutable state -- convert string keys back to ints.
        raw_gold = data.get("gold", {})
        system.gold = {int(k): v for k, v in raw_gold.items()}

        raw_gpt = data.get("gold_per_turn", {})
        system.gold_per_turn = {int(k): v for k, v in raw_gpt.items()}

        return system
