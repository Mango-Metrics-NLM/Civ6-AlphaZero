"""Composable city growth system for Civ6 AlphaZero.

This module implements population growth, food stockpiling, starvation,
and production bonuses for cities. All parameters are driven by the
CityGrowthConfig dataclass -- no hardcoded values.

Typical usage:

    config = CityGrowthConfig(max_population=20)
    system = CityGrowthSystem(config)

    # Each turn, for every city:
    surplus = system.calculate_food_surplus(city, worked_tiles)
    result = system.process_city_growth(city_id, population, surplus)
    if result["grew"]:
        print("City grew!")

    # Serialize / restore
    saved = system.to_dict()
    system = CityGrowthSystem.from_dict(saved)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class CityGrowthConfig:
    """All tunable parameters for the city growth system.

    Attributes:
        food_consumption_per_pop: Food consumed per population per turn.
        growth_threshold_base: Food needed to grow from pop 1 to 2.
        growth_threshold_scaling: Additional food needed per subsequent
            population level.
        max_population: Hard cap on city population.
        base_food_yield: Food produced by the city center tile each turn.
        production_per_population: Production bonus granted per population
            point.
        starvation_damage: Population lost per turn when the food
            stockpile is negative.
        growth_reward: Reward signal emitted per population growth event.
    """

    food_consumption_per_pop: int = 2
    growth_threshold_base: int = 15
    growth_threshold_scaling: int = 8
    max_population: int = 15
    base_food_yield: int = 2
    production_per_population: int = 1
    starvation_damage: int = 1
    growth_reward: float = 0.05


class CityGrowthSystem:
    """Composable city growth system.

    Tracks per-city food stockpiles and processes population changes
    each turn based on food surplus or deficit.  Every numeric constant
    is sourced from :class:`CityGrowthConfig`.

    Attributes:
        food_stockpile: Mapping of city id to accumulated food.
        config: The active growth configuration.
    """

    def __init__(self, config: CityGrowthConfig | None = None) -> None:
        """Initialise the growth system.

        Args:
            config: Growth parameters.  If ``None``, default values are
                used.
        """
        self.config: CityGrowthConfig = config if config is not None else CityGrowthConfig()
        self.food_stockpile: dict[int, int] = {}
        logger.debug(
            "CityGrowthSystem initialised (max_pop=%d, threshold_base=%d)",
            self.config.max_population,
            self.config.growth_threshold_base,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all food stockpiles."""
        self.food_stockpile.clear()
        logger.debug("CityGrowthSystem reset -- all stockpiles cleared")

    def get_growth_threshold(self, current_population: int) -> int:
        """Return the food needed to grow to the next population level.

        Formula::

            growth_threshold_base + (current_population - 1) * growth_threshold_scaling

        Args:
            current_population: The city's current population.

        Returns:
            Food required for the next population growth.
        """
        return (
            self.config.growth_threshold_base
            + (current_population - 1) * self.config.growth_threshold_scaling
        )

    def calculate_food_surplus(self, city: dict[str, Any], tile_yields: list[dict[str, int]]) -> int:
        """Calculate the per-turn food surplus for a city.

        Total food is the sum of the ``"food"`` key from every worked
        tile in *tile_yields* plus the city center's
        :pyattr:`CityGrowthConfig.base_food_yield`.  Food consumption is
        ``food_consumption_per_pop * population``.

        Args:
            city: A dictionary describing the city.  Must contain a
                ``"population"`` key with an integer value.
            tile_yields: List of yield dictionaries for each worked tile.
                Each dictionary should have a ``"food"`` key.

        Returns:
            The food surplus (positive) or deficit (negative).
        """
        total_food = self.config.base_food_yield
        for tile in tile_yields:
            total_food += tile.get("food", 0)

        population: int = city.get("population", 1)
        consumption = self.config.food_consumption_per_pop * population
        surplus = total_food - consumption

        logger.debug(
            "Food surplus: total_food=%d, consumption=%d, surplus=%d",
            total_food,
            consumption,
            surplus,
        )
        return surplus

    def process_city_growth(
        self, city_id: int, population: int, food_surplus: int
    ) -> dict[str, Any]:
        """Process one turn of growth for a single city.

        The *food_surplus* is added to the city's stockpile.  Then:

        * **Growth**: if the stockpile meets or exceeds the growth
          threshold *and* the population is below
          :pyattr:`CityGrowthConfig.max_population`, population
          increases by 1 and the stockpile is reduced by the threshold
          (overflow is kept).
        * **Starvation**: if the stockpile drops below 0 and population
          is greater than 1, population decreases by
          :pyattr:`CityGrowthConfig.starvation_damage` (but never below
          1) and the stockpile resets to 0.

        Args:
            city_id: Unique identifier for the city.
            population: The city's current population before this turn.
            food_surplus: Food surplus (or deficit) for this turn,
                as returned by :meth:`calculate_food_surplus`.

        Returns:
            A dictionary with the following keys:

            * ``population_change`` (int): Net change in population
              (positive for growth, negative for starvation, 0
              otherwise).
            * ``new_food_stockpile`` (int): The food stockpile after
              processing.
            * ``grew`` (bool): Whether the city grew this turn.
            * ``starved`` (bool): Whether the city lost population this
              turn.
        """
        stockpile = self.food_stockpile.get(city_id, 0) + food_surplus
        population_change = 0
        grew = False
        starved = False

        threshold = self.get_growth_threshold(population)

        # Growth check
        if stockpile >= threshold and population < self.config.max_population:
            stockpile -= threshold
            population_change = 1
            grew = True
            logger.info(
                "City %d grew: pop %d -> %d (stockpile overflow=%d)",
                city_id,
                population,
                population + 1,
                stockpile,
            )

        # Starvation check
        if stockpile < 0 and population > 1:
            pop_loss = min(self.config.starvation_damage, population - 1)
            population_change = -pop_loss
            starved = True
            stockpile = 0
            logger.warning(
                "City %d starved: pop %d -> %d",
                city_id,
                population,
                population + population_change,
            )

        self.food_stockpile[city_id] = stockpile

        return {
            "population_change": population_change,
            "new_food_stockpile": stockpile,
            "grew": grew,
            "starved": starved,
        }

    def get_production_bonus(self, population: int) -> int:
        """Return the production bonus for a given population.

        Args:
            population: The city's current population.

        Returns:
            Production bonus equal to
            ``population * production_per_population``.
        """
        return population * self.config.production_per_population

    # ------------------------------------------------------------------
    # Stockpile accessors
    # ------------------------------------------------------------------

    def get_food_stockpile(self, city_id: int) -> int:
        """Return the current food stockpile for a city.

        Args:
            city_id: The city identifier.

        Returns:
            Stored food for the city, defaulting to 0 if untracked.
        """
        return self.food_stockpile.get(city_id, 0)

    def set_food_stockpile(self, city_id: int, amount: int) -> None:
        """Set the food stockpile for a city.

        Args:
            city_id: The city identifier.
            amount: New stockpile value.
        """
        self.food_stockpile[city_id] = amount

    # ------------------------------------------------------------------
    # Encoding for neural network vector input
    # ------------------------------------------------------------------

    def encode_for_vector(self, city_id: int, population: int) -> dict[str, float]:
        """Return normalised values suitable for a neural-network vector input.

        Args:
            city_id: The city identifier (used to look up the stockpile).
            population: The city's current population.

        Returns:
            A dictionary with:

            * ``food_progress``: stockpile / growth_threshold (clipped
              to [0, 1] implicitly by the growth system).
            * ``population_normalized``: population / max_population.
        """
        threshold = self.get_growth_threshold(population)
        stockpile = self.food_stockpile.get(city_id, 0)

        food_progress = stockpile / threshold if threshold > 0 else 0.0
        population_normalized = population / self.config.max_population

        return {
            "food_progress": float(food_progress),
            "population_normalized": float(population_normalized),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the system state to a plain dictionary.

        Returns:
            A dictionary containing the config fields and the current
            food stockpiles.  Suitable for JSON serialization.
        """
        return {
            "config": {
                "food_consumption_per_pop": self.config.food_consumption_per_pop,
                "growth_threshold_base": self.config.growth_threshold_base,
                "growth_threshold_scaling": self.config.growth_threshold_scaling,
                "max_population": self.config.max_population,
                "base_food_yield": self.config.base_food_yield,
                "production_per_population": self.config.production_per_population,
                "starvation_damage": self.config.starvation_damage,
                "growth_reward": self.config.growth_reward,
            },
            "food_stockpile": {str(k): v for k, v in self.food_stockpile.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CityGrowthSystem:
        """Restore a CityGrowthSystem from a dictionary.

        Args:
            data: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            A fully restored CityGrowthSystem instance.
        """
        config_data = data.get("config", {})
        config = CityGrowthConfig(**config_data)
        system = cls(config)
        raw_stockpile = data.get("food_stockpile", {})
        system.food_stockpile = {int(k): v for k, v in raw_stockpile.items()}
        return system
