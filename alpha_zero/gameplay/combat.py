"""Enhanced composable combat system for Civilization 6 AlphaZero.

This module implements combat resolution with configurable parameters for
terrain modifiers, flanking bonuses, ranged vs melee distinctions, city
attacks, and shaped reward computation. All values are driven by
CombatConfig -- no hardcoded magic numbers.

Combat flow:
    1. Calculate effective strengths (health scaling, terrain, flanking).
    2. Compute damage using strength ratio formula.
    3. Apply ranged-no-retaliation rule when applicable.
    4. Determine kills and city captures.
    5. Compute shaped reward signal for RL training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CombatConfig:
    """Configuration for the combat system.

    All combat parameters are exposed here so that the system contains
    no hardcoded values.  Defaults provide a reasonable baseline inspired
    by Civilization 6 mechanics.
    """

    base_damage: float = 30.0
    base_retaliation_damage: float = 20.0
    terrain_defense_bonuses: dict[str, float] = field(
        default_factory=lambda: {"hills": 0.25, "forest": 0.15}
    )
    flanking_bonus_per_unit: float = 0.1
    max_flanking_bonus: float = 0.3
    ranged_attack_range: int = 2
    ranged_no_retaliation: bool = True
    city_defense_strength: int = 25
    city_ranged_attack_range: int = 2
    health_damage_scaling: bool = True
    kill_reward: float = 0.1
    death_penalty: float = 0.1
    city_capture_reward: float = 0.5
    damage_reward_scaling: float = 0.001


@dataclass
class CombatResult:
    """Result of a single combat engagement.

    Captures all information needed for game-state updates and reward
    computation.
    """

    attacker_damage_dealt: float = 0.0
    defender_damage_dealt: float = 0.0
    attacker_killed: bool = False
    defender_killed: bool = False
    city_captured: bool = False
    reward: float = 0.0


class CombatSystem:
    """Composable combat resolution engine.

    Usage::

        config = CombatConfig(base_damage=35.0)
        combat = CombatSystem(config)

        result = combat.resolve_combat(
            attacker={"strength": 30, "health": 100, "unit_type": "warrior",
                      "position": (3, 4)},
            defender={"strength": 25, "health": 80, "unit_type": "warrior",
                      "position": (3, 5)},
            terrain_type="hills",
            flanking_units=2,
        )
    """

    # Default mapping from unit type to attack range category.
    # "ranged" types use the configured ranged_attack_range; everything
    # else defaults to melee (range 1).
    _RANGED_UNIT_TYPES: set[str] = {"archer"}

    def __init__(self, config: CombatConfig | None = None) -> None:
        self.config = config if config is not None else CombatConfig()
        logger.debug("CombatSystem initialised with config: %s", self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_combat(
        self,
        attacker: dict[str, Any],
        defender: dict[str, Any],
        terrain_type: str = "plains",
        terrain_features: list[str] | None = None,
        flanking_units: int = 0,
    ) -> CombatResult:
        """Resolve a combat engagement between two units.

        Args:
            attacker: Dict with keys ``strength``, ``health``,
                ``unit_type``, ``position``.
            defender: Dict with the same keys as *attacker*.
            terrain_type: Base terrain of the defender's tile
                (e.g. ``"plains"``, ``"hills"``).
            terrain_features: Additional features on the tile
                (e.g. ``["forest"]``).
            flanking_units: Number of friendly units adjacent to the
                defender (excluding the attacker).

        Returns:
            A :class:`CombatResult` with damage, kill, and reward info.
        """

        cfg = self.config

        # --- effective strengths ---
        attacker_eff = self._effective_strength(attacker)
        defender_eff = self._effective_strength(defender)

        # Terrain defence bonus for the defender.
        terrain_mod = self.calculate_terrain_modifier(terrain_type, terrain_features)
        defender_eff *= 1.0 + terrain_mod

        # Flanking bonus for the attacker.
        flanking_mod = self.calculate_flanking_bonus(flanking_units)
        attacker_eff *= 1.0 + flanking_mod

        # --- damage calculation ---
        strength_ratio = attacker_eff / (attacker_eff + defender_eff + 1e-8)

        rng = np.random.default_rng()
        random_variance = rng.uniform(-2.0, 2.0)

        damage_to_defender = cfg.base_damage * strength_ratio + random_variance
        damage_to_attacker = cfg.base_retaliation_damage * (1.0 - strength_ratio)

        # Ranged units do not receive retaliation damage (if configured).
        if cfg.ranged_no_retaliation and self._is_ranged(attacker):
            damage_to_attacker = 0.0

        # Clamp to non-negative values.
        damage_to_defender = max(damage_to_defender, 0.0)
        damage_to_attacker = max(damage_to_attacker, 0.0)

        # --- determine kills ---
        attacker_health: float = float(attacker.get("health", 100))
        defender_health: float = float(defender.get("health", 100))

        attacker_killed = (attacker_health - damage_to_attacker) <= 0.0
        defender_killed = (defender_health - damage_to_defender) <= 0.0

        result = CombatResult(
            attacker_damage_dealt=damage_to_defender,
            defender_damage_dealt=damage_to_attacker,
            attacker_killed=attacker_killed,
            defender_killed=defender_killed,
            city_captured=False,
        )

        result.reward = self.compute_reward(result)

        logger.debug(
            "Combat resolved: attacker %s vs defender %s -> %s",
            attacker.get("unit_type"),
            defender.get("unit_type"),
            result,
        )

        return result

    def can_attack(self, attacker: dict[str, Any], target_pos: tuple[int, int]) -> bool:
        """Check whether *attacker* can reach *target_pos*.

        Melee units must be adjacent (Chebyshev distance <= 1).
        Ranged units can attack within their configured range.

        Args:
            attacker: Dict with at least ``unit_type`` and ``position``.
            target_pos: ``(row, col)`` of the intended target.

        Returns:
            ``True`` if the attack is in range.
        """

        pos = attacker.get("position")
        if pos is None:
            return False

        distance = self._chebyshev_distance(pos, target_pos)
        attack_range = self.get_attack_range(attacker.get("unit_type", ""))
        return distance <= attack_range

    def get_attack_range(self, unit_type: str) -> int:
        """Return the attack range for the given *unit_type*.

        Ranged types (currently ``"archer"``) use
        :pyattr:`CombatConfig.ranged_attack_range`.  All other types
        default to melee range (1).

        Args:
            unit_type: The unit's type string.

        Returns:
            Integer attack range.
        """

        if unit_type in self._RANGED_UNIT_TYPES:
            return self.config.ranged_attack_range
        return 1

    def calculate_terrain_modifier(
        self, terrain_type: str, features: list[str] | None = None
    ) -> float:
        """Calculate the combined additive terrain defence modifier.

        Args:
            terrain_type: Base terrain (e.g. ``"hills"``).
            features: Additional tile features (e.g. ``["forest"]``).

        Returns:
            Combined terrain bonus as a fraction (e.g. ``0.40``).
        """

        bonuses = self.config.terrain_defense_bonuses
        modifier = bonuses.get(terrain_type, 0.0)

        if features is not None:
            for feat in features:
                modifier += bonuses.get(feat, 0.0)

        return modifier

    def calculate_flanking_bonus(self, num_flanking_units: int) -> float:
        """Calculate the flanking bonus for the attacker.

        Args:
            num_flanking_units: Number of friendly units adjacent to the
                defender (excluding the attacker itself).

        Returns:
            Flanking bonus capped at :pyattr:`CombatConfig.max_flanking_bonus`.
        """

        cfg = self.config
        return min(
            num_flanking_units * cfg.flanking_bonus_per_unit,
            cfg.max_flanking_bonus,
        )

    def resolve_city_attack(
        self,
        attacker: dict[str, Any],
        city: dict[str, Any],
        terrain_type: str = "plains",
    ) -> CombatResult:
        """Resolve an attack against a city.

        Cities use :pyattr:`CombatConfig.city_defense_strength` as their
        base strength.  If the city's health drops to zero or below the
        city is considered captured.

        Args:
            attacker: Dict with ``strength``, ``health``, ``unit_type``,
                ``position``.
            city: Dict with at least ``defense_strength`` (optional,
                defaults to config), ``health``, and ``position``.
            terrain_type: Base terrain of the city tile.

        Returns:
            A :class:`CombatResult` — ``city_captured`` is ``True`` when
            the city's remaining health is <= 0.
        """

        cfg = self.config

        # Build a pseudo-defender dict from the city.
        city_strength = float(city.get("defense_strength", cfg.city_defense_strength))
        city_health = float(city.get("health", 100))

        # Effective attacker strength.
        attacker_eff = self._effective_strength(attacker)

        # Terrain modifier for the city tile.
        terrain_mod = self.calculate_terrain_modifier(terrain_type)
        city_eff = city_strength * (1.0 + terrain_mod)

        if cfg.health_damage_scaling:
            city_eff *= city_health / 100.0

        # Damage calculation.
        strength_ratio = attacker_eff / (attacker_eff + city_eff + 1e-8)

        rng = np.random.default_rng()
        random_variance = rng.uniform(-2.0, 2.0)

        damage_to_city = cfg.base_damage * strength_ratio + random_variance
        damage_to_attacker = cfg.base_retaliation_damage * (1.0 - strength_ratio)

        # Ranged attackers avoid retaliation if configured.
        if cfg.ranged_no_retaliation and self._is_ranged(attacker):
            damage_to_attacker = 0.0

        damage_to_city = max(damage_to_city, 0.0)
        damage_to_attacker = max(damage_to_attacker, 0.0)

        attacker_health = float(attacker.get("health", 100))
        attacker_killed = (attacker_health - damage_to_attacker) <= 0.0
        city_captured = (city_health - damage_to_city) <= 0.0

        result = CombatResult(
            attacker_damage_dealt=damage_to_city,
            defender_damage_dealt=damage_to_attacker,
            attacker_killed=attacker_killed,
            defender_killed=False,
            city_captured=city_captured,
        )

        result.reward = self.compute_reward(result)

        logger.debug(
            "City attack resolved: attacker %s vs city (str=%s, hp=%s) -> %s",
            attacker.get("unit_type"),
            city_strength,
            city_health,
            result,
        )

        return result

    def compute_reward(self, result: CombatResult) -> float:
        """Compute a shaped reward signal from a combat result.

        The reward combines:
        - A small per-damage bonus scaled by
          :pyattr:`CombatConfig.damage_reward_scaling`.
        - A bonus for killing the defender.
        - A penalty for losing the attacker.
        - A large bonus for capturing a city.

        Args:
            result: A :class:`CombatResult` instance.

        Returns:
            Scalar reward value.
        """

        cfg = self.config
        reward = 0.0

        # Reward proportional to damage dealt.
        reward += result.attacker_damage_dealt * cfg.damage_reward_scaling

        # Penalty proportional to damage received.
        reward -= result.defender_damage_dealt * cfg.damage_reward_scaling

        # Kill / death bonuses.
        if result.defender_killed:
            reward += cfg.kill_reward
        if result.attacker_killed:
            reward -= cfg.death_penalty

        # City capture bonus.
        if result.city_captured:
            reward += cfg.city_capture_reward

        return reward

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _effective_strength(self, unit: dict[str, Any]) -> float:
        """Return the health-scaled effective strength of *unit*."""

        strength = float(unit.get("strength", 0))
        if self.config.health_damage_scaling:
            health = float(unit.get("health", 100))
            strength *= health / 100.0
        return strength

    def _is_ranged(self, unit: dict[str, Any]) -> bool:
        """Return ``True`` if *unit* is a ranged type."""

        return unit.get("unit_type", "") in self._RANGED_UNIT_TYPES

    @staticmethod
    def _chebyshev_distance(
        pos_a: tuple[int, int], pos_b: tuple[int, int]
    ) -> int:
        """Compute the Chebyshev (chessboard) distance between two positions."""

        return max(abs(pos_a[0] - pos_b[0]), abs(pos_a[1] - pos_b[1]))
