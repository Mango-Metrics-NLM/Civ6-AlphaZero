"""Unit Promotion System for Civ6-AlphaZero.

This module implements unit promotions including:
- Experience point tracking
- Promotion trees by unit class
- Stat modifiers and special abilities
- Promotion prerequisites

All mechanics are designed to be:
- Fully configurable via dataclasses
- Testable in isolation
- Compatible with the existing encoder system
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class UnitClass(Enum):
    """Unit classes for promotion trees."""

    MELEE = auto()
    RANGED = auto()
    CAVALRY = auto()
    ANTI_CAVALRY = auto()
    SIEGE = auto()
    NAVAL_MELEE = auto()
    NAVAL_RANGED = auto()
    NAVAL_RAIDER = auto()
    AIR_FIGHTER = auto()
    AIR_BOMBER = auto()
    SUPPORT = auto()
    RECON = auto()


class PromotionTier(Enum):
    """Promotion tiers."""

    TIER_1 = auto()
    TIER_2 = auto()
    TIER_3 = auto()
    TIER_4 = auto()


class StatModifierType(Enum):
    """Types of stat modifications."""

    COMBAT_STRENGTH = auto()
    RANGED_STRENGTH = auto()
    MOVEMENT = auto()
    SIGHT = auto()
    RANGE = auto()
    HEAL_PER_TURN = auto()
    ATTACK_BONUS = auto()
    DEFENSE_BONUS = auto()


class AbilityType(Enum):
    """Types of special abilities."""

    FLANKING_BONUS = auto()
    FORTIFICATION_BONUS = auto()
    TERRAIN_BONUS = auto()
    FIRST_STRIKE = auto()
    IGNORE_TERRAIN_COST = auto()
    PILLAGE_BONUS = auto()
    CAPTURE_BONUS = auto()
    ZONE_OF_CONTROL = auto()
    HEAL_ADJACENT = auto()
    BONUS_VS_UNIT_TYPE = auto()


@dataclass
class PromotionConfig:
    """Configuration for promotion mechanics.

    Attributes:
        xp_per_combat: Base XP from combat
        xp_from_kill: Bonus XP for killing
        xp_from_city_capture: XP from capturing city
        xp_from_pillage: XP from pillaging
        xp_per_level: XP thresholds for each level
        max_promotions: Maximum promotions per unit
        heal_on_promotion: HP healed when promoted
        barracks_xp_bonus: XP multiplier from barracks
        armory_xp_bonus: XP multiplier from armory
        military_academy_xp_bonus: XP multiplier from academy
    """

    xp_per_combat: int = 5
    xp_from_kill: int = 10
    xp_from_city_capture: int = 20
    xp_from_pillage: int = 3
    xp_per_level: list[int] = field(default_factory=lambda: [15, 30, 45, 60, 75, 90, 105])
    max_promotions: int = 7
    heal_on_promotion: int = 50
    barracks_xp_bonus: float = 0.25
    armory_xp_bonus: float = 0.25
    military_academy_xp_bonus: float = 0.25


@dataclass
class StatModifier:
    """A stat modification from a promotion.

    Attributes:
        modifier_type: Type of stat modified
        value: Modification value
        condition: Optional condition for the modifier
        condition_value: Value for conditional modifiers
    """

    modifier_type: StatModifierType
    value: float
    condition: str | None = None  # e.g., "vs_ranged", "in_forest"
    condition_value: str | None = None


@dataclass
class Ability:
    """A special ability from a promotion.

    Attributes:
        ability_type: Type of ability
        value: Ability value/magnitude
        description: Human-readable description
        conditions: Conditions for ability activation
    """

    ability_type: AbilityType
    value: float = 1.0
    description: str = ""
    conditions: list[str] = field(default_factory=list)


@dataclass
class Promotion:
    """Definition of a promotion.

    Attributes:
        promotion_id: Unique identifier
        name: Display name
        tier: Promotion tier
        unit_classes: Classes that can use this promotion
        description: Human-readable description
        stat_modifiers: List of stat modifications
        abilities: List of abilities granted
        prerequisites: Required promotions
        excludes: Promotions that prevent this one
    """

    promotion_id: str
    name: str
    tier: PromotionTier
    unit_classes: list[UnitClass]
    description: str
    stat_modifiers: list[StatModifier] = field(default_factory=list)
    abilities: list[Ability] = field(default_factory=list)
    prerequisites: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)


# Default promotions
DEFAULT_PROMOTIONS: dict[str, Promotion] = {
    # Melee Tier 1
    "battlecry": Promotion(
        promotion_id="battlecry",
        name="Battlecry",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.MELEE, UnitClass.ANTI_CAVALRY],
        description="+7 Combat Strength vs melee and ranged units",
        stat_modifiers=[
            StatModifier(StatModifierType.COMBAT_STRENGTH, 7.0, "vs_melee"),
            StatModifier(StatModifierType.COMBAT_STRENGTH, 7.0, "vs_ranged"),
        ],
    ),
    "tortoise": Promotion(
        promotion_id="tortoise",
        name="Tortoise",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.MELEE, UnitClass.ANTI_CAVALRY],
        description="+10 Combat Strength when defending against ranged attacks",
        stat_modifiers=[
            StatModifier(StatModifierType.DEFENSE_BONUS, 10.0, "vs_ranged"),
        ],
    ),
    # Melee Tier 2
    "commando": Promotion(
        promotion_id="commando",
        name="Commando",
        tier=PromotionTier.TIER_2,
        unit_classes=[UnitClass.MELEE, UnitClass.ANTI_CAVALRY],
        description="Can scale Cliff walls. +1 Movement.",
        stat_modifiers=[
            StatModifier(StatModifierType.MOVEMENT, 1.0),
        ],
        abilities=[
            Ability(AbilityType.IGNORE_TERRAIN_COST, 1.0, "Can scale cliffs"),
        ],
        prerequisites=["battlecry"],
    ),
    "amphibious": Promotion(
        promotion_id="amphibious",
        name="Amphibious",
        tier=PromotionTier.TIER_2,
        unit_classes=[UnitClass.MELEE, UnitClass.ANTI_CAVALRY],
        description="No Combat Strength penalty when attacking across river or from sea",
        abilities=[
            Ability(AbilityType.IGNORE_TERRAIN_COST, 1.0, "No amphibious penalty", ["river", "embark"]),
        ],
        prerequisites=["tortoise"],
    ),
    # Ranged Tier 1
    "volley": Promotion(
        promotion_id="volley",
        name="Volley",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.RANGED],
        description="+5 Ranged Strength vs land units",
        stat_modifiers=[
            StatModifier(StatModifierType.RANGED_STRENGTH, 5.0, "vs_land"),
        ],
    ),
    "garrison": Promotion(
        promotion_id="garrison",
        name="Garrison",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.RANGED],
        description="+10 Combat Strength when occupying a district or fort",
        stat_modifiers=[
            StatModifier(StatModifierType.COMBAT_STRENGTH, 10.0, "in_district"),
        ],
    ),
    # Ranged Tier 2
    "arrow_storm": Promotion(
        promotion_id="arrow_storm",
        name="Arrow Storm",
        tier=PromotionTier.TIER_2,
        unit_classes=[UnitClass.RANGED],
        description="+7 Ranged Strength vs land and naval units",
        stat_modifiers=[
            StatModifier(StatModifierType.RANGED_STRENGTH, 7.0, "vs_land"),
            StatModifier(StatModifierType.RANGED_STRENGTH, 7.0, "vs_naval"),
        ],
        prerequisites=["volley"],
    ),
    "incendiaries": Promotion(
        promotion_id="incendiaries",
        name="Incendiaries",
        tier=PromotionTier.TIER_2,
        unit_classes=[UnitClass.RANGED],
        description="+7 Ranged Strength vs district defenses",
        stat_modifiers=[
            StatModifier(StatModifierType.RANGED_STRENGTH, 7.0, "vs_district"),
        ],
        prerequisites=["garrison"],
    ),
    # Cavalry Tier 1
    "caparison": Promotion(
        promotion_id="caparison",
        name="Caparison",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.CAVALRY],
        description="+5 Combat Strength when attacking",
        stat_modifiers=[
            StatModifier(StatModifierType.ATTACK_BONUS, 5.0),
        ],
    ),
    "coursers": Promotion(
        promotion_id="coursers",
        name="Coursers",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.CAVALRY],
        description="+5 Combat Strength when defending",
        stat_modifiers=[
            StatModifier(StatModifierType.DEFENSE_BONUS, 5.0),
        ],
    ),
    # Recon Tier 1
    "ranger": Promotion(
        promotion_id="ranger",
        name="Ranger",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.RECON],
        description="Faster Movement in Woods and Jungle terrain",
        abilities=[
            Ability(AbilityType.TERRAIN_BONUS, 1.0, "Ignore forest/jungle movement cost", ["forest", "jungle"]),
        ],
    ),
    "alpine": Promotion(
        promotion_id="alpine",
        name="Alpine",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.RECON],
        description="Faster Movement on Hills terrain",
        abilities=[
            Ability(AbilityType.TERRAIN_BONUS, 1.0, "Ignore hills movement cost", ["hills"]),
        ],
    ),
    # Siege Tier 1
    "grape_shot": Promotion(
        promotion_id="grape_shot",
        name="Grape Shot",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.SIEGE],
        description="+7 Combat Strength vs land units",
        stat_modifiers=[
            StatModifier(StatModifierType.RANGED_STRENGTH, 7.0, "vs_land"),
        ],
    ),
    "crew_weapons": Promotion(
        promotion_id="crew_weapons",
        name="Crew Weapons",
        tier=PromotionTier.TIER_1,
        unit_classes=[UnitClass.SIEGE],
        description="+7 Combat Strength when defending",
        stat_modifiers=[
            StatModifier(StatModifierType.DEFENSE_BONUS, 7.0),
        ],
    ),
}


@dataclass
class UnitExperience:
    """Experience and promotion state for a unit.

    Attributes:
        unit_id: Unit identifier
        unit_class: Class of unit
        experience: Total XP earned
        level: Current level
        promotions: List of earned promotion IDs
        pending_promotion: Whether unit can promote
    """

    unit_id: str
    unit_class: UnitClass
    experience: int = 0
    level: int = 0
    promotions: list[str] = field(default_factory=list)
    pending_promotion: bool = False


class PromotionManager:
    """Manages unit promotion mechanics.

    This class handles:
    - XP tracking and level ups
    - Promotion selection
    - Stat modifier calculations
    - Ability management

    Example:
        >>> config = PromotionConfig(xp_per_combat=5)
        >>> manager = PromotionManager(config)
        >>> manager.register_unit("warrior_1", UnitClass.MELEE)
        >>> manager.add_experience("warrior_1", 20)
        >>> available = manager.get_available_promotions("warrior_1")
        >>> manager.apply_promotion("warrior_1", "battlecry")
    """

    def __init__(
        self,
        config: PromotionConfig | None = None,
        promotions: dict[str, Promotion] | None = None,
    ) -> None:
        """Initialize promotion manager.

        Args:
            config: Promotion configuration
            promotions: Available promotions
        """
        self.config = config or PromotionConfig()
        self._promotions = promotions or DEFAULT_PROMOTIONS

        self._units: dict[str, UnitExperience] = {}

        logger.debug(f"PromotionManager initialized with {len(self._promotions)} promotions")

    def register_unit(
        self,
        unit_id: str,
        unit_class: UnitClass,
        starting_xp: int = 0,
    ) -> UnitExperience:
        """Register a new unit for promotion tracking.

        Args:
            unit_id: Unique unit identifier
            unit_class: Class of unit
            starting_xp: Initial XP (from buildings)

        Returns:
            Created UnitExperience
        """
        unit = UnitExperience(
            unit_id=unit_id,
            unit_class=unit_class,
            experience=starting_xp,
        )

        self._units[unit_id] = unit
        self._check_level_up(unit)

        logger.debug(f"Registered unit {unit_id} ({unit_class.name}) with {starting_xp} XP")
        return unit

    def add_experience(
        self,
        unit_id: str,
        amount: int,
        source: str = "combat",
        xp_multiplier: float = 1.0,
    ) -> int:
        """Add experience to a unit.

        Args:
            unit_id: Unit to receive XP
            amount: Base XP amount
            source: Source of XP
            xp_multiplier: Multiplier from buildings/policies

        Returns:
            Total XP after addition
        """
        if unit_id not in self._units:
            logger.warning(f"Unknown unit: {unit_id}")
            return 0

        unit = self._units[unit_id]
        actual_xp = int(amount * xp_multiplier)
        unit.experience += actual_xp

        logger.debug(f"Unit {unit_id} gained {actual_xp} XP from {source} (total: {unit.experience})")

        self._check_level_up(unit)
        return unit.experience

    def _check_level_up(self, unit: UnitExperience) -> bool:
        """Check if unit can level up.

        Args:
            unit: Unit to check

        Returns:
            True if unit leveled up
        """
        if unit.level >= len(self.config.xp_per_level):
            return False

        threshold = self.config.xp_per_level[unit.level]
        if unit.experience >= threshold and len(unit.promotions) < self.config.max_promotions:
            unit.pending_promotion = True
            logger.info(f"Unit {unit.unit_id} ready for promotion (level {unit.level} -> {unit.level + 1})")
            return True

        return False

    def get_available_promotions(self, unit_id: str) -> list[Promotion]:
        """Get promotions available to a unit.

        Args:
            unit_id: Unit to check

        Returns:
            List of available promotions
        """
        if unit_id not in self._units:
            return []

        unit = self._units[unit_id]
        if not unit.pending_promotion:
            return []

        available: list[Promotion] = []

        for promotion_id, promotion in self._promotions.items():
            # Check unit class
            if unit.unit_class not in promotion.unit_classes:
                continue

            # Check not already earned
            if promotion_id in unit.promotions:
                continue

            # Check prerequisites
            if promotion.prerequisites:
                if not all(p in unit.promotions for p in promotion.prerequisites):
                    continue

            # Check exclusions
            if promotion.excludes:
                if any(p in unit.promotions for p in promotion.excludes):
                    continue

            # Check tier (must have lower tier first)
            tier_value = promotion.tier.value
            if tier_value > 1:
                # Need at least one promotion from previous tier
                has_previous_tier = any(
                    self._promotions[p].tier.value == tier_value - 1
                    for p in unit.promotions
                    if p in self._promotions
                )
                if not has_previous_tier:
                    continue

            available.append(promotion)

        return available

    def apply_promotion(
        self,
        unit_id: str,
        promotion_id: str,
    ) -> dict[str, Any] | None:
        """Apply a promotion to a unit.

        Args:
            unit_id: Unit to promote
            promotion_id: Promotion to apply

        Returns:
            Dictionary of changes, or None if invalid
        """
        if unit_id not in self._units:
            logger.warning(f"Unknown unit: {unit_id}")
            return None

        unit = self._units[unit_id]

        if not unit.pending_promotion:
            logger.warning(f"Unit {unit_id} not ready for promotion")
            return None

        if promotion_id not in self._promotions:
            logger.warning(f"Unknown promotion: {promotion_id}")
            return None

        promotion = self._promotions[promotion_id]

        # Verify promotion is available
        available = self.get_available_promotions(unit_id)
        if promotion not in available:
            logger.warning(f"Promotion {promotion_id} not available for unit {unit_id}")
            return None

        # Apply promotion
        unit.promotions.append(promotion_id)
        unit.level += 1
        unit.pending_promotion = False

        # Check for next level
        self._check_level_up(unit)

        result = {
            "unit_id": unit_id,
            "promotion": promotion_id,
            "promotion_name": promotion.name,
            "new_level": unit.level,
            "heal_amount": self.config.heal_on_promotion,
            "stat_modifiers": [
                {
                    "type": m.modifier_type.name,
                    "value": m.value,
                    "condition": m.condition,
                }
                for m in promotion.stat_modifiers
            ],
            "abilities": [
                {
                    "type": a.ability_type.name,
                    "value": a.value,
                    "description": a.description,
                }
                for a in promotion.abilities
            ],
        }

        logger.info(f"Applied promotion {promotion.name} to unit {unit_id} (level {unit.level})")
        return result

    def get_unit_stat_modifiers(
        self,
        unit_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[StatModifierType, float]:
        """Get total stat modifiers for a unit.

        Args:
            unit_id: Unit to check
            context: Combat/terrain context for conditional modifiers

        Returns:
            Dictionary of modifier type to total value
        """
        if unit_id not in self._units:
            return {}

        unit = self._units[unit_id]
        modifiers: dict[StatModifierType, float] = {}
        context = context or {}

        for promotion_id in unit.promotions:
            if promotion_id not in self._promotions:
                continue

            promotion = self._promotions[promotion_id]
            for mod in promotion.stat_modifiers:
                # Check condition
                if mod.condition:
                    if mod.condition not in context:
                        continue
                    if mod.condition_value and context.get(mod.condition) != mod.condition_value:
                        continue

                current = modifiers.get(mod.modifier_type, 0.0)
                modifiers[mod.modifier_type] = current + mod.value

        return modifiers

    def get_unit_abilities(self, unit_id: str) -> list[Ability]:
        """Get all abilities for a unit.

        Args:
            unit_id: Unit to check

        Returns:
            List of abilities
        """
        if unit_id not in self._units:
            return []

        unit = self._units[unit_id]
        abilities: list[Ability] = []

        for promotion_id in unit.promotions:
            if promotion_id not in self._promotions:
                continue

            promotion = self._promotions[promotion_id]
            abilities.extend(promotion.abilities)

        return abilities

    def has_ability(
        self,
        unit_id: str,
        ability_type: AbilityType,
        context: dict[str, Any] | None = None,
    ) -> bool:
        """Check if unit has a specific ability.

        Args:
            unit_id: Unit to check
            ability_type: Ability to check for
            context: Context for conditional abilities

        Returns:
            True if unit has the ability
        """
        abilities = self.get_unit_abilities(unit_id)
        context = context or {}

        for ability in abilities:
            if ability.ability_type != ability_type:
                continue

            # Check conditions
            if ability.conditions:
                if not any(c in context for c in ability.conditions):
                    continue

            return True

        return False

    def get_unit_info(self, unit_id: str) -> dict[str, Any] | None:
        """Get complete promotion info for a unit.

        Args:
            unit_id: Unit to check

        Returns:
            Dictionary of promotion information
        """
        if unit_id not in self._units:
            return None

        unit = self._units[unit_id]

        return {
            "unit_id": unit_id,
            "unit_class": unit.unit_class.name,
            "experience": unit.experience,
            "level": unit.level,
            "promotions": unit.promotions,
            "pending_promotion": unit.pending_promotion,
            "next_level_xp": (
                self.config.xp_per_level[unit.level]
                if unit.level < len(self.config.xp_per_level)
                else None
            ),
        }

    def remove_unit(self, unit_id: str) -> bool:
        """Remove a unit from tracking.

        Args:
            unit_id: Unit to remove

        Returns:
            True if unit was removed
        """
        if unit_id in self._units:
            del self._units[unit_id]
            return True
        return False

    def encode_for_state(self, player_id: int, unit_ids: list[str]) -> dict[str, Any]:
        """Encode promotion state for neural network input.

        Args:
            player_id: Player perspective
            unit_ids: List of unit IDs to encode

        Returns:
            Dictionary of encoded features
        """
        total_promotions = 0
        pending_promotions = 0
        avg_level = 0.0

        for unit_id in unit_ids:
            if unit_id not in self._units:
                continue

            unit = self._units[unit_id]
            total_promotions += len(unit.promotions)
            if unit.pending_promotion:
                pending_promotions += 1
            avg_level += unit.level

        num_units = len(unit_ids)
        if num_units > 0:
            avg_level /= num_units

        return {
            "total_promotions": total_promotions,
            "pending_promotions": pending_promotions,
            "avg_unit_level": avg_level,
            "units_tracked": num_units,
        }

    def reset(self) -> None:
        """Reset all promotion state."""
        self._units.clear()
        logger.debug("PromotionManager reset")
