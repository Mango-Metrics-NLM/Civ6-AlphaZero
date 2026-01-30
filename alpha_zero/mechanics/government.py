"""Government and Policy System for Civ6-AlphaZero.

This module implements government types and policy cards including:
- Government types with slot configurations
- Policy cards with effects
- Government legacy bonuses
- Policy slot management

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


class GovernmentTier(Enum):
    """Government tier/era."""

    ANCIENT = auto()
    CLASSICAL = auto()
    MEDIEVAL = auto()
    RENAISSANCE = auto()
    MODERN = auto()
    INFORMATION = auto()


class PolicySlotType(Enum):
    """Types of policy card slots."""

    MILITARY = auto()
    ECONOMIC = auto()
    DIPLOMATIC = auto()
    WILDCARD = auto()


class PolicyEffectType(Enum):
    """Types of policy effects."""

    FLAT_BONUS = auto()  # +X to something
    PERCENTAGE_BONUS = auto()  # +X% to something
    YIELD_MODIFIER = auto()  # Modify yields
    UNIT_MODIFIER = auto()  # Modify unit stats
    DISCOUNT = auto()  # Reduce costs
    SPECIAL = auto()  # Custom effect


@dataclass
class GovernmentConfig:
    """Configuration for the government system.

    Attributes:
        change_cooldown_turns: Turns before can change government again
        anarchy_turns: Turns of reduced output during transition
        anarchy_reduction_per_change: Reduction per previous change
        legacy_bonus_per_turn: Legacy points per turn in government
        max_legacy_bonus: Maximum legacy bonus achievable
    """

    change_cooldown_turns: int = 10
    anarchy_turns: int = 1
    anarchy_reduction_per_change: int = 0  # Modern: no anarchy
    legacy_bonus_per_turn: float = 0.1
    max_legacy_bonus: float = 15.0


@dataclass
class PolicySlots:
    """Policy slot configuration for a government.

    Attributes:
        military: Number of military slots
        economic: Number of economic slots
        diplomatic: Number of diplomatic slots
        wildcard: Number of wildcard slots
    """

    military: int = 1
    economic: int = 1
    diplomatic: int = 0
    wildcard: int = 0

    def total(self) -> int:
        """Get total number of slots."""
        return self.military + self.economic + self.diplomatic + self.wildcard

    def get_slot_count(self, slot_type: PolicySlotType) -> int:
        """Get count for a specific slot type."""
        mapping = {
            PolicySlotType.MILITARY: self.military,
            PolicySlotType.ECONOMIC: self.economic,
            PolicySlotType.DIPLOMATIC: self.diplomatic,
            PolicySlotType.WILDCARD: self.wildcard,
        }
        return mapping.get(slot_type, 0)


@dataclass
class GovernmentType:
    """Definition of a government type.

    Attributes:
        government_id: Unique identifier
        name: Display name
        tier: Government tier/era
        slots: Policy slot configuration
        inherent_bonus: Description of inherent bonus
        inherent_effects: Dictionary of effects
        legacy_name: Name of legacy bonus
        legacy_effects: Legacy bonus effects
        required_civic: Civic required to unlock
    """

    government_id: str
    name: str
    tier: GovernmentTier
    slots: PolicySlots
    inherent_bonus: str
    inherent_effects: dict[str, float] = field(default_factory=dict)
    legacy_name: str = ""
    legacy_effects: dict[str, float] = field(default_factory=dict)
    required_civic: str | None = None


# Default government types
DEFAULT_GOVERNMENTS: dict[str, GovernmentType] = {
    "chiefdom": GovernmentType(
        government_id="chiefdom",
        name="Chiefdom",
        tier=GovernmentTier.ANCIENT,
        slots=PolicySlots(military=1, economic=1),
        inherent_bonus="No inherent bonus (starting government)",
        inherent_effects={},
    ),
    "autocracy": GovernmentType(
        government_id="autocracy",
        name="Autocracy",
        tier=GovernmentTier.ANCIENT,
        slots=PolicySlots(military=2, economic=1),
        inherent_bonus="+1 to all yields for capital",
        inherent_effects={"capital_yield_bonus": 1.0},
        legacy_name="Autocratic Legacy",
        legacy_effects={"wonder_production_bonus": 0.1},
        required_civic="Political Philosophy",
    ),
    "oligarchy": GovernmentType(
        government_id="oligarchy",
        name="Oligarchy",
        tier=GovernmentTier.ANCIENT,
        slots=PolicySlots(military=1, economic=1, wildcard=1),
        inherent_bonus="+4 Combat Strength to all melee class units",
        inherent_effects={"melee_combat_bonus": 4.0},
        legacy_name="Oligarchic Legacy",
        legacy_effects={"experience_bonus": 0.2},
        required_civic="Political Philosophy",
    ),
    "classical_republic": GovernmentType(
        government_id="classical_republic",
        name="Classical Republic",
        tier=GovernmentTier.ANCIENT,
        slots=PolicySlots(economic=2, wildcard=1),
        inherent_bonus="+1 Amenity in cities with districts",
        inherent_effects={"amenity_bonus": 1.0},
        legacy_name="Republican Legacy",
        legacy_effects={"housing_bonus": 0.15},
        required_civic="Political Philosophy",
    ),
    "monarchy": GovernmentType(
        government_id="monarchy",
        name="Monarchy",
        tier=GovernmentTier.MEDIEVAL,
        slots=PolicySlots(military=1, economic=1, diplomatic=1, wildcard=1),
        inherent_bonus="+2 Housing in cities with Medieval Walls",
        inherent_effects={"wall_housing_bonus": 2.0},
        legacy_name="Monarchic Legacy",
        legacy_effects={"influence_bonus": 0.2},
        required_civic="Divine Right",
    ),
    "merchant_republic": GovernmentType(
        government_id="merchant_republic",
        name="Merchant Republic",
        tier=GovernmentTier.MEDIEVAL,
        slots=PolicySlots(military=1, economic=2, diplomatic=1, wildcard=1),
        inherent_bonus="+2 Trade Routes",
        inherent_effects={"trade_route_bonus": 2.0},
        legacy_name="Mercantile Legacy",
        legacy_effects={"gold_bonus": 0.15},
        required_civic="Exploration",
    ),
    "democracy": GovernmentType(
        government_id="democracy",
        name="Democracy",
        tier=GovernmentTier.MODERN,
        slots=PolicySlots(military=2, economic=2, diplomatic=2, wildcard=2),
        inherent_bonus="+2 Great People points per specialty district",
        inherent_effects={"great_people_bonus": 2.0},
        legacy_name="Democratic Legacy",
        legacy_effects={"tourism_bonus": 0.2},
        required_civic="Suffrage",
    ),
    "communism": GovernmentType(
        government_id="communism",
        name="Communism",
        tier=GovernmentTier.MODERN,
        slots=PolicySlots(military=3, economic=3, wildcard=2),
        inherent_bonus="+10% Production towards all",
        inherent_effects={"production_bonus": 0.1},
        legacy_name="Communist Legacy",
        legacy_effects={"defense_bonus": 0.15},
        required_civic="Class Struggle",
    ),
    "fascism": GovernmentType(
        government_id="fascism",
        name="Fascism",
        tier=GovernmentTier.MODERN,
        slots=PolicySlots(military=4, economic=1, wildcard=3),
        inherent_bonus="+5 Combat Strength to all units",
        inherent_effects={"combat_bonus": 5.0},
        legacy_name="Fascist Legacy",
        legacy_effects={"unit_production_bonus": 0.2},
        required_civic="Totalitarianism",
    ),
}


@dataclass
class PolicyCard:
    """Definition of a policy card.

    Attributes:
        policy_id: Unique identifier
        name: Display name
        slot_type: Type of slot this card fits in
        effect_description: Human-readable effect description
        effects: Dictionary of effects
        effect_type: Type of effect
        required_civic: Civic required to unlock
        obsolete_civic: Civic that obsoletes this card
        is_dark_age: Whether this is a dark age policy
    """

    policy_id: str
    name: str
    slot_type: PolicySlotType
    effect_description: str
    effects: dict[str, float] = field(default_factory=dict)
    effect_type: PolicyEffectType = PolicyEffectType.FLAT_BONUS
    required_civic: str | None = None
    obsolete_civic: str | None = None
    is_dark_age: bool = False


# Example policy cards
DEFAULT_POLICIES: dict[str, PolicyCard] = {
    # Military Policies
    "discipline": PolicyCard(
        policy_id="discipline",
        name="Discipline",
        slot_type=PolicySlotType.MILITARY,
        effect_description="+5 Combat Strength vs Barbarians",
        effects={"barbarian_combat_bonus": 5.0},
        required_civic="Code of Laws",
    ),
    "survey": PolicyCard(
        policy_id="survey",
        name="Survey",
        slot_type=PolicySlotType.MILITARY,
        effect_description="+2 Movement for Recon units",
        effects={"recon_movement_bonus": 2.0},
        required_civic="Code of Laws",
    ),
    "maneuver": PolicyCard(
        policy_id="maneuver",
        name="Maneuver",
        slot_type=PolicySlotType.MILITARY,
        effect_description="+50% Production towards light cavalry",
        effects={"light_cavalry_production_bonus": 0.5},
        required_civic="Military Tradition",
    ),
    "conscription": PolicyCard(
        policy_id="conscription",
        name="Conscription",
        slot_type=PolicySlotType.MILITARY,
        effect_description="-1 Gold maintenance for units",
        effects={"unit_maintenance_reduction": 1.0},
        required_civic="State Workforce",
    ),
    # Economic Policies
    "god_king": PolicyCard(
        policy_id="god_king",
        name="God King",
        slot_type=PolicySlotType.ECONOMIC,
        effect_description="+1 Faith, +1 Gold in capital",
        effects={"capital_faith_bonus": 1.0, "capital_gold_bonus": 1.0},
        required_civic="Code of Laws",
    ),
    "urban_planning": PolicyCard(
        policy_id="urban_planning",
        name="Urban Planning",
        slot_type=PolicySlotType.ECONOMIC,
        effect_description="+1 Production in all cities",
        effects={"city_production_bonus": 1.0},
        required_civic="Code of Laws",
    ),
    "ilkum": PolicyCard(
        policy_id="ilkum",
        name="Ilkum",
        slot_type=PolicySlotType.ECONOMIC,
        effect_description="+30% Production towards Builders",
        effects={"builder_production_bonus": 0.3},
        required_civic="Craftsmanship",
    ),
    "colonization": PolicyCard(
        policy_id="colonization",
        name="Colonization",
        slot_type=PolicySlotType.ECONOMIC,
        effect_description="+50% Production towards Settlers",
        effects={"settler_production_bonus": 0.5},
        required_civic="Early Empire",
    ),
    # Diplomatic Policies
    "diplomatic_league": PolicyCard(
        policy_id="diplomatic_league",
        name="Diplomatic League",
        slot_type=PolicySlotType.DIPLOMATIC,
        effect_description="+1 Influence Point per turn per city-state met",
        effects={"influence_per_city_state": 1.0},
        required_civic="Political Philosophy",
    ),
    "charismatic_leader": PolicyCard(
        policy_id="charismatic_leader",
        name="Charismatic Leader",
        slot_type=PolicySlotType.DIPLOMATIC,
        effect_description="+2 Influence Points per turn",
        effects={"influence_bonus": 2.0},
        required_civic="Political Philosophy",
    ),
}


@dataclass
class ActivePolicy:
    """A policy currently in effect.

    Attributes:
        policy: The policy card
        slot_type: The slot type it's assigned to
        turns_active: How long it's been active
    """

    policy: PolicyCard
    slot_type: PolicySlotType
    turns_active: int = 0


class GovernmentManager:
    """Manages government and policy mechanics.

    This class handles:
    - Government selection and transitions
    - Policy card management
    - Effect calculations
    - Legacy bonuses

    Example:
        >>> config = GovernmentConfig(change_cooldown_turns=5)
        >>> manager = GovernmentManager(config)
        >>> manager.set_government(player_id=0, government_id="autocracy")
        >>> manager.slot_policy(player_id=0, policy_id="discipline", slot_idx=0)
        >>> effects = manager.get_total_effects(player_id=0)
    """

    def __init__(
        self,
        config: GovernmentConfig | None = None,
        governments: dict[str, GovernmentType] | None = None,
        policies: dict[str, PolicyCard] | None = None,
    ) -> None:
        """Initialize government manager.

        Args:
            config: Government configuration
            governments: Available government types
            policies: Available policy cards
        """
        self.config = config or GovernmentConfig()
        self._governments = governments or DEFAULT_GOVERNMENTS
        self._policies = policies or DEFAULT_POLICIES

        self._player_governments: dict[int, str] = {}  # player_id -> government_id
        self._player_policies: dict[
            int, dict[int, ActivePolicy]
        ] = {}  # player_id -> slot -> policy
        self._player_legacy: dict[int, dict[str, float]] = {}  # player_id -> gov_id -> bonus
        self._change_cooldowns: dict[int, int] = {}  # player_id -> turns remaining
        self._turns_in_government: dict[int, int] = {}  # player_id -> turns

        logger.debug(f"GovernmentManager initialized with {len(self._governments)} governments")

    def set_government(
        self,
        player_id: int,
        government_id: str,
        force: bool = False,
    ) -> bool:
        """Set government for a player.

        Args:
            player_id: Player ID
            government_id: ID of new government
            force: Bypass cooldown check

        Returns:
            True if government was changed
        """
        if government_id not in self._governments:
            logger.warning(f"Unknown government: {government_id}")
            return False

        # Check cooldown
        if not force:
            cooldown = self._change_cooldowns.get(player_id, 0)
            if cooldown > 0:
                logger.warning(
                    f"Player {player_id} on government change cooldown ({cooldown} turns)"
                )
                return False

        old_government = self._player_governments.get(player_id)

        # Calculate legacy bonus for old government
        if old_government:
            turns = self._turns_in_government.get(player_id, 0)
            bonus = min(
                turns * self.config.legacy_bonus_per_turn,
                self.config.max_legacy_bonus,
            )

            if player_id not in self._player_legacy:
                self._player_legacy[player_id] = {}
            self._player_legacy[player_id][old_government] = bonus

            logger.info(f"Player {player_id} legacy bonus for {old_government}: {bonus:.1f}")

        self._player_governments[player_id] = government_id
        self._turns_in_government[player_id] = 0
        self._change_cooldowns[player_id] = self.config.change_cooldown_turns

        # Clear incompatible policies
        self._clear_incompatible_policies(player_id)

        logger.info(f"Player {player_id} changed government to {government_id}")
        return True

    def _clear_incompatible_policies(self, player_id: int) -> None:
        """Clear policies that don't fit in new government's slots."""
        government_id = self._player_governments.get(player_id)
        if not government_id:
            return

        government = self._governments[government_id]
        player_policies = self._player_policies.get(player_id, {})

        # Count slots by type
        slot_counts: dict[PolicySlotType, int] = {
            PolicySlotType.MILITARY: government.slots.military,
            PolicySlotType.ECONOMIC: government.slots.economic,
            PolicySlotType.DIPLOMATIC: government.slots.diplomatic,
            PolicySlotType.WILDCARD: government.slots.wildcard,
        }

        used_counts: dict[PolicySlotType, int] = dict.fromkeys(PolicySlotType, 0)

        # Remove policies that exceed slot limits
        to_remove: list[int] = []
        for slot_idx, active_policy in player_policies.items():
            slot_type = active_policy.slot_type
            if used_counts[slot_type] < slot_counts[slot_type]:
                used_counts[slot_type] += 1
            elif slot_counts[PolicySlotType.WILDCARD] > used_counts[PolicySlotType.WILDCARD]:
                used_counts[PolicySlotType.WILDCARD] += 1
            else:
                to_remove.append(slot_idx)

        for slot_idx in to_remove:
            del player_policies[slot_idx]

        self._player_policies[player_id] = player_policies

    def slot_policy(
        self,
        player_id: int,
        policy_id: str,
        slot_idx: int,
    ) -> bool:
        """Slot a policy card.

        Args:
            player_id: Player ID
            policy_id: ID of policy to slot
            slot_idx: Slot index

        Returns:
            True if policy was slotted
        """
        if policy_id not in self._policies:
            logger.warning(f"Unknown policy: {policy_id}")
            return False

        policy = self._policies[policy_id]
        government_id = self._player_governments.get(player_id)

        if not government_id:
            logger.warning(f"Player {player_id} has no government")
            return False

        government = self._governments[government_id]

        # Check if slot is valid
        total_slots = government.slots.total()
        if slot_idx >= total_slots:
            logger.warning(f"Invalid slot index {slot_idx} (max: {total_slots - 1})")
            return False

        # Determine slot type based on government configuration
        slot_type = self._get_slot_type(government.slots, slot_idx)

        # Check if policy fits in slot (or wildcard)
        if policy.slot_type != slot_type and slot_type != PolicySlotType.WILDCARD:
            logger.warning(
                f"Policy {policy_id} ({policy.slot_type.name}) doesn't fit in "
                f"slot {slot_idx} ({slot_type.name})",
            )
            return False

        if player_id not in self._player_policies:
            self._player_policies[player_id] = {}

        self._player_policies[player_id][slot_idx] = ActivePolicy(
            policy=policy,
            slot_type=slot_type,
            turns_active=0,
        )

        logger.info(f"Player {player_id} slotted {policy_id} in slot {slot_idx}")
        return True

    def _get_slot_type(self, slots: PolicySlots, slot_idx: int) -> PolicySlotType:
        """Determine slot type from index."""
        cumulative = 0

        if slot_idx < cumulative + slots.military:
            return PolicySlotType.MILITARY
        cumulative += slots.military

        if slot_idx < cumulative + slots.economic:
            return PolicySlotType.ECONOMIC
        cumulative += slots.economic

        if slot_idx < cumulative + slots.diplomatic:
            return PolicySlotType.DIPLOMATIC
        cumulative += slots.diplomatic

        return PolicySlotType.WILDCARD

    def unslot_policy(self, player_id: int, slot_idx: int) -> bool:
        """Remove a policy from a slot.

        Args:
            player_id: Player ID
            slot_idx: Slot to clear

        Returns:
            True if policy was removed
        """
        player_policies = self._player_policies.get(player_id, {})

        if slot_idx not in player_policies:
            return False

        del player_policies[slot_idx]
        return True

    def get_total_effects(self, player_id: int) -> dict[str, float]:
        """Get combined effects from government and policies.

        Args:
            player_id: Player ID

        Returns:
            Dictionary of effect name to total value
        """
        effects: dict[str, float] = {}

        # Government inherent effects
        government_id = self._player_governments.get(player_id)
        if government_id:
            government = self._governments[government_id]
            for effect, value in government.inherent_effects.items():
                effects[effect] = effects.get(effect, 0.0) + value

            # Legacy effects from previous governments
            for gov_id, bonus_mult in self._player_legacy.get(player_id, {}).items():
                legacy_gov = self._governments.get(gov_id)
                if legacy_gov:
                    for effect, value in legacy_gov.legacy_effects.items():
                        effects[effect] = effects.get(effect, 0.0) + value * (bonus_mult / 100)

        # Policy effects
        player_policies = self._player_policies.get(player_id, {})
        for active_policy in player_policies.values():
            for effect, value in active_policy.policy.effects.items():
                effects[effect] = effects.get(effect, 0.0) + value

        return effects

    def process_turn(self) -> None:
        """Process turn for government mechanics."""
        # Update cooldowns
        for player_id in list(self._change_cooldowns.keys()):
            self._change_cooldowns[player_id] -= 1
            if self._change_cooldowns[player_id] <= 0:
                del self._change_cooldowns[player_id]

        # Update turns in government
        for player_id in self._player_governments:
            self._turns_in_government[player_id] = self._turns_in_government.get(player_id, 0) + 1

        # Update policy active turns
        for player_policies in self._player_policies.values():
            for active_policy in player_policies.values():
                active_policy.turns_active += 1

    def get_available_governments(
        self,
        player_id: int,
        unlocked_civics: set[str],
    ) -> list[str]:
        """Get governments available to a player.

        Args:
            player_id: Player ID
            unlocked_civics: Set of unlocked civic names

        Returns:
            List of available government IDs
        """
        available: list[str] = []

        for gov_id, government in self._governments.items():
            if government.required_civic is None:
                available.append(gov_id)
            elif government.required_civic in unlocked_civics:
                available.append(gov_id)

        return available

    def get_available_policies(
        self,
        player_id: int,
        unlocked_civics: set[str],
        slot_type: PolicySlotType | None = None,
    ) -> list[str]:
        """Get policies available to a player.

        Args:
            player_id: Player ID
            unlocked_civics: Set of unlocked civic names
            slot_type: Optional filter by slot type

        Returns:
            List of available policy IDs
        """
        available: list[str] = []

        for policy_id, policy in self._policies.items():
            # Check required civic
            if policy.required_civic and policy.required_civic not in unlocked_civics:
                continue

            # Check obsolete civic
            if policy.obsolete_civic and policy.obsolete_civic in unlocked_civics:
                continue

            # Check slot type filter
            if slot_type and policy.slot_type != slot_type:
                continue

            available.append(policy_id)

        return available

    def get_player_government(self, player_id: int) -> GovernmentType | None:
        """Get current government for a player."""
        gov_id = self._player_governments.get(player_id)
        return self._governments.get(gov_id) if gov_id else None

    def get_player_policies(self, player_id: int) -> list[PolicyCard]:
        """Get active policies for a player."""
        return [ap.policy for ap in self._player_policies.get(player_id, {}).values()]

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode government state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded government features
        """
        government = self.get_player_government(player_id)
        policies = self.get_player_policies(player_id)
        effects = self.get_total_effects(player_id)

        # One-hot encode government tier
        tier_encoding = [0.0] * len(GovernmentTier)
        if government:
            tier_encoding[government.tier.value - 1] = 1.0

        return {
            "government_tier": government.tier.value if government else 0,
            "government_tier_encoding": tier_encoding,
            "num_policy_slots": government.slots.total() if government else 0,
            "policies_slotted": len(policies),
            "military_policies": len(
                [p for p in policies if p.slot_type == PolicySlotType.MILITARY]
            ),
            "economic_policies": len(
                [p for p in policies if p.slot_type == PolicySlotType.ECONOMIC]
            ),
            "diplomatic_policies": len(
                [p for p in policies if p.slot_type == PolicySlotType.DIPLOMATIC]
            ),
            "legacy_bonuses": len(self._player_legacy.get(player_id, {})),
            "turns_in_government": self._turns_in_government.get(player_id, 0),
            "total_effects": effects,
        }

    def reset(self) -> None:
        """Reset all government state."""
        self._player_governments.clear()
        self._player_policies.clear()
        self._player_legacy.clear()
        self._change_cooldowns.clear()
        self._turns_in_government.clear()
        logger.debug("GovernmentManager reset")
