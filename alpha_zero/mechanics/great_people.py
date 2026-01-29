"""Great People System for Civ6-AlphaZero.

This module implements the Great People mechanics including:
- Great Person point accumulation
- Recruitment and competition
- Unique abilities and activations
- Era-specific Great People

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


class GreatPersonType(Enum):
    """Types of Great People."""

    GREAT_GENERAL = auto()
    GREAT_ADMIRAL = auto()
    GREAT_SCIENTIST = auto()
    GREAT_ENGINEER = auto()
    GREAT_MERCHANT = auto()
    GREAT_ARTIST = auto()
    GREAT_MUSICIAN = auto()
    GREAT_WRITER = auto()
    GREAT_PROPHET = auto()


class GreatPersonEra(Enum):
    """Historical eras for Great People."""

    ANCIENT = auto()
    CLASSICAL = auto()
    MEDIEVAL = auto()
    RENAISSANCE = auto()
    INDUSTRIAL = auto()
    MODERN = auto()
    ATOMIC = auto()
    INFORMATION = auto()


class ActivationType(Enum):
    """Types of Great Person activations."""

    INSTANT = auto()  # Immediate one-time effect
    PASSIVE = auto()  # Ongoing bonus while alive
    RETIRE = auto()  # Consume to provide effect
    AREA_EFFECT = auto()  # Affects nearby units/tiles


@dataclass
class GreatPeopleConfig:
    """Configuration for Great People mechanics.

    Attributes:
        base_points_per_district: Points generated per relevant district
        points_per_building: Additional points from buildings
        competition_visible_turns: Turns before recruitment becomes visible
        recruit_cost_gold_multiplier: Gold cost = points * multiplier
        patronage_bonus: Bonus points from city-state suzerainty
        max_active_per_type: Maximum active Great People of one type
        prophet_faith_cost_base: Base faith cost for Great Prophet
        prophet_faith_cost_increase: Cost increase per prophet recruited
    """

    base_points_per_district: float = 2.0
    points_per_building: float = 1.0
    competition_visible_turns: int = 10
    recruit_cost_gold_multiplier: float = 0.5
    patronage_bonus: float = 2.0
    max_active_per_type: int = 10
    prophet_faith_cost_base: int = 25
    prophet_faith_cost_increase: int = 15


@dataclass
class GreatPersonAbility:
    """Definition of a Great Person's ability.

    Attributes:
        ability_id: Unique identifier
        name: Ability name
        description: Human-readable description
        activation_type: How the ability is used
        effects: Dictionary of effect name to value
        charges: Number of uses (0 = unlimited/passive)
        target_required: Whether a target is needed
    """

    ability_id: str
    name: str
    description: str
    activation_type: ActivationType
    effects: dict[str, float] = field(default_factory=dict)
    charges: int = 1
    target_required: bool = False


@dataclass
class GreatPersonDefinition:
    """Definition of a specific Great Person.

    Attributes:
        person_id: Unique identifier
        name: Historical name
        person_type: Type of Great Person
        era: Historical era
        passive_bonus: Description of passive bonus
        passive_effects: Passive effect values
        active_ability: Active ability definition
        recruitment_cost: Base points needed to recruit
    """

    person_id: str
    name: str
    person_type: GreatPersonType
    era: GreatPersonEra
    passive_bonus: str = ""
    passive_effects: dict[str, float] = field(default_factory=dict)
    active_ability: GreatPersonAbility | None = None
    recruitment_cost: int = 60


# Example Great People definitions
DEFAULT_GREAT_PEOPLE: list[GreatPersonDefinition] = [
    # Great Scientists
    GreatPersonDefinition(
        person_id="euclid",
        name="Euclid",
        person_type=GreatPersonType.GREAT_SCIENTIST,
        era=GreatPersonEra.ANCIENT,
        passive_bonus="+1 Science per library",
        passive_effects={"library_science_bonus": 1.0},
        active_ability=GreatPersonAbility(
            ability_id="euclid_ability",
            name="Trigger Eureka",
            description="Triggers Eureka for Mathematics and one random tech",
            activation_type=ActivationType.RETIRE,
            effects={"eureka_triggers": 2.0},
        ),
        recruitment_cost=60,
    ),
    GreatPersonDefinition(
        person_id="hypatia",
        name="Hypatia",
        person_type=GreatPersonType.GREAT_SCIENTIST,
        era=GreatPersonEra.CLASSICAL,
        passive_bonus="+1 Science to campus buildings",
        passive_effects={"campus_building_science_bonus": 1.0},
        active_ability=GreatPersonAbility(
            ability_id="hypatia_ability",
            name="Gain Library and University",
            description="Grants a free Library and University in target campus",
            activation_type=ActivationType.RETIRE,
            effects={"free_buildings": 2.0},
            target_required=True,
        ),
        recruitment_cost=60,
    ),
    # Great Engineers
    GreatPersonDefinition(
        person_id="imhotep",
        name="Imhotep",
        person_type=GreatPersonType.GREAT_ENGINEER,
        era=GreatPersonEra.ANCIENT,
        passive_bonus="+2 Production towards wonders",
        passive_effects={"wonder_production_bonus": 2.0},
        active_ability=GreatPersonAbility(
            ability_id="imhotep_ability",
            name="Wonder Production",
            description="Grants 100 Production towards wonder construction",
            activation_type=ActivationType.RETIRE,
            effects={"wonder_production": 100.0},
            target_required=True,
            charges=1,
        ),
        recruitment_cost=60,
    ),
    # Great Generals
    GreatPersonDefinition(
        person_id="sun_tzu",
        name="Sun Tzu",
        person_type=GreatPersonType.GREAT_GENERAL,
        era=GreatPersonEra.ANCIENT,
        passive_bonus="+5 Combat Strength to nearby land units",
        passive_effects={"land_combat_aura": 5.0},
        active_ability=GreatPersonAbility(
            ability_id="sun_tzu_ability",
            name="Art of War",
            description="Grants The Art of War great work and +5 Combat to all units",
            activation_type=ActivationType.RETIRE,
            effects={"great_work": 1.0, "global_combat_bonus": 5.0},
        ),
        recruitment_cost=60,
    ),
    # Great Merchants
    GreatPersonDefinition(
        person_id="marcus_crassus",
        name="Marcus Licinius Crassus",
        person_type=GreatPersonType.GREAT_MERCHANT,
        era=GreatPersonEra.CLASSICAL,
        passive_bonus="+4 Gold from trade routes",
        passive_effects={"trade_route_gold_bonus": 4.0},
        active_ability=GreatPersonAbility(
            ability_id="crassus_ability",
            name="Gain Gold",
            description="Grants 60 Gold",
            activation_type=ActivationType.RETIRE,
            effects={"gold": 60.0},
            charges=2,
        ),
        recruitment_cost=60,
    ),
    # Great Artists
    GreatPersonDefinition(
        person_id="andrei_rublev",
        name="Andrei Rublev",
        person_type=GreatPersonType.GREAT_ARTIST,
        era=GreatPersonEra.MEDIEVAL,
        passive_bonus="+2 Culture in cities with art museums",
        passive_effects={"museum_culture_bonus": 2.0},
        active_ability=GreatPersonAbility(
            ability_id="rublev_ability",
            name="Create Religious Art",
            description="Creates Religious Art great work",
            activation_type=ActivationType.RETIRE,
            effects={"religious_great_work": 1.0},
            target_required=True,
            charges=3,
        ),
        recruitment_cost=60,
    ),
]


@dataclass
class GreatPerson:
    """An instance of a recruited Great Person.

    Attributes:
        instance_id: Unique instance identifier
        definition: The Great Person definition
        owner_id: Player who recruited this person
        position: Current position (x, y)
        charges_remaining: Remaining ability uses
        is_active: Whether still in play
        turns_alive: Turns since recruitment
    """

    instance_id: str
    definition: GreatPersonDefinition
    owner_id: int
    position: tuple[int, int] = (0, 0)
    charges_remaining: int = 0
    is_active: bool = True
    turns_alive: int = 0

    def __post_init__(self) -> None:
        """Initialize charges from ability definition."""
        if self.definition.active_ability:
            self.charges_remaining = self.definition.active_ability.charges

    @property
    def name(self) -> str:
        """Get the Great Person's name."""
        return self.definition.name

    @property
    def person_type(self) -> GreatPersonType:
        """Get the type of Great Person."""
        return self.definition.person_type


@dataclass
class RecruitmentProgress:
    """Tracks recruitment progress for a Great Person.

    Attributes:
        definition: Great Person being competed for
        player_points: Dictionary of player_id to accumulated points
        turns_available: Turns this person has been available
    """

    definition: GreatPersonDefinition
    player_points: dict[int, float] = field(default_factory=dict)
    turns_available: int = 0


class GreatPeopleManager:
    """Manages Great People mechanics.

    This class handles:
    - Point accumulation and tracking
    - Recruitment competition
    - Great Person abilities
    - Patronage and purchasing

    Example:
        >>> config = GreatPeopleConfig(base_points_per_district=2.0)
        >>> manager = GreatPeopleManager(config)
        >>> manager.add_points(player_id=0, person_type=GreatPersonType.GREAT_SCIENTIST, amount=10)
        >>> available = manager.get_available_for_recruitment(GreatPersonType.GREAT_SCIENTIST)
        >>> person = manager.recruit(player_id=0, person_id="euclid", position=(5, 5))
    """

    def __init__(
        self,
        config: GreatPeopleConfig | None = None,
        definitions: list[GreatPersonDefinition] | None = None,
    ) -> None:
        """Initialize Great People manager.

        Args:
            config: Configuration for mechanics
            definitions: List of Great Person definitions
        """
        self.config = config or GreatPeopleConfig()

        # Build definitions dictionary
        self._definitions: dict[str, GreatPersonDefinition] = {}
        for defn in (definitions or DEFAULT_GREAT_PEOPLE):
            self._definitions[defn.person_id] = defn

        # Group definitions by type and era for recruitment order
        self._by_type_era: dict[GreatPersonType, dict[GreatPersonEra, list[str]]] = {}
        for person_id, defn in self._definitions.items():
            if defn.person_type not in self._by_type_era:
                self._by_type_era[defn.person_type] = {}
            if defn.era not in self._by_type_era[defn.person_type]:
                self._by_type_era[defn.person_type][defn.era] = []
            self._by_type_era[defn.person_type][defn.era].append(person_id)

        # State tracking
        self._player_points: dict[int, dict[GreatPersonType, float]] = {}
        self._recruited: dict[str, GreatPerson] = {}  # instance_id -> GreatPerson
        self._available: dict[GreatPersonType, RecruitmentProgress] = {}
        self._recruited_definitions: set[str] = set()  # Already recruited person_ids
        self._instance_counter = 0

        # Initialize first available Great People
        self._initialize_available()

        logger.debug(f"GreatPeopleManager initialized with {len(self._definitions)} definitions")

    def _initialize_available(self) -> None:
        """Initialize the first available Great Person for each type."""
        for person_type in GreatPersonType:
            self._advance_available(person_type)

    def _advance_available(self, person_type: GreatPersonType) -> bool:
        """Make the next Great Person available for a type.

        Args:
            person_type: Type to advance

        Returns:
            True if a new person became available
        """
        if person_type not in self._by_type_era:
            return False

        # Find next unrecruted person in era order
        for era in GreatPersonEra:
            if era not in self._by_type_era[person_type]:
                continue

            for person_id in self._by_type_era[person_type][era]:
                if person_id not in self._recruited_definitions:
                    defn = self._definitions[person_id]
                    self._available[person_type] = RecruitmentProgress(
                        definition=defn,
                        player_points={},
                    )
                    logger.info(f"New {person_type.name} available: {defn.name}")
                    return True

        # No more available of this type
        if person_type in self._available:
            del self._available[person_type]

        return False

    def add_points(
        self,
        player_id: int,
        person_type: GreatPersonType,
        amount: float,
    ) -> float:
        """Add Great Person points for a player.

        Args:
            player_id: Player receiving points
            person_type: Type of points
            amount: Points to add

        Returns:
            New total points
        """
        if player_id not in self._player_points:
            self._player_points[player_id] = {}

        current = self._player_points[player_id].get(person_type, 0.0)
        new_total = current + amount
        self._player_points[player_id][person_type] = new_total

        # Also add to recruitment progress
        if person_type in self._available:
            progress = self._available[person_type]
            progress.player_points[player_id] = progress.player_points.get(player_id, 0.0) + amount

        return new_total

    def get_points(self, player_id: int, person_type: GreatPersonType) -> float:
        """Get current points for a player and type."""
        return self._player_points.get(player_id, {}).get(person_type, 0.0)

    def get_available_for_recruitment(
        self,
        person_type: GreatPersonType,
    ) -> GreatPersonDefinition | None:
        """Get the currently available Great Person for recruitment.

        Args:
            person_type: Type to check

        Returns:
            Definition of available person, or None
        """
        if person_type not in self._available:
            return None
        return self._available[person_type].definition

    def can_recruit(
        self,
        player_id: int,
        person_id: str,
    ) -> tuple[bool, str]:
        """Check if a player can recruit a Great Person.

        Args:
            player_id: Player to check
            person_id: Great Person to recruit

        Returns:
            Tuple of (can_recruit, reason)
        """
        defn = self._definitions.get(person_id)
        if not defn:
            return False, "Unknown Great Person"

        if person_id in self._recruited_definitions:
            return False, "Already recruited by another player"

        # Check if this is the currently available one
        if defn.person_type not in self._available:
            return False, f"No {defn.person_type.name} currently available"

        available = self._available[defn.person_type]
        if available.definition.person_id != person_id:
            return False, f"Not currently available (current: {available.definition.name})"

        # Check points
        player_points = available.player_points.get(player_id, 0.0)
        if player_points < defn.recruitment_cost:
            return False, f"Insufficient points ({player_points:.0f}/{defn.recruitment_cost})"

        return True, "Can recruit"

    def recruit(
        self,
        player_id: int,
        person_id: str,
        position: tuple[int, int],
    ) -> GreatPerson | None:
        """Recruit a Great Person.

        Args:
            player_id: Recruiting player
            person_id: Great Person to recruit
            position: Starting position

        Returns:
            Recruited Great Person, or None if invalid
        """
        can, reason = self.can_recruit(player_id, person_id)
        if not can:
            logger.warning(f"Cannot recruit {person_id}: {reason}")
            return None

        defn = self._definitions[person_id]

        # Create instance
        self._instance_counter += 1
        instance_id = f"gp_{self._instance_counter}"

        person = GreatPerson(
            instance_id=instance_id,
            definition=defn,
            owner_id=player_id,
            position=position,
        )

        # Update state
        self._recruited[instance_id] = person
        self._recruited_definitions.add(person_id)

        # Deduct points
        if defn.person_type in self._available:
            progress = self._available[defn.person_type]
            if player_id in progress.player_points:
                progress.player_points[player_id] -= defn.recruitment_cost
                if player_id in self._player_points:
                    self._player_points[player_id][defn.person_type] -= defn.recruitment_cost

        # Advance to next available
        self._advance_available(defn.person_type)

        logger.info(f"Player {player_id} recruited {defn.name} at {position}")
        return person

    def purchase_with_gold(
        self,
        player_id: int,
        person_id: str,
        position: tuple[int, int],
        player_gold: int,
    ) -> tuple[GreatPerson | None, int]:
        """Purchase a Great Person with gold (patronage).

        Args:
            player_id: Purchasing player
            person_id: Great Person to purchase
            position: Starting position
            player_gold: Player's available gold

        Returns:
            Tuple of (Great Person, gold cost) or (None, 0)
        """
        defn = self._definitions.get(person_id)
        if not defn:
            return None, 0

        if defn.person_type not in self._available:
            return None, 0

        available = self._available[defn.person_type]
        if available.definition.person_id != person_id:
            return None, 0

        # Calculate gold cost
        current_points = available.player_points.get(player_id, 0.0)
        points_needed = max(0, defn.recruitment_cost - current_points)
        gold_cost = int(points_needed * self.config.recruit_cost_gold_multiplier)

        if player_gold < gold_cost:
            logger.warning(f"Insufficient gold ({player_gold}/{gold_cost}) for {defn.name}")
            return None, 0

        # Grant enough points to recruit
        self.add_points(player_id, defn.person_type, points_needed)

        # Recruit
        person = self.recruit(player_id, person_id, position)
        return person, gold_cost

    def activate_ability(
        self,
        instance_id: str,
        target: tuple[int, int] | str | None = None,
    ) -> dict[str, Any] | None:
        """Activate a Great Person's ability.

        Args:
            instance_id: Instance to activate
            target: Optional target position or city name

        Returns:
            Dictionary of effects, or None if invalid
        """
        person = self._recruited.get(instance_id)
        if not person:
            return None

        if not person.is_active:
            logger.warning(f"Great Person {instance_id} is not active")
            return None

        ability = person.definition.active_ability
        if not ability:
            logger.warning(f"Great Person {instance_id} has no active ability")
            return None

        if ability.target_required and target is None:
            logger.warning("Ability requires target")
            return None

        if ability.charges > 0 and person.charges_remaining <= 0:
            logger.warning("No charges remaining")
            return None

        # Consume charge
        if ability.charges > 0:
            person.charges_remaining -= 1

        # Check if should retire
        if ability.activation_type == ActivationType.RETIRE:
            person.is_active = False

        effects: dict[str, Any] = dict(ability.effects)
        effects["person_name"] = person.name
        effects["ability_name"] = ability.name

        logger.info(f"Activated {ability.name} for {person.name}: {effects}")
        return effects

    def get_passive_effects(self, player_id: int) -> dict[str, float]:
        """Get total passive effects from all Great People.

        Args:
            player_id: Player to check

        Returns:
            Dictionary of effect name to total value
        """
        effects: dict[str, float] = {}

        for person in self._recruited.values():
            if person.owner_id != player_id or not person.is_active:
                continue

            for effect, value in person.definition.passive_effects.items():
                effects[effect] = effects.get(effect, 0.0) + value

        return effects

    def process_turn(self) -> None:
        """Process turn for Great People mechanics."""
        # Increment turns for available
        for progress in self._available.values():
            progress.turns_available += 1

        # Increment turns alive for recruited
        for person in self._recruited.values():
            if person.is_active:
                person.turns_alive += 1

    def get_player_great_people(
        self,
        player_id: int,
        active_only: bool = True,
    ) -> list[GreatPerson]:
        """Get all Great People for a player.

        Args:
            player_id: Player to check
            active_only: Only return active Great People

        Returns:
            List of Great People
        """
        people: list[GreatPerson] = []
        for person in self._recruited.values():
            if person.owner_id != player_id:
                continue
            if active_only and not person.is_active:
                continue
            people.append(person)
        return people

    def get_recruitment_progress(
        self,
        person_type: GreatPersonType,
        player_id: int,
    ) -> dict[str, Any] | None:
        """Get recruitment progress for a type.

        Args:
            person_type: Type to check
            player_id: Player perspective

        Returns:
            Progress dictionary
        """
        if person_type not in self._available:
            return None

        progress = self._available[person_type]
        player_points = progress.player_points.get(player_id, 0.0)

        # Find leader
        leader_id = None
        leader_points = 0.0
        for pid, points in progress.player_points.items():
            if points > leader_points:
                leader_points = points
                leader_id = pid

        return {
            "person_name": progress.definition.name,
            "person_type": person_type.name,
            "era": progress.definition.era.name,
            "cost": progress.definition.recruitment_cost,
            "player_points": player_points,
            "leader_player_id": leader_id,
            "leader_points": leader_points,
            "turns_available": progress.turns_available,
        }

    def encode_for_state(self, player_id: int) -> dict[str, Any]:
        """Encode Great People state for neural network input.

        Args:
            player_id: Player perspective

        Returns:
            Dictionary of encoded features
        """
        people = self.get_player_great_people(player_id)
        passive_effects = self.get_passive_effects(player_id)

        # Points by type
        points = self._player_points.get(player_id, {})

        return {
            "num_great_people": len(people),
            "generals": len([p for p in people if p.person_type == GreatPersonType.GREAT_GENERAL]),
            "scientists": len([p for p in people if p.person_type == GreatPersonType.GREAT_SCIENTIST]),
            "engineers": len([p for p in people if p.person_type == GreatPersonType.GREAT_ENGINEER]),
            "merchants": len([p for p in people if p.person_type == GreatPersonType.GREAT_MERCHANT]),
            "artists": len([p for p in people if p.person_type == GreatPersonType.GREAT_ARTIST]),
            "points_scientist": points.get(GreatPersonType.GREAT_SCIENTIST, 0.0),
            "points_engineer": points.get(GreatPersonType.GREAT_ENGINEER, 0.0),
            "points_merchant": points.get(GreatPersonType.GREAT_MERCHANT, 0.0),
            "points_general": points.get(GreatPersonType.GREAT_GENERAL, 0.0),
            "passive_effects": passive_effects,
        }

    def reset(self) -> None:
        """Reset all state."""
        self._player_points.clear()
        self._recruited.clear()
        self._available.clear()
        self._recruited_definitions.clear()
        self._instance_counter = 0
        self._initialize_available()
        logger.debug("GreatPeopleManager reset")
