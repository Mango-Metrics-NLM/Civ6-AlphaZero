"""
Religion System for Civ6-AlphaZero

Implements religious mechanics including founding, spreading, and beliefs.
All values are configurable - no hardcoded constants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol


class BeliefType(Enum):
    """Types of religious beliefs."""

    FOUNDER = auto()      # Bonus for founding religion
    FOLLOWER = auto()     # Bonus for followers
    WORSHIP = auto()       # Building unlocks
    ENHANCER = auto()      # Spread bonuses


class ReligiousUnitType(Enum):
    """Types of religious units."""

    MISSIONARY = auto()
    APOSTLE = auto()
    INQUISITOR = auto()
    GURU = auto()


@dataclass
class ReligionConfig:
    """Configuration for the religion system."""

    # Religion limits
    max_religions: int = 7
    max_beliefs_per_religion: int = 4

    # Faith costs
    great_prophet_faith_cost: int = 200
    missionary_faith_cost: int = 100
    apostle_faith_cost: int = 200
    inquisitor_faith_cost: int = 100

    # Religious pressure
    base_pressure_per_citizen: float = 1.0
    holy_city_pressure_multiplier: float = 4.0
    trade_route_pressure: float = 0.5
    missionary_pressure_strength: int = 100
    apostle_pressure_strength: int = 150

    # Conversion thresholds
    conversion_threshold: float = 50.0
    majority_threshold: float = 0.5

    # Unit charges
    missionary_charges: int = 3
    apostle_charges: int = 3
    inquisitor_charges: int = 3

    # Spread range
    missionary_range: int = 10
    natural_spread_range: int = 6


@dataclass
class ReligiousBelief:
    """A religious belief that provides bonuses.

    Attributes:
        name: Belief name
        belief_type: Category of belief
        effect_type: Type of bonus provided
        effect_value: Numeric bonus value
        description: Human-readable description
    """

    name: str
    belief_type: BeliefType
    effect_type: str
    effect_value: float
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "belief_type": self.belief_type.name,
            "effect_type": self.effect_type,
            "effect_value": self.effect_value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReligiousBelief:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            belief_type=BeliefType[data["belief_type"]],
            effect_type=data["effect_type"],
            effect_value=data["effect_value"],
            description=data.get("description", ""),
        )


@dataclass
class Religion:
    """A founded religion.

    Attributes:
        id: Unique religion ID
        name: Religion name
        founder_player: Player who founded
        holy_city: Position of holy city (x, y)
        beliefs: List of chosen beliefs
        founded_turn: Turn when founded
    """

    id: int
    name: str
    founder_player: int
    holy_city: tuple[int, int]
    beliefs: list[ReligiousBelief] = field(default_factory=list)
    founded_turn: int = 0

    def add_belief(self, belief: ReligiousBelief) -> bool:
        """Add a belief to this religion.

        Args:
            belief: Belief to add.

        Returns:
            True if belief was added.
        """
        # Check if belief type already exists
        for existing in self.beliefs:
            if existing.belief_type == belief.belief_type:
                return False
        self.beliefs.append(belief)
        return True

    def get_belief(self, belief_type: BeliefType) -> ReligiousBelief | None:
        """Get a belief by type."""
        for belief in self.beliefs:
            if belief.belief_type == belief_type:
                return belief
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "founder_player": self.founder_player,
            "holy_city": self.holy_city,
            "beliefs": [b.to_dict() for b in self.beliefs],
            "founded_turn": self.founded_turn,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Religion:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            founder_player=data["founder_player"],
            holy_city=tuple(data["holy_city"]),
            beliefs=[ReligiousBelief.from_dict(b) for b in data.get("beliefs", [])],
            founded_turn=data.get("founded_turn", 0),
        )


@dataclass
class ReligiousUnit:
    """A religious unit.

    Attributes:
        id: Unique unit ID
        unit_type: Type of religious unit
        player: Owner player
        position: Current position (x, y)
        religion_id: Religion this unit spreads
        charges_remaining: Spread charges left
        strength: Religious combat strength
    """

    id: int
    unit_type: ReligiousUnitType
    player: int
    position: tuple[int, int]
    religion_id: int
    charges_remaining: int
    strength: int = 100

    def use_charge(self) -> bool:
        """Use one spread charge.

        Returns:
            True if charge was available.
        """
        if self.charges_remaining > 0:
            self.charges_remaining -= 1
            return True
        return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "unit_type": self.unit_type.name,
            "player": self.player,
            "position": self.position,
            "religion_id": self.religion_id,
            "charges_remaining": self.charges_remaining,
            "strength": self.strength,
        }


@dataclass
class ReligiousPressure:
    """Religious pressure on a city.

    Attributes:
        city_position: City location
        pressure_by_religion: Pressure from each religion
        citizens_by_religion: Followers of each religion
        total_citizens: Total city population
    """

    city_position: tuple[int, int]
    pressure_by_religion: dict[int, float] = field(default_factory=dict)
    citizens_by_religion: dict[int, int] = field(default_factory=dict)
    total_citizens: int = 1

    @property
    def majority_religion(self) -> int | None:
        """Get the majority religion in this city."""
        if not self.citizens_by_religion:
            return None

        max_followers = 0
        majority = None
        for religion_id, followers in self.citizens_by_religion.items():
            if followers > max_followers:
                max_followers = followers
                majority = religion_id

        # Check if it's actually a majority
        if majority and max_followers > self.total_citizens / 2:
            return majority
        return None

    def get_religion_percentage(self, religion_id: int) -> float:
        """Get percentage of population following a religion."""
        if self.total_citizens == 0:
            return 0.0
        followers = self.citizens_by_religion.get(religion_id, 0)
        return followers / self.total_citizens

    def add_pressure(self, religion_id: int, amount: float) -> None:
        """Add religious pressure."""
        self.pressure_by_religion[religion_id] = (
            self.pressure_by_religion.get(religion_id, 0.0) + amount
        )

    def convert_citizens(self, threshold: float) -> None:
        """Convert citizens based on accumulated pressure."""
        if not self.pressure_by_religion:
            return

        total_pressure = sum(self.pressure_by_religion.values())
        if total_pressure == 0:
            return

        # Reset followers and redistribute based on pressure
        self.citizens_by_religion.clear()
        for religion_id, pressure in self.pressure_by_religion.items():
            ratio = pressure / total_pressure
            followers = int(self.total_citizens * ratio)
            if followers > 0:
                self.citizens_by_religion[religion_id] = followers

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "city_position": self.city_position,
            "pressure_by_religion": self.pressure_by_religion,
            "citizens_by_religion": self.citizens_by_religion,
            "total_citizens": self.total_citizens,
        }


class CityProtocol(Protocol):
    """Protocol for city objects."""

    position: tuple[int, int]
    player: int
    population: int


class ReligionManager:
    """Manages the religion system.

    Features:
    - Track founded religions
    - Calculate religious pressure spread
    - Manage religious units
    - Generate religion-related action masks
    """

    def __init__(self, config: ReligionConfig | None = None) -> None:
        """Initialize the religion manager.

        Args:
            config: Religion configuration.
        """
        self.config = config or ReligionConfig()
        self._religions: dict[int, Religion] = {}
        self._religious_units: dict[int, ReligiousUnit] = {}
        self._city_pressure: dict[tuple[int, int], ReligiousPressure] = {}
        self._next_religion_id: int = 1
        self._next_unit_id: int = 1
        self._player_faith: dict[int, int] = {}

    def get_founded_religions(self) -> list[Religion]:
        """Get all founded religions."""
        return list(self._religions.values())

    def can_found_religion(self, player: int) -> tuple[bool, str]:
        """Check if a player can found a religion.

        Args:
            player: Player ID.

        Returns:
            Tuple of (can_found, reason).
        """
        # Check max religions limit
        if len(self._religions) >= self.config.max_religions:
            return False, "Maximum religions reached"

        # Check if player already founded
        for religion in self._religions.values():
            if religion.founder_player == player:
                return False, "Player already founded a religion"

        # Check faith
        faith = self._player_faith.get(player, 0)
        if faith < self.config.great_prophet_faith_cost:
            return False, "Not enough faith"

        return True, ""

    def found_religion(
        self,
        player: int,
        name: str,
        holy_city_position: tuple[int, int],
        founder_belief: ReligiousBelief | None = None,
        current_turn: int = 0,
    ) -> Religion | None:
        """Found a new religion.

        Args:
            player: Founding player.
            name: Religion name.
            holy_city_position: Position of the holy city.
            founder_belief: Optional founder belief.
            current_turn: Current game turn.

        Returns:
            The founded Religion or None if failed.
        """
        can_found, _ = self.can_found_religion(player)
        if not can_found:
            return None

        religion = Religion(
            id=self._next_religion_id,
            name=name,
            founder_player=player,
            holy_city=holy_city_position,
            founded_turn=current_turn,
        )

        if founder_belief:
            religion.add_belief(founder_belief)

        self._religions[religion.id] = religion
        self._next_religion_id += 1

        # Deduct faith cost
        self._player_faith[player] = (
            self._player_faith.get(player, 0) - self.config.great_prophet_faith_cost
        )

        # Initialize holy city pressure
        self._ensure_city_pressure(holy_city_position)
        self._city_pressure[holy_city_position].add_pressure(
            religion.id,
            self.config.base_pressure_per_citizen * self.config.holy_city_pressure_multiplier * 100,
        )

        return religion

    def get_player_religion(self, player: int) -> Religion | None:
        """Get the religion founded by a player."""
        for religion in self._religions.values():
            if religion.founder_player == player:
                return religion
        return None

    def get_religion(self, religion_id: int) -> Religion | None:
        """Get a religion by ID."""
        return self._religions.get(religion_id)

    def add_faith(self, player: int, amount: int) -> None:
        """Add faith to a player."""
        self._player_faith[player] = self._player_faith.get(player, 0) + amount

    def get_faith(self, player: int) -> int:
        """Get a player's faith."""
        return self._player_faith.get(player, 0)

    def _ensure_city_pressure(self, position: tuple[int, int]) -> None:
        """Ensure a city has a pressure tracker."""
        if position not in self._city_pressure:
            self._city_pressure[position] = ReligiousPressure(city_position=position)

    def create_religious_unit(
        self,
        unit_type: ReligiousUnitType,
        player: int,
        position: tuple[int, int],
        religion_id: int,
    ) -> ReligiousUnit | None:
        """Create a religious unit.

        Args:
            unit_type: Type of unit.
            player: Owner player.
            position: Starting position.
            religion_id: Religion to spread.

        Returns:
            Created unit or None if failed.
        """
        # Check faith cost
        cost_map = {
            ReligiousUnitType.MISSIONARY: self.config.missionary_faith_cost,
            ReligiousUnitType.APOSTLE: self.config.apostle_faith_cost,
            ReligiousUnitType.INQUISITOR: self.config.inquisitor_faith_cost,
        }
        cost = cost_map.get(unit_type, 100)

        if self._player_faith.get(player, 0) < cost:
            return None

        # Get charges
        charges_map = {
            ReligiousUnitType.MISSIONARY: self.config.missionary_charges,
            ReligiousUnitType.APOSTLE: self.config.apostle_charges,
            ReligiousUnitType.INQUISITOR: self.config.inquisitor_charges,
        }
        charges = charges_map.get(unit_type, 3)

        unit = ReligiousUnit(
            id=self._next_unit_id,
            unit_type=unit_type,
            player=player,
            position=position,
            religion_id=religion_id,
            charges_remaining=charges,
        )

        self._religious_units[unit.id] = unit
        self._next_unit_id += 1
        self._player_faith[player] -= cost

        return unit

    def spread_religion(
        self,
        unit_id: int,
        target_city_position: tuple[int, int],
    ) -> bool:
        """Spread religion to a city using a religious unit.

        Args:
            unit_id: Religious unit ID.
            target_city_position: Target city position.

        Returns:
            True if spread was successful.
        """
        unit = self._religious_units.get(unit_id)
        if not unit or not unit.use_charge():
            return False

        # Calculate spread strength
        if unit.unit_type == ReligiousUnitType.MISSIONARY:
            strength = self.config.missionary_pressure_strength
        elif unit.unit_type == ReligiousUnitType.APOSTLE:
            strength = self.config.apostle_pressure_strength
        else:
            strength = self.config.missionary_pressure_strength

        # Apply pressure
        self._ensure_city_pressure(target_city_position)
        self._city_pressure[target_city_position].add_pressure(
            unit.religion_id, float(strength),
        )

        # Remove unit if out of charges
        if unit.charges_remaining <= 0:
            del self._religious_units[unit_id]

        return True

    def calculate_natural_spread(
        self,
        cities: list[CityProtocol],
    ) -> None:
        """Calculate natural religious pressure spread between cities.

        Args:
            cities: List of all cities.
        """
        for city in cities:
            self._ensure_city_pressure(city.position)
            pressure = self._city_pressure[city.position]
            pressure.total_citizens = city.population

            # Add pressure from majority religion citizens
            majority = pressure.majority_religion
            if majority:
                pressure.add_pressure(
                    majority,
                    self.config.base_pressure_per_citizen * city.population,
                )

        # Spread pressure between nearby cities
        for source_city in cities:
            source_pressure = self._city_pressure[source_city.position]
            majority = source_pressure.majority_religion
            if not majority:
                continue

            for target_city in cities:
                if source_city.position == target_city.position:
                    continue

                # Check distance
                dx = abs(source_city.position[0] - target_city.position[0])
                dy = abs(source_city.position[1] - target_city.position[1])
                distance = max(dx, dy)

                if distance <= self.config.natural_spread_range:
                    # Pressure falls off with distance
                    spread_amount = (
                        self.config.base_pressure_per_citizen *
                        source_city.population *
                        (1.0 - distance / (self.config.natural_spread_range + 1))
                    )
                    self._city_pressure[target_city.position].add_pressure(
                        majority, spread_amount,
                    )

    def process_turn(self) -> None:
        """Process end-of-turn religion updates."""
        for pressure in self._city_pressure.values():
            pressure.convert_citizens(self.config.conversion_threshold)

    def get_religious_vector(self, player: int, max_religions: int = 7) -> list[float]:
        """Get religion state as a feature vector.

        Args:
            player: Player to get features for.
            max_religions: Maximum religions for padding.

        Returns:
            Flat vector of religious features.
        """
        features: list[float] = []

        # Player's faith (normalized)
        features.append(self._player_faith.get(player, 0) / 1000.0)

        # Player's religion (one-hot)
        player_religion = self.get_player_religion(player)
        religion_onehot = [0.0] * max_religions
        if player_religion:
            religion_onehot[min(player_religion.id - 1, max_religions - 1)] = 1.0
        features.extend(religion_onehot)

        # Number of religions founded (normalized)
        features.append(len(self._religions) / max_religions)

        return features

    def get_city_religious_state(
        self,
        city_position: tuple[int, int],
        max_religions: int = 7,
    ) -> list[float]:
        """Get religious state for a specific city.

        Args:
            city_position: City position.
            max_religions: Maximum religions for padding.

        Returns:
            Vector of religious features for the city.
        """
        self._ensure_city_pressure(city_position)
        pressure = self._city_pressure[city_position]

        features: list[float] = []

        # Follower percentage for each religion
        for religion_id in range(1, max_religions + 1):
            pct = pressure.get_religion_percentage(religion_id)
            features.append(pct)

        # Is holy city
        is_holy = 0.0
        for religion in self._religions.values():
            if religion.holy_city == city_position:
                is_holy = 1.0
                break
        features.append(is_holy)

        return features

    def to_dict(self) -> dict[str, Any]:
        """Serialize religion state."""
        return {
            "religions": {
                str(k): v.to_dict() for k, v in self._religions.items()
            },
            "religious_units": {
                str(k): v.to_dict() for k, v in self._religious_units.items()
            },
            "city_pressure": {
                f"{k[0]}_{k[1]}": v.to_dict() for k, v in self._city_pressure.items()
            },
            "player_faith": self._player_faith,
            "next_religion_id": self._next_religion_id,
            "next_unit_id": self._next_unit_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: ReligionConfig | None = None,
    ) -> ReligionManager:
        """Deserialize religion state."""
        manager = cls(config=config)

        for _key, rel_data in data.get("religions", {}).items():
            religion = Religion.from_dict(rel_data)
            manager._religions[religion.id] = religion

        manager._player_faith = data.get("player_faith", {})
        manager._next_religion_id = data.get("next_religion_id", 1)
        manager._next_unit_id = data.get("next_unit_id", 1)

        return manager
