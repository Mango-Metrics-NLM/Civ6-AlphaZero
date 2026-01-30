"""
Culture and Tourism System for Civ6-AlphaZero

Implements cultural victory mechanics including great works, tourism, and
cultural influence. All values are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class GreatWorkType(Enum):
    """Types of great works."""

    WRITING = auto()
    MUSIC = auto()
    ART = auto()
    ARTIFACT = auto()
    RELIC = auto()


class GreatPersonType(Enum):
    """Types of great people for culture."""

    GREAT_WRITER = auto()
    GREAT_ARTIST = auto()
    GREAT_MUSICIAN = auto()


class WonderType(Enum):
    """Types of wonders (simplified)."""

    ANCIENT = auto()
    CLASSICAL = auto()
    MEDIEVAL = auto()
    RENAISSANCE = auto()
    INDUSTRIAL = auto()
    MODERN = auto()
    ATOMIC = auto()
    INFORMATION = auto()


@dataclass
class CultureConfig:
    """Configuration for the culture system."""

    # Tourism base values
    base_tourism_per_great_work: int = 2
    artifact_tourism: int = 3
    relic_tourism: int = 4
    wonder_tourism_base: int = 2

    # Theme bonuses
    theme_bonus_multiplier: float = 2.0

    # Victory thresholds
    cultural_victory_percentage: float = 1.0  # Must influence all civs

    # Domestic tourism
    domestic_tourism_per_culture: float = 0.1
    domestic_tourism_wonder_bonus: int = 4

    # Influence thresholds (cumulative tourism needed)
    influence_familiar: int = 100
    influence_exotic: int = 200
    influence_influential: int = 400
    influence_dominant: int = 600

    # Modifiers
    open_borders_tourism_modifier: float = 0.25
    trade_route_tourism_modifier: float = 0.25
    shared_religion_tourism_modifier: float = 0.5


@dataclass
class GreatWork:
    """A great work of art, writing, or music.

    Attributes:
        id: Unique ID
        work_type: Type of great work
        name: Name of the work
        creator_player: Player who created it
        era: Era when created
        tourism_value: Tourism generated
        culture_value: Culture generated
        location: Current location (building ID or None if unslotted)
    """

    id: int
    work_type: GreatWorkType
    name: str
    creator_player: int
    era: int = 0
    tourism_value: int = 2
    culture_value: int = 2
    location: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "work_type": self.work_type.name,
            "name": self.name,
            "creator_player": self.creator_player,
            "era": self.era,
            "tourism_value": self.tourism_value,
            "culture_value": self.culture_value,
            "location": self.location,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GreatWork:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            work_type=GreatWorkType[data["work_type"]],
            name=data["name"],
            creator_player=data["creator_player"],
            era=data.get("era", 0),
            tourism_value=data.get("tourism_value", 2),
            culture_value=data.get("culture_value", 2),
            location=data.get("location"),
        )


@dataclass
class TourismSource:
    """A source of tourism.

    Attributes:
        source_type: Type of tourism source
        source_id: ID of the source (great work, wonder, etc.)
        tourism_per_turn: Tourism generated per turn
        modifiers: Active modifiers
    """

    source_type: str
    source_id: int
    tourism_per_turn: int
    modifiers: dict[str, float] = field(default_factory=dict)

    @property
    def modified_tourism(self) -> float:
        """Get tourism with all modifiers applied."""
        total = float(self.tourism_per_turn)
        for modifier in self.modifiers.values():
            total *= 1.0 + modifier
        return total


@dataclass
class CulturalInfluence:
    """Cultural influence on another civilization.

    Attributes:
        source_player: Player generating influence
        target_player: Player being influenced
        cumulative_tourism: Total tourism accumulated
        influence_level: Current influence level (0-4)
    """

    source_player: int
    target_player: int
    cumulative_tourism: float = 0.0
    influence_level: int = 0  # 0=unknown, 1=exotic, 2=familiar, 3=influential, 4=dominant

    def add_tourism(self, amount: float) -> bool:
        """Add tourism and check for level up.

        Args:
            amount: Tourism to add.

        Returns:
            True if influence level increased.
        """
        self.cumulative_tourism += amount
        old_level = self.influence_level
        self._update_level()
        return self.influence_level > old_level

    def _update_level(self) -> None:
        """Update influence level based on cumulative tourism."""
        # Thresholds from config would be injected here
        # Using defaults for now
        if self.cumulative_tourism >= 600:
            self.influence_level = 4
        elif self.cumulative_tourism >= 400:
            self.influence_level = 3
        elif self.cumulative_tourism >= 200:
            self.influence_level = 2
        elif self.cumulative_tourism >= 100:
            self.influence_level = 1
        else:
            self.influence_level = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_player": self.source_player,
            "target_player": self.target_player,
            "cumulative_tourism": self.cumulative_tourism,
            "influence_level": self.influence_level,
        }


@dataclass
class CulturalVictoryProgress:
    """Tracks progress toward cultural victory.

    Attributes:
        player: Player being tracked
        domestic_tourists: Number of domestic tourists
        foreign_tourists: Tourists from each other civ
        influence_on_others: Influence level on each other civ
        is_victory_achieved: Whether victory conditions are met
    """

    player: int
    domestic_tourists: float = 0.0
    foreign_tourists: dict[int, float] = field(default_factory=dict)
    influence_on_others: dict[int, int] = field(default_factory=dict)
    is_victory_achieved: bool = False

    def get_total_foreign_tourists(self) -> float:
        """Get total tourists attracted from other civs."""
        return sum(self.foreign_tourists.values())

    def check_victory(self, other_players_domestic: dict[int, float]) -> bool:
        """Check if cultural victory is achieved.

        Victory requires more foreign tourists than any other civ's domestic tourists.

        Args:
            other_players_domestic: Domestic tourist counts for other players.

        Returns:
            True if victory achieved.
        """
        total_foreign = self.get_total_foreign_tourists()

        for _other_player, their_domestic in other_players_domestic.items():
            if total_foreign <= their_domestic:
                self.is_victory_achieved = False
                return False

        self.is_victory_achieved = True
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player": self.player,
            "domestic_tourists": self.domestic_tourists,
            "foreign_tourists": self.foreign_tourists,
            "influence_on_others": self.influence_on_others,
            "is_victory_achieved": self.is_victory_achieved,
        }


class CultureManager:
    """Manages the culture and tourism system.

    Features:
    - Track great works and their locations
    - Calculate tourism per turn
    - Track cultural influence between civs
    - Determine cultural victory progress
    """

    def __init__(self, config: CultureConfig | None = None) -> None:
        """Initialize the culture manager.

        Args:
            config: Culture configuration.
        """
        self.config = config or CultureConfig()
        self._great_works: dict[int, GreatWork] = {}
        self._player_culture: dict[int, float] = {}
        self._player_tourism: dict[int, float] = {}
        self._influences: dict[tuple[int, int], CulturalInfluence] = {}
        self._victory_progress: dict[int, CulturalVictoryProgress] = {}
        self._next_work_id: int = 1

    def create_great_work(
        self,
        work_type: GreatWorkType,
        name: str,
        creator_player: int,
        era: int = 0,
    ) -> GreatWork:
        """Create a new great work.

        Args:
            work_type: Type of great work.
            name: Name of the work.
            creator_player: Player who created it.
            era: Era when created.

        Returns:
            The created GreatWork.
        """
        # Determine tourism based on type
        tourism_map = {
            GreatWorkType.WRITING: self.config.base_tourism_per_great_work,
            GreatWorkType.MUSIC: self.config.base_tourism_per_great_work,
            GreatWorkType.ART: self.config.base_tourism_per_great_work,
            GreatWorkType.ARTIFACT: self.config.artifact_tourism,
            GreatWorkType.RELIC: self.config.relic_tourism,
        }
        tourism = tourism_map.get(work_type, self.config.base_tourism_per_great_work)

        work = GreatWork(
            id=self._next_work_id,
            work_type=work_type,
            name=name,
            creator_player=creator_player,
            era=era,
            tourism_value=tourism,
            culture_value=tourism,  # Culture equals tourism for simplicity
        )

        self._great_works[work.id] = work
        self._next_work_id += 1

        return work

    def get_player_great_works(self, player: int) -> list[GreatWork]:
        """Get all great works created by a player."""
        return [w for w in self._great_works.values() if w.creator_player == player]

    def get_great_work(self, work_id: int) -> GreatWork | None:
        """Get a great work by ID."""
        return self._great_works.get(work_id)

    def add_culture(self, player: int, amount: float) -> None:
        """Add culture to a player."""
        self._player_culture[player] = self._player_culture.get(player, 0) + amount

    def get_culture(self, player: int) -> float:
        """Get a player's total culture."""
        return self._player_culture.get(player, 0)

    def calculate_tourism_per_turn(self, player: int) -> float:
        """Calculate a player's tourism per turn.

        Args:
            player: Player ID.

        Returns:
            Total tourism per turn.
        """
        total = 0.0

        # Tourism from great works
        for work in self.get_player_great_works(player):
            total += work.tourism_value

        # Store for tracking
        self._player_tourism[player] = total

        return total

    def get_influence(self, source: int, target: int) -> CulturalInfluence:
        """Get influence from source player on target player.

        Creates new influence tracker if none exists.

        Args:
            source: Influencing player.
            target: Influenced player.

        Returns:
            CulturalInfluence object.
        """
        key = (source, target)
        if key not in self._influences:
            self._influences[key] = CulturalInfluence(
                source_player=source,
                target_player=target,
            )
        return self._influences[key]

    def apply_tourism(
        self,
        source_player: int,
        target_players: list[int],
        modifiers: dict[int, float] | None = None,
    ) -> dict[int, bool]:
        """Apply a player's tourism to other players.

        Args:
            source_player: Player generating tourism.
            target_players: Players to influence.
            modifiers: Per-player tourism modifiers.

        Returns:
            Dictionary mapping target player to whether influence leveled up.
        """
        base_tourism = self.calculate_tourism_per_turn(source_player)
        modifiers = modifiers or {}
        results: dict[int, bool] = {}

        for target in target_players:
            if target == source_player:
                continue

            # Apply modifiers
            modifier = modifiers.get(target, 0.0)
            tourism = base_tourism * (1.0 + modifier)

            # Apply to influence
            influence = self.get_influence(source_player, target)
            leveled_up = influence.add_tourism(tourism)
            results[target] = leveled_up

        return results

    def get_victory_progress(self, player: int) -> CulturalVictoryProgress:
        """Get cultural victory progress for a player.

        Args:
            player: Player ID.

        Returns:
            CulturalVictoryProgress object.
        """
        if player not in self._victory_progress:
            self._victory_progress[player] = CulturalVictoryProgress(player=player)

        progress = self._victory_progress[player]

        # Update domestic tourists (from accumulated culture)
        progress.domestic_tourists = (
            self._player_culture.get(player, 0) * self.config.domestic_tourism_per_culture
        )

        # Update foreign tourists and influence
        for key, influence in self._influences.items():
            if key[0] == player:
                target = key[1]
                # Foreign tourists are proportional to influence
                progress.foreign_tourists[target] = influence.cumulative_tourism * 0.1
                progress.influence_on_others[target] = influence.influence_level

        return progress

    def check_cultural_victory(self, player: int, all_players: list[int]) -> bool:
        """Check if a player has achieved cultural victory.

        Args:
            player: Player to check.
            all_players: List of all player IDs.

        Returns:
            True if cultural victory achieved.
        """
        progress = self.get_victory_progress(player)

        # Get domestic tourists for all other players
        other_domestic = {}
        for other in all_players:
            if other != player:
                other_progress = self.get_victory_progress(other)
                other_domestic[other] = other_progress.domestic_tourists

        return progress.check_victory(other_domestic)

    def process_turn(self, all_players: list[int]) -> None:
        """Process end-of-turn culture updates.

        Args:
            all_players: List of all player IDs.
        """
        for player in all_players:
            # Apply tourism
            targets = [p for p in all_players if p != player]
            self.apply_tourism(player, targets)

    def get_cultural_vector(
        self,
        player: int,
        num_players: int,
        max_great_works: int = 20,
    ) -> list[float]:
        """Get culture state as a feature vector.

        Args:
            player: Player to get features for.
            num_players: Total number of players.
            max_great_works: Max great works for normalization.

        Returns:
            Flat vector of cultural features.
        """
        features: list[float] = []

        # Player's culture (normalized)
        features.append(min(self._player_culture.get(player, 0) / 10000.0, 1.0))

        # Player's tourism per turn (normalized)
        tourism = self.calculate_tourism_per_turn(player)
        features.append(min(tourism / 100.0, 1.0))

        # Number of great works (normalized)
        num_works = len(self.get_player_great_works(player))
        features.append(num_works / max_great_works)

        # Influence level on each other player (one-hot per level)
        for target in range(num_players):
            if target == player:
                features.extend([0.0] * 5)  # 5 influence levels
                continue

            influence = self.get_influence(player, target)
            level_onehot = [0.0] * 5
            level_onehot[min(influence.influence_level, 4)] = 1.0
            features.extend(level_onehot)

        return features

    def to_dict(self) -> dict[str, Any]:
        """Serialize culture state."""
        return {
            "great_works": {str(k): v.to_dict() for k, v in self._great_works.items()},
            "player_culture": self._player_culture,
            "player_tourism": self._player_tourism,
            "influences": {f"{k[0]}_{k[1]}": v.to_dict() for k, v in self._influences.items()},
            "next_work_id": self._next_work_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: CultureConfig | None = None,
    ) -> CultureManager:
        """Deserialize culture state."""
        manager = cls(config=config)

        for _key, work_data in data.get("great_works", {}).items():
            work = GreatWork.from_dict(work_data)
            manager._great_works[work.id] = work

        manager._player_culture = {int(k): v for k, v in data.get("player_culture", {}).items()}
        manager._player_tourism = {int(k): v for k, v in data.get("player_tourism", {}).items()}
        manager._next_work_id = data.get("next_work_id", 1)

        return manager
