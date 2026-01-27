"""Base classes for game interfaces.

Provides abstract base classes and common data structures for all
game interface implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, TypeVar, Generic
import json
import logging

logger = logging.getLogger(__name__)


class InterfaceError(Exception):
    """Base exception for interface errors."""
    pass


class StateReadError(InterfaceError):
    """Raised when reading game state fails."""
    pass


class ActionSendError(InterfaceError):
    """Raised when sending an action fails."""
    pass


class ActionType(Enum):
    """Types of game actions."""
    END_TURN = "end_turn"
    MOVE = "move"
    ATTACK = "attack"
    SKIP = "skip"
    PRODUCE = "produce"
    RESEARCH = "research"
    BUILD = "build"
    SETTLE = "settle"
    DIPLOMACY = "diplomacy"


class UnitType(Enum):
    """Types of units."""
    WARRIOR = "warrior"
    SETTLER = "settler"
    ARCHER = "archer"
    SCOUT = "scout"
    BUILDER = "builder"
    CAVALRY = "cavalry"
    SIEGE = "siege"
    NAVAL = "naval"
    AIR = "air"
    CIVILIAN = "civilian"


class TerrainType(Enum):
    """Types of terrain."""
    PLAINS = "plains"
    GRASSLAND = "grassland"
    HILLS = "hills"
    MOUNTAIN = "mountain"
    DESERT = "desert"
    TUNDRA = "tundra"
    SNOW = "snow"
    COAST = "coast"
    OCEAN = "ocean"


@dataclass
class Position:
    """A position on the game map."""
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def to_tuple(self) -> tuple[int, int]:
        """Convert to tuple."""
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, t: tuple[int, int]) -> Position:
        """Create from tuple."""
        return cls(x=t[0], y=t[1])

    def distance_to(self, other: Position) -> int:
        """Calculate Manhattan distance to another position."""
        return abs(self.x - other.x) + abs(self.y - other.y)


@dataclass
class Unit:
    """Represents a game unit."""
    id: str
    type: str
    player: int
    position: Position
    health: int = 100
    movement: int = 2
    strength: int = 20
    can_attack: bool = True
    can_move: bool = True
    promotions: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by encoder."""
        return {
            "id": self.id,
            "type": self.type,
            "player": self.player,
            "position": self.position.to_tuple(),
            "health": self.health,
            "movement": self.movement,
            "strength": self.strength,
            "can_attack": self.can_attack,
            "can_move": self.can_move,
            "promotions": self.promotions,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Unit:
        """Create from dictionary."""
        pos = data.get("position", (0, 0))
        if isinstance(pos, (list, tuple)):
            position = Position(x=pos[0], y=pos[1])
        else:
            position = Position(x=0, y=0)

        return cls(
            id=str(data.get("id", "")),
            type=data.get("type", "warrior"),
            player=data.get("player", 0),
            position=position,
            health=data.get("health", 100),
            movement=data.get("movement", 2),
            strength=data.get("strength", 20),
            can_attack=data.get("can_attack", True),
            can_move=data.get("can_move", True),
            promotions=data.get("promotions", []),
            extra={k: v for k, v in data.items() if k not in [
                "id", "type", "player", "position", "health",
                "movement", "strength", "can_attack", "can_move", "promotions"
            ]},
        )


@dataclass
class City:
    """Represents a city."""
    id: str
    name: str
    player: int
    position: Position
    population: int = 1
    production: int = 0
    food: int = 0
    defense: int = 20
    districts: list[str] = field(default_factory=list)
    buildings: list[str] = field(default_factory=list)
    producing: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by encoder."""
        return {
            "id": self.id,
            "name": self.name,
            "player": self.player,
            "position": self.position.to_tuple(),
            "population": self.population,
            "production": self.production,
            "food": self.food,
            "defense": self.defense,
            "districts": self.districts,
            "buildings": self.buildings,
            "producing": self.producing,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> City:
        """Create from dictionary."""
        pos = data.get("position", (0, 0))
        if isinstance(pos, (list, tuple)):
            position = Position(x=pos[0], y=pos[1])
        else:
            position = Position(x=0, y=0)

        return cls(
            id=str(data.get("id", "")),
            name=data.get("name", "City"),
            player=data.get("player", 0),
            position=position,
            population=data.get("population", 1),
            production=data.get("production", 0),
            food=data.get("food", 0),
            defense=data.get("defense", 20),
            districts=data.get("districts", []),
            buildings=data.get("buildings", []),
            producing=data.get("producing"),
            extra={k: v for k, v in data.items() if k not in [
                "id", "name", "player", "position", "population",
                "production", "food", "defense", "districts",
                "buildings", "producing"
            ]},
        )


@dataclass
class Tile:
    """Represents a map tile."""
    position: Position
    terrain: str = "plains"
    features: list[str] = field(default_factory=list)
    resource: Optional[str] = None
    improvement: Optional[str] = None
    yields: dict[str, int] = field(default_factory=dict)
    owner: Optional[int] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.terrain,
            "features": self.features,
            "resource": self.resource,
            "improvement": self.improvement,
            "yields": self.yields,
            "owner": self.owner,
        }


@dataclass
class GameState:
    """Complete game state."""
    turn: int = 0
    current_player: int = 0
    num_players: int = 2
    map_width: int = 16
    map_height: int = 16
    units: list[Unit] = field(default_factory=list)
    cities: list[City] = field(default_factory=list)
    terrain: dict[tuple[int, int], Tile] = field(default_factory=dict)
    visibility: dict[tuple[int, int], str] = field(default_factory=dict)
    resources: dict[str, int] = field(default_factory=dict)
    technologies: list[int] = field(default_factory=list)
    civics: list[int] = field(default_factory=list)
    era: int = 0
    gold: int = 0
    gold_per_turn: int = 0
    science_per_turn: int = 0
    culture_per_turn: int = 0
    faith_per_turn: int = 0
    military_strength: int = 0
    victory_progress: dict[str, float] = field(default_factory=dict)
    game_over: bool = False
    winner: Optional[int] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_encoder_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by the encoder."""
        terrain_dict = {}
        for pos, tile in self.terrain.items():
            terrain_dict[pos] = tile.to_dict()

        return {
            "turn": self.turn,
            "current_player": self.current_player,
            "terrain": terrain_dict,
            "units": [u.to_dict() for u in self.units],
            "cities": [c.to_dict() for c in self.cities],
            "visibility": self.visibility,
            "resources": self.resources,
            "technologies": self.technologies,
            "civics": self.civics,
            "era": self.era,
            "gold": self.gold,
            "gold_per_turn": self.gold_per_turn,
            "science_per_turn": self.science_per_turn,
            "culture_per_turn": self.culture_per_turn,
            "faith_per_turn": self.faith_per_turn,
            "military_strength": self.military_strength,
            "victory_progress": self.victory_progress,
            **self.extra,
        }

    def get_unit_by_id(self, unit_id: str) -> Optional[Unit]:
        """Get a unit by its ID."""
        for unit in self.units:
            if unit.id == unit_id:
                return unit
        return None

    def get_city_by_id(self, city_id: str) -> Optional[City]:
        """Get a city by its ID."""
        for city in self.cities:
            if city.id == city_id:
                return city
        return None

    def get_units_at(self, position: Position) -> list[Unit]:
        """Get all units at a position."""
        return [u for u in self.units if u.position == position]

    def get_city_at(self, position: Position) -> Optional[City]:
        """Get city at a position."""
        for city in self.cities:
            if city.position == position:
                return city
        return None

    def get_own_units(self) -> list[Unit]:
        """Get units belonging to current player."""
        return [u for u in self.units if u.player == self.current_player]

    def get_own_cities(self) -> list[City]:
        """Get cities belonging to current player."""
        return [c for c in self.cities if c.player == self.current_player]


@dataclass
class GameAction:
    """Represents an action to be taken in the game."""
    action_type: ActionType
    unit_id: Optional[str] = None
    city_id: Optional[str] = None
    target_position: Optional[Position] = None
    target_id: Optional[str] = None
    production_type: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "type": self.action_type.value,
        }
        if self.unit_id:
            result["unit_id"] = self.unit_id
        if self.city_id:
            result["city_id"] = self.city_id
        if self.target_position:
            result["target"] = self.target_position.to_tuple()
        if self.target_id:
            result["target_id"] = self.target_id
        if self.production_type:
            result["production"] = self.production_type
        result.update(self.extra)
        return result

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameAction:
        """Create from dictionary."""
        action_type = ActionType(data.get("type", "end_turn"))

        target_pos = None
        if "target" in data:
            t = data["target"]
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                target_pos = Position(x=t[0], y=t[1])

        return cls(
            action_type=action_type,
            unit_id=data.get("unit_id"),
            city_id=data.get("city_id"),
            target_position=target_pos,
            target_id=data.get("target_id"),
            production_type=data.get("production"),
            extra={k: v for k, v in data.items() if k not in [
                "type", "unit_id", "city_id", "target",
                "target_id", "production"
            ]},
        )

    @classmethod
    def end_turn(cls) -> GameAction:
        """Create an end turn action."""
        return cls(action_type=ActionType.END_TURN)

    @classmethod
    def move(cls, unit_id: str, target: Position) -> GameAction:
        """Create a move action."""
        return cls(
            action_type=ActionType.MOVE,
            unit_id=unit_id,
            target_position=target,
        )

    @classmethod
    def attack(cls, unit_id: str, target: Position) -> GameAction:
        """Create an attack action."""
        return cls(
            action_type=ActionType.ATTACK,
            unit_id=unit_id,
            target_position=target,
        )

    @classmethod
    def skip(cls, unit_id: str) -> GameAction:
        """Create a skip action."""
        return cls(
            action_type=ActionType.SKIP,
            unit_id=unit_id,
        )

    @classmethod
    def produce(cls, city_id: str, production_type: str) -> GameAction:
        """Create a produce action."""
        return cls(
            action_type=ActionType.PRODUCE,
            city_id=city_id,
            production_type=production_type,
        )


@dataclass
class InterfaceConfig:
    """Configuration for game interfaces."""
    # File paths
    state_dir: Optional[Path] = None
    action_dir: Optional[Path] = None
    log_dir: Optional[Path] = None
    save_path: Optional[Path] = None

    # Timing
    poll_interval_ms: int = 100
    timeout_seconds: float = 30.0
    retry_count: int = 3
    retry_delay_seconds: float = 1.0

    # Validation
    validate_actions: bool = True
    validate_state: bool = True

    # Game settings
    board_size: int = 16
    max_turns: int = 500

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterfaceConfig:
        """Create from dictionary."""
        config = cls()
        if "state_dir" in data:
            config.state_dir = Path(data["state_dir"])
        if "action_dir" in data:
            config.action_dir = Path(data["action_dir"])
        if "log_dir" in data:
            config.log_dir = Path(data["log_dir"])
        if "save_path" in data:
            config.save_path = Path(data["save_path"])
        if "poll_interval_ms" in data:
            config.poll_interval_ms = int(data["poll_interval_ms"])
        if "timeout_seconds" in data:
            config.timeout_seconds = float(data["timeout_seconds"])
        if "retry_count" in data:
            config.retry_count = int(data["retry_count"])
        if "retry_delay_seconds" in data:
            config.retry_delay_seconds = float(data["retry_delay_seconds"])
        if "validate_actions" in data:
            config.validate_actions = bool(data["validate_actions"])
        if "validate_state" in data:
            config.validate_state = bool(data["validate_state"])
        if "board_size" in data:
            config.board_size = int(data["board_size"])
        if "max_turns" in data:
            config.max_turns = int(data["max_turns"])
        return config


class GameInterface(ABC):
    """Abstract base class for game interfaces.

    Provides the common interface for reading game state and sending actions
    to different game implementations (Civ6, Unciv, etc.).
    """

    def __init__(self, config: Optional[InterfaceConfig] = None):
        """Initialize the interface.

        Args:
            config: Interface configuration.
        """
        self.config = config or InterfaceConfig()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if interface is connected to game."""
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the game.

        Returns:
            True if connection successful.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the game."""
        pass

    @abstractmethod
    def read_state(self) -> GameState:
        """Read current game state.

        Returns:
            Current game state.

        Raises:
            StateReadError: If reading state fails.
        """
        pass

    @abstractmethod
    def send_action(self, action: GameAction) -> bool:
        """Send an action to the game.

        Args:
            action: Action to send.

        Returns:
            True if action was accepted.

        Raises:
            ActionSendError: If sending action fails.
        """
        pass

    @abstractmethod
    def get_valid_actions(self, state: GameState) -> list[GameAction]:
        """Get list of valid actions for current state.

        Args:
            state: Current game state.

        Returns:
            List of valid actions.
        """
        pass

    def wait_for_state_change(
        self,
        timeout_seconds: Optional[float] = None,
    ) -> Optional[GameState]:
        """Wait for game state to change.

        Args:
            timeout_seconds: Maximum time to wait.

        Returns:
            New state if changed, None if timeout.
        """
        import time

        timeout = timeout_seconds or self.config.timeout_seconds
        start_time = time.time()
        poll_interval = self.config.poll_interval_ms / 1000.0

        try:
            initial_state = self.read_state()
            initial_turn = initial_state.turn
        except StateReadError:
            return None

        while time.time() - start_time < timeout:
            time.sleep(poll_interval)
            try:
                current_state = self.read_state()
                if current_state.turn != initial_turn:
                    return current_state
            except StateReadError:
                continue

        return None

    def validate_action(self, action: GameAction, state: GameState) -> bool:
        """Validate that an action is legal in the current state.

        Args:
            action: Action to validate.
            state: Current game state.

        Returns:
            True if action is valid.
        """
        if not self.config.validate_actions:
            return True

        valid_actions = self.get_valid_actions(state)

        # Check if action matches any valid action
        for valid in valid_actions:
            if self._actions_match(action, valid):
                return True

        return False

    def _actions_match(self, a1: GameAction, a2: GameAction) -> bool:
        """Check if two actions are equivalent.

        Args:
            a1: First action.
            a2: Second action.

        Returns:
            True if actions are equivalent.
        """
        if a1.action_type != a2.action_type:
            return False
        if a1.unit_id != a2.unit_id:
            return False
        if a1.city_id != a2.city_id:
            return False
        if a1.target_position != a2.target_position:
            return False
        return True

    def __enter__(self) -> GameInterface:
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
