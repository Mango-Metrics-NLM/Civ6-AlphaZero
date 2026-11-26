"""Entity and command dataclasses for the Civ6 bridge protocol.

Layer 1 of the split bridge protocol (depends only on Layer 0
:mod:`alpha_zero.bridge.protocol_validation` and leaf helpers; never imports
:mod:`alpha_zero.bridge.protocol`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.bridge.protocol_validation import (
    ActionType,
    ProtocolValidationError,
    _optional_int,
    _optional_number,
    _validate_dict,
    _validate_float,
    _validate_int,
    _validate_list,
    _validate_required,
    _validate_string,
    _validate_type,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CityData",
    "LegalAction",
    "LuaCommand",
    "MapData",
    "PlayerData",
    "Position",
    "ResearchStatus",
    "TileData",
    "UnitData",
]


@dataclass
class Position:
    """A position on the game map."""

    x: int
    y: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {"x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Position:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="Position.from_dict")
        x = _validate_int(data.get("x"), "x", context="Position")
        y = _validate_int(data.get("y"), "y", context="Position")
        return cls(x=x, y=y)


#: Action types whose ``item_id`` may be a game type name instead of a number.
_NAMED_ITEM_ACTIONS = frozenset(
    {ActionType.RESEARCH_TECH, ActionType.RESEARCH_CIVIC, ActionType.CITY_PRODUCE}
)
_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class LuaCommand:
    """A command to be executed by the Lua mod.

    Attributes:
        action_type: Type of action to execute.
        unit_id: Unit ID for unit actions.
        city_id: City ID for city actions.
        target: Target position for movement/attack.
        item_id: Item ID for production/research. An int, or for research the
            game's type name (``"TECH_MINING"``), which the mod resolves itself.
        item_type: Type of item (unit, building, wonder, district).
        metadata: Additional action-specific data.
    """

    action_type: ActionType
    unit_id: int | None = None
    city_id: int | None = None
    target: Position | None = None
    item_id: int | str | None = None
    item_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {"action_type": self.action_type.value}

        if self.unit_id is not None:
            result["unit_id"] = self.unit_id
        if self.city_id is not None:
            result["city_id"] = self.city_id
        if self.target is not None:
            result["target"] = self.target.to_dict()
        if self.item_id is not None:
            result["item_id"] = self.item_id
        if self.item_type is not None:
            result["item_type"] = self.item_type
        if self.metadata:
            result["metadata"] = self.metadata

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LuaCommand:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="LuaCommand.from_dict")

        action_type_raw = _validate_required(data, "action_type", context="LuaCommand")
        try:
            action_type = ActionType(action_type_raw)
        except ValueError as err:
            valid_actions = [a.value for a in ActionType]
            raise ProtocolValidationError(
                "Invalid action type",
                field_name="action_type",
                expected=f"one of {valid_actions}",
                actual=action_type_raw,
            ) from err

        target = None
        if "target" in data and data["target"] is not None:
            target_data = _validate_type(
                data["target"], dict, "target", context="LuaCommand", allow_none=True
            )
            if target_data:
                target = Position.from_dict(target_data)

        unit_id = data.get("unit_id")
        if unit_id is not None:
            unit_id = _validate_int(unit_id, "unit_id", context="LuaCommand")

        city_id = data.get("city_id")
        if city_id is not None:
            city_id = _validate_int(city_id, "city_id", context="LuaCommand")

        item_id = data.get("item_id")
        if (
            isinstance(item_id, str)
            and action_type in _NAMED_ITEM_ACTIONS
            and _TYPE_NAME.fullmatch(item_id)
        ):
            pass  # a game type name ("TECH_MINING"); the mod resolves it
        elif item_id is not None:
            item_id = _validate_int(item_id, "item_id", context="LuaCommand")

        metadata = _validate_dict(data.get("metadata"), "metadata", context="LuaCommand")

        return cls(
            action_type=action_type,
            unit_id=unit_id,
            city_id=city_id,
            target=target,
            item_id=item_id,
            item_type=data.get("item_type"),
            metadata=metadata,
        )


@dataclass
class TileData:
    """Data for a single map tile."""

    x: int
    y: int
    terrain: str
    features: list[str] = field(default_factory=list)
    resource: str | None = None
    improvement: str | None = None
    owner_id: int | None = None
    is_visible: bool = True
    is_revealed: bool = True
    yields: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "x": self.x,
            "y": self.y,
            "terrain": self.terrain,
            "is_visible": self.is_visible,
            "is_revealed": self.is_revealed,
        }
        if self.features:
            result["features"] = self.features
        if self.resource is not None:
            result["resource"] = self.resource
        if self.improvement is not None:
            result["improvement"] = self.improvement
        if self.owner_id is not None:
            result["owner_id"] = self.owner_id
        if self.yields:
            result["yields"] = self.yields
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TileData:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="TileData.from_dict")

        x = _validate_int(data.get("x"), "x", context="TileData", default=0)
        y = _validate_int(data.get("y"), "y", context="TileData", default=0)

        # Handle both 'owner' and 'owner_id' keys (consistent with UnitData/CityData)
        raw_owner = data.get("owner", data.get("owner_id", -1))
        owner_id = None
        if raw_owner is not None and raw_owner != -1:
            owner_id = _validate_int(raw_owner, "owner_id", context="TileData", default=-1)
            if owner_id == -1:
                owner_id = None

        features = _validate_list(data.get("features"), "features", context="TileData")
        yields = _validate_dict(data.get("yields"), "yields", context="TileData")

        return cls(
            x=x,
            y=y,
            terrain=_validate_string(
                data.get("terrain"), "terrain", context="TileData", default="TERRAIN_PLAINS"
            ),
            features=features,
            resource=data.get("resource"),
            improvement=data.get("improvement"),
            owner_id=owner_id,
            is_visible=bool(data.get("is_visible", True)),
            is_revealed=bool(data.get("is_revealed", True)),
            yields=yields,
        )


@dataclass
class UnitData:
    """Data for a game unit."""

    id: int
    unit_type: str
    owner_id: int
    x: int
    y: int
    health: int = 100
    max_health: int = 100
    damage: int = 0
    moves_remaining: float = 0.0
    max_moves: float = 2.0
    combat_strength: int = 0
    ranged_strength: int = 0
    range: int = 0
    is_civilian: bool = False
    can_attack: bool = True
    promotions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "unit_type": self.unit_type,
            "owner_id": self.owner_id,
            "x": self.x,
            "y": self.y,
            "health": self.health,
            "max_health": self.max_health,
            "damage": self.damage,
            "moves_remaining": self.moves_remaining,
            "max_moves": self.max_moves,
            "combat_strength": self.combat_strength,
            "ranged_strength": self.ranged_strength,
            "range": self.range,
            "is_civilian": self.is_civilian,
            "can_attack": self.can_attack,
        }
        if self.promotions:
            result["promotions"] = self.promotions
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UnitData:
        """Create from dictionary.

        Field priority (backward compatibility with Lua mod protocol versions):
            unit_type: "unit_type" (current) > "type" (legacy v1) > "type_name" (legacy v0)
        """
        _validate_type(data, dict, "data", context="UnitData.from_dict")

        unit_id = _validate_int(data.get("id"), "id", context="UnitData", default=0)
        raw_unit_type = data.get("unit_type") or data.get("type") or data.get("type_name")
        unit_type = _validate_string(
            raw_unit_type,
            "unit_type",
            context="UnitData",
            default="UNIT_WARRIOR",
        )
        owner_id = _validate_int(
            data.get("owner", data.get("owner_id")),
            "owner_id",
            context="UnitData",
            default=0,
        )
        x = _validate_int(data.get("x"), "x", context="UnitData", default=0)
        y = _validate_int(data.get("y"), "y", context="UnitData", default=0)
        health = _validate_int(
            data.get("hp", data.get("health")),
            "health",
            context="UnitData",
            default=100,
            min_val=0,
        )
        max_health = _validate_int(
            data.get("max_hp", data.get("max_health")),
            "max_health",
            context="UnitData",
            default=100,
            min_val=1,
        )
        damage = _validate_int(
            data.get("damage"), "damage", context="UnitData", default=0, min_val=0
        )
        moves_remaining = _validate_float(
            data.get("movement", data.get("moves_remaining")),
            "moves_remaining",
            context="UnitData",
            default=0.0,
        )
        max_moves = _validate_float(
            data.get("max_movement", data.get("max_moves")),
            "max_moves",
            context="UnitData",
            default=2.0,
        )
        combat_strength = _validate_int(
            data.get("combat_strength"), "combat_strength", context="UnitData", default=0
        )
        ranged_strength = _validate_int(
            data.get("ranged_strength"), "ranged_strength", context="UnitData", default=0
        )
        range_val = _validate_int(data.get("range"), "range", context="UnitData", default=0)
        promotions = _validate_list(data.get("promotions"), "promotions", context="UnitData")

        return cls(
            id=unit_id,
            unit_type=unit_type,
            owner_id=owner_id,
            x=x,
            y=y,
            health=health,
            max_health=max_health,
            damage=damage,
            moves_remaining=moves_remaining,
            max_moves=max_moves,
            combat_strength=combat_strength,
            ranged_strength=ranged_strength,
            range=range_val,
            is_civilian=bool(data.get("is_civilian", False)),
            can_attack=bool(data.get("can_attack", True)),
            promotions=promotions,
        )

    @property
    def position(self) -> tuple[int, int]:
        """Get position as tuple."""
        return (self.x, self.y)


@dataclass
class CityData:
    """Data for a city."""

    id: int
    name: str
    owner_id: int
    x: int
    y: int
    population: int = 1
    is_capital: bool = False
    defense_strength: int = 0
    buildings: list[str] = field(default_factory=list)
    districts: list[str] = field(default_factory=list)
    current_production: str | None = None
    production_progress: int = 0
    available_production: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "owner_id": self.owner_id,
            "x": self.x,
            "y": self.y,
            "population": self.population,
            "is_capital": self.is_capital,
            "defense_strength": self.defense_strength,
            "production_progress": self.production_progress,
        }
        if self.buildings:
            result["buildings"] = self.buildings
        if self.districts:
            result["districts"] = self.districts
        if self.current_production is not None:
            result["current_production"] = self.current_production
        if self.available_production:
            result["available_production"] = self.available_production
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CityData:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="CityData.from_dict")

        city_id = _validate_int(data.get("id"), "id", context="CityData", default=0)
        name = _validate_string(data.get("name"), "name", context="CityData", default="Unknown")
        owner_id = _validate_int(
            data.get("owner", data.get("owner_id")),
            "owner_id",
            context="CityData",
            default=0,
        )
        x = _validate_int(data.get("x"), "x", context="CityData", default=0)
        y = _validate_int(data.get("y"), "y", context="CityData", default=0)
        population = _validate_int(
            data.get("population"), "population", context="CityData", default=1, min_val=1
        )
        defense_strength = _validate_int(
            data.get("defense_strength", data.get("defense")),
            "defense_strength",
            context="CityData",
            default=0,
        )
        production_progress = _validate_int(
            data.get("production_progress"),
            "production_progress",
            context="CityData",
            default=0,
        )
        buildings = _validate_list(data.get("buildings"), "buildings", context="CityData")
        districts = _validate_list(data.get("districts"), "districts", context="CityData")
        available_production = _validate_list(
            data.get("available_production"), "available_production", context="CityData"
        )

        return cls(
            id=city_id,
            name=name,
            owner_id=owner_id,
            x=x,
            y=y,
            population=population,
            is_capital=bool(data.get("is_capital", False)),
            defense_strength=defense_strength,
            buildings=buildings,
            districts=districts,
            current_production=data.get("current_production"),
            production_progress=production_progress,
            available_production=available_production,
        )

    @property
    def position(self) -> tuple[int, int]:
        """Get position as tuple."""
        return (self.x, self.y)


@dataclass
class ResearchStatus:
    """The tech or civic a player is working on (state schema 2).

    Every field but ``type`` is optional: the mod leaves one out when the game
    call behind it is missing in the game's Lua context.
    """

    type: str
    index: int | None = None
    cost: float | None = None
    progress: float | None = None
    turns_left: int | None = None
    boost_triggered: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, leaving out the fields that are None."""
        result: dict[str, Any] = {"type": self.type}
        for key in ("index", "cost", "progress", "turns_left", "boost_triggered"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchStatus:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="ResearchStatus.from_dict")
        boost = data.get("boost_triggered")
        return cls(
            type=_validate_string(
                data.get("type"), "type", context="ResearchStatus", default="UNKNOWN"
            ),
            index=_optional_int(data.get("index")),
            cost=_optional_number(data.get("cost")),
            progress=_optional_number(data.get("progress")),
            turns_left=_optional_int(data.get("turns_left")),
            boost_triggered=boost if isinstance(boost, bool) else None,
        )


def _parse_research_status(value: Any) -> ResearchStatus | None:
    """A status object, or None when the payload has none or it is not an object."""
    if not isinstance(value, dict):
        return None
    return ResearchStatus.from_dict(value)


@dataclass
class PlayerData:
    """Data for a player."""

    id: int
    civilization: str
    gold: float = 0.0
    gold_per_turn: float = 0.0
    science_per_turn: float = 0.0
    culture_per_turn: float = 0.0
    faith: float = 0.0
    score: int = 0
    is_alive: bool = True
    is_human: bool = True
    techs_researched: list[str] = field(default_factory=list)
    current_research: str | None = None
    civics_researched: list[str] = field(default_factory=list)
    current_civic: str | None = None
    research_status: ResearchStatus | None = None
    civic_status: ResearchStatus | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "id": self.id,
            "civilization": self.civilization,
            "gold": self.gold,
            "gold_per_turn": self.gold_per_turn,
            "science_per_turn": self.science_per_turn,
            "culture_per_turn": self.culture_per_turn,
            "faith": self.faith,
            "score": self.score,
            "is_alive": self.is_alive,
            "is_human": self.is_human,
        }
        if self.techs_researched:
            result["techs_researched"] = self.techs_researched
        if self.current_research is not None:
            result["current_research"] = self.current_research
        if self.civics_researched:
            result["civics_researched"] = self.civics_researched
        if self.current_civic is not None:
            result["current_civic"] = self.current_civic
        if self.research_status is not None:
            result["research_status"] = self.research_status.to_dict()
        if self.civic_status is not None:
            result["civic_status"] = self.civic_status.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlayerData:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="PlayerData.from_dict")

        player_id = _validate_int(data.get("id"), "id", context="PlayerData", default=0)
        civilization = _validate_string(
            data.get("civilization", data.get("civ")),
            "civilization",
            context="PlayerData",
            default="UNKNOWN",
        )
        gold = _validate_float(data.get("gold"), "gold", context="PlayerData", default=0.0)
        gold_per_turn = _validate_float(
            data.get("gold_per_turn"), "gold_per_turn", context="PlayerData", default=0.0
        )
        science_per_turn = _validate_float(
            data.get("science_per_turn", data.get("science")),
            "science_per_turn",
            context="PlayerData",
            default=0.0,
        )
        culture_per_turn = _validate_float(
            data.get("culture_per_turn", data.get("culture")),
            "culture_per_turn",
            context="PlayerData",
            default=0.0,
        )
        faith = _validate_float(data.get("faith"), "faith", context="PlayerData", default=0.0)
        score = _validate_int(data.get("score"), "score", context="PlayerData", default=0)
        techs_researched = _validate_list(
            data.get("techs_researched"), "techs_researched", context="PlayerData"
        )
        civics_researched = _validate_list(
            data.get("civics_researched"), "civics_researched", context="PlayerData"
        )

        return cls(
            id=player_id,
            civilization=civilization,
            gold=gold,
            gold_per_turn=gold_per_turn,
            science_per_turn=science_per_turn,
            culture_per_turn=culture_per_turn,
            faith=faith,
            score=score,
            is_alive=bool(data.get("is_alive", True)),
            is_human=bool(data.get("is_human", True)),
            techs_researched=techs_researched,
            current_research=data.get("current_research"),
            civics_researched=civics_researched,
            current_civic=data.get("current_civic"),
            research_status=_parse_research_status(data.get("research_status")),
            civic_status=_parse_research_status(data.get("civic_status")),
        )


@dataclass
class LegalAction:
    """A legal action from the Lua mod."""

    action_type: str
    unit_id: int | None = None
    city_id: int | None = None
    targets: list[tuple[int, int]] = field(default_factory=list)
    items: list[int | str] = field(default_factory=list)
    item_type: str | None = None
    item_types: dict[str, str] = field(default_factory=dict)
    item_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Dictionary representation of the legal action, for JSON."""
        result: dict[str, Any] = {"action_type": self.action_type}
        if self.unit_id is not None:
            result["unit_id"] = self.unit_id
        if self.city_id is not None:
            result["city_id"] = self.city_id
        if self.targets:
            result["targets"] = [{"x": t[0], "y": t[1]} for t in self.targets]
        if self.items:
            result["items"] = self.items
        if self.item_type is not None:
            result["item_type"] = self.item_type
        if self.item_types:
            result["item_types"] = self.item_types
        if self.item_names:
            result["item_names"] = self.item_names
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LegalAction:
        """Create from dictionary.

        Field priority (backward compatibility with Lua mod protocol versions):
            action_type: "action_type" (current) > "action" (legacy v1) > "type" (legacy v0)
        """
        _validate_type(data, dict, "data", context="LegalAction.from_dict")

        targets = []
        raw_targets = data.get("targets", [])
        if raw_targets is not None:
            _validate_type(raw_targets, list, "targets", context="LegalAction")
            for i, t in enumerate(raw_targets):
                if isinstance(t, dict):
                    x = _validate_int(
                        t.get("x"), f"targets[{i}].x", context="LegalAction", default=0
                    )
                    y = _validate_int(
                        t.get("y"), f"targets[{i}].y", context="LegalAction", default=0
                    )
                    targets.append((x, y))
                elif isinstance(t, (list, tuple)) and len(t) >= 2:
                    x = _validate_int(t[0], f"targets[{i}][0]", context="LegalAction", default=0)
                    y = _validate_int(t[1], f"targets[{i}][1]", context="LegalAction", default=0)
                    targets.append((x, y))
                else:
                    logger.warning(f"Skipping invalid target format at index {i}: {t}")

        raw_items = data.get("items", data.get("item", []))
        if raw_items is None:
            raw_items = []
        elif not isinstance(raw_items, list):
            raw_items = [raw_items]

        items: list[int | str] = []
        for item in raw_items:
            if isinstance(item, int):
                items.append(item)
            elif isinstance(item, str):
                if item.isdigit() or (item.startswith("-") and item[1:].isdigit()):
                    items.append(int(item))
                else:
                    items.append(item)
            else:
                logger.warning(f"Skipping invalid item type: {type(item).__name__}")

        unit_id = data.get("unit_id")
        if unit_id is not None:
            unit_id = _validate_int(unit_id, "unit_id", context="LegalAction")

        city_id = data.get("city_id")
        if city_id is not None:
            city_id = _validate_int(city_id, "city_id", context="LegalAction")

        raw_action_type = data.get("action_type") or data.get("action") or data.get("type")
        action_type = _validate_string(
            raw_action_type,
            "action_type",
            context="LegalAction",
            default="end_turn",
        )

        item_types = _validate_dict(data.get("item_types"), "item_types", context="LegalAction")

        raw_item_names = data.get("item_names")
        item_names = (
            [name for name in raw_item_names if isinstance(name, str)]
            if isinstance(raw_item_names, list)
            else []
        )

        return cls(
            action_type=action_type,
            unit_id=unit_id,
            city_id=city_id,
            targets=targets,
            items=items,
            item_type=data.get("item_type"),
            item_types=item_types,
            item_names=item_names,
        )


@dataclass
class MapData:
    """Data for the game map."""

    width: int
    height: int
    tiles: list[TileData] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapData:
        """Create from dictionary."""
        _validate_type(data, dict, "data", context="MapData.from_dict")

        if "width" not in data:
            raise ProtocolValidationError(
                "MapData.from_dict: missing required field 'width' in payload",
                field_name="width",
                expected="int (explicit, may be 0 for pre-game)",
                actual=None,
            )
        if "height" not in data:
            raise ProtocolValidationError(
                "MapData.from_dict: missing required field 'height' in payload",
                field_name="height",
                expected="int (explicit, may be 0 for pre-game)",
                actual=None,
            )
        width = _validate_int(data["width"], "width", context="MapData", min_val=0)
        height = _validate_int(data["height"], "height", context="MapData", min_val=0)

        raw_tiles = _validate_list(data.get("tiles"), "tiles", context="MapData")
        tiles = []
        for i, t in enumerate(raw_tiles):
            try:
                tiles.append(TileData.from_dict(t))
            except ProtocolValidationError as e:
                logger.warning(f"Skipping invalid tile at index {i}: {e}")

        return cls(width=width, height=height, tiles=tiles)
