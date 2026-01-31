"""Unciv JSON game interface for Civ6-AlphaZero.

This module provides an interface to interact with Unciv, an open-source
Civilization V clone that uses JSON for game state serialization.

The interface supports:
- Loading and saving Unciv game states from JSON files
- Converting between Unciv JSON format and internal encoder format
- Generating valid action masks from game state
- Applying actions and updating game state

Unciv GitHub: https://github.com/yairm210/Unciv
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)


# =============================================================================
# Enums for Unciv game elements
# =============================================================================


class UncivTerrainType(str, Enum):
    """Terrain base types in Unciv."""

    GRASSLAND = "Grassland"
    PLAINS = "Plains"
    DESERT = "Desert"
    TUNDRA = "Tundra"
    SNOW = "Snow"
    COAST = "Coast"
    OCEAN = "Ocean"
    LAKES = "Lakes"


class UncivTerrainFeature(str, Enum):
    """Terrain features in Unciv."""

    FOREST = "Forest"
    JUNGLE = "Jungle"
    MARSH = "Marsh"
    FLOOD_PLAINS = "Flood plains"
    OASIS = "Oasis"
    HILLS = "Hill"
    MOUNTAIN = "Mountain"
    ICE = "Ice"
    FALLOUT = "Fallout"
    ATOLL = "Atoll"


class UncivResource(str, Enum):
    """Resources in Unciv."""

    # Strategic
    HORSES = "Horses"
    IRON = "Iron"
    COAL = "Coal"
    OIL = "Oil"
    ALUMINUM = "Aluminum"
    URANIUM = "Uranium"

    # Luxury
    GOLD_RES = "Gold"
    SILVER = "Silver"
    GEMS = "Gems"
    MARBLE = "Marble"
    IVORY = "Ivory"
    PEARLS = "Pearls"
    SILK = "Silk"
    SPICES = "Spices"
    SUGAR = "Sugar"
    COTTON = "Cotton"
    WINE = "Wine"
    INCENSE = "Incense"
    DYES = "Dyes"
    FURS = "Furs"
    WHALES = "Whales"
    CITRUS = "Citrus"
    TRUFFLES = "Truffles"
    CRAB = "Crab"
    COPPER = "Copper"
    SALT = "Salt"
    COCOA = "Cocoa"
    PORCELAIN = "Porcelain"
    JEWELRY = "Jewelry"
    NUTMEG = "Nutmeg"
    CLOVES = "Cloves"
    PEPPER = "Pepper"

    # Bonus
    WHEAT = "Wheat"
    CATTLE = "Cattle"
    SHEEP = "Sheep"
    DEER = "Deer"
    BANANAS = "Bananas"
    FISH = "Fish"
    STONE = "Stone"
    BISON = "Bison"


class UncivUnitType(str, Enum):
    """Unit types in Unciv."""

    # Land Melee
    WARRIOR = "Warrior"
    SPEARMAN = "Spearman"
    SWORDSMAN = "Swordsman"
    PIKEMAN = "Pikeman"
    LONGSWORDSMAN = "Longswordsman"
    MUSKETMAN = "Musketman"
    RIFLEMAN = "Rifleman"
    INFANTRY = "Infantry"
    MECHANIZED_INFANTRY = "Mechanized Infantry"

    # Land Ranged
    ARCHER = "Archer"
    CHARIOT_ARCHER = "Chariot Archer"
    COMPOSITE_BOWMAN = "Composite Bowman"
    CROSSBOWMAN = "Crossbowman"
    GATLING_GUN = "Gatling Gun"
    MACHINE_GUN = "Machine Gun"

    # Mounted
    HORSEMAN = "Horseman"
    KNIGHT = "Knight"
    CAVALRY = "Cavalry"
    TANK = "Tank"
    MODERN_ARMOR = "Modern Armor"

    # Siege
    CATAPULT = "Catapult"
    TREBUCHET = "Trebuchet"
    CANNON = "Cannon"
    ARTILLERY = "Artillery"
    ROCKET_ARTILLERY = "Rocket Artillery"

    # Naval
    TRIREME = "Trireme"
    CARAVEL = "Caravel"
    FRIGATE = "Frigate"
    BATTLESHIP = "Battleship"
    CARRIER = "Carrier"
    SUBMARINE = "Submarine"
    DESTROYER = "Destroyer"

    # Civilian
    SETTLER = "Settler"
    WORKER = "Worker"
    WORK_BOAT = "Work Boats"
    GREAT_GENERAL = "Great General"
    GREAT_PROPHET = "Great Prophet"
    GREAT_ENGINEER = "Great Engineer"
    GREAT_MERCHANT = "Great Merchant"
    GREAT_SCIENTIST = "Great Scientist"
    GREAT_ARTIST = "Great Artist"
    GREAT_MUSICIAN = "Great Musician"
    GREAT_WRITER = "Great Writer"

    # Air
    FIGHTER = "Fighter"
    BOMBER = "Bomber"
    JET_FIGHTER = "Jet Fighter"
    STEALTH_BOMBER = "Stealth Bomber"


class UncivImprovementType(str, Enum):
    """Improvement types in Unciv."""

    FARM = "Farm"
    MINE = "Mine"
    TRADING_POST = "Trading post"
    LUMBER_MILL = "Lumber mill"
    PLANTATION = "Plantation"
    PASTURE = "Pasture"
    CAMP = "Camp"
    QUARRY = "Quarry"
    FISHING_BOATS = "Fishing Boats"
    OIL_WELL = "Oil well"
    OFFSHORE_PLATFORM = "Offshore Platform"
    FORT = "Fort"
    CITADEL = "Citadel"
    ROADS = "Road"
    RAILROAD = "Railroad"
    LANDMARK = "Landmark"
    ACADEMY = "Academy"
    CUSTOMS_HOUSE = "Customs house"
    MANUFACTORY = "Manufactory"
    HOLY_SITE = "Holy site"


# =============================================================================
# Dataclasses for Unciv game elements
# =============================================================================


@dataclass
class UncivTile:
    """Represents a single tile in Unciv.

    Attributes:
        position: (x, y) coordinates of the tile
        base_terrain: Base terrain type (e.g., Grassland, Plains)
        terrain_features: List of terrain features (e.g., Hills, Forest)
        resource: Resource on the tile, if any
        improvement: Improvement built on the tile, if any
        road_status: Type of road (none, road, railroad)
        owner: Civilization that owns this tile, if any
        worked_by_city: Name of city working this tile, if any
        military_unit: Military unit on this tile, if any
        civilian_unit: Civilian unit on this tile, if any
    """

    position: tuple[int, int]
    base_terrain: str = "Grassland"
    terrain_features: list[str] = field(default_factory=list)
    resource: str | None = None
    improvement: str | None = None
    road_status: str = "None"
    owner: str | None = None
    worked_by_city: str | None = None
    military_unit: str | None = None
    civilian_unit: str | None = None

    @classmethod
    def from_unciv_dict(cls, tile_data: dict, position: tuple[int, int]) -> UncivTile:
        """Create UncivTile from Unciv JSON tile data.

        Args:
            tile_data: Dictionary from Unciv JSON
            position: (x, y) position of the tile

        Returns:
            UncivTile instance
        """
        return cls(
            position=position,
            base_terrain=tile_data.get("baseTerrain", "Grassland"),
            terrain_features=tile_data.get("terrainFeatures", []),
            resource=tile_data.get("resource"),
            improvement=tile_data.get("improvement"),
            road_status=tile_data.get("roadStatus", "None"),
            owner=tile_data.get("owningCity"),
            worked_by_city=tile_data.get("workingCity"),
            military_unit=tile_data.get("militaryUnit", {}).get("name")
            if tile_data.get("militaryUnit")
            else None,
            civilian_unit=tile_data.get("civilianUnit", {}).get("name")
            if tile_data.get("civilianUnit")
            else None,
        )

    def to_unciv_dict(self) -> dict:
        """Convert to Unciv JSON format.

        Returns:
            Dictionary in Unciv JSON format
        """
        result: dict[str, Any] = {
            "baseTerrain": self.base_terrain,
        }
        if self.terrain_features:
            result["terrainFeatures"] = self.terrain_features
        if self.resource:
            result["resource"] = self.resource
        if self.improvement:
            result["improvement"] = self.improvement
        if self.road_status != "None":
            result["roadStatus"] = self.road_status
        if self.owner:
            result["owningCity"] = self.owner
        if self.worked_by_city:
            result["workingCity"] = self.worked_by_city
        return result


@dataclass
class UncivUnit:
    """Represents a unit in Unciv.

    Attributes:
        id: Unique identifier for the unit
        name: Unit type name (e.g., "Warrior", "Settler")
        owner: Civilization that owns this unit
        position: (x, y) position on the map
        health: Current health (0-100)
        current_movement: Movement points remaining
        max_movement: Maximum movement points
        strength: Combat strength (0 for civilians)
        ranged_strength: Ranged combat strength (0 if not ranged)
        range: Attack range (1 for melee)
        promotions: List of promotions the unit has
        is_embarked: Whether the unit is on water
        action: Current action (e.g., "Fortify", "Sleep")
    """

    id: str
    name: str
    owner: str
    position: tuple[int, int]
    health: int = 100
    current_movement: float = 2.0
    max_movement: float = 2.0
    strength: int = 0
    ranged_strength: int = 0
    range: int = 1
    promotions: list[str] = field(default_factory=list)
    is_embarked: bool = False
    action: str | None = None

    @classmethod
    def from_unciv_dict(cls, unit_data: dict) -> UncivUnit:
        """Create UncivUnit from Unciv JSON unit data.

        Args:
            unit_data: Dictionary from Unciv JSON

        Returns:
            UncivUnit instance
        """
        position = unit_data.get("currentTile", {})
        x = position.get("x", 0) if isinstance(position, dict) else 0
        y = position.get("y", 0) if isinstance(position, dict) else 0

        return cls(
            id=unit_data.get("id", str(id(unit_data))),
            name=unit_data.get("name", "Unknown"),
            owner=unit_data.get("owner", "Unknown"),
            position=(x, y),
            health=unit_data.get("health", 100),
            current_movement=unit_data.get("currentMovement", 2.0),
            max_movement=unit_data.get("maxMovement", 2.0),
            strength=unit_data.get("baseUnit", {}).get("strength", 0),
            ranged_strength=unit_data.get("baseUnit", {}).get("rangedStrength", 0),
            range=unit_data.get("baseUnit", {}).get("range", 1),
            promotions=unit_data.get("promotions", {}).get("promotions", []),
            is_embarked=unit_data.get("isEmbarked", False),
            action=unit_data.get("action"),
        )

    def to_unciv_dict(self) -> dict:
        """Convert to Unciv JSON format.

        Returns:
            Dictionary in Unciv JSON format
        """
        return {
            "id": self.id,
            "name": self.name,
            "owner": self.owner,
            "currentTile": {"x": self.position[0], "y": self.position[1]},
            "health": self.health,
            "currentMovement": self.current_movement,
            "promotions": {"promotions": self.promotions},
            "isEmbarked": self.is_embarked,
            "action": self.action,
        }

    @property
    def is_civilian(self) -> bool:
        """Check if this is a civilian unit."""
        civilian_types = {
            "Settler",
            "Worker",
            "Work Boats",
            "Great General",
            "Great Prophet",
            "Great Engineer",
            "Great Merchant",
            "Great Scientist",
            "Great Artist",
            "Great Musician",
            "Great Writer",
        }
        return self.name in civilian_types or self.strength == 0

    @property
    def has_moves(self) -> bool:
        """Check if the unit has movement points remaining."""
        return self.current_movement > 0


@dataclass
class UncivCity:
    """Represents a city in Unciv.

    Attributes:
        name: City name
        owner: Civilization that owns this city
        position: (x, y) position on the map
        population: City population
        health: Current city health/defense
        is_capital: Whether this is the civilization's capital
        is_being_razed: Whether the city is being razed
        is_puppet: Whether this is a puppet city
        current_construction: What is being built
        production_stored: Production accumulated for current item
        worked_tiles: List of (x, y) positions of worked tiles
        buildings: List of building names in the city
        religion: Primary religion, if any
        food_stored: Food accumulated towards growth
    """

    name: str
    owner: str
    position: tuple[int, int]
    population: int = 1
    health: int = 200
    is_capital: bool = False
    is_being_razed: bool = False
    is_puppet: bool = False
    current_construction: str | None = None
    production_stored: int = 0
    worked_tiles: list[tuple[int, int]] = field(default_factory=list)
    buildings: list[str] = field(default_factory=list)
    religion: str | None = None
    food_stored: int = 0

    @classmethod
    def from_unciv_dict(cls, city_data: dict) -> UncivCity:
        """Create UncivCity from Unciv JSON city data.

        Args:
            city_data: Dictionary from Unciv JSON

        Returns:
            UncivCity instance
        """
        location = city_data.get("location", {})
        x = location.get("x", 0)
        y = location.get("y", 0)

        worked_tiles = []
        for tile in city_data.get("workedTiles", []):
            if isinstance(tile, dict):
                worked_tiles.append((tile.get("x", 0), tile.get("y", 0)))

        return cls(
            name=city_data.get("name", "Unknown"),
            owner=city_data.get("civName", city_data.get("foundingCiv", "Unknown")),
            position=(x, y),
            population=city_data.get("population", 1),
            health=city_data.get("health", 200),
            is_capital=city_data.get("isCapital", False),
            is_being_razed=city_data.get("isBeingRazed", False),
            is_puppet=city_data.get("isPuppet", False),
            current_construction=city_data.get("cityConstructions", {}).get(
                "currentConstructionFromQueue",
            ),
            production_stored=city_data.get("cityConstructions", {}).get(
                "productionOverflow",
                0,
            ),
            worked_tiles=worked_tiles,
            buildings=city_data.get("cityConstructions", {}).get("builtBuildings", []),
            religion=city_data.get("religion", {}).get("religionThisIsTheHolyCityOf"),
            food_stored=city_data.get("population", {}).get("foodStored", 0)
            if isinstance(city_data.get("population"), dict)
            else 0,
        )

    def to_unciv_dict(self) -> dict:
        """Convert to Unciv JSON format.

        Returns:
            Dictionary in Unciv JSON format
        """
        return {
            "name": self.name,
            "civName": self.owner,
            "location": {"x": self.position[0], "y": self.position[1]},
            "population": self.population,
            "health": self.health,
            "isCapital": self.is_capital,
            "isBeingRazed": self.is_being_razed,
            "isPuppet": self.is_puppet,
            "cityConstructions": {
                "currentConstructionFromQueue": self.current_construction,
                "productionOverflow": self.production_stored,
                "builtBuildings": self.buildings,
            },
            "workedTiles": [{"x": x, "y": y} for x, y in self.worked_tiles],
        }


@dataclass
class UncivPlayer:
    """Represents a player/civilization in Unciv.

    Attributes:
        civ_name: Civilization name (e.g., "Rome", "Greece")
        player_type: "Human" or "AI"
        gold: Current gold stockpile
        gold_per_turn: Gold income per turn
        science: Current science stockpile
        culture: Current culture
        faith: Current faith
        happiness: Current happiness
        tech_in_progress: Technology being researched
        tech_progress: Beakers accumulated for current tech
        researched_techs: List of researched technology names
        policies_adopted: List of adopted social policies
        cities: List of city names owned
        units: List of unit IDs owned
        is_alive: Whether this civilization is still in the game
        is_at_war_with: List of civilizations at war with
        has_met: List of civilizations encountered
    """

    civ_name: str
    player_type: str = "AI"
    gold: int = 0
    gold_per_turn: int = 0
    science: int = 0
    culture: int = 0
    faith: int = 0
    happiness: int = 0
    tech_in_progress: str | None = None
    tech_progress: int = 0
    researched_techs: list[str] = field(default_factory=list)
    policies_adopted: list[str] = field(default_factory=list)
    cities: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    is_alive: bool = True
    is_at_war_with: list[str] = field(default_factory=list)
    has_met: list[str] = field(default_factory=list)

    @classmethod
    def from_unciv_dict(cls, civ_data: dict) -> UncivPlayer:
        """Create UncivPlayer from Unciv JSON civilization data.

        Args:
            civ_data: Dictionary from Unciv JSON

        Returns:
            UncivPlayer instance
        """
        # Extract gold and resources
        gold = civ_data.get("gold", 0)

        # Extract tech info
        tech_manager = civ_data.get("tech", {})
        researched = tech_manager.get("techsResearched", [])
        in_progress = tech_manager.get("currentTechnologyName")
        tech_progress = (
            tech_manager.get("techsInProgress", {}).get(in_progress, 0) if in_progress else 0
        )

        # Extract policies
        policies = civ_data.get("policies", {})
        adopted = policies.get("adoptedPolicies", [])

        # Extract diplomacy
        diplomacy = civ_data.get("diplomacy", {})
        at_war = []
        has_met = []
        for other_civ, relation in diplomacy.items():
            if isinstance(relation, dict):
                has_met.append(other_civ)
                if relation.get("diplomaticStatus") == "War":
                    at_war.append(other_civ)

        return cls(
            civ_name=civ_data.get("civName", "Unknown"),
            player_type=civ_data.get("playerType", "AI"),
            gold=gold,
            gold_per_turn=civ_data.get("statsForNextTurn", {}).get("gold", 0),
            science=civ_data.get("statsForNextTurn", {}).get("science", 0),
            culture=civ_data.get("statsForNextTurn", {}).get("culture", 0),
            faith=civ_data.get("statsForNextTurn", {}).get("faith", 0),
            happiness=civ_data.get("happiness", 0),
            tech_in_progress=in_progress,
            tech_progress=tech_progress,
            researched_techs=researched,
            policies_adopted=adopted,
            cities=[city.get("name", "") for city in civ_data.get("cities", [])],
            units=[unit.get("id", "") for unit in civ_data.get("units", [])],
            is_alive=civ_data.get("isAlive", True),
            is_at_war_with=at_war,
            has_met=has_met,
        )

    def to_unciv_dict(self) -> dict:
        """Convert to Unciv JSON format.

        Returns:
            Dictionary in Unciv JSON format
        """
        return {
            "civName": self.civ_name,
            "playerType": self.player_type,
            "gold": self.gold,
            "statsForNextTurn": {
                "gold": self.gold_per_turn,
                "science": self.science,
                "culture": self.culture,
                "faith": self.faith,
            },
            "happiness": self.happiness,
            "tech": {
                "currentTechnologyName": self.tech_in_progress,
                "techsResearched": self.researched_techs,
                "techsInProgress": {self.tech_in_progress: self.tech_progress}
                if self.tech_in_progress
                else {},
            },
            "policies": {"adoptedPolicies": self.policies_adopted},
            "isAlive": self.is_alive,
        }


@dataclass
class UncivGameState:
    """Represents the complete Unciv game state.

    Attributes:
        version: Unciv game version
        game_id: Unique game identifier
        current_player: Name of the civilization whose turn it is
        turn: Current turn number
        difficulty: Game difficulty setting
        game_speed: Game speed (Quick, Standard, Epic, Marathon)
        map_size: Map dimensions (width, height)
        tiles: Dictionary mapping (x, y) to UncivTile
        players: Dictionary mapping civilization name to UncivPlayer
        units: Dictionary mapping unit ID to UncivUnit
        cities: Dictionary mapping city name to UncivCity
        barbarians_enabled: Whether barbarians are in the game
        victory_types: List of enabled victory types
    """

    version: str = ""
    game_id: str = ""
    current_player: str = ""
    turn: int = 0
    difficulty: str = "Prince"
    game_speed: str = "Standard"
    map_size: tuple[int, int] = (40, 25)
    tiles: dict[tuple[int, int], UncivTile] = field(default_factory=dict)
    players: dict[str, UncivPlayer] = field(default_factory=dict)
    units: dict[str, UncivUnit] = field(default_factory=dict)
    cities: dict[str, UncivCity] = field(default_factory=dict)
    barbarians_enabled: bool = True
    victory_types: list[str] = field(default_factory=list)

    @classmethod
    def from_unciv_json(cls, json_data: dict) -> UncivGameState:
        """Create UncivGameState from Unciv JSON save data.

        Args:
            json_data: Parsed JSON from Unciv save file

        Returns:
            UncivGameState instance
        """
        game_info = json_data.get("gameInfo", json_data)

        # Parse tiles
        tiles = {}
        tile_map = json_data.get("tileMap", {})
        tile_list = tile_map.get("tileList", [])

        # Unciv uses a hexagonal coordinate system
        # We need to extract positions from the tile data
        for tile_data in tile_list:
            pos = tile_data.get("position", {})
            x = pos.get("x", 0)
            y = pos.get("y", 0)
            tiles[(x, y)] = UncivTile.from_unciv_dict(tile_data, (x, y))

        # Calculate map size from tiles
        if tiles:
            xs = [p[0] for p in tiles.keys()]
            ys = [p[1] for p in tiles.keys()]
            map_size = (max(xs) - min(xs) + 1, max(ys) - min(ys) + 1)
        else:
            map_size = (40, 25)

        # Parse civilizations and their units/cities
        players = {}
        units = {}
        cities = {}

        for civ_data in json_data.get("civilizations", []):
            civ_name = civ_data.get("civName", "Unknown")
            players[civ_name] = UncivPlayer.from_unciv_dict(civ_data)

            # Parse units for this civilization
            for unit_data in civ_data.get("units", []):
                unit = UncivUnit.from_unciv_dict(unit_data)
                unit.owner = civ_name
                units[unit.id] = unit

            # Parse cities for this civilization
            for city_data in civ_data.get("cities", []):
                city = UncivCity.from_unciv_dict(city_data)
                city.owner = civ_name
                cities[city.name] = city

        # Get current player
        current_player = json_data.get("currentPlayer", "")
        if not current_player and players:
            # Find first human player or first player
            for name, player in players.items():
                if player.player_type == "Human":
                    current_player = name
                    break
            if not current_player:
                current_player = next(iter(players.keys()))

        return cls(
            version=json_data.get("version", {}).get("number", ""),
            game_id=game_info.get("gameId", ""),
            current_player=current_player,
            turn=json_data.get("turns", 0),
            difficulty=game_info.get("difficulty", "Prince"),
            game_speed=game_info.get("speed", "Standard"),
            map_size=map_size,
            tiles=tiles,
            players=players,
            units=units,
            cities=cities,
            barbarians_enabled=game_info.get("barbarians", True),
            victory_types=game_info.get(
                "victoryTypes", ["Domination", "Science", "Cultural", "Diplomatic"]
            ),
        )

    def to_unciv_json(self) -> dict:
        """Convert to Unciv JSON format.

        Returns:
            Dictionary in Unciv JSON format suitable for saving
        """
        # Build tile list
        tile_list = []
        for pos, tile in self.tiles.items():
            tile_dict = tile.to_unciv_dict()
            tile_dict["position"] = {"x": pos[0], "y": pos[1]}
            tile_list.append(tile_dict)

        # Build civilizations list
        civilizations = []
        for civ_name, player in self.players.items():
            civ_dict = player.to_unciv_dict()

            # Add units for this civilization
            civ_dict["units"] = [
                unit.to_unciv_dict() for unit in self.units.values() if unit.owner == civ_name
            ]

            # Add cities for this civilization
            civ_dict["cities"] = [
                city.to_unciv_dict() for city in self.cities.values() if city.owner == civ_name
            ]

            civilizations.append(civ_dict)

        return {
            "version": {"number": self.version},
            "gameInfo": {
                "gameId": self.game_id,
                "difficulty": self.difficulty,
                "speed": self.game_speed,
                "barbarians": self.barbarians_enabled,
                "victoryTypes": self.victory_types,
            },
            "currentPlayer": self.current_player,
            "turns": self.turn,
            "tileMap": {
                "tileList": tile_list,
            },
            "civilizations": civilizations,
        }


# =============================================================================
# Mapping tables for conversion between Unciv and internal formats
# =============================================================================


class UncivMappings:
    """Mapping tables for converting between Unciv and internal formats."""

    # Terrain base types to internal format
    TERRAIN_TO_INTERNAL = {
        "Grassland": "grassland",
        "Plains": "plains",
        "Desert": "desert",
        "Tundra": "tundra",
        "Snow": "snow",
        "Coast": "coast",
        "Ocean": "ocean",
        "Lakes": "coast",  # Treat lakes as coast
    }

    INTERNAL_TO_TERRAIN = {v: k for k, v in TERRAIN_TO_INTERNAL.items()}

    # Terrain features to internal format
    FEATURE_TO_INTERNAL = {
        "Forest": "forest",
        "Jungle": "jungle",
        "Marsh": "marsh",
        "Flood plains": "floodplains",
        "Oasis": "floodplains",  # Treat as floodplains
        "Hill": "hills",
        "Mountain": "mountains",
        "Ice": "ice",
    }

    INTERNAL_TO_FEATURE = {v: k for k, v in FEATURE_TO_INTERNAL.items()}

    # Unit types to internal categories
    UNIT_TO_CATEGORY = {
        # Melee
        "Warrior": "melee",
        "Spearman": "melee",
        "Swordsman": "melee",
        "Pikeman": "melee",
        "Longswordsman": "melee",
        "Musketman": "melee",
        "Rifleman": "melee",
        "Infantry": "melee",
        "Mechanized Infantry": "melee",
        # Ranged
        "Archer": "ranged",
        "Chariot Archer": "ranged",
        "Composite Bowman": "ranged",
        "Crossbowman": "ranged",
        "Gatling Gun": "ranged",
        "Machine Gun": "ranged",
        # Cavalry
        "Horseman": "cavalry",
        "Knight": "cavalry",
        "Cavalry": "cavalry",
        "Tank": "cavalry",
        "Modern Armor": "cavalry",
        # Siege
        "Catapult": "siege",
        "Trebuchet": "siege",
        "Cannon": "siege",
        "Artillery": "siege",
        "Rocket Artillery": "siege",
        # Naval Melee
        "Trireme": "naval_melee",
        "Caravel": "naval_melee",
        "Destroyer": "naval_melee",
        # Naval Ranged
        "Frigate": "naval_ranged",
        "Battleship": "naval_ranged",
        "Carrier": "naval_ranged",
        "Submarine": "naval_ranged",
        # Civilian
        "Settler": "settler",
        "Worker": "civilian",
        "Work Boats": "civilian",
        # Great People
        "Great General": "great_person",
        "Great Prophet": "religious",
        "Great Engineer": "great_person",
        "Great Merchant": "great_person",
        "Great Scientist": "great_person",
        "Great Artist": "great_person",
        "Great Musician": "great_person",
        "Great Writer": "great_person",
        # Air
        "Fighter": "air",
        "Bomber": "air",
        "Jet Fighter": "air",
        "Stealth Bomber": "air",
    }

    # Era mapping (based on tech count)
    ERA_THRESHOLDS = [
        (0, 0),  # Ancient
        (10, 1),  # Classical
        (20, 2),  # Medieval
        (35, 3),  # Renaissance
        (50, 4),  # Industrial
        (65, 5),  # Modern
        (75, 6),  # Atomic
        (80, 7),  # Information
    ]

    @classmethod
    def get_era_from_tech_count(cls, tech_count: int) -> int:
        """Determine era based on number of researched technologies."""
        era = 0
        for threshold, era_num in cls.ERA_THRESHOLDS:
            if tech_count >= threshold:
                era = era_num
        return era


# =============================================================================
# Main Interface Class
# =============================================================================


class UncivInterfaceError(Exception):
    """Base exception for Unciv interface errors."""


class UncivLoadError(UncivInterfaceError):
    """Error loading Unciv game file."""


class UncivSaveError(UncivInterfaceError):
    """Error saving Unciv game file."""


class UncivConversionError(UncivInterfaceError):
    """Error converting between formats."""


class UncivActionError(UncivInterfaceError):
    """Error applying action to game state."""


class UncivInterface:
    """Interface for Unciv JSON game state files.

    This class provides methods to load, save, and convert Unciv game states
    to and from the internal format used by the Civ6-AlphaZero encoder and
    action space.

    Example:
        >>> interface = UncivInterface(board_size=16)
        >>> state = interface.load_game("game.json")
        >>> internal = interface.convert_to_internal(state)
        >>> mask = interface.get_valid_actions(internal)
        >>> new_internal = interface.apply_action(internal, action)
        >>> interface.save_game(state, "game_modified.json")
    """

    def __init__(
        self,
        board_size: int = 16,
        max_units: int = 100,
        max_cities: int = 20,
    ):
        """Initialize the Unciv interface.

        Args:
            board_size: Size of the internal board representation
            max_units: Maximum number of units to track
            max_cities: Maximum number of cities to track
        """
        self.board_size = board_size
        self.max_units = max_units
        self.max_cities = max_cities

        # Import action space lazily to avoid circular imports
        self._action_space = None

        logger.info(
            f"UncivInterface initialized with board_size={board_size}, "
            f"max_units={max_units}, max_cities={max_cities}",
        )

    def _load_actions_module(self):
        """Load the civ6_actions module without triggering torch import.

        Returns:
            The loaded civ6_actions module
        """
        import importlib.util
        import sys

        module_name = "alpha_zero.core.civ6_actions"

        # Check if already imported
        if module_name in sys.modules:
            return sys.modules[module_name]

        # Load the module directly from file
        module_path = Path(__file__).parent.parent / "core" / "civ6_actions.py"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        actions_module = importlib.util.module_from_spec(spec)

        # Register in sys.modules BEFORE exec to allow self-references
        sys.modules[module_name] = actions_module
        spec.loader.exec_module(actions_module)

        return actions_module

    @property
    def action_space(self):
        """Lazy load the action space."""
        if self._action_space is None:
            actions_module = self._load_actions_module()
            ActionSpace = actions_module.ActionSpace
            ActionSpaceConfig = actions_module.ActionSpaceConfig

            self._action_space = ActionSpace(
                ActionSpaceConfig(
                    board_size=self.board_size,
                    max_units=self.max_units,
                    max_cities=self.max_cities,
                ),
            )
        return self._action_space

    def load_game(self, path: str) -> UncivGameState:
        """Load an Unciv game from a JSON file.

        Args:
            path: Path to the Unciv JSON save file

        Returns:
            UncivGameState object representing the game

        Raises:
            UncivLoadError: If the file cannot be loaded or parsed
        """
        file_path = Path(path)
        if not file_path.exists():
            raise UncivLoadError(f"Game file not found: {path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                json_data = json.load(f)
        except json.JSONDecodeError as e:
            raise UncivLoadError(f"Invalid JSON in game file: {e}") from e
        except OSError as e:
            raise UncivLoadError(f"Cannot read game file: {e}") from e

        try:
            game_state = UncivGameState.from_unciv_json(json_data)
            logger.info(
                f"Loaded game: turn {game_state.turn}, "
                f"{len(game_state.players)} players, "
                f"{len(game_state.cities)} cities, "
                f"{len(game_state.units)} units",
            )
            return game_state
        except Exception as e:
            raise UncivLoadError(f"Error parsing game state: {e}") from e

    def save_game(self, state: UncivGameState, path: str) -> None:
        """Save an Unciv game state to a JSON file.

        Args:
            state: UncivGameState to save
            path: Path for the output JSON file

        Raises:
            UncivSaveError: If the file cannot be saved
        """
        file_path = Path(path)

        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            json_data = state.to_unciv_json()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)

            logger.info(f"Saved game to {path}")
        except OSError as e:
            raise UncivSaveError(f"Cannot write game file: {e}") from e
        except Exception as e:
            raise UncivSaveError(f"Error saving game state: {e}") from e

    def convert_to_internal(self, unciv_state: UncivGameState | dict) -> dict:
        """Convert Unciv game state to internal encoder format.

        This method converts the Unciv JSON format to the dictionary format
        expected by Civ6StateEncoder.encode().

        Args:
            unciv_state: UncivGameState object or raw Unciv JSON dict

        Returns:
            Dictionary in internal format suitable for the encoder

        Raises:
            UncivConversionError: If conversion fails
        """
        try:
            # Handle raw dict input
            if isinstance(unciv_state, dict):
                unciv_state = UncivGameState.from_unciv_json(unciv_state)

            # Find current player index
            player_names = list(unciv_state.players.keys())
            current_player_idx = 0
            if unciv_state.current_player in player_names:
                current_player_idx = player_names.index(unciv_state.current_player)

            current_player = unciv_state.players.get(unciv_state.current_player)

            # Convert terrain
            terrain = {}
            for pos, tile in unciv_state.tiles.items():
                # Map position to internal board coordinates
                internal_pos = self._map_position_to_internal(pos, unciv_state.map_size)
                if internal_pos is None:
                    continue

                terrain[internal_pos] = {
                    "type": UncivMappings.TERRAIN_TO_INTERNAL.get(
                        tile.base_terrain,
                        "plains",
                    ),
                    "features": [
                        UncivMappings.FEATURE_TO_INTERNAL.get(f, f.lower())
                        for f in tile.terrain_features
                        if f in UncivMappings.FEATURE_TO_INTERNAL
                    ],
                    "yields": self._calculate_tile_yields(tile),
                }

            # Convert units
            units = []
            unit_index = 0
            for unit_id, unit in unciv_state.units.items():
                if unit_index >= self.max_units:
                    break

                internal_pos = self._map_position_to_internal(
                    unit.position,
                    unciv_state.map_size,
                )
                if internal_pos is None:
                    continue

                owner_idx = player_names.index(unit.owner) if unit.owner in player_names else 0

                units.append(
                    {
                        "id": unit_id,
                        "position": internal_pos,
                        "type": UncivMappings.UNIT_TO_CATEGORY.get(unit.name, "melee"),
                        "player": owner_idx,
                        "health": unit.health,
                        "strength": unit.strength or unit.ranged_strength,
                        "moves_remaining": int(unit.current_movement),
                        "max_moves": int(unit.max_movement),
                        "is_ranged": unit.ranged_strength > 0,
                        "range": unit.range,
                        "promotions": unit.promotions,
                        "original_name": unit.name,
                    }
                )
                unit_index += 1

            # Convert cities
            cities = []
            city_index = 0
            for city_name, city in unciv_state.cities.items():
                if city_index >= self.max_cities:
                    break

                internal_pos = self._map_position_to_internal(
                    city.position,
                    unciv_state.map_size,
                )
                if internal_pos is None:
                    continue

                owner_idx = player_names.index(city.owner) if city.owner in player_names else 0

                cities.append(
                    {
                        "name": city_name,
                        "position": internal_pos,
                        "player": owner_idx,
                        "population": city.population,
                        "production": city.production_stored,
                        "defense": city.health,
                        "is_capital": city.is_capital,
                        "current_production": city.current_construction,
                        "buildings": city.buildings,
                        "available_production": self._get_available_production(city),
                    }
                )
                city_index += 1

            # Convert visibility (full visibility for now)
            visibility = {}
            for y in range(self.board_size):
                for x in range(self.board_size):
                    visibility[(x, y)] = "visible"

            # Calculate era from tech count
            tech_count = len(current_player.researched_techs) if current_player else 0
            era = UncivMappings.get_era_from_tech_count(tech_count)

            # Build technology list (as indices)
            technologies = []
            if current_player:
                # Convert tech names to indices (simplified mapping)
                for i, _tech in enumerate(current_player.researched_techs):
                    if i < 100:  # Max tech slots in encoder
                        technologies.append(i)

            # Build civics list (policies)
            civics = []
            if current_player:
                for i, _policy in enumerate(current_player.policies_adopted):
                    if i < 100:  # Max civic slots in encoder
                        civics.append(i)

            # Build resources dict
            resources = {}
            # Unciv doesn't separate resources the same way, use defaults
            resources["gold"] = current_player.gold if current_player else 0

            # Build victory progress (simplified)
            victory_progress = {
                "domination": self._calculate_domination_progress(
                    unciv_state,
                    current_player_idx,
                ),
                "science": self._calculate_science_progress(current_player),
                "culture": self._calculate_culture_progress(current_player),
                "religious": 0.0,
                "diplomatic": 0.0,
            }

            internal_state = {
                "current_player": current_player_idx,
                "turn": unciv_state.turn,
                "era": era,
                "terrain": terrain,
                "units": units,
                "cities": cities,
                "visibility": visibility,
                "technologies": technologies,
                "civics": civics,
                "resources": resources,
                "gold": current_player.gold if current_player else 0,
                "gold_per_turn": current_player.gold_per_turn if current_player else 0,
                "science_per_turn": current_player.science if current_player else 0,
                "culture_per_turn": current_player.culture if current_player else 0,
                "faith_per_turn": current_player.faith if current_player else 0,
                "military_strength": self._calculate_military_strength(
                    unciv_state,
                    current_player_idx,
                ),
                "victory_progress": victory_progress,
                "current_tech": current_player.tech_in_progress if current_player else None,
                "current_civic": None,  # Not directly available in Unciv
                "num_players": len(player_names),
                "available_techs": self._get_available_techs(current_player),
                "available_civics": self._get_available_civics(current_player),
                # Store metadata for reverse conversion
                "_unciv_player_names": player_names,
                "_unciv_map_size": unciv_state.map_size,
            }

            logger.debug(
                f"Converted to internal: {len(units)} units, {len(cities)} cities",
            )
            return internal_state

        except Exception as e:
            raise UncivConversionError(f"Error converting to internal format: {e}") from e

    def convert_to_unciv(self, internal_state: dict) -> dict:
        """Convert internal game state back to Unciv JSON format.

        Args:
            internal_state: Dictionary in internal encoder format

        Returns:
            Dictionary in Unciv JSON format

        Raises:
            UncivConversionError: If conversion fails
        """
        try:
            # Get metadata from internal state
            player_names = internal_state.get("_unciv_player_names", ["Player 1", "Player 2"])
            map_size = internal_state.get("_unciv_map_size", (40, 25))

            current_player_idx = internal_state.get("current_player", 0)
            current_player_name = (
                player_names[current_player_idx]
                if current_player_idx < len(player_names)
                else player_names[0]
                if player_names
                else "Unknown"
            )

            # Convert tiles
            tile_list = []
            for pos, tile_data in internal_state.get("terrain", {}).items():
                unciv_pos = self._map_position_from_internal(pos, map_size)
                if unciv_pos is None:
                    continue

                tile_dict = {
                    "position": {"x": unciv_pos[0], "y": unciv_pos[1]},
                    "baseTerrain": UncivMappings.INTERNAL_TO_TERRAIN.get(
                        tile_data.get("type", "plains"),
                        "Plains",
                    ),
                    "terrainFeatures": [
                        UncivMappings.INTERNAL_TO_FEATURE.get(f, f.title())
                        for f in tile_data.get("features", [])
                    ],
                }
                tile_list.append(tile_dict)

            # Convert units grouped by owner
            units_by_owner: dict[str, list] = {name: [] for name in player_names}
            for unit in internal_state.get("units", []):
                owner_idx = unit.get("player", 0)
                owner_name = (
                    player_names[owner_idx] if owner_idx < len(player_names) else player_names[0]
                )

                unciv_pos = self._map_position_from_internal(
                    unit.get("position", (0, 0)),
                    map_size,
                )
                if unciv_pos is None:
                    continue

                unit_dict = {
                    "id": unit.get("id", str(id(unit))),
                    "name": unit.get("original_name", "Warrior"),
                    "owner": owner_name,
                    "currentTile": {"x": unciv_pos[0], "y": unciv_pos[1]},
                    "health": unit.get("health", 100),
                    "currentMovement": unit.get("moves_remaining", 2),
                    "promotions": {"promotions": unit.get("promotions", [])},
                }
                units_by_owner[owner_name].append(unit_dict)

            # Convert cities grouped by owner
            cities_by_owner: dict[str, list] = {name: [] for name in player_names}
            for city in internal_state.get("cities", []):
                owner_idx = city.get("player", 0)
                owner_name = (
                    player_names[owner_idx] if owner_idx < len(player_names) else player_names[0]
                )

                unciv_pos = self._map_position_from_internal(
                    city.get("position", (0, 0)),
                    map_size,
                )
                if unciv_pos is None:
                    continue

                city_dict = {
                    "name": city.get("name", "City"),
                    "civName": owner_name,
                    "location": {"x": unciv_pos[0], "y": unciv_pos[1]},
                    "population": city.get("population", 1),
                    "health": city.get("defense", 200),
                    "isCapital": city.get("is_capital", False),
                    "cityConstructions": {
                        "currentConstructionFromQueue": city.get("current_production"),
                        "productionOverflow": city.get("production", 0),
                        "builtBuildings": city.get("buildings", []),
                    },
                }
                cities_by_owner[owner_name].append(city_dict)

            # Build civilizations
            civilizations = []
            for i, name in enumerate(player_names):
                civ_dict = {
                    "civName": name,
                    "playerType": "Human" if i == current_player_idx else "AI",
                    "gold": internal_state.get("gold", 0) if i == current_player_idx else 0,
                    "units": units_by_owner.get(name, []),
                    "cities": cities_by_owner.get(name, []),
                    "isAlive": True,
                }
                civilizations.append(civ_dict)

            unciv_json = {
                "version": {"number": "4.0"},
                "gameInfo": {
                    "gameId": "",
                    "difficulty": "Prince",
                    "speed": "Standard",
                },
                "currentPlayer": current_player_name,
                "turns": internal_state.get("turn", 0),
                "tileMap": {"tileList": tile_list},
                "civilizations": civilizations,
            }

            return unciv_json

        except Exception as e:
            raise UncivConversionError(f"Error converting to Unciv format: {e}") from e

    def get_valid_actions(self, state: dict) -> np.ndarray:
        """Get valid action mask for the current game state.

        Args:
            state: Game state in internal format

        Returns:
            Numpy array of shape (action_space_size,) with 1.0 for valid actions
        """
        return self.action_space.get_full_mask(state)

    def _get_action_enums(self):
        """Get action enum classes, loading them if needed."""
        actions_module = self._load_actions_module()
        return (
            actions_module.ActionCategory,
            actions_module.UnitActionType,
            actions_module.CityActionType,
        )

    def apply_action(self, state: dict, action: int) -> dict:
        """Apply an action to the game state.

        This method decodes the flat action index, applies it to the state,
        and returns the new state.

        Args:
            state: Game state in internal format
            action: Flat action index

        Returns:
            New game state after applying the action

        Raises:
            UncivActionError: If the action cannot be applied
        """
        try:
            ActionCategory, UnitActionType, CityActionType = self._get_action_enums()

            # Decode the action
            decoded = self.action_space.decode(action)

            # Make a deep copy of the state
            new_state = copy.deepcopy(state)
            current_player = new_state.get("current_player", 0)

            if decoded.category == ActionCategory.END_TURN:
                # End turn - switch to next player
                num_players = new_state.get("num_players", 2)
                new_state["current_player"] = (current_player + 1) % num_players

                # If we wrapped around to player 0, increment turn
                if new_state["current_player"] == 0:
                    new_state["turn"] = new_state.get("turn", 0) + 1

                # Reset unit moves for new player
                for unit in new_state.get("units", []):
                    if unit.get("player") == new_state["current_player"]:
                        unit["moves_remaining"] = unit.get("max_moves", 2)

                logger.debug(f"End turn: now player {new_state['current_player']}")

            elif decoded.category == ActionCategory.UNIT_ACTION:
                # Get the unit
                units = new_state.get("units", [])
                if decoded.selector >= len(units):
                    raise UncivActionError(
                        f"Invalid unit index: {decoded.selector}",
                    )

                unit = units[decoded.selector]
                if unit.get("player") != current_player:
                    raise UncivActionError("Cannot control opponent's unit")

                action_type = decoded.action_type
                target = decoded.target

                if action_type == UnitActionType.MOVE:
                    # Move unit to target position
                    if isinstance(target, tuple):
                        unit["position"] = target
                        unit["moves_remaining"] = max(
                            0,
                            unit.get("moves_remaining", 0) - 1,
                        )
                        logger.debug(f"Unit {decoded.selector} moved to {target}")

                elif action_type == UnitActionType.ATTACK:
                    # Attack enemy unit at target position
                    if isinstance(target, tuple):
                        # Find enemy unit at target
                        for _i, enemy in enumerate(units):
                            if (
                                enemy.get("player") != current_player
                                and tuple(enemy.get("position", ())) == target
                            ):
                                # Simple combat: reduce health
                                attacker_strength = unit.get("strength", 10)
                                defender_strength = enemy.get("strength", 10)

                                # Damage calculation (simplified)
                                damage_to_defender = max(
                                    10,
                                    attacker_strength - defender_strength // 2,
                                )
                                damage_to_attacker = max(
                                    5,
                                    defender_strength // 2 - attacker_strength // 4,
                                )

                                enemy["health"] = max(
                                    0,
                                    enemy.get("health", 100) - damage_to_defender,
                                )
                                unit["health"] = max(
                                    0,
                                    unit.get("health", 100) - damage_to_attacker,
                                )

                                # Remove dead units
                                if enemy["health"] <= 0:
                                    units.remove(enemy)
                                    logger.debug(f"Enemy unit destroyed at {target}")
                                if unit["health"] <= 0:
                                    units.remove(unit)
                                    logger.debug("Attacker unit destroyed")

                                unit["moves_remaining"] = 0
                                break

                elif action_type == UnitActionType.SKIP:
                    # Skip turn for this unit
                    unit["moves_remaining"] = 0
                    logger.debug(f"Unit {decoded.selector} skipped")

                elif action_type == UnitActionType.FORTIFY:
                    # Fortify unit
                    unit["moves_remaining"] = 0
                    unit["action"] = "Fortify"
                    logger.debug(f"Unit {decoded.selector} fortifying")

                elif action_type == UnitActionType.FOUND_CITY:
                    # Found a city (settler only)
                    if unit.get("type") == "settler":
                        pos = unit.get("position", (0, 0))
                        city_name = f"City_{len(new_state.get('cities', []))}"
                        new_city = {
                            "name": city_name,
                            "position": pos,
                            "player": current_player,
                            "population": 1,
                            "production": 0,
                            "defense": 200,
                            "is_capital": len(
                                [
                                    c
                                    for c in new_state.get("cities", [])
                                    if c.get("player") == current_player
                                ],
                            )
                            == 0,
                            "current_production": None,
                            "buildings": [],
                            "available_production": list(range(10)),
                        }
                        new_state.setdefault("cities", []).append(new_city)

                        # Remove settler
                        units.remove(unit)
                        logger.debug(f"Founded city {city_name} at {pos}")

            elif decoded.category == ActionCategory.CITY_ACTION:
                # Get the city
                cities = new_state.get("cities", [])
                if decoded.selector >= len(cities):
                    raise UncivActionError(
                        f"Invalid city index: {decoded.selector}",
                    )

                city = cities[decoded.selector]
                if city.get("player") != current_player:
                    raise UncivActionError("Cannot control opponent's city")

                action_type = decoded.action_type
                item_id = decoded.target if isinstance(decoded.target, int) else 0

                if action_type == CityActionType.PRODUCE_UNIT:
                    # Start producing a unit
                    unit_names = [
                        "Warrior",
                        "Settler",
                        "Archer",
                        "Spearman",
                        "Swordsman",
                    ]
                    if item_id < len(unit_names):
                        city["current_production"] = unit_names[item_id]
                        logger.debug(
                            f"City {city.get('name')} producing {unit_names[item_id]}",
                        )

                elif action_type == CityActionType.PRODUCE_BUILDING:
                    # Start producing a building
                    building_names = [
                        "Monument",
                        "Granary",
                        "Library",
                        "Barracks",
                        "Walls",
                    ]
                    if item_id < len(building_names):
                        city["current_production"] = building_names[item_id]
                        logger.debug(
                            f"City {city.get('name')} producing {building_names[item_id]}",
                        )

            elif decoded.category == ActionCategory.RESEARCH_TECH:
                # Start researching a technology
                tech_names = [
                    "Agriculture",
                    "Mining",
                    "Pottery",
                    "Animal Husbandry",
                    "Archery",
                    "Sailing",
                    "Writing",
                    "Bronze Working",
                    "Masonry",
                    "The Wheel",
                ]
                if decoded.selector < len(tech_names):
                    new_state["current_tech"] = tech_names[decoded.selector]
                    logger.debug(f"Researching {tech_names[decoded.selector]}")

            elif decoded.category == ActionCategory.RESEARCH_CIVIC:
                # Start researching a civic
                civic_names = [
                    "Code of Laws",
                    "Craftsmanship",
                    "Foreign Trade",
                    "Military Tradition",
                    "State Workforce",
                ]
                if decoded.selector < len(civic_names):
                    new_state["current_civic"] = civic_names[decoded.selector]
                    logger.debug(f"Adopting civic {civic_names[decoded.selector]}")

            return new_state

        except UncivActionError:
            raise
        except Exception as e:
            raise UncivActionError(f"Error applying action {action}: {e}") from e

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _map_position_to_internal(
        self, pos: tuple[int, int], map_size: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Map Unciv coordinates to internal board coordinates.

        Args:
            pos: (x, y) position in Unciv coordinates
            map_size: (width, height) of the Unciv map

        Returns:
            (x, y) position in internal coordinates, or None if out of bounds
        """
        # Scale position to fit internal board
        scale_x = self.board_size / max(map_size[0], 1)
        scale_y = self.board_size / max(map_size[1], 1)

        internal_x = int(pos[0] * scale_x)
        internal_y = int(pos[1] * scale_y)

        # Clamp to board bounds
        internal_x = max(0, min(self.board_size - 1, internal_x))
        internal_y = max(0, min(self.board_size - 1, internal_y))

        return (internal_x, internal_y)

    def _map_position_from_internal(
        self, pos: tuple[int, int], map_size: tuple[int, int]
    ) -> tuple[int, int] | None:
        """Map internal board coordinates back to Unciv coordinates.

        Args:
            pos: (x, y) position in internal coordinates
            map_size: (width, height) of the Unciv map

        Returns:
            (x, y) position in Unciv coordinates
        """
        scale_x = map_size[0] / max(self.board_size, 1)
        scale_y = map_size[1] / max(self.board_size, 1)

        unciv_x = int(pos[0] * scale_x)
        unciv_y = int(pos[1] * scale_y)

        return (unciv_x, unciv_y)

    def _calculate_tile_yields(self, tile: UncivTile) -> dict:
        """Calculate base yields for a tile.

        Args:
            tile: UncivTile object

        Returns:
            Dictionary with yield values
        """
        yields = {"food": 0, "production": 0, "gold": 0, "science": 0, "culture": 0}

        # Base terrain yields
        terrain_yields = {
            "Grassland": {"food": 2},
            "Plains": {"food": 1, "production": 1},
            "Desert": {},
            "Tundra": {"food": 1},
            "Snow": {},
            "Coast": {"food": 1, "gold": 1},
            "Ocean": {"food": 1},
        }

        base = terrain_yields.get(tile.base_terrain, {})
        for k, v in base.items():
            yields[k] = v

        # Feature modifications
        for feature in tile.terrain_features:
            if feature in ("Forest", "Jungle"):
                yields["production"] += 1
            elif feature == "Hill":
                yields["production"] += 2
                yields["food"] = max(0, yields["food"] - 1)
            elif feature == "Flood plains":
                yields["food"] += 2
            elif feature == "Oasis":
                yields["food"] += 3
                yields["gold"] += 1

        # Improvement bonuses (simplified)
        if tile.improvement:
            if tile.improvement == "Farm":
                yields["food"] += 1
            elif tile.improvement == "Mine":
                yields["production"] += 1
            elif tile.improvement == "Trading post":
                yields["gold"] += 1

        return yields

    def _get_available_production(self, city: UncivCity) -> list[int]:
        """Get available production items for a city.

        Args:
            city: UncivCity object

        Returns:
            List of production item indices
        """
        # Simplified: return first 10 items as available
        # In a full implementation, this would check requirements
        return list(range(10))

    def _calculate_military_strength(
        self,
        state: UncivGameState,
        player_idx: int,
    ) -> int:
        """Calculate total military strength for a player.

        Args:
            state: UncivGameState
            player_idx: Player index

        Returns:
            Total military strength
        """
        player_names = list(state.players.keys())
        if player_idx >= len(player_names):
            return 0

        player_name = player_names[player_idx]
        strength = 0

        for unit in state.units.values():
            if unit.owner == player_name:
                strength += max(unit.strength, unit.ranged_strength)

        return strength

    def _calculate_domination_progress(
        self,
        state: UncivGameState,
        player_idx: int,
    ) -> float:
        """Calculate domination victory progress.

        Args:
            state: UncivGameState
            player_idx: Player index

        Returns:
            Progress value between 0 and 1
        """
        player_names = list(state.players.keys())
        if player_idx >= len(player_names):
            return 0.0

        player_name = player_names[player_idx]

        # Count capitals controlled
        total_players = len([p for p in state.players.values() if p.is_alive])
        capitals_held = 0

        for city in state.cities.values():
            if city.is_capital and city.owner == player_name:
                capitals_held += 1

        if total_players <= 1:
            return 1.0 if capitals_held > 0 else 0.0

        return capitals_held / total_players

    def _calculate_science_progress(self, player: UncivPlayer | None) -> float:
        """Calculate science victory progress.

        Args:
            player: UncivPlayer object

        Returns:
            Progress value between 0 and 1
        """
        if not player:
            return 0.0

        # Based on tech count (simplified)
        total_techs = 75  # Approximate total techs in game
        return min(1.0, len(player.researched_techs) / total_techs)

    def _calculate_culture_progress(self, player: UncivPlayer | None) -> float:
        """Calculate culture victory progress.

        Args:
            player: UncivPlayer object

        Returns:
            Progress value between 0 and 1
        """
        if not player:
            return 0.0

        # Based on policies adopted (simplified)
        total_policies = 30  # Approximate total policies
        return min(1.0, len(player.policies_adopted) / total_policies)

    def _get_available_techs(self, player: UncivPlayer | None) -> list[int]:
        """Get available technologies for research.

        Args:
            player: UncivPlayer object

        Returns:
            List of technology indices
        """
        if not player:
            return list(range(5))

        # Simplified: return techs not yet researched
        researched_count = len(player.researched_techs)
        # Return next 5 available techs
        return list(range(researched_count, min(researched_count + 5, 73)))

    def _get_available_civics(self, player: UncivPlayer | None) -> list[int]:
        """Get available civics for adoption.

        Args:
            player: UncivPlayer object

        Returns:
            List of civic indices
        """
        if not player:
            return list(range(3))

        # Simplified: return civics not yet adopted
        adopted_count = len(player.policies_adopted)
        return list(range(adopted_count, min(adopted_count + 3, 56)))


# =============================================================================
# Factory function for convenience
# =============================================================================


def create_unciv_interface(
    board_size: int = 16,
    max_units: int = 100,
    max_cities: int = 20,
) -> UncivInterface:
    """Factory function to create an UncivInterface.

    Args:
        board_size: Size of the internal board representation
        max_units: Maximum number of units to track
        max_cities: Maximum number of cities to track

    Returns:
        Configured UncivInterface instance
    """
    return UncivInterface(
        board_size=board_size,
        max_units=max_units,
        max_cities=max_cities,
    )
