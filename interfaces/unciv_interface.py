"""Unciv JSON Interface Implementation.

Provides direct manipulation of Unciv save files for training.
Unciv is an open-source Civilization V clone with JSON-based save files,
making it ideal for rapid self-play iteration.

Unciv Save Format:
    Save files are gzipped JSON with the following structure:
    {
        "gameInfo": {...},
        "civilizations": [...],
        "tileMap": {...},
        ...
    }
"""

from __future__ import annotations

import gzip
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import (
    ActionSendError,
    ActionType,
    City,
    GameAction,
    GameInterface,
    GameState,
    InterfaceConfig,
    InterfaceError,
    Position,
    StateReadError,
    Tile,
    Unit,
)

logger = logging.getLogger(__name__)


class UncivError(InterfaceError):
    """Unciv interface specific error."""
    pass


class SaveFileError(UncivError):
    """Error reading or writing save file."""
    pass


@dataclass
class UncivStateParser:
    """Parser for Unciv JSON save files.

    Converts Unciv save format to standardized GameState objects.
    """

    player_civ: str = "Babylon"

    def parse(self, data: dict[str, Any]) -> GameState:
        """Parse Unciv save data into GameState.

        Args:
            data: Parsed JSON from save file.

        Returns:
            GameState object.
        """
        state = GameState()

        game_info = data.get("gameInfo", {})
        state.turn = game_info.get("turns", 0)

        # Find current player
        civilizations = data.get("civilizations", [])
        for i, civ in enumerate(civilizations):
            if civ.get("civName") == self.player_civ:
                state.current_player = i
                break

        state.num_players = len([c for c in civilizations if not c.get("isDefeated", False)])

        # Parse tile map
        tile_map = data.get("tileMap", {})
        state.map_width = tile_map.get("mapParameters", {}).get("size", {}).get("width", 40)
        state.map_height = tile_map.get("mapParameters", {}).get("size", {}).get("height", 25)
        state.terrain = self._parse_tiles(tile_map.get("tileList", []))

        # Parse units from all civilizations
        state.units = []
        for i, civ in enumerate(civilizations):
            civ_units = self._parse_units(civ.get("units", []), i)
            state.units.extend(civ_units)

        # Parse cities from all civilizations
        state.cities = []
        for i, civ in enumerate(civilizations):
            civ_cities = self._parse_cities(civ.get("cities", []), i)
            state.cities.extend(civ_cities)

        # Parse resources for current player
        if 0 <= state.current_player < len(civilizations):
            current_civ = civilizations[state.current_player]
            state.resources = self._parse_resources(current_civ)
            state.technologies = self._parse_technologies(current_civ)
            state.gold = current_civ.get("gold", 0)
            state.gold_per_turn = current_civ.get("statsForNextTurn", {}).get("gold", 0)
            state.science_per_turn = current_civ.get("statsForNextTurn", {}).get("science", 0)
            state.culture_per_turn = current_civ.get("statsForNextTurn", {}).get("culture", 0)
            state.faith_per_turn = current_civ.get("statsForNextTurn", {}).get("faith", 0)

        # Check game over conditions
        state.game_over = game_info.get("gameStatus", "") != ""
        if state.game_over:
            winner_civ = game_info.get("winnerCiv", "")
            for i, civ in enumerate(civilizations):
                if civ.get("civName") == winner_civ:
                    state.winner = i
                    break

        # Store extra data
        state.extra = {
            "gameId": data.get("gameId", ""),
            "difficulty": game_info.get("difficulty", ""),
            "gameSpeed": game_info.get("speed", ""),
        }

        return state

    def _parse_tiles(self, tile_list: list[dict[str, Any]]) -> dict[tuple[int, int], Tile]:
        """Parse tiles from Unciv format."""
        terrain = {}
        for tile_data in tile_list:
            try:
                pos_str = tile_data.get("position", "")
                # Unciv uses "x,y" format
                x, y = map(int, pos_str.split(","))

                tile = Tile(
                    position=Position(x=x, y=y),
                    terrain=self._map_terrain(tile_data.get("baseTerrain", "Grassland")),
                    features=tile_data.get("terrainFeatures", []),
                    resource=tile_data.get("resource"),
                    improvement=tile_data.get("improvement"),
                    owner=None,  # Set from cities later
                    yields={
                        "food": tile_data.get("stats", {}).get("food", 0),
                        "production": tile_data.get("stats", {}).get("production", 0),
                        "gold": tile_data.get("stats", {}).get("gold", 0),
                        "science": tile_data.get("stats", {}).get("science", 0),
                        "culture": tile_data.get("stats", {}).get("culture", 0),
                        "faith": tile_data.get("stats", {}).get("faith", 0),
                    }
                )
                terrain[(x, y)] = tile
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse tile: {e}")
                continue
        return terrain

    def _parse_units(
        self,
        units_data: list[dict[str, Any]],
        player_id: int
    ) -> list[Unit]:
        """Parse units from Unciv format."""
        units = []
        for u in units_data:
            try:
                # Parse position
                tile_pos = u.get("currentTile", "")
                x, y = 0, 0
                if tile_pos:
                    x, y = map(int, tile_pos.split(","))

                unit = Unit(
                    id=u.get("id", str(len(units))),
                    type=u.get("name", "Warrior"),
                    player=player_id,
                    position=Position(x=x, y=y),
                    health=u.get("health", 100),
                    movement=u.get("currentMovement", 0),
                    strength=u.get("baseUnit", {}).get("strength", 20),
                    can_attack=not u.get("hasAttacked", False),
                    can_move=u.get("currentMovement", 0) > 0,
                    promotions=u.get("promotions", {}).get("promotions", []),
                    extra={
                        "maxMovement": u.get("baseUnit", {}).get("movement", 2),
                        "rangedStrength": u.get("baseUnit", {}).get("rangedStrength", 0),
                        "range": u.get("baseUnit", {}).get("range", 0),
                    }
                )
                units.append(unit)
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse unit: {e}")
                continue
        return units

    def _parse_cities(
        self,
        cities_data: list[dict[str, Any]],
        player_id: int
    ) -> list[City]:
        """Parse cities from Unciv format."""
        cities = []
        for c in cities_data:
            try:
                # Parse position
                location = c.get("location", "")
                x, y = 0, 0
                if location:
                    x, y = map(int, location.split(","))

                city = City(
                    id=c.get("id", str(len(cities))),
                    name=c.get("name", "City"),
                    player=player_id,
                    position=Position(x=x, y=y),
                    population=c.get("population", 1),
                    production=c.get("cityConstructions", {}).get("productionProgress", 0),
                    food=c.get("population", {}).get("foodStored", 0),
                    defense=c.get("cityStrength", 20),
                    districts=[],  # Unciv doesn't have districts
                    buildings=c.get("cityConstructions", {}).get("builtBuildings", []),
                    producing=c.get("cityConstructions", {}).get("currentConstruction"),
                    extra={
                        "isCapital": c.get("isCapital", False),
                        "resistanceCounter": c.get("resistanceCounter", 0),
                        "turnsToGrow": c.get("population", {}).get("turnsToGrow", 0),
                    }
                )
                cities.append(city)
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse city: {e}")
                continue
        return cities

    def _parse_resources(self, civ_data: dict[str, Any]) -> dict[str, int]:
        """Parse resources from civilization data."""
        resources = {}
        civ_resources = civ_data.get("resourcesForDisplay", {})
        for resource, amount in civ_resources.items():
            if amount > 0:
                resources[resource] = amount
        return resources

    def _parse_technologies(self, civ_data: dict[str, Any]) -> list[int]:
        """Parse researched technologies."""
        tech_manager = civ_data.get("tech", {})
        techs_researched = tech_manager.get("techsResearched", [])
        # Return indices (for compatibility with encoding)
        return list(range(len(techs_researched)))

    def _map_terrain(self, unciv_terrain: str) -> str:
        """Map Unciv terrain names to standard format."""
        terrain_map = {
            "Grassland": "TERRAIN_GRASS",
            "Plains": "TERRAIN_PLAINS",
            "Desert": "TERRAIN_DESERT",
            "Tundra": "TERRAIN_TUNDRA",
            "Snow": "TERRAIN_SNOW",
            "Coast": "TERRAIN_COAST",
            "Ocean": "TERRAIN_OCEAN",
            "Mountain": "TERRAIN_MOUNTAIN",
            "Hill": "TERRAIN_HILLS",
        }
        return terrain_map.get(unciv_terrain, "TERRAIN_PLAINS")


@dataclass
class UncivActionSerializer:
    """Serializer for converting GameActions to Unciv format.

    Modifies the save file to apply actions directly.
    """

    def serialize(
        self,
        action: GameAction,
        save_data: dict[str, Any],
        player_id: int
    ) -> dict[str, Any]:
        """Apply an action to the save data.

        Args:
            action: Action to apply.
            save_data: Current save data.
            player_id: Player applying the action.

        Returns:
            Modified save data.
        """
        # Make a copy to avoid modifying original
        data = json.loads(json.dumps(save_data))

        if action.action_type == ActionType.END_TURN:
            return self._apply_end_turn(data, player_id)
        elif action.action_type == ActionType.MOVE:
            return self._apply_move(data, action, player_id)
        elif action.action_type == ActionType.ATTACK:
            return self._apply_attack(data, action, player_id)
        elif action.action_type == ActionType.SKIP:
            return self._apply_skip(data, action, player_id)
        elif action.action_type == ActionType.PRODUCE:
            return self._apply_produce(data, action, player_id)
        elif action.action_type == ActionType.SETTLE:
            return self._apply_settle(data, action, player_id)
        else:
            logger.warning(f"Unsupported action type: {action.action_type}")
            return data

    def _apply_end_turn(
        self,
        data: dict[str, Any],
        player_id: int
    ) -> dict[str, Any]:
        """Apply end turn action."""
        # Mark current player as having ended turn
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            civs[player_id]["hasPlayedTurn"] = True

        # Check if all players have ended turn
        all_ended = all(c.get("hasPlayedTurn", False) for c in civs if not c.get("isDefeated", False))
        if all_ended:
            # Advance turn
            game_info = data.get("gameInfo", {})
            game_info["turns"] = game_info.get("turns", 0) + 1

            # Reset turn flags
            for civ in civs:
                if not civ.get("isDefeated", False):
                    civ["hasPlayedTurn"] = False

        return data

    def _apply_move(
        self,
        data: dict[str, Any],
        action: GameAction,
        player_id: int
    ) -> dict[str, Any]:
        """Apply move action."""
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            units = civs[player_id].get("units", [])
            for unit in units:
                if unit.get("id") == action.unit_id:
                    if action.target_position:
                        # Update unit position
                        new_pos = f"{action.target_position.x},{action.target_position.y}"
                        unit["currentTile"] = new_pos
                        # Reduce movement
                        unit["currentMovement"] = max(0, unit.get("currentMovement", 1) - 1)
                    break
        return data

    def _apply_attack(
        self,
        data: dict[str, Any],
        action: GameAction,
        player_id: int
    ) -> dict[str, Any]:
        """Apply attack action."""
        if not action.target_position:
            return data

        target_pos = f"{action.target_position.x},{action.target_position.y}"
        attacker_strength = 20
        defender_strength = 20

        # Find attacker
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            units = civs[player_id].get("units", [])
            for unit in units:
                if unit.get("id") == action.unit_id:
                    attacker_strength = unit.get("baseUnit", {}).get("strength", 20)
                    unit["hasAttacked"] = True
                    unit["currentMovement"] = 0
                    break

        # Find defender and apply damage
        for i, civ in enumerate(civs):
            if i == player_id:
                continue
            units = civ.get("units", [])
            for unit in units:
                if unit.get("currentTile") == target_pos:
                    defender_strength = unit.get("baseUnit", {}).get("strength", 20)
                    # Simple damage calculation
                    damage_ratio = attacker_strength / max(defender_strength, 1)
                    damage = int(30 * damage_ratio)
                    unit["health"] = max(0, unit.get("health", 100) - damage)
                    if unit["health"] <= 0:
                        units.remove(unit)
                    break

        return data

    def _apply_skip(
        self,
        data: dict[str, Any],
        action: GameAction,
        player_id: int
    ) -> dict[str, Any]:
        """Apply skip action."""
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            units = civs[player_id].get("units", [])
            for unit in units:
                if unit.get("id") == action.unit_id:
                    unit["currentMovement"] = 0
                    break
        return data

    def _apply_produce(
        self,
        data: dict[str, Any],
        action: GameAction,
        player_id: int
    ) -> dict[str, Any]:
        """Apply produce action."""
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            cities = civs[player_id].get("cities", [])
            for city in cities:
                if city.get("id") == action.city_id:
                    constructions = city.setdefault("cityConstructions", {})
                    constructions["currentConstruction"] = action.production_type
                    constructions["productionProgress"] = 0
                    break
        return data

    def _apply_settle(
        self,
        data: dict[str, Any],
        action: GameAction,
        player_id: int
    ) -> dict[str, Any]:
        """Apply settle action (found city)."""
        civs = data.get("civilizations", [])
        if 0 <= player_id < len(civs):
            units = civs[player_id].get("units", [])
            settler = None
            settler_pos = None

            # Find settler unit
            for unit in units:
                if unit.get("id") == action.unit_id:
                    if "Settler" in unit.get("name", ""):
                        settler = unit
                        settler_pos = unit.get("currentTile", "0,0")
                        break

            if settler and settler_pos:
                # Remove settler
                units.remove(settler)

                # Create city
                cities = civs[player_id].setdefault("cities", [])
                city_name = f"City {len(cities) + 1}"
                new_city = {
                    "id": str(len(cities)),
                    "name": city_name,
                    "location": settler_pos,
                    "population": 1,
                    "isCapital": len(cities) == 0,
                    "cityConstructions": {
                        "builtBuildings": [],
                        "currentConstruction": None,
                        "productionProgress": 0,
                    },
                }
                cities.append(new_city)

        return data


class UncivInterface(GameInterface):
    """Interface for Unciv save file manipulation.

    This interface directly manipulates Unciv JSON save files,
    allowing rapid self-play without running the actual game.

    Example:
        interface = UncivInterface(
            save_path=Path("game.json"),
            player_civ="Babylon",
        )
        with interface:
            state = interface.read_state()
            interface.send_action(GameAction.move(unit_id="0", target=Position(5, 5)))
            interface.save()
    """

    def __init__(
        self,
        config: InterfaceConfig | None = None,
        save_path: Path | None = None,
        player_civ: str = "Babylon",
        auto_save: bool = True,
        backup_saves: bool = True,
    ) -> None:
        """Initialize the Unciv interface.

        Args:
            config: Interface configuration.
            save_path: Path to save file (overrides config).
            player_civ: Name of the civilization to control.
            auto_save: Automatically save after each action.
            backup_saves: Create backup before modifications.
        """
        super().__init__(config)

        self.save_path = save_path or self.config.save_path or Path("game.json")
        self.player_civ = player_civ
        self.auto_save = auto_save
        self.backup_saves = backup_saves

        self.parser = UncivStateParser(player_civ=player_civ)
        self.serializer = UncivActionSerializer()

        self._save_data: dict[str, Any] | None = None
        self._player_id: int = 0
        self._modified = False

    def connect(self) -> bool:
        """Load the save file.

        Returns:
            True if file loaded successfully.
        """
        try:
            self._save_data = self._load_save()
            self._connected = True

            # Find player ID
            civs = self._save_data.get("civilizations", [])
            for i, civ in enumerate(civs):
                if civ.get("civName") == self.player_civ:
                    self._player_id = i
                    break

            logger.info(f"Connected to Unciv save: {self.save_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load save: {e}")
            return False

    def disconnect(self) -> None:
        """Save and close the file."""
        if self._modified and self._save_data:
            self._write_save(self._save_data)
        self._connected = False
        self._save_data = None
        logger.info("Disconnected from Unciv save")

    def read_state(self) -> GameState:
        """Read current game state from save data.

        Returns:
            Current game state.
        """
        if not self._save_data:
            raise StateReadError("No save data loaded")

        return self.parser.parse(self._save_data)

    def send_action(self, action: GameAction) -> bool:
        """Apply an action to the save data.

        Args:
            action: Action to apply.

        Returns:
            True if action was applied.
        """
        if not self._save_data:
            raise ActionSendError("No save data loaded")

        try:
            self._save_data = self.serializer.serialize(
                action, self._save_data, self._player_id
            )
            self._modified = True

            if self.auto_save:
                self._write_save(self._save_data)

            logger.debug(f"Applied action: {action.action_type.value}")
            return True

        except Exception as e:
            raise ActionSendError(f"Failed to apply action: {e}") from e

    def get_valid_actions(self, state: GameState) -> list[GameAction]:
        """Get list of valid actions for current state.

        Args:
            state: Current game state.

        Returns:
            List of valid actions.
        """
        actions: list[GameAction] = []

        # End turn is always valid
        actions.append(GameAction.end_turn())

        # Unit actions
        for unit in state.get_own_units():
            actions.append(GameAction.skip(unit.id))

            if unit.can_move:
                # Movement - 6 hex directions (Unciv uses hexagonal grid)
                hex_directions = [
                    (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)
                ]
                for dx, dy in hex_directions:
                    target = Position(
                        x=unit.position.x + dx,
                        y=unit.position.y + dy
                    )
                    tile = state.terrain.get(target.to_tuple())
                    if tile and tile.terrain != "TERRAIN_MOUNTAIN":
                        actions.append(GameAction.move(unit.id, target))

            if unit.can_attack:
                # Attack adjacent enemies
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        target = Position(
                            x=unit.position.x + dx,
                            y=unit.position.y + dy
                        )
                        enemies = [
                            u for u in state.get_units_at(target)
                            if u.player != state.current_player
                        ]
                        if enemies:
                            actions.append(GameAction.attack(unit.id, target))

            # Settle action for settlers
            if "Settler" in unit.type:
                actions.append(GameAction(
                    action_type=ActionType.SETTLE,
                    unit_id=unit.id,
                ))

        # City production
        for city in state.get_own_cities():
            if not city.producing:
                for unit_type in ["Warrior", "Settler", "Worker", "Archer"]:
                    actions.append(GameAction.produce(city.id, unit_type))

        return actions

    def save(self) -> None:
        """Force save the current state."""
        if self._save_data:
            self._write_save(self._save_data)
            self._modified = False

    def load_game_state(self) -> GameState:
        """Load game state (alias for read_state after connect)."""
        if not self._connected:
            self.connect()
        return self.read_state()

    def apply_action(self, action: GameAction) -> bool:
        """Apply action (alias for send_action)."""
        return self.send_action(action)

    def _load_save(self) -> dict[str, Any]:
        """Load save file from disk."""
        if not self.save_path.exists():
            raise SaveFileError(f"Save file not found: {self.save_path}")

        try:
            # Try gzipped format first (standard Unciv format)
            if self.save_path.suffix == ".gz" or self._is_gzipped():
                with gzip.open(self.save_path, "rt", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # Plain JSON
                return json.loads(self.save_path.read_text())

        except json.JSONDecodeError as e:
            raise SaveFileError(f"Invalid JSON in save file: {e}") from e
        except Exception as e:
            raise SaveFileError(f"Failed to load save file: {e}") from e

    def _write_save(self, data: dict[str, Any]) -> None:
        """Write save file to disk."""
        try:
            # Create backup if enabled
            if self.backup_saves and self.save_path.exists():
                backup_path = self.save_path.with_suffix(".backup")
                shutil.copy2(self.save_path, backup_path)

            # Write in same format as loaded
            if self.save_path.suffix == ".gz" or self._is_gzipped():
                with gzip.open(self.save_path, "wt", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            else:
                self.save_path.write_text(json.dumps(data, indent=2))

            logger.debug(f"Saved to: {self.save_path}")

        except Exception as e:
            raise SaveFileError(f"Failed to write save file: {e}") from e

    def _is_gzipped(self) -> bool:
        """Check if save file is gzipped."""
        try:
            with open(self.save_path, "rb") as f:
                magic = f.read(2)
                return magic == b"\x1f\x8b"
        except Exception:
            return False

    def create_new_game(
        self,
        player_civs: list[str],
        map_size: tuple[int, int] = (40, 25),
        difficulty: str = "Prince",
    ) -> GameState:
        """Create a new game save.

        Args:
            player_civs: List of civilization names.
            map_size: Map width and height.
            difficulty: Difficulty level.

        Returns:
            Initial game state.
        """
        # Create minimal save structure
        self._save_data = {
            "gameId": f"alphazero_{hash(tuple(player_civs)) % 100000}",
            "gameInfo": {
                "turns": 0,
                "difficulty": difficulty,
                "speed": "Quick",
                "gameStatus": "",
            },
            "civilizations": [
                {
                    "civName": civ,
                    "isDefeated": False,
                    "hasPlayedTurn": False,
                    "gold": 0,
                    "units": [],
                    "cities": [],
                    "tech": {"techsResearched": []},
                }
                for civ in player_civs
            ],
            "tileMap": {
                "mapParameters": {
                    "size": {"width": map_size[0], "height": map_size[1]},
                },
                "tileList": self._generate_basic_map(map_size),
            },
        }

        # Add starting units and cities
        for i, _civ in enumerate(player_civs):
            start_x = 5 + (i * 10) % map_size[0]
            start_y = map_size[1] // 2

            # Starting settler
            self._save_data["civilizations"][i]["units"].append({
                "id": f"{i}_settler",
                "name": "Settler",
                "currentTile": f"{start_x},{start_y}",
                "health": 100,
                "currentMovement": 2,
                "baseUnit": {"movement": 2, "strength": 0},
            })

            # Starting warrior
            self._save_data["civilizations"][i]["units"].append({
                "id": f"{i}_warrior",
                "name": "Warrior",
                "currentTile": f"{start_x + 1},{start_y}",
                "health": 100,
                "currentMovement": 2,
                "baseUnit": {"movement": 2, "strength": 20},
            })

        self._connected = True
        self._player_id = 0 if player_civs else -1
        self.player_civ = player_civs[0] if player_civs else ""

        return self.read_state()

    def _generate_basic_map(
        self,
        size: tuple[int, int]
    ) -> list[dict[str, Any]]:
        """Generate a basic map for new games."""
        tiles = []
        for x in range(size[0]):
            for y in range(size[1]):
                terrain = "Grassland"
                # Add some variety
                if (x + y) % 7 == 0:
                    terrain = "Plains"
                elif (x + y) % 11 == 0:
                    terrain = "Hill"
                elif x == 0 or x == size[0] - 1 or y == 0 or y == size[1] - 1:
                    terrain = "Ocean"

                tiles.append({
                    "position": f"{x},{y}",
                    "baseTerrain": terrain,
                    "terrainFeatures": [],
                    "stats": {
                        "food": 2 if terrain == "Grassland" else 1,
                        "production": 2 if terrain == "Hill" else 1,
                    },
                })
        return tiles
