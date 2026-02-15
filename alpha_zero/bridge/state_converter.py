"""State converter for Civ6 bridge.

Converts raw Civ6 Lua mod JSON state to the format expected by the
Civ6StateEncoder for neural network input.

The Civ6 mod exports state in a specific format, which needs to be
transformed to match the encoder's expectations:
- Terrain types need mapping (TERRAIN_GRASS -> grassland)
- Unit types need mapping (UNIT_WARRIOR -> warrior/melee)
- Coordinates need scaling from Civ6 map size to board_size
- Visibility needs to be extracted per tile

Usage:
    from alpha_zero.bridge.state_converter import StateConverter

    converter = StateConverter(board_size=16)

    # Convert Civ6 JSON to encoder format
    encoder_state = converter.convert(civ6_state)

    # Or convert and encode to tensors directly
    spatial, vector = converter.convert_and_encode(civ6_state)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from alpha_zero.bridge.config import MapSizeConfig, get_map_config
from alpha_zero.bridge.protocol import (
    GameStateData,
)
from alpha_zero.envs.civ6_encoder import Civ6StateEncoder, EncoderConfig

logger = logging.getLogger(__name__)

# Named constants for victory progress normalization.
# Base game has ~67 techs; use 70 as a round estimate.
_ESTIMATED_TOTAL_TECHS: int = 70
# High score estimate for normalizing score-based victory progress.
_ESTIMATED_MAX_SCORE: int = 2000


@dataclass
class TerrainMapping:
    """Mapping from Civ6 terrain types to internal types."""

    # Civ6 terrain type -> internal type (matches Civ6StateEncoder.TERRAIN_TYPES)
    TERRAIN: dict[str, str] = field(
        default_factory=lambda: {
            # Base terrains - handle both naming conventions (GRASS vs GRASSLAND)
            "TERRAIN_GRASS": "grassland",
            "TERRAIN_GRASSLAND": "grassland",
            "TERRAIN_GRASS_HILLS": "grassland",  # Hills treated as feature
            "TERRAIN_GRASSLAND_HILLS": "grassland",
            "TERRAIN_GRASS_MOUNTAIN": "grassland",  # Mountains treated as feature
            "TERRAIN_GRASSLAND_MOUNTAIN": "grassland",
            "TERRAIN_PLAINS": "plains",
            "TERRAIN_PLAINS_HILLS": "plains",
            "TERRAIN_PLAINS_MOUNTAIN": "plains",
            "TERRAIN_DESERT": "desert",
            "TERRAIN_DESERT_HILLS": "desert",
            "TERRAIN_DESERT_MOUNTAIN": "desert",
            "TERRAIN_TUNDRA": "tundra",
            "TERRAIN_TUNDRA_HILLS": "tundra",
            "TERRAIN_TUNDRA_MOUNTAIN": "tundra",
            "TERRAIN_SNOW": "snow",
            "TERRAIN_SNOW_HILLS": "snow",
            "TERRAIN_SNOW_MOUNTAIN": "snow",
            "TERRAIN_COAST": "coast",
            "TERRAIN_OCEAN": "ocean",
        }
    )

    # Civ6 feature type -> internal type (matches Civ6StateEncoder.FEATURE_TYPES)
    FEATURES: dict[str, str] = field(
        default_factory=lambda: {
            "FEATURE_FOREST": "forest",
            "FEATURE_JUNGLE": "jungle",
            "FEATURE_MARSH": "marsh",
            "FEATURE_FLOODPLAINS": "floodplains",
            "FEATURE_FLOODPLAINS_GRASSLAND": "floodplains",
            "FEATURE_FLOODPLAINS_PLAINS": "floodplains",
            "FEATURE_OASIS": "floodplains",  # Map to similar
            "FEATURE_ICE": "ice",
            "FEATURE_REEF": "river",  # Approximate
            "FEATURE_VOLCANIC_SOIL": "floodplains",  # Approximate
            "FEATURE_GEOTHERMAL_FISSURE": "hills",  # Approximate
        }
    )


@dataclass
class UnitMapping:
    """Mapping from Civ6 unit types to internal unit categories."""

    # Civ6 unit type -> internal category (matches Civ6StateEncoder.UNIT_TYPES)
    UNITS: dict[str, str] = field(
        default_factory=lambda: {
            # Melee units
            "UNIT_WARRIOR": "melee",
            "UNIT_SWORDSMAN": "melee",
            "UNIT_MAN_AT_ARMS": "melee",
            "UNIT_MUSKETMAN": "melee",
            "UNIT_LINE_INFANTRY": "melee",
            "UNIT_INFANTRY": "melee",
            "UNIT_MECHANIZED_INFANTRY": "melee",
            # Anti-cavalry / spearmen
            "UNIT_SPEARMAN": "melee",
            "UNIT_PIKEMAN": "melee",
            "UNIT_PIKE_AND_SHOT": "melee",
            "UNIT_AT_CREW": "melee",
            # Ranged units
            "UNIT_SLINGER": "ranged",
            "UNIT_ARCHER": "ranged",
            "UNIT_CROSSBOWMAN": "ranged",
            "UNIT_FIELD_CANNON": "ranged",
            "UNIT_MACHINE_GUN": "ranged",
            # Cavalry
            "UNIT_HORSEMAN": "cavalry",
            "UNIT_KNIGHT": "cavalry",
            "UNIT_CAVALRY": "cavalry",
            "UNIT_TANK": "cavalry",
            "UNIT_MODERN_ARMOR": "cavalry",
            "UNIT_HORSE_ARCHER": "cavalry",
            # Siege
            "UNIT_CATAPULT": "siege",
            "UNIT_BOMBARD": "siege",
            "UNIT_ARTILLERY": "siege",
            "UNIT_ROCKET_ARTILLERY": "siege",
            "UNIT_BATTERING_RAM": "siege",
            "UNIT_SIEGE_TOWER": "siege",
            # Naval melee
            "UNIT_GALLEY": "naval_melee",
            "UNIT_CARAVEL": "naval_melee",
            "UNIT_IRONCLAD": "naval_melee",
            "UNIT_DESTROYER": "naval_melee",
            # Naval ranged
            "UNIT_QUADRIREME": "naval_ranged",
            "UNIT_FRIGATE": "naval_ranged",
            "UNIT_BATTLESHIP": "naval_ranged",
            "UNIT_MISSILE_CRUISER": "naval_ranged",
            "UNIT_AIRCRAFT_CARRIER": "naval_ranged",
            "UNIT_SUBMARINE": "naval_ranged",
            "UNIT_NUCLEAR_SUBMARINE": "naval_ranged",
            # Air
            "UNIT_BIPLANE": "air",
            "UNIT_FIGHTER": "air",
            "UNIT_JET_FIGHTER": "air",
            "UNIT_BOMBER": "air",
            "UNIT_JET_BOMBER": "air",
            # Support
            "UNIT_MEDIC": "support",
            "UNIT_OBSERVATION_BALLOON": "support",
            "UNIT_DRONE": "support",
            # Civilian
            "UNIT_SETTLER": "civilian",
            "UNIT_BUILDER": "civilian",
            "UNIT_TRADER": "civilian",
            "UNIT_SCOUT": "civilian",
            "UNIT_RANGER": "civilian",
            "UNIT_SPY": "civilian",
            # Religious
            "UNIT_MISSIONARY": "religious",
            "UNIT_APOSTLE": "religious",
            "UNIT_INQUISITOR": "religious",
            "UNIT_GURU": "religious",
            # Great people
            "UNIT_GREAT_GENERAL": "great_person",
            "UNIT_GREAT_ADMIRAL": "great_person",
            "UNIT_GREAT_ENGINEER": "great_person",
            "UNIT_GREAT_MERCHANT": "great_person",
            "UNIT_GREAT_PROPHET": "great_person",
            "UNIT_GREAT_SCIENTIST": "great_person",
            "UNIT_GREAT_WRITER": "great_person",
            "UNIT_GREAT_ARTIST": "great_person",
            "UNIT_GREAT_MUSICIAN": "great_person",
        }
    )

    def get_unit_type(self, civ6_type: str) -> str:
        """Get internal unit type for Civ6 type.

        Args:
            civ6_type: Civ6 unit type string.

        Returns:
            Internal unit type (defaults to 'melee' if unknown).
        """
        # Try exact match
        if civ6_type in self.UNITS:
            return self.UNITS[civ6_type]

        # Try partial match (some units have unique names)
        civ6_upper = civ6_type.upper()
        for key, value in self.UNITS.items():
            if key in civ6_upper or civ6_upper in key:
                return value

        # Log warning for unknown unit types to help identify mapping gaps
        logger.warning(
            f"Unknown Civ6 unit type '{civ6_type}', defaulting to 'melee'. "
            "Consider adding this unit to the UnitMapping."
        )

        # Default to melee for unknown military units
        return "melee"


class StateConverter:
    """Converts Civ6 Lua mod state to encoder-compatible format.

    This class handles the transformation from the raw JSON format
    output by the Civ6 Lua mod to the dictionary format expected
    by Civ6StateEncoder.

    Attributes:
        board_size: Target board size for coordinate scaling.
        terrain_mapping: Terrain type mappings.
        unit_mapping: Unit type mappings.
        encoder: Optional Civ6StateEncoder for direct tensor output.
    """

    def __init__(
        self,
        board_size: int = 16,
        terrain_mapping: TerrainMapping | None = None,
        unit_mapping: UnitMapping | None = None,
        create_encoder: bool = True,
    ) -> None:
        """Initialize the state converter.

        Args:
            board_size: Target board size for the neural network.
            terrain_mapping: Custom terrain mappings (uses defaults if None).
            unit_mapping: Custom unit mappings (uses defaults if None).
            create_encoder: Whether to create an internal encoder.
        """
        self.board_size = board_size
        self.terrain_mapping = terrain_mapping or TerrainMapping()
        self.unit_mapping = unit_mapping or UnitMapping()

        self._encoder: Civ6StateEncoder | None = None
        if create_encoder:
            self._encoder = Civ6StateEncoder(
                EncoderConfig(
                    board_size=board_size,
                    spatial_channels=84,
                    vector_dim=512,
                )
            )

        self._map_config: MapSizeConfig | None = None
        self._scale_x: float = 1.0
        self._scale_y: float = 1.0

    def convert(self, civ6_state: GameStateData | dict[str, Any]) -> dict[str, Any]:
        """Convert Civ6 state to encoder-compatible format.

        Args:
            civ6_state: State from Civ6 Lua mod (GameStateData or dict).

        Returns:
            Dictionary compatible with Civ6StateEncoder.encode().
        """
        # Handle both GameStateData and raw dict
        if isinstance(civ6_state, dict):
            from alpha_zero.bridge.protocol import GameStateData

            state = GameStateData.from_dict(civ6_state)
        else:
            state = civ6_state

        # Validate map size before scaling
        if state.map_data.width <= 0 or state.map_data.height <= 0:
            raise ValueError(
                f"Invalid map size: {state.map_data.width}x{state.map_data.height}. "
                "Map dimensions must be positive."
            )

        # Set up coordinate scaling
        self._setup_scaling(state.map_data.width, state.map_data.height)

        # Convert each component
        terrain = self._convert_terrain(state)
        units = self._convert_units(state)
        cities = self._convert_cities(state)
        visibility = self._convert_visibility(state)

        # Build player info
        current_player = self._get_current_player_id(state)
        player_data = self._extract_player_data(state, current_player)

        return {
            "turn": state.turn_id,
            "current_player": current_player,
            "terrain": terrain,
            "units": units,
            "cities": cities,
            "visibility": visibility,
            # Player economy
            "gold": player_data.get("gold", 0),
            "gold_per_turn": player_data.get("gold_per_turn", 0),
            "science_per_turn": player_data.get("science_per_turn", 0),
            "culture_per_turn": player_data.get("culture_per_turn", 0),
            "faith_per_turn": player_data.get("faith", 0),
            # Military
            "military_strength": player_data.get("military_strength", 0),
            # Score (for reward calculation)
            "score": player_data.get("score", 0),
            # Era and progress
            "era": state.era,
            "resources": self._extract_resources(state),
            "technologies": self._extract_technologies(state),
            "civics": self._extract_civics(state),
            "victory_progress": self._extract_victory_progress(state),
            # Store original legal actions for action mask generation
            "_legal_actions": state.legal_actions,
            "_map_size": (state.map_data.width, state.map_data.height),
            "_player_id": state.player_id,
        }

    def convert_and_encode(
        self,
        civ6_state: GameStateData | dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert Civ6 state and encode to tensors.

        Args:
            civ6_state: State from Civ6 Lua mod.

        Returns:
            Tuple of (spatial_tensor, vector_tensor).
        """
        if self._encoder is None:
            raise RuntimeError("Encoder not created. Use create_encoder=True.")

        encoder_state = self.convert(civ6_state)
        return self._encoder.encode(encoder_state)

    def _setup_scaling(self, map_width: int, map_height: int) -> None:
        """Set up coordinate scaling for the map.

        Args:
            map_width: Original map width.
            map_height: Original map height.
        """
        if (
            self._map_config is None
            or self._map_config.width != map_width
            or self._map_config.height != map_height
        ):
            self._map_config = get_map_config(map_width, map_height)
            self._scale_x, self._scale_y = self._map_config.get_scale_factor(self.board_size)

    def _scale_coords(self, x: int, y: int) -> tuple[int, int]:
        """Scale coordinates from Civ6 map to board_size.

        Uses proper rounding (round half up) instead of truncation to avoid
        systematic bias toward (0,0). This ensures more accurate position
        mapping especially for entities near tile boundaries.

        Args:
            x: Original X coordinate.
            y: Original Y coordinate.

        Returns:
            Scaled (x, y) coordinates clamped to valid range.
        """
        # Use round() for proper rounding instead of int() truncation
        # This avoids systematic error where entities cluster toward origin
        scaled_x = round(x * self._scale_x)
        scaled_y = round(y * self._scale_y)

        # Clamp to valid range [0, board_size - 1]
        scaled_x = max(0, min(self.board_size - 1, scaled_x))
        scaled_y = max(0, min(self.board_size - 1, scaled_y))

        return (scaled_x, scaled_y)

    def _convert_terrain(self, state: GameStateData) -> dict[tuple[int, int], dict]:
        """Convert terrain data with collision handling.

        When multiple Civ6 tiles map to the same scaled position (due to
        coordinate scaling), this method uses majority voting for terrain type
        and aggregates features from all contributing tiles.

        Args:
            state: Game state data.

        Returns:
            Dictionary mapping (x, y) to terrain info.
        """
        terrain: dict[tuple[int, int], dict] = {}

        # Initialize with default terrain
        for y in range(self.board_size):
            for x in range(self.board_size):
                terrain[(x, y)] = {
                    "type": "plains",
                    "features": [],
                    "yields": {},
                }

        # Track collisions: position -> list of (terrain_type, features, yields)
        collision_data: dict[tuple[int, int], list[tuple[str, list[str], dict]]] = {}

        # Process tiles from state, collecting collision data
        for tile in state.map_data.tiles:
            scaled_pos = self._scale_coords(tile.x, tile.y)

            # Get terrain type
            terrain_type = self.terrain_mapping.TERRAIN.get(tile.terrain, "plains")

            # Check for hills/mountains in terrain name
            features = list(tile.features) if tile.features else []
            if "HILLS" in tile.terrain.upper():
                features.append("hills")
            if "MOUNTAIN" in tile.terrain.upper():
                features.append("mountains")

            # Map features
            mapped_features = []
            for feature in features:
                if feature in self.terrain_mapping.FEATURES:
                    mapped_features.append(self.terrain_mapping.FEATURES[feature])
                elif feature.lower() in ["hills", "mountains", "forest", "jungle", "marsh"]:
                    mapped_features.append(feature.lower())

            # Track collision data
            if scaled_pos not in collision_data:
                collision_data[scaled_pos] = []
            collision_data[scaled_pos].append((terrain_type, mapped_features, tile.yields))

        # Resolve collisions using majority voting and feature aggregation
        collision_count = 0
        for pos, tiles_data in collision_data.items():
            if len(tiles_data) == 1:
                # No collision - use the single tile directly
                terrain_type, mapped_features, yields = tiles_data[0]
                terrain[pos] = {
                    "type": terrain_type,
                    "features": mapped_features,
                    "yields": yields,
                }
            else:
                # Collision detected - resolve it
                collision_count += 1
                resolved = self._resolve_terrain_collision(tiles_data)
                terrain[pos] = resolved

        if collision_count > 0:
            logger.debug(
                f"Resolved {collision_count} terrain collisions during coordinate scaling "
                f"(map {state.map_data.width}x{state.map_data.height} -> {self.board_size}x{self.board_size})"
            )

        return terrain

    def _resolve_terrain_collision(
        self,
        tiles_data: list[tuple[str, list[str], dict]],
    ) -> dict[str, Any]:
        """Resolve terrain collision when multiple tiles map to same position.

        Uses majority voting for terrain type and aggregates all features.
        For yields, uses average values.

        Args:
            tiles_data: List of (terrain_type, features, yields) tuples.

        Returns:
            Resolved terrain dictionary.
        """
        from collections import Counter

        # Majority vote for terrain type
        terrain_types = [t[0] for t in tiles_data]
        terrain_counter = Counter(terrain_types)
        # Get most common, with fallback priority for strategic terrains
        most_common = terrain_counter.most_common()

        # Terrain priority for ties (prefer more strategic/visible terrain)
        terrain_priority = {
            "mountains": 10,  # Impassable - important to preserve
            "coast": 9,
            "ocean": 8,
            "desert": 7,
            "tundra": 6,
            "snow": 5,
            "plains": 4,
            "grassland": 3,
        }

        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            # Tie - use priority
            tied_terrains = [t for t, c in most_common if c == most_common[0][1]]
            resolved_terrain = max(tied_terrains, key=lambda t: terrain_priority.get(t, 0))
        else:
            resolved_terrain = most_common[0][0]

        # Aggregate features (union of all features, deduplicated)
        all_features: set[str] = set()
        for _, features, _ in tiles_data:
            all_features.update(features)

        # Average yields
        yield_sums: dict[str, float] = {}
        yield_counts: dict[str, int] = {}
        for _, _, yields in tiles_data:
            for key, value in yields.items():
                yield_sums[key] = yield_sums.get(key, 0) + value
                yield_counts[key] = yield_counts.get(key, 0) + 1

        averaged_yields = {key: int(yield_sums[key] / yield_counts[key]) for key in yield_sums}

        return {
            "type": resolved_terrain,
            "features": list(all_features),
            "yields": averaged_yields,
        }

    def _convert_units(self, state: GameStateData) -> list[dict]:
        """Convert unit data.

        Args:
            state: Game state data.

        Returns:
            List of unit dictionaries.
        """
        units = []
        current_player = self._get_current_player_id(state)

        for unit in state.units:
            scaled_pos = self._scale_coords(unit.x, unit.y)
            unit_type = self.unit_mapping.get_unit_type(unit.unit_type)

            # Determine player index (0 = current player, 1+ = others)
            if unit.owner_id == state.player_id:
                player = current_player
            else:
                player = 1 if current_player == 0 else 0

            health = unit.health
            if unit.max_health > 0:
                health = int(100 * unit.health / unit.max_health)

            units.append(
                {
                    "position": scaled_pos,
                    "type": unit_type,
                    "player": player,
                    "health": health,
                    "strength": unit.combat_strength,
                    "moves_remaining": unit.moves_remaining,
                    "max_moves": unit.max_moves,
                    # Preserve original data for action conversion
                    "_id": unit.id,
                    "_original_pos": (unit.x, unit.y),
                    "_owner_id": unit.owner_id,
                }
            )

        return units

    def _convert_cities(self, state: GameStateData) -> list[dict]:
        """Convert city data.

        Args:
            state: Game state data.

        Returns:
            List of city dictionaries.
        """
        cities = []
        current_player = self._get_current_player_id(state)

        for city in state.cities:
            scaled_pos = self._scale_coords(city.x, city.y)

            # Determine player index
            if city.owner_id == state.player_id:
                player = current_player
            else:
                player = 1 if current_player == 0 else 0

            cities.append(
                {
                    "position": scaled_pos,
                    "player": player,
                    "name": city.name,
                    "population": city.population,
                    "defense": city.defense_strength,
                    "production": city.production_progress,
                    "is_capital": city.is_capital,
                    # Preserve original data
                    "_id": city.id,
                    "_original_pos": (city.x, city.y),
                    "_owner_id": city.owner_id,
                    "_buildings": city.buildings,
                    "_available_production": city.available_production,
                }
            )

        return cities

    def _convert_visibility(self, state: GameStateData) -> dict[tuple[int, int], str]:
        """Convert visibility data with collision handling.

        When multiple tiles map to the same position, uses the highest
        visibility level (visible > seen > fog) to ensure important
        visibility information is preserved.

        Args:
            state: Game state data.

        Returns:
            Dictionary mapping (x, y) to visibility state.
        """
        visibility: dict[tuple[int, int], str] = {}

        # Visibility priority (higher = more visible)
        visibility_priority = {"fog": 0, "seen": 1, "visible": 2}

        # Default to fog
        for y in range(self.board_size):
            for x in range(self.board_size):
                visibility[(x, y)] = "fog"

        # Update from tiles, keeping highest visibility on collision
        for tile in state.map_data.tiles:
            scaled_pos = self._scale_coords(tile.x, tile.y)

            if tile.is_visible:
                new_vis = "visible"
            elif tile.is_revealed:
                new_vis = "seen"
            else:
                new_vis = "fog"

            # Only update if new visibility is higher priority (handles collisions)
            current_vis = visibility[scaled_pos]
            if visibility_priority[new_vis] > visibility_priority[current_vis]:
                visibility[scaled_pos] = new_vis

        return visibility

    def _get_current_player_id(self, state: GameStateData) -> int:
        """Get the current player index (0-based for encoder).

        Args:
            state: Game state data.

        Returns:
            Player index (0 = AI player, 1 = opponent).
        """
        # For the encoder, we always treat the AI as player 0
        return 0

    def _extract_player_data(
        self,
        state: GameStateData,
        player_idx: int,
    ) -> dict[str, Any]:
        """Extract player-specific data.

        Args:
            state: Game state data.
            player_idx: Player index to extract.

        Returns:
            Dictionary of player data.
        """
        # Try to find player in players list
        for player in state.players:
            if player.id == state.player_id:
                return {
                    "gold": player.gold,
                    "gold_per_turn": 0,  # Not directly available
                    "science_per_turn": player.science_per_turn,
                    "culture_per_turn": player.culture_per_turn,
                    "faith": player.faith,
                    "score": player.score,
                    "military_strength": 0,  # Calculate from units if needed
                }

        # Default values
        return {
            "gold": 0,
            "gold_per_turn": 0,
            "science_per_turn": 0,
            "culture_per_turn": 0,
            "faith": 0,
            "score": 0,
            "military_strength": 0,
        }

    def _extract_resources(self, state: GameStateData) -> dict[str, int]:
        """Extract resource counts from map tiles.

        Collects all resources visible on the map tiles and counts
        how many of each resource type are controlled by the player.

        Args:
            state: Game state data.

        Returns:
            Dictionary mapping resource type to count.
        """
        resources: dict[str, int] = {}

        for tile in state.map_data.tiles:
            # Only count resources on tiles we own
            if tile.resource and tile.owner_id == state.player_id:
                resource_name = self._normalize_resource_name(tile.resource)
                resources[resource_name] = resources.get(resource_name, 0) + 1

        return resources

    def _normalize_resource_name(self, resource: str) -> str:
        """Normalize Civ6 resource name to internal format.

        Args:
            resource: Civ6 resource name (e.g., "RESOURCE_IRON").

        Returns:
            Normalized name (e.g., "iron").
        """
        # Remove common prefixes and convert to lowercase
        name = resource.lower()
        for prefix in ("resource_", "res_"):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        return name

    def _extract_technologies(self, state: GameStateData) -> list[str]:
        """Extract researched technologies for the current player.

        Args:
            state: Game state data.

        Returns:
            List of technology names that have been researched.
        """
        for player in state.players:
            if player.id == state.player_id:
                return [self._normalize_tech_name(tech) for tech in player.techs_researched]
        return []

    def _normalize_tech_name(self, tech: str) -> str:
        """Normalize Civ6 technology name to internal format.

        Args:
            tech: Civ6 technology name (e.g., "TECH_POTTERY").

        Returns:
            Normalized name (e.g., "pottery").
        """
        name = tech.lower()
        for prefix in ("tech_",):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        return name

    def _extract_civics(self, state: GameStateData) -> list[str]:
        """Extract researched civics for the current player.

        Args:
            state: Game state data.

        Returns:
            List of civic names that have been researched.
        """
        for player in state.players:
            if player.id == state.player_id:
                return [self._normalize_civic_name(civic) for civic in player.civics_researched]
        return []

    def _normalize_civic_name(self, civic: str) -> str:
        """Normalize Civ6 civic name to internal format.

        Args:
            civic: Civ6 civic name (e.g., "CIVIC_CODE_OF_LAWS").

        Returns:
            Normalized name (e.g., "code_of_laws").
        """
        name = civic.lower()
        for prefix in ("civic_",):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        return name

    def _extract_victory_progress(self, state: GameStateData) -> dict[str, float]:
        """Extract victory progress indicators.

        Victory progress is calculated from available game state data:
        - Science victory: tech count / total techs
        - Domination: controlled capitals / total capitals
        - Score: normalized current score

        Args:
            state: Game state data.

        Returns:
            Dictionary mapping victory type to progress (0.0 to 1.0).
        """
        progress: dict[str, float] = {}

        # Science progress (rough estimate based on techs)
        techs = self._extract_technologies(state)
        progress["science"] = min(1.0, len(techs) / _ESTIMATED_TOTAL_TECHS)

        # Domination progress (count our cities vs total cities)
        total_cities = len(state.cities)
        our_cities = sum(1 for c in state.cities if c.owner_id == state.player_id)
        if total_cities > 0:
            progress["domination"] = our_cities / total_cities
        else:
            progress["domination"] = 0.0

        # Score progress (normalized by max reasonable score)
        for player in state.players:
            if player.id == state.player_id:
                progress["score"] = min(1.0, player.score / _ESTIMATED_MAX_SCORE)
                break
        else:
            progress["score"] = 0.0

        # Game completion indicator
        if state.is_game_over:
            if state.winner_id == state.player_id:
                progress["complete"] = 1.0  # Victory
            else:
                progress["complete"] = -1.0  # Defeat
        else:
            progress["complete"] = 0.0

        return progress

    def inverse_scale_coords(self, x: int, y: int) -> tuple[int, int]:
        """Convert scaled coordinates back to original Civ6 coordinates.

        Args:
            x: Scaled X coordinate.
            y: Scaled Y coordinate.

        Returns:
            Original (x, y) coordinates.
        """
        if self._map_config is None:
            return (x, y)

        orig_x = round(x / self._scale_x)
        orig_y = round(y / self._scale_y)

        # Clamp to original map bounds
        orig_x = max(0, min(self._map_config.width - 1, orig_x))
        orig_y = max(0, min(self._map_config.height - 1, orig_y))

        return (orig_x, orig_y)

    @property
    def map_config(self) -> MapSizeConfig | None:
        """Get the current map configuration."""
        return self._map_config

    @property
    def encoder(self) -> Civ6StateEncoder | None:
        """Get the internal encoder."""
        return self._encoder
