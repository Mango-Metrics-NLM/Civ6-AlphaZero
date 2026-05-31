"""Tests for alpha_zero.bridge.state_converter module.

Covers:
- TerrainMapping defaults and custom
- UnitMapping and get_unit_type
- StateConverter.__init__
- StateConverter.convert with GameStateData and dict input
- StateConverter._setup_scaling
- StateConverter._convert_terrain
- StateConverter._convert_units
- StateConverter._convert_cities
- StateConverter._scale_coords / inverse_scale_coords
- Named constants _ESTIMATED_TOTAL_TECHS, _ESTIMATED_MAX_SCORE
- Multiplayer helpers (build_player_id_mapping, _remap_player_ids)
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.protocol import (
    CityData,
    GameStateData,
    LegalAction,
    MapData,
    PlayerData,
    TileData,
    UnitData,
)
from alpha_zero.bridge.state_converter import (
    GameNotStartedError,
    StateConverter,
    TerrainMapping,
    UnitMapping,
)
from alpha_zero.bridge.config import GameConstants

# ---------------------------------------------------------------------------
# Helpers to build minimal game state data
# ---------------------------------------------------------------------------


def _make_tile(
    x: int = 0,
    y: int = 0,
    terrain: str = "TERRAIN_PLAINS",
    features: list[str] | None = None,
    resource: str | None = None,
    owner_id: int | None = None,
    is_visible: bool = True,
    is_revealed: bool = True,
    yields: dict[str, int] | None = None,
) -> TileData:
    return TileData(
        x=x,
        y=y,
        terrain=terrain,
        features=features or [],
        resource=resource,
        owner_id=owner_id,
        is_visible=is_visible,
        is_revealed=is_revealed,
        yields=yields or {},
    )


def _make_unit(
    id: int = 1,
    unit_type: str = "UNIT_WARRIOR",
    owner_id: int = 0,
    x: int = 0,
    y: int = 0,
    health: int = 100,
    max_health: int = 100,
    combat_strength: int = 20,
    moves_remaining: float = 2.0,
    max_moves: float = 2.0,
) -> UnitData:
    return UnitData(
        id=id,
        unit_type=unit_type,
        owner_id=owner_id,
        x=x,
        y=y,
        health=health,
        max_health=max_health,
        combat_strength=combat_strength,
        moves_remaining=moves_remaining,
        max_moves=max_moves,
    )


def _make_city(
    id: int = 1,
    name: str = "TestCity",
    owner_id: int = 0,
    x: int = 5,
    y: int = 5,
    population: int = 3,
    is_capital: bool = True,
    defense_strength: int = 20,
    production_progress: int = 10,
    buildings: list[str] | None = None,
    available_production: list[int] | None = None,
) -> CityData:
    return CityData(
        id=id,
        name=name,
        owner_id=owner_id,
        x=x,
        y=y,
        population=population,
        is_capital=is_capital,
        defense_strength=defense_strength,
        production_progress=production_progress,
        buildings=buildings or [],
        available_production=available_production or [],
    )


def _make_player(
    id: int = 0,
    civilization: str = "ROME",
    gold: float = 100.0,
    science_per_turn: float = 10.0,
    culture_per_turn: float = 5.0,
    faith: float = 0.0,
    score: int = 50,
    techs_researched: list[str] | None = None,
    civics_researched: list[str] | None = None,
) -> PlayerData:
    return PlayerData(
        id=id,
        civilization=civilization,
        gold=gold,
        science_per_turn=science_per_turn,
        culture_per_turn=culture_per_turn,
        faith=faith,
        score=score,
        techs_researched=techs_researched or [],
        civics_researched=civics_researched or [],
    )


def _make_game_state(
    turn_id: int = 1,
    player_id: int = 0,
    map_width: int = 20,
    map_height: int = 20,
    tiles: list[TileData] | None = None,
    units: list[UnitData] | None = None,
    cities: list[CityData] | None = None,
    players: list[PlayerData] | None = None,
    legal_actions: list[LegalAction] | None = None,
    era: int = 0,
    is_game_over: bool = False,
    winner_id: int | None = None,
) -> GameStateData:
    if tiles is None:
        tiles = [_make_tile(x=0, y=0)]
    if units is None:
        units = []
    if cities is None:
        cities = []
    if players is None:
        players = [_make_player(id=player_id)]
    if legal_actions is None:
        legal_actions = []

    return GameStateData(
        turn_id=turn_id,
        player_id=player_id,
        map_data=MapData(width=map_width, height=map_height, tiles=tiles),
        units=units,
        cities=cities,
        players=players,
        legal_actions=legal_actions,
        era=era,
        is_game_over=is_game_over,
        winner_id=winner_id,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def converter() -> StateConverter:
    """StateConverter with create_encoder=False for unit tests."""
    return StateConverter(board_size=16, create_encoder=False)


@pytest.fixture
def converter_8() -> StateConverter:
    """Small board StateConverter."""
    return StateConverter(board_size=8, create_encoder=False)


# ===========================================================================
# 1. TerrainMapping
# ===========================================================================


class TestTerrainMapping:
    """Test terrain mapping defaults and custom overrides."""

    def test_default_terrain_mapping(self) -> None:
        mapping = TerrainMapping()
        assert "TERRAIN_GRASS" in mapping.TERRAIN
        assert mapping.TERRAIN["TERRAIN_GRASS"] == "grassland"

    def test_plains_mapping(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_PLAINS"] == "plains"

    def test_desert_mapping(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_DESERT"] == "desert"

    def test_coast_and_ocean(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_COAST"] == "coast"
        assert mapping.TERRAIN["TERRAIN_OCEAN"] == "ocean"

    def test_hills_variant_mapped_to_base(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_PLAINS_HILLS"] == "plains"

    def test_mountain_variant_mapped_to_base(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_DESERT_MOUNTAIN"] == "desert"

    def test_feature_mapping_forest(self) -> None:
        mapping = TerrainMapping()
        assert mapping.FEATURES["FEATURE_FOREST"] == "forest"

    def test_feature_mapping_jungle(self) -> None:
        mapping = TerrainMapping()
        assert mapping.FEATURES["FEATURE_JUNGLE"] == "jungle"

    def test_feature_mapping_ice(self) -> None:
        mapping = TerrainMapping()
        assert mapping.FEATURES["FEATURE_ICE"] == "ice"

    def test_custom_terrain_mapping(self) -> None:
        custom_terrain = {"MY_TERRAIN": "custom"}
        mapping = TerrainMapping(TERRAIN=custom_terrain)
        assert mapping.TERRAIN == {"MY_TERRAIN": "custom"}
        # Default features should still be present
        assert "FEATURE_FOREST" in mapping.FEATURES

    def test_custom_feature_mapping(self) -> None:
        custom_features = {"MY_FEATURE": "custom_feat"}
        mapping = TerrainMapping(FEATURES=custom_features)
        assert mapping.FEATURES == {"MY_FEATURE": "custom_feat"}
        # Default terrain should still be present
        assert "TERRAIN_GRASS" in mapping.TERRAIN

    def test_grassland_alternate_key(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_GRASSLAND"] == "grassland"

    def test_tundra_and_snow(self) -> None:
        mapping = TerrainMapping()
        assert mapping.TERRAIN["TERRAIN_TUNDRA"] == "tundra"
        assert mapping.TERRAIN["TERRAIN_SNOW"] == "snow"


# ===========================================================================
# 2. UnitMapping
# ===========================================================================


class TestUnitMapping:
    """Test unit type mapping and get_unit_type fallback logic."""

    def test_warrior_maps_to_melee(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_WARRIOR"] == "melee"

    def test_archer_maps_to_ranged(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_ARCHER"] == "ranged"

    def test_horseman_maps_to_cavalry(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_HORSEMAN"] == "cavalry"

    def test_catapult_maps_to_siege(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_CATAPULT"] == "siege"

    def test_galley_maps_to_naval_melee(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_GALLEY"] == "naval_melee"

    def test_frigate_maps_to_naval_ranged(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_FRIGATE"] == "naval_ranged"

    def test_fighter_maps_to_air(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_FIGHTER"] == "air"

    def test_settler_maps_to_civilian(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_SETTLER"] == "civilian"

    def test_missionary_maps_to_religious(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_MISSIONARY"] == "religious"

    def test_great_general_maps_to_great_person(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_GREAT_GENERAL"] == "great_person"

    def test_medic_maps_to_support(self) -> None:
        mapping = UnitMapping()
        assert mapping.UNITS["UNIT_MEDIC"] == "support"

    def test_get_unit_type_exact_match(self) -> None:
        mapping = UnitMapping()
        assert mapping.get_unit_type("UNIT_WARRIOR") == "melee"

    def test_get_unit_type_unknown_defaults_to_melee(self) -> None:
        mapping = UnitMapping()
        result = mapping.get_unit_type("UNIT_TOTALLY_UNKNOWN")
        assert result == "melee"

    def test_get_unit_type_partial_match(self) -> None:
        """Partial matching: 'UNIT_WARRIOR' should be found in 'MY_CUSTOM_UNIT_WARRIOR_V2'."""
        mapping = UnitMapping()
        # The partial match logic: civ6_upper in key or key in civ6_upper
        # "UNIT_WARRIOR" is in "MY_CUSTOM_UNIT_WARRIOR_V2" -> True
        result = mapping.get_unit_type("MY_CUSTOM_UNIT_WARRIOR_V2")
        # Should find UNIT_WARRIOR as a partial match
        assert result == "melee"

    def test_get_unit_type_case_insensitive_partial(self) -> None:
        """Partial match is case insensitive (uppercased)."""
        mapping = UnitMapping()
        result = mapping.get_unit_type("unit_archer")
        # "UNIT_ARCHER" is in "UNIT_ARCHER" (exact after upper)
        assert result == "ranged"


# ===========================================================================
# 3. StateConverter.__init__
# ===========================================================================


class TestStateConverterInit:
    """StateConverter initialization with various parameters."""

    def test_defaults(self) -> None:
        conv = StateConverter(create_encoder=False)
        assert conv.board_size == 16
        assert conv.num_players == 2
        assert isinstance(conv.terrain_mapping, TerrainMapping)
        assert isinstance(conv.unit_mapping, UnitMapping)
        assert conv._encoder is None

    def test_custom_board_size(self) -> None:
        conv = StateConverter(board_size=32, create_encoder=False)
        assert conv.board_size == 32

    def test_custom_num_players(self) -> None:
        conv = StateConverter(num_players=4, create_encoder=False)
        assert conv.num_players == 4

    def test_custom_terrain_mapping(self) -> None:
        custom = TerrainMapping(TERRAIN={"MY_T": "custom"})
        conv = StateConverter(terrain_mapping=custom, create_encoder=False)
        assert conv.terrain_mapping.TERRAIN == {"MY_T": "custom"}

    def test_custom_unit_mapping(self) -> None:
        custom = UnitMapping(UNITS={"MY_UNIT": "melee"})
        conv = StateConverter(unit_mapping=custom, create_encoder=False)
        assert conv.unit_mapping.UNITS == {"MY_UNIT": "melee"}

    def test_create_encoder_false(self) -> None:
        conv = StateConverter(create_encoder=False)
        assert conv._encoder is None
        assert conv.encoder is None

    def test_create_encoder_true(self) -> None:
        conv = StateConverter(create_encoder=True)
        assert conv._encoder is not None
        assert conv.encoder is not None

    def test_initial_scale_factors(self) -> None:
        conv = StateConverter(create_encoder=False)
        assert conv._scale_x == 1.0
        assert conv._scale_y == 1.0

    def test_initial_map_config_none(self) -> None:
        conv = StateConverter(create_encoder=False)
        assert conv._map_config is None
        assert conv.map_config is None


# ===========================================================================
# 4. StateConverter.convert
# ===========================================================================


class TestStateConverterConvert:
    """Tests for the main convert method."""

    def test_convert_valid_game_state_data(self, converter: StateConverter) -> None:
        state = _make_game_state(
            turn_id=5,
            player_id=0,
            map_width=20,
            map_height=20,
            tiles=[_make_tile(x=0, y=0, terrain="TERRAIN_GRASS")],
            units=[_make_unit(id=1, owner_id=0, x=3, y=4)],
            cities=[_make_city(id=10, owner_id=0, x=5, y=5)],
            players=[
                _make_player(
                    id=0,
                    gold=200,
                    score=100,
                    techs_researched=["TECH_POTTERY"],
                    civics_researched=["CIVIC_CODE_OF_LAWS"],
                )
            ],
        )
        result = converter.convert(state)

        assert result["turn"] == 5
        assert result["current_player"] == 0
        assert isinstance(result["terrain"], dict)
        assert isinstance(result["units"], list)
        assert isinstance(result["cities"], list)
        assert isinstance(result["visibility"], dict)
        assert result["gold"] == 200
        assert result["score"] == 100
        assert result["_map_size"] == (20, 20)
        assert result["_player_id"] == 0

    def test_convert_dict_input(self, converter: StateConverter) -> None:
        """Convert should accept raw dict and parse via GameStateData.from_dict."""
        raw_dict: dict[str, Any] = {
            "turn_id": 3,
            "player_id": 0,
            "map": {
                "width": 20,
                "height": 20,
                "tiles": [{"x": 0, "y": 0, "terrain": "TERRAIN_PLAINS"}],
            },
            "units": [],
            "cities": [],
            "players": [{"id": 0, "civilization": "ROME"}],
            "legal_actions": [],
        }
        result = converter.convert(raw_dict)
        assert result["turn"] == 3
        assert result["current_player"] == 0

    def test_convert_invalid_map_size_raises(self, converter: StateConverter) -> None:
        state = _make_game_state(map_width=0, map_height=20)
        with pytest.raises(GameNotStartedError, match="Invalid map size"):
            converter.convert(state)

    def test_convert_negative_map_height_raises(self, converter: StateConverter) -> None:
        state = _make_game_state(map_width=20, map_height=-1)
        with pytest.raises(GameNotStartedError, match="Invalid map size"):
            converter.convert(state)

    def test_convert_stores_legal_actions(self, converter: StateConverter) -> None:
        state = _make_game_state(
            legal_actions=[LegalAction(action_type="end_turn")]
        )
        result = converter.convert(state)
        assert len(result["_legal_actions"]) == 1

    def test_convert_extracts_technologies(self, converter: StateConverter) -> None:
        state = _make_game_state(
            players=[
                _make_player(
                    id=0, techs_researched=["TECH_POTTERY", "TECH_MINING"]
                )
            ]
        )
        result = converter.convert(state)
        assert "pottery" in result["technologies"]
        assert "mining" in result["technologies"]

    def test_convert_extracts_civics(self, converter: StateConverter) -> None:
        state = _make_game_state(
            players=[
                _make_player(
                    id=0, civics_researched=["CIVIC_CODE_OF_LAWS"]
                )
            ]
        )
        result = converter.convert(state)
        assert "code_of_laws" in result["civics"]

    def test_convert_extracts_era(self, converter: StateConverter) -> None:
        state = _make_game_state(era=2)
        result = converter.convert(state)
        assert result["era"] == 2

    def test_convert_extracts_victory_progress(self, converter: StateConverter) -> None:
        state = _make_game_state(
            players=[_make_player(id=0, score=500, techs_researched=["TECH_A"])],
            cities=[_make_city(id=1, owner_id=0), _make_city(id=2, owner_id=1)],
        )
        result = converter.convert(state)
        vp = result["victory_progress"]
        assert "science" in vp
        assert "domination" in vp
        assert "score" in vp
        assert "complete" in vp
        assert vp["complete"] == 0.0  # game not over

    def test_convert_victory_progress_game_won(self, converter: StateConverter) -> None:
        state = _make_game_state(
            player_id=0,
            is_game_over=True,
            winner_id=0,
            players=[_make_player(id=0, score=100)],
        )
        result = converter.convert(state)
        assert result["victory_progress"]["complete"] == 1.0

    def test_convert_victory_progress_game_lost(self, converter: StateConverter) -> None:
        state = _make_game_state(
            player_id=0,
            is_game_over=True,
            winner_id=1,
            players=[_make_player(id=0, score=50)],
        )
        result = converter.convert(state)
        assert result["victory_progress"]["complete"] == -1.0

    def test_convert_default_player_data_when_not_found(
        self, converter: StateConverter
    ) -> None:
        """When player not found in players list, defaults to zeros."""
        state = _make_game_state(
            player_id=0,
            players=[_make_player(id=99)],  # Different ID
        )
        result = converter.convert(state)
        assert result["gold"] == 0
        assert result["score"] == 0

    def test_convert_resources_extraction(self, converter: StateConverter) -> None:
        state = _make_game_state(
            player_id=0,
            tiles=[
                _make_tile(x=1, y=1, resource="RESOURCE_IRON", owner_id=0),
                _make_tile(x=2, y=2, resource="RESOURCE_IRON", owner_id=0),
                _make_tile(x=3, y=3, resource="RESOURCE_GOLD", owner_id=1),
            ],
        )
        result = converter.convert(state)
        assert result["resources"].get("iron", 0) == 2
        assert "gold" not in result["resources"]  # Owned by player 1


# ===========================================================================
# 5. StateConverter._setup_scaling
# ===========================================================================


class TestSetupScaling:
    """Test coordinate scaling factor computation."""

    def test_setup_scaling_sets_factors(self, converter: StateConverter) -> None:
        converter._setup_scaling(32, 32)
        assert converter._scale_x == 16 / 32  # 0.5
        assert converter._scale_y == 16 / 32

    def test_setup_scaling_non_square(self, converter: StateConverter) -> None:
        converter._setup_scaling(84, 54)
        assert converter._scale_x == pytest.approx(16 / 84)
        assert converter._scale_y == pytest.approx(16 / 54)

    def test_setup_scaling_creates_map_config(self, converter: StateConverter) -> None:
        converter._setup_scaling(44, 26)
        assert converter._map_config is not None
        assert converter._map_config.width == 44
        assert converter._map_config.height == 26

    def test_setup_scaling_reuses_if_same_size(self, converter: StateConverter) -> None:
        converter._setup_scaling(44, 26)
        config1 = converter._map_config
        converter._setup_scaling(44, 26)
        # Same object should be reused (or at least values match)
        assert converter._map_config is config1

    def test_setup_scaling_changes_on_new_size(self, converter: StateConverter) -> None:
        converter._setup_scaling(44, 26)
        converter._setup_scaling(84, 54)
        assert converter._map_config is not None
        assert converter._map_config.width == 84


# ===========================================================================
# 6. StateConverter._convert_terrain
# ===========================================================================


class TestConvertTerrain:
    """Test terrain conversion with scaling and collision handling."""

    def test_basic_terrain_conversion(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[
                _make_tile(x=0, y=0, terrain="TERRAIN_GRASS"),
                _make_tile(x=1, y=0, terrain="TERRAIN_PLAINS"),
            ],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert terrain[(0, 0)]["type"] == "grassland"
        assert terrain[(1, 0)]["type"] == "plains"

    def test_default_terrain_is_plains(self, converter: StateConverter) -> None:
        """Tiles not covered by map data default to plains."""
        state = _make_game_state(
            map_width=16, map_height=16, tiles=[]
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        # Every position should default to plains
        assert terrain[(5, 5)]["type"] == "plains"

    def test_hills_feature_extracted(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=2, y=3, terrain="TERRAIN_PLAINS_HILLS")],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert "hills" in terrain[(2, 3)]["features"]

    def test_mountain_feature_extracted(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=4, y=4, terrain="TERRAIN_DESERT_MOUNTAIN")],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert terrain[(4, 4)]["type"] == "desert"
        assert "mountains" in terrain[(4, 4)]["features"]

    def test_feature_mapping(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[
                _make_tile(
                    x=1, y=1, terrain="TERRAIN_GRASS", features=["FEATURE_FOREST"]
                )
            ],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert "forest" in terrain[(1, 1)]["features"]

    def test_unknown_terrain_defaults_to_plains(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=0, y=0, terrain="TERRAIN_UNKNOWN")],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert terrain[(0, 0)]["type"] == "plains"

    def test_terrain_collision_majority_vote(self, converter_8: StateConverter) -> None:
        """When map is larger than board, multiple tiles may map to same pos."""
        # 16x16 map -> 8x8 board, so tiles scale by 0.5
        tiles = [
            _make_tile(x=0, y=0, terrain="TERRAIN_GRASS"),
            _make_tile(x=1, y=0, terrain="TERRAIN_GRASS"),
            _make_tile(x=0, y=1, terrain="TERRAIN_PLAINS"),
            _make_tile(x=1, y=1, terrain="TERRAIN_PLAINS"),
        ]
        state = _make_game_state(
            map_width=16, map_height=16, tiles=tiles
        )
        converter_8._setup_scaling(16, 16)
        terrain = converter_8._convert_terrain(state)
        # All 4 tiles map to (0, 0). 2 grass + 2 plains is a tie.
        # Tie-breaking uses priority (grassland=3, plains=4) so plains wins.
        assert terrain[(0, 0)]["type"] in ("grassland", "plains")

    def test_yields_preserved(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[
                _make_tile(x=3, y=3, terrain="TERRAIN_GRASS", yields={"food": 2, "production": 1})
            ],
        )
        converter._setup_scaling(16, 16)
        terrain = converter._convert_terrain(state)
        assert terrain[(3, 3)]["yields"]["food"] == 2


# ===========================================================================
# 7. StateConverter._convert_units
# ===========================================================================


class TestConvertUnits:
    """Test unit conversion with player mapping and scaling."""

    def test_basic_unit_conversion(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[_make_unit(id=1, unit_type="UNIT_WARRIOR", owner_id=0, x=3, y=4)],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert len(units) == 1
        assert units[0]["type"] == "melee"
        assert units[0]["_id"] == 1
        assert units[0]["player"] == 0
        assert units[0]["position"] == (3, 4)

    def test_enemy_unit_mapping_2player(self, converter: StateConverter) -> None:
        """In 2-player mode, non-current-player units are assigned player=1."""
        state = _make_game_state(
            player_id=0,
            map_width=16,
            map_height=16,
            units=[_make_unit(id=2, owner_id=1, x=5, y=5)],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["player"] == 1

    def test_unit_health_normalized(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[_make_unit(id=1, health=50, max_health=100)],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["health"] == 50  # 100 * 50/100

    def test_unit_health_normalized_different_max(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[_make_unit(id=1, health=75, max_health=150)],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["health"] == 50  # int(100 * 75/150)

    def test_unit_health_zero_max(self, converter: StateConverter) -> None:
        """If max_health is 0, raw health should be used."""
        unit = _make_unit(id=1, health=50, max_health=0)
        state = _make_game_state(map_width=16, map_height=16, units=[unit])
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["health"] == 50

    def test_unit_preserves_original_data(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[_make_unit(id=42, owner_id=0, x=7, y=8)],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["_id"] == 42
        assert units[0]["_original_pos"] == (7, 8)
        assert units[0]["_owner_id"] == 0

    def test_unit_type_mapping(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[
                _make_unit(id=1, unit_type="UNIT_ARCHER", owner_id=0),
                _make_unit(id=2, unit_type="UNIT_SETTLER", owner_id=0),
            ],
        )
        converter._setup_scaling(16, 16)
        units = converter._convert_units(state)
        assert units[0]["type"] == "ranged"
        assert units[1]["type"] == "civilian"

    def test_multiplayer_mode_uses_direct_owner_id(self) -> None:
        """In N-player mode (num_players>2), owner_id is used directly."""
        conv = StateConverter(board_size=16, num_players=4, create_encoder=False)
        state = _make_game_state(
            player_id=0,
            map_width=16,
            map_height=16,
            units=[
                _make_unit(id=1, owner_id=2, x=3, y=3),
                _make_unit(id=2, owner_id=3, x=4, y=4),
            ],
        )
        conv._setup_scaling(16, 16)
        units = conv._convert_units(state)
        assert units[0]["player"] == 2
        assert units[1]["player"] == 3

    def test_coordinate_scaling(self, converter_8: StateConverter) -> None:
        """Units on a 16x16 map should be scaled to 8x8 board."""
        state = _make_game_state(
            map_width=16,
            map_height=16,
            units=[_make_unit(id=1, x=8, y=8, owner_id=0)],
        )
        converter_8._setup_scaling(16, 16)
        units = converter_8._convert_units(state)
        assert units[0]["position"] == (4, 4)


# ===========================================================================
# 8. StateConverter._convert_cities
# ===========================================================================


class TestConvertCities:
    """Test city conversion with player mapping."""

    def test_basic_city_conversion(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            cities=[_make_city(id=10, name="Rome", owner_id=0, x=5, y=5, population=5)],
        )
        converter._setup_scaling(16, 16)
        cities = converter._convert_cities(state)
        assert len(cities) == 1
        assert cities[0]["name"] == "Rome"
        assert cities[0]["population"] == 5
        assert cities[0]["player"] == 0
        assert cities[0]["_id"] == 10

    def test_enemy_city_2player(self, converter: StateConverter) -> None:
        state = _make_game_state(
            player_id=0,
            map_width=16,
            map_height=16,
            cities=[_make_city(id=20, owner_id=1, x=10, y=10)],
        )
        converter._setup_scaling(16, 16)
        cities = converter._convert_cities(state)
        assert cities[0]["player"] == 1

    def test_city_preserves_original_data(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            cities=[
                _make_city(
                    id=10,
                    owner_id=0,
                    x=5,
                    y=5,
                    buildings=["BUILDING_MONUMENT"],
                    available_production=[1, 2, 3],
                )
            ],
        )
        converter._setup_scaling(16, 16)
        cities = converter._convert_cities(state)
        assert cities[0]["_original_pos"] == (5, 5)
        assert cities[0]["_owner_id"] == 0
        assert "BUILDING_MONUMENT" in cities[0]["_buildings"]
        assert cities[0]["_available_production"] == [1, 2, 3]

    def test_city_is_capital(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            cities=[_make_city(id=10, is_capital=True)],
        )
        converter._setup_scaling(16, 16)
        cities = converter._convert_cities(state)
        assert cities[0]["is_capital"] is True

    def test_multiplayer_city_uses_direct_owner(self) -> None:
        conv = StateConverter(board_size=16, num_players=4, create_encoder=False)
        state = _make_game_state(
            player_id=0,
            map_width=16,
            map_height=16,
            cities=[_make_city(id=10, owner_id=3, x=5, y=5)],
        )
        conv._setup_scaling(16, 16)
        cities = conv._convert_cities(state)
        assert cities[0]["player"] == 3

    def test_city_coordinate_scaling(self, converter_8: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            cities=[_make_city(id=10, x=8, y=8, owner_id=0)],
        )
        converter_8._setup_scaling(16, 16)
        cities = converter_8._convert_cities(state)
        assert cities[0]["position"] == (4, 4)


# ===========================================================================
# 9. StateConverter._scale_coords / inverse_scale_coords
# ===========================================================================


class TestScalePosition:
    """Test coordinate scaling in both directions."""

    def test_scale_coords_identity(self, converter: StateConverter) -> None:
        """When map == board_size, scaling is identity."""
        converter._setup_scaling(16, 16)
        assert converter._scale_coords(5, 5) == (5, 5)

    def test_scale_coords_downscale(self, converter: StateConverter) -> None:
        """32x32 map -> 16x16 board: divide by 2."""
        converter._setup_scaling(32, 32)
        assert converter._scale_coords(16, 16) == (8, 8)

    def test_scale_coords_clamps_to_bounds(self, converter: StateConverter) -> None:
        converter._setup_scaling(16, 16)
        # x=100 should clamp to board_size-1=15
        sx, sy = converter._scale_coords(100, 100)
        assert sx <= 15
        assert sy <= 15

    def test_scale_coords_zero(self, converter: StateConverter) -> None:
        converter._setup_scaling(32, 32)
        assert converter._scale_coords(0, 0) == (0, 0)

    def test_inverse_scale_coords(self, converter: StateConverter) -> None:
        converter._setup_scaling(32, 32)
        # After scaling 16->8, inverse should go 8->16
        ox, oy = converter.inverse_scale_coords(8, 8)
        assert ox == 16
        assert oy == 16

    def test_inverse_scale_no_map_config(self, converter: StateConverter) -> None:
        """When no map config is set, inverse returns input unchanged."""
        assert converter._map_config is None
        assert converter.inverse_scale_coords(5, 5) == (5, 5)

    def test_inverse_scale_clamps(self, converter: StateConverter) -> None:
        converter._setup_scaling(20, 20)
        ox, oy = converter.inverse_scale_coords(100, 100)
        assert ox <= 19
        assert oy <= 19


# ===========================================================================
# 10. Named constants
# ===========================================================================


class TestNamedConstants:
    """Test module-level constants."""

    def test_estimated_total_techs(self):
        """Test ESTIMATED_TOTAL_TECHS constant."""
        assert GameConstants.ESTIMATED_TOTAL_TECHS == 70
        assert isinstance(GameConstants.ESTIMATED_TOTAL_TECHS, int)

    def test_estimated_max_score(self):
        """Test ESTIMATED_MAX_SCORE constant."""
        assert GameConstants.ESTIMATED_MAX_SCORE == 2000
        assert isinstance(GameConstants.ESTIMATED_MAX_SCORE, int)


# ===========================================================================
# 11. Visibility conversion
# ===========================================================================


class TestConvertVisibility:
    """Test visibility conversion with priority handling."""

    def test_default_visibility_is_fog(self, converter: StateConverter) -> None:
        state = _make_game_state(map_width=16, map_height=16, tiles=[])
        converter._setup_scaling(16, 16)
        vis = converter._convert_visibility(state)
        assert vis[(0, 0)] == "fog"
        assert vis[(5, 5)] == "fog"

    def test_visible_tile(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=3, y=3, is_visible=True, is_revealed=True)],
        )
        converter._setup_scaling(16, 16)
        vis = converter._convert_visibility(state)
        assert vis[(3, 3)] == "visible"

    def test_seen_tile(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=3, y=3, is_visible=False, is_revealed=True)],
        )
        converter._setup_scaling(16, 16)
        vis = converter._convert_visibility(state)
        assert vis[(3, 3)] == "seen"

    def test_fog_tile(self, converter: StateConverter) -> None:
        state = _make_game_state(
            map_width=16,
            map_height=16,
            tiles=[_make_tile(x=3, y=3, is_visible=False, is_revealed=False)],
        )
        converter._setup_scaling(16, 16)
        vis = converter._convert_visibility(state)
        assert vis[(3, 3)] == "fog"

    def test_visibility_collision_takes_highest(self, converter_8: StateConverter) -> None:
        """When tiles collide, visible > seen > fog."""
        tiles = [
            _make_tile(x=0, y=0, is_visible=False, is_revealed=True),  # seen
            _make_tile(x=1, y=0, is_visible=True, is_revealed=True),  # visible
        ]
        state = _make_game_state(map_width=16, map_height=16, tiles=tiles)
        converter_8._setup_scaling(16, 16)
        vis = converter_8._convert_visibility(state)
        # Both tiles map to (0, 0) on 8x8 board. Visible wins.
        assert vis[(0, 0)] == "visible"


# ===========================================================================
# 12. Multiplayer helpers
# ===========================================================================


class TestMultiplayerHelpers:
    """Test build_player_id_mapping and _remap_player_ids."""

    def test_build_player_id_mapping_contiguous(self) -> None:
        mapping = StateConverter.build_player_id_mapping([0, 1, 2])
        assert mapping == {0: 0, 1: 1, 2: 2}

    def test_build_player_id_mapping_non_contiguous(self) -> None:
        mapping = StateConverter.build_player_id_mapping([0, 1, 3, 5])
        assert mapping == {0: 0, 1: 1, 3: 2, 5: 3}

    def test_build_player_id_mapping_deduplicates(self) -> None:
        mapping = StateConverter.build_player_id_mapping([0, 0, 1, 1])
        assert mapping == {0: 0, 1: 1}

    def test_build_player_id_mapping_sorts(self) -> None:
        mapping = StateConverter.build_player_id_mapping([5, 3, 1])
        assert mapping == {1: 0, 3: 1, 5: 2}

    def test_remap_player_ids(self) -> None:
        raw_state: dict[str, Any] = {
            "player_id": 5,
            "players": [{"id": 5}, {"id": 10}],
            "units": [{"owner_id": 5}, {"owner_id": 10}],
            "cities": [{"owner_id": 5}],
            "winner_id": 10,
        }
        mapping = {5: 0, 10: 1}
        remapped = StateConverter._remap_player_ids(raw_state, mapping)
        assert remapped["player_id"] == 0
        assert remapped["players"][0]["id"] == 0
        assert remapped["players"][1]["id"] == 1
        assert remapped["units"][0]["owner_id"] == 0
        assert remapped["units"][1]["owner_id"] == 1
        assert remapped["cities"][0]["owner_id"] == 0
        assert remapped["winner_id"] == 1

    def test_remap_preserves_original(self) -> None:
        raw_state: dict[str, Any] = {
            "player_id": 5,
            "players": [{"id": 5}],
            "units": [],
            "cities": [],
        }
        mapping = {5: 0}
        StateConverter._remap_player_ids(raw_state, mapping)
        # Original should not be modified
        assert raw_state["player_id"] == 5
        assert raw_state["players"][0]["id"] == 5

    def test_remap_missing_winner_id(self) -> None:
        raw_state: dict[str, Any] = {
            "players": [],
            "units": [],
            "cities": [],
        }
        mapping: dict[int, int] = {}
        remapped = StateConverter._remap_player_ids(raw_state, mapping)
        assert "winner_id" not in remapped


# ===========================================================================
# 13. convert_and_encode
# ===========================================================================


class TestConvertAndEncode:
    """Test convert_and_encode requires encoder."""

    def test_no_encoder_raises(self, converter: StateConverter) -> None:
        state = _make_game_state()
        with pytest.raises(RuntimeError, match="Encoder not created"):
            converter.convert_and_encode(state)

    def test_with_encoder_returns_tensors(self) -> None:
        """With create_encoder=True, should return (spatial, vector) tensors."""
        conv = StateConverter(board_size=16, create_encoder=True)
        state = _make_game_state(
            map_width=20,
            map_height=20,
            tiles=[_make_tile(x=0, y=0)],
            units=[_make_unit(id=1, owner_id=0)],
            cities=[_make_city(id=10, owner_id=0)],
        )
        spatial, vector = conv.convert_and_encode(state)
        assert spatial.shape[0] == 84  # spatial_channels
        assert spatial.shape[1] == 16  # board_size
        assert spatial.shape[2] == 16
        assert vector.ndim == 1


# ===========================================================================
# 14. Resource / tech / civic normalization
# ===========================================================================


class TestNormalization:
    """Test the various _normalize_* methods."""

    def test_normalize_resource_name(self, converter: StateConverter) -> None:
        assert converter._normalize_resource_name("RESOURCE_IRON") == "iron"
        assert converter._normalize_resource_name("RES_HORSES") == "horses"
        assert converter._normalize_resource_name("gold") == "gold"

    def test_normalize_tech_name(self, converter: StateConverter) -> None:
        assert converter._normalize_tech_name("TECH_POTTERY") == "pottery"
        assert converter._normalize_tech_name("pottery") == "pottery"

    def test_normalize_civic_name(self, converter: StateConverter) -> None:
        assert converter._normalize_civic_name("CIVIC_CODE_OF_LAWS") == "code_of_laws"
        assert converter._normalize_civic_name("mysticism") == "mysticism"


# ===========================================================================
# 15. Terrain collision resolution
# ===========================================================================


class TestResolveTerrainCollision:
    """Test _resolve_terrain_collision majority vote logic."""

    def test_clear_majority(self, converter: StateConverter) -> None:
        tiles_data = [
            ("grassland", ["forest"], {"food": 2}),
            ("grassland", [], {"food": 1}),
            ("plains", [], {"production": 1}),
        ]
        result = converter._resolve_terrain_collision(tiles_data)
        assert result["type"] == "grassland"

    def test_tie_uses_priority(self, converter: StateConverter) -> None:
        tiles_data = [
            ("grassland", [], {}),
            ("mountains", [], {}),
        ]
        result = converter._resolve_terrain_collision(tiles_data)
        # mountains has priority 10, grassland has 3
        assert result["type"] == "mountains"

    def test_features_aggregated(self, converter: StateConverter) -> None:
        tiles_data = [
            ("plains", ["forest"], {}),
            ("plains", ["hills"], {}),
        ]
        result = converter._resolve_terrain_collision(tiles_data)
        assert "forest" in result["features"]
        assert "hills" in result["features"]

    def test_yields_averaged(self, converter: StateConverter) -> None:
        tiles_data = [
            ("plains", [], {"food": 4}),
            ("plains", [], {"food": 2}),
        ]
        result = converter._resolve_terrain_collision(tiles_data)
        assert result["yields"]["food"] == 3  # int(6/2)
