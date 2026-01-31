"""Unit tests for the Unciv interface.

Tests the UncivInterface class and related dataclasses for:
- Loading and saving game states
- Converting between Unciv and internal formats
- Generating valid action masks
- Applying actions to game states
- Error handling
"""

import os
import tempfile

import numpy as np
import pytest

from alpha_zero.interfaces import (
    UncivActionError,
    UncivCity,
    UncivGameState,
    UncivInterface,
    UncivLoadError,
    UncivMappings,
    UncivPlayer,
    UncivTile,
    UncivUnit,
    create_unciv_interface,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def interface():
    """Create a default UncivInterface."""
    return create_unciv_interface(board_size=16)


@pytest.fixture
def sample_game_state():
    """Create a sample UncivGameState for testing."""
    game_state = UncivGameState(
        version="4.0",
        game_id="test_game",
        current_player="Rome",
        turn=10,
        map_size=(40, 25),
    )

    # Add tiles
    for x in range(10):
        for y in range(10):
            terrain = "Plains" if (x + y) % 2 == 0 else "Grassland"
            features = ["Hill"] if (x + y) % 5 == 0 else []
            game_state.tiles[(x, y)] = UncivTile(
                position=(x, y),
                base_terrain=terrain,
                terrain_features=features,
            )

    # Add players
    game_state.players["Rome"] = UncivPlayer(
        civ_name="Rome",
        player_type="Human",
        gold=500,
        gold_per_turn=25,
        researched_techs=["Agriculture", "Mining", "Pottery"],
    )
    game_state.players["Greece"] = UncivPlayer(
        civ_name="Greece",
        player_type="AI",
        gold=300,
        researched_techs=["Agriculture", "Animal Husbandry"],
    )

    # Add units
    game_state.units["unit_1"] = UncivUnit(
        id="unit_1",
        name="Warrior",
        owner="Rome",
        position=(2, 3),
        health=100,
        current_movement=2,
        strength=8,
    )
    game_state.units["unit_2"] = UncivUnit(
        id="unit_2",
        name="Settler",
        owner="Rome",
        position=(3, 3),
        health=100,
        current_movement=2,
        strength=0,
    )
    game_state.units["unit_3"] = UncivUnit(
        id="unit_3",
        name="Warrior",
        owner="Greece",
        position=(7, 7),
        health=100,
        strength=8,
    )

    # Add cities
    game_state.cities["Rome"] = UncivCity(
        name="Rome",
        owner="Rome",
        position=(2, 2),
        population=5,
        is_capital=True,
    )
    game_state.cities["Athens"] = UncivCity(
        name="Athens",
        owner="Greece",
        position=(8, 8),
        population=3,
        is_capital=True,
    )

    return game_state


# =============================================================================
# Test Dataclasses
# =============================================================================


class TestUncivTile:
    """Tests for UncivTile dataclass."""

    def test_create_tile(self):
        """Test creating a tile with default values."""
        tile = UncivTile(position=(5, 5))
        assert tile.position == (5, 5)
        assert tile.base_terrain == "Grassland"
        assert tile.terrain_features == []
        assert tile.resource is None

    def test_tile_with_features(self):
        """Test creating a tile with features."""
        tile = UncivTile(
            position=(3, 4),
            base_terrain="Plains",
            terrain_features=["Hill", "Forest"],
            resource="Iron",
            improvement="Mine",
        )
        assert tile.base_terrain == "Plains"
        assert "Hill" in tile.terrain_features
        assert tile.resource == "Iron"
        assert tile.improvement == "Mine"

    def test_tile_to_unciv_dict(self):
        """Test converting tile to Unciv JSON format."""
        tile = UncivTile(
            position=(1, 2),
            base_terrain="Desert",
            terrain_features=["Oasis"],
            road_status="Road",
        )
        result = tile.to_unciv_dict()
        assert result["baseTerrain"] == "Desert"
        assert "Oasis" in result["terrainFeatures"]
        assert result["roadStatus"] == "Road"


class TestUncivUnit:
    """Tests for UncivUnit dataclass."""

    def test_create_unit(self):
        """Test creating a unit with required fields."""
        unit = UncivUnit(
            id="u1",
            name="Warrior",
            owner="Rome",
            position=(5, 5),
        )
        assert unit.id == "u1"
        assert unit.name == "Warrior"
        assert unit.owner == "Rome"
        assert unit.health == 100

    def test_is_civilian(self):
        """Test civilian unit detection."""
        warrior = UncivUnit(id="w1", name="Warrior", owner="Rome", position=(0, 0), strength=8)
        settler = UncivUnit(id="s1", name="Settler", owner="Rome", position=(0, 0), strength=0)
        assert not warrior.is_civilian
        assert settler.is_civilian

    def test_has_moves(self):
        """Test movement point detection."""
        fresh = UncivUnit(
            id="u1",
            name="Warrior",
            owner="Rome",
            position=(0, 0),
            current_movement=2,
        )
        exhausted = UncivUnit(
            id="u2",
            name="Warrior",
            owner="Rome",
            position=(0, 0),
            current_movement=0,
        )
        assert fresh.has_moves
        assert not exhausted.has_moves


class TestUncivCity:
    """Tests for UncivCity dataclass."""

    def test_create_city(self):
        """Test creating a city."""
        city = UncivCity(
            name="Rome",
            owner="Rome",
            position=(10, 10),
            population=5,
            is_capital=True,
        )
        assert city.name == "Rome"
        assert city.population == 5
        assert city.is_capital


class TestUncivPlayer:
    """Tests for UncivPlayer dataclass."""

    def test_create_player(self):
        """Test creating a player."""
        player = UncivPlayer(
            civ_name="Rome",
            player_type="Human",
            gold=1000,
            researched_techs=["Agriculture", "Mining"],
        )
        assert player.civ_name == "Rome"
        assert player.gold == 1000
        assert len(player.researched_techs) == 2


# =============================================================================
# Test UncivGameState
# =============================================================================


class TestUncivGameState:
    """Tests for UncivGameState dataclass."""

    def test_create_game_state(self, sample_game_state):
        """Test creating a game state."""
        assert sample_game_state.turn == 10
        assert sample_game_state.current_player == "Rome"
        assert len(sample_game_state.players) == 2
        assert len(sample_game_state.units) == 3
        assert len(sample_game_state.cities) == 2

    def test_to_json_and_back(self, sample_game_state):
        """Test round-trip conversion to JSON."""
        json_data = sample_game_state.to_unciv_json()
        restored = UncivGameState.from_unciv_json(json_data)

        assert restored.turn == sample_game_state.turn
        assert restored.current_player == sample_game_state.current_player


# =============================================================================
# Test UncivInterface
# =============================================================================


class TestUncivInterface:
    """Tests for UncivInterface class."""

    def test_create_interface(self):
        """Test creating an interface."""
        interface = UncivInterface(board_size=16)
        assert interface.board_size == 16
        assert interface.max_units == 100
        assert interface.max_cities == 20

    def test_factory_function(self):
        """Test the factory function."""
        interface = create_unciv_interface(board_size=8, max_units=50)
        assert interface.board_size == 8
        assert interface.max_units == 50

    def test_convert_to_internal(self, interface, sample_game_state):
        """Test converting to internal format."""
        json_data = sample_game_state.to_unciv_json()
        internal = interface.convert_to_internal(json_data)

        # Check required fields
        assert "current_player" in internal
        assert "turn" in internal
        assert "terrain" in internal
        assert "units" in internal
        assert "cities" in internal

        assert internal["turn"] == 10
        assert len(internal["units"]) == 3
        assert len(internal["cities"]) == 2

    def test_convert_back_to_unciv(self, interface, sample_game_state):
        """Test converting back to Unciv format."""
        json_data = sample_game_state.to_unciv_json()
        internal = interface.convert_to_internal(json_data)
        unciv_out = interface.convert_to_unciv(internal)

        assert "civilizations" in unciv_out
        assert "tileMap" in unciv_out
        assert unciv_out["turns"] == internal["turn"]

    def test_get_valid_actions(self, interface, sample_game_state):
        """Test getting valid action mask."""
        json_data = sample_game_state.to_unciv_json()
        internal = interface.convert_to_internal(json_data)
        mask = interface.get_valid_actions(internal)

        # Mask should be a numpy array
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.float32

        # Should have at least end turn as valid
        assert np.sum(mask > 0) >= 1

    def test_apply_end_turn(self, interface, sample_game_state):
        """Test applying end turn action."""
        json_data = sample_game_state.to_unciv_json()
        internal = interface.convert_to_internal(json_data)

        # Get end turn action index
        ActionCategory, _, _ = interface._get_action_enums()
        end_turn = interface.action_space.category_offsets[ActionCategory.END_TURN]

        new_state = interface.apply_action(internal, end_turn)

        # Player should have changed
        assert new_state["current_player"] != internal["current_player"]


class TestUncivInterfaceIO:
    """Tests for UncivInterface file I/O operations."""

    def test_save_and_load(self, interface, sample_game_state):
        """Test saving and loading a game."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            interface.save_game(sample_game_state, temp_path)
            loaded = interface.load_game(temp_path)

            assert loaded.turn == sample_game_state.turn
            assert loaded.current_player == sample_game_state.current_player
        finally:
            os.unlink(temp_path)

    def test_load_nonexistent_file(self, interface):
        """Test loading a file that doesn't exist."""
        with pytest.raises(UncivLoadError):
            interface.load_game("/nonexistent/path/game.json")

    def test_load_invalid_json(self, interface):
        """Test loading an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("this is not valid json {{{")
            temp_path = f.name

        try:
            with pytest.raises(UncivLoadError):
                interface.load_game(temp_path)
        finally:
            os.unlink(temp_path)


class TestUncivMappings:
    """Tests for terrain and unit mappings."""

    def test_terrain_mapping(self):
        """Test terrain type mappings."""
        assert UncivMappings.TERRAIN_TO_INTERNAL["Grassland"] == "grassland"
        assert UncivMappings.TERRAIN_TO_INTERNAL["Plains"] == "plains"
        assert UncivMappings.TERRAIN_TO_INTERNAL["Ocean"] == "ocean"

    def test_feature_mapping(self):
        """Test terrain feature mappings."""
        assert UncivMappings.FEATURE_TO_INTERNAL["Forest"] == "forest"
        assert UncivMappings.FEATURE_TO_INTERNAL["Hill"] == "hills"
        assert UncivMappings.FEATURE_TO_INTERNAL["Mountain"] == "mountains"

    def test_unit_category_mapping(self):
        """Test unit type to category mapping."""
        assert UncivMappings.UNIT_TO_CATEGORY["Warrior"] == "melee"
        assert UncivMappings.UNIT_TO_CATEGORY["Archer"] == "ranged"
        assert UncivMappings.UNIT_TO_CATEGORY["Settler"] == "settler"

    def test_era_from_tech_count(self):
        """Test era calculation from tech count."""
        assert UncivMappings.get_era_from_tech_count(0) == 0  # Ancient
        assert UncivMappings.get_era_from_tech_count(10) == 1  # Classical
        assert UncivMappings.get_era_from_tech_count(50) == 4  # Industrial


class TestPositionMapping:
    """Tests for position coordinate mapping."""

    def test_map_to_internal(self, interface):
        """Test mapping Unciv coords to internal coords."""
        # Origin
        pos = interface._map_position_to_internal((0, 0), (40, 25))
        assert pos == (0, 0)

        # Max position
        pos = interface._map_position_to_internal((39, 24), (40, 25))
        assert pos[0] <= 15
        assert pos[1] <= 15

    def test_map_from_internal(self, interface):
        """Test mapping internal coords back to Unciv coords."""
        pos = interface._map_position_from_internal((0, 0), (40, 25))
        assert pos == (0, 0)

        pos = interface._map_position_from_internal((15, 15), (40, 25))
        assert pos[0] >= 0
        assert pos[1] >= 0


class TestErrorHandling:
    """Tests for error handling."""

    def test_convert_empty_state(self, interface):
        """Test converting an empty state."""
        # Should not raise, but handle gracefully
        internal = interface.convert_to_internal({})
        assert "current_player" in internal

    def test_apply_invalid_unit_action(self, interface):
        """Test applying action for non-existent unit."""
        minimal_state = {
            "current_player": 0,
            "num_players": 2,
            "units": [],
            "cities": [],
        }

        # Get a unit action index
        ActionCategory, _, _ = interface._get_action_enums()
        unit_offset = interface.action_space.category_offsets[ActionCategory.UNIT_ACTION]

        with pytest.raises(UncivActionError):
            interface.apply_action(minimal_state, unit_offset + 100)
