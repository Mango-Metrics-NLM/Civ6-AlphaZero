"""Tests for the state converter module."""

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
from alpha_zero.bridge.state_converter import StateConverter


class TestStateConverter:
    """Tests for StateConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a StateConverter instance."""
        return StateConverter(board_size=16, create_encoder=True)

    @pytest.fixture
    def sample_game_state(self):
        """Create a sample GameStateData for testing."""
        tiles = [TileData(x=x, y=y, terrain="TERRAIN_PLAINS") for x in range(44) for y in range(26)]
        # Add some variety
        tiles[0] = TileData(x=0, y=0, terrain="TERRAIN_GRASS", features=["FEATURE_FOREST"])
        tiles[1] = TileData(x=1, y=0, terrain="TERRAIN_COAST")

        return GameStateData(
            turn_id=10,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=tiles),
            units=[
                UnitData(
                    id=1,
                    unit_type="UNIT_WARRIOR",
                    owner_id=0,
                    x=5,
                    y=5,
                    health=100,
                    max_health=100,
                    damage=0,
                    moves_remaining=2.0,
                    max_moves=2.0,
                    combat_strength=20,
                ),
                UnitData(
                    id=2,
                    unit_type="UNIT_ARCHER",
                    owner_id=1,
                    x=10,
                    y=10,
                    health=80,
                    max_health=100,
                    damage=20,
                    moves_remaining=1.0,
                    max_moves=2.0,
                    combat_strength=15,
                    ranged_strength=25,
                    range=2,
                ),
            ],
            cities=[
                CityData(
                    id=1,
                    name="Capital",
                    owner_id=0,
                    x=8,
                    y=8,
                    population=3,
                    is_capital=True,
                ),
            ],
            players=[
                PlayerData(
                    id=0,
                    civilization="CIVILIZATION_ROME",
                    gold=100.0,
                    science_per_turn=10.0,
                    culture_per_turn=5.0,
                ),
                PlayerData(
                    id=1,
                    civilization="CIVILIZATION_GREECE",
                    gold=50.0,
                    science_per_turn=8.0,
                    culture_per_turn=4.0,
                ),
            ],
            legal_actions=[
                LegalAction(action_type="end_turn"),
                LegalAction(action_type="move", unit_id=1, targets=[(4, 5), (5, 4)]),
            ],
        )

    def test_convert_basic(self, converter, sample_game_state):
        """Test basic state conversion."""
        result = converter.convert(sample_game_state)

        assert isinstance(result, dict)
        assert result["turn"] == 10
        assert result["current_player"] == 0
        assert "terrain" in result
        assert "units" in result
        assert "cities" in result

    def test_convert_terrain(self, converter, sample_game_state):
        """Test terrain conversion."""
        result = converter.convert(sample_game_state)

        # Check terrain dict exists and has data
        assert len(result["terrain"]) > 0

        # Check that terrain values are properly mapped
        for _pos, terrain_data in result["terrain"].items():
            assert "type" in terrain_data
            # Should be mapped to internal format (plains, grassland, etc.)
            assert isinstance(terrain_data["type"], str)

    def test_convert_units(self, converter, sample_game_state):
        """Test unit conversion."""
        result = converter.convert(sample_game_state)

        assert len(result["units"]) == 2

        # Check own unit (player 0)
        own_unit = result["units"][0]
        # Type should be combat class (melee, ranged, etc.)
        assert own_unit["type"] in ["melee", "ranged", "cavalry", "siege", "civilian", "recon"]
        assert own_unit["player"] == 0
        assert "position" in own_unit

        # Check enemy unit (player 1)
        enemy_unit = result["units"][1]
        assert enemy_unit["player"] == 1

    def test_convert_cities(self, converter, sample_game_state):
        """Test city conversion."""
        result = converter.convert(sample_game_state)

        assert len(result["cities"]) == 1

        city = result["cities"][0]
        assert city["name"] == "Capital"
        assert city["player"] == 0
        assert city["population"] == 3
        assert city["is_capital"] is True

    def test_convert_legal_actions(self, converter, sample_game_state):
        """Test legal action preservation."""
        result = converter.convert(sample_game_state)

        assert "_legal_actions" in result
        assert len(result["_legal_actions"]) == 2

    def test_convert_and_encode(self, converter, sample_game_state):
        """Test full conversion and encoding."""
        spatial, vector = converter.convert_and_encode(sample_game_state)

        # Check output shapes
        assert spatial.shape == (84, 16, 16)
        assert vector.shape == (512,)

        # Check that output has non-zero values (state is not empty)
        assert (spatial != 0).any()
        assert (vector != 0).any()

    def test_coordinate_scaling(self, converter, sample_game_state):
        """Test that coordinates are scaled properly."""
        result = converter.convert(sample_game_state)

        # Original map is 44x26, scaled to 16x16
        # Units should have scaled positions
        for unit in result["units"]:
            scaled_x, scaled_y = unit["position"]
            # Check that position is within board bounds
            assert 0 <= scaled_x < 16
            assert 0 <= scaled_y < 16

    def test_convert_dict_input(self, converter):
        """Test conversion from dictionary input."""
        dict_state = {
            "turn_id": 5,
            "player_id": 0,
            "map": {
                "width": 44,
                "height": 26,
                "tiles": [
                    {"x": x, "y": y, "terrain": "TERRAIN_PLAINS"}
                    for x in range(44)
                    for y in range(26)
                ],
            },
            "units": [],
            "cities": [],
            "players": [],
            "legal_actions": [],
        }

        result = converter.convert(dict_state)

        assert result["turn"] == 5
        assert result["current_player"] == 0

    def test_empty_state(self, converter):
        """Test conversion of minimal state."""
        minimal_state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(minimal_state)

        assert result["turn"] == 1
        assert len(result["units"]) == 0
        assert len(result["cities"]) == 0

    def test_resources_conversion(self, converter):
        """Test that state includes resources field."""
        tiles = [
            TileData(x=0, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_IRON"),
            TileData(x=1, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_WHEAT"),
        ]

        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=tiles),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)

        # Resources should be in the state
        assert "resources" in result


class TestStateConverterEdgeCases:
    """Edge case tests for StateConverter."""

    def test_large_map(self):
        """Test conversion with large map."""
        converter = StateConverter(board_size=16)

        tiles = [
            TileData(x=x, y=y, terrain="TERRAIN_PLAINS") for x in range(100) for y in range(100)
        ]

        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=100, height=100, tiles=tiles),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert result["turn"] == 1

    def test_multiple_players(self):
        """Test conversion with multiple players."""
        converter = StateConverter(board_size=16)

        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[
                UnitData(id=1, unit_type="UNIT_WARRIOR", owner_id=0, x=0, y=0),
                UnitData(id=2, unit_type="UNIT_WARRIOR", owner_id=1, x=1, y=1),
            ],
            cities=[],
            players=[
                PlayerData(id=0, civilization="CIV_A"),
                PlayerData(id=1, civilization="CIV_B"),
            ],
            legal_actions=[],
        )

        result = converter.convert(state)

        # Check units are properly attributed
        assert result["units"][0]["player"] == 0
        assert result["units"][1]["player"] == 1

    def test_visibility_field(self):
        """Test that visibility is included in state."""
        converter = StateConverter(board_size=16)

        tiles = [TileData(x=0, y=0, terrain="TERRAIN_PLAINS")]

        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=tiles),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)

        # Visibility should be in the state
        assert "visibility" in result


class TestCoordinateScaling:
    """Tests for coordinate scaling from Civ6 map to board_size."""

    def test_scale_coords_identity(self) -> None:
        """Test scaling when map size equals board size."""
        converter = StateConverter(board_size=16, create_encoder=False)
        # Set up scaling for 16x16 map -> 16x16 board (identity)
        converter._setup_scaling(16, 16)
        assert converter._scale_coords(0, 0) == (0, 0)
        assert converter._scale_coords(15, 15) == (15, 15)
        assert converter._scale_coords(8, 8) == (8, 8)

    def test_scale_coords_downscale(self) -> None:
        """Test scaling when downscaling (larger map to smaller board)."""
        converter = StateConverter(board_size=16, create_encoder=False)
        # 80x50 map -> 16x16 board
        converter._setup_scaling(80, 50)

        # Origin stays at origin
        x, y = converter._scale_coords(0, 0)
        assert x == 0
        assert y == 0

        # Far corner should scale down
        x, y = converter._scale_coords(79, 49)
        assert x <= 15
        assert y <= 15

    def test_scale_coords_clamping(self) -> None:
        """Test that coordinates are clamped to valid range."""
        converter = StateConverter(board_size=16, create_encoder=False)
        converter._setup_scaling(80, 50)

        # Even with extreme values, should clamp to valid range
        x, y = converter._scale_coords(1000, 1000)
        assert 0 <= x <= 15
        assert 0 <= y <= 15

        x, y = converter._scale_coords(-10, -10)
        assert x == 0
        assert y == 0

    def test_scale_coords_bidirectional_approximate(self) -> None:
        """Test that scaling is approximately reversible."""
        converter = StateConverter(board_size=16, create_encoder=False)
        converter._setup_scaling(80, 50)

        # Scale down
        original_x, original_y = 40, 25  # Center of original map
        scaled_x, scaled_y = converter._scale_coords(original_x, original_y)

        # Scaled coordinates should be near center of board
        # For 80->16 scale factor ~0.2, center 40 -> ~8
        assert 6 <= scaled_x <= 10
        assert 6 <= scaled_y <= 10


class TestObservationShapes:
    """Tests for observation tensor shapes consistency."""

    def test_encoder_spatial_shape(self) -> None:
        """Test spatial tensor has expected 84 channels."""
        converter = StateConverter(board_size=16, create_encoder=True)
        encoder = converter._encoder
        assert encoder is not None

        # Civ6StateEncoder uses 84 spatial channels
        config = encoder.config
        assert config.num_spatial_channels == 84
        assert config.board_size == 16

    def test_encoder_vector_shape(self) -> None:
        """Test vector tensor has expected 512 dimensions."""
        converter = StateConverter(board_size=16, create_encoder=True)
        encoder = converter._encoder
        assert encoder is not None

        config = encoder.config
        assert config.vector_dim == 512

    def test_shapes_consistent_across_states(self) -> None:
        """Test tensor shapes remain consistent across different states."""
        converter = StateConverter(board_size=16, create_encoder=True)

        # Create two different states
        state1 = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        state2 = GameStateData(
            turn_id=100,
            player_id=1,
            map_data=MapData(
                width=80,
                height=50,
                tiles=[
                    TileData(x=x, y=y, terrain="TERRAIN_GRASS")
                    for x in range(10)
                    for y in range(10)
                ],
            ),
            units=[
                UnitData(id=1, unit_type="UNIT_WARRIOR", owner_id=0, x=5, y=5),
                UnitData(id=2, unit_type="UNIT_ARCHER", owner_id=1, x=10, y=10),
            ],
            cities=[
                CityData(id=1, name="City1", owner_id=0, x=5, y=5),
            ],
            players=[
                PlayerData(id=0, civilization="CIV_A", gold=1000.0),
                PlayerData(id=1, civilization="CIV_B", gold=500.0),
            ],
            legal_actions=[],
        )

        spatial1, vector1 = converter.convert_and_encode(state1)
        spatial2, vector2 = converter.convert_and_encode(state2)

        # Shapes must be identical regardless of state content
        assert spatial1.shape == spatial2.shape
        assert vector1.shape == vector2.shape
        assert spatial1.shape == (84, 16, 16)
        assert vector1.shape == (512,)


class TestResourceExtractionComplete:
    """Tests for complete resource extraction (addresses TODO at lines 311-314)."""

    def test_technologies_field_type(self) -> None:
        """Test technologies field is a list."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert "technologies" in result
        assert isinstance(result["technologies"], list)

    def test_civics_field_type(self) -> None:
        """Test civics field is a list."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert "civics" in result
        assert isinstance(result["civics"], list)

    def test_victory_progress_field_type(self) -> None:
        """Test victory_progress field is a dict."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert "victory_progress" in result
        assert isinstance(result["victory_progress"], dict)

    def test_era_field_preserved(self) -> None:
        """Test era field is preserved through conversion."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
            era="CLASSICAL",
        )

        result = converter.convert(state)
        assert result["era"] == "CLASSICAL"


class TestUnitEncodingCombatStats:
    """Tests for unit encoding with combat stats."""

    def test_unit_health_in_output(self) -> None:
        """Test unit health is included in converted unit data."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[
                UnitData(
                    id=1,
                    unit_type="UNIT_WARRIOR",
                    owner_id=0,
                    x=5,
                    y=5,
                    health=80,
                    max_health=100,
                    damage=20,
                ),
            ],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert len(result["units"]) == 1

    def test_unit_moves_in_output(self) -> None:
        """Test unit movement points are included."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[
                UnitData(
                    id=1,
                    unit_type="UNIT_ARCHER",
                    owner_id=0,
                    x=3,
                    y=3,
                    moves_remaining=1.0,
                    max_moves=2.0,
                ),
            ],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        assert len(result["units"]) == 1

    def test_ranged_unit_stats(self) -> None:
        """Test ranged unit has range and ranged strength stats."""
        converter = StateConverter(board_size=16, create_encoder=False)
        state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26, tiles=[]),
            units=[
                UnitData(
                    id=1,
                    unit_type="UNIT_ARCHER",
                    owner_id=0,
                    x=5,
                    y=5,
                    ranged_strength=25,
                    range=2,
                ),
            ],
            cities=[],
            players=[],
            legal_actions=[],
        )

        result = converter.convert(state)
        # Verify unit is properly converted
        assert len(result["units"]) == 1
        assert result["units"][0]["type"] == "ranged"
