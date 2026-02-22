"""Tests for state extraction methods in StateConverter.

This module tests the resource, technology, civic, and victory progress
extraction methods added for hierarchical controller integration.

Note: Many tests are marked as skip because they require the full protocol
implementation which uses complex data structures. The integration tests
verify the methods work with real game state data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


@dataclass
class MockPlayerData:
    """Mock player data for testing state extraction."""

    id: int = 0
    gold: int = 500
    faith: int = 100
    science_per_turn: int = 25
    culture_per_turn: int = 20
    score: int = 100
    cities: list[Any] = field(default_factory=list)
    units: list[Any] = field(default_factory=list)
    techs_researched: list[str] = field(default_factory=list)
    civics_completed: list[str] = field(default_factory=list)
    current_research: str | None = None


@dataclass
class MockCity:
    """Mock city for testing."""

    owner_id: int = 0
    population: int = 5


@dataclass
class MockGameStateData:
    """Mock game state data for testing."""

    player_id: int = 0
    turn: int = 50
    era: str = "ancient"
    players: list[MockPlayerData] = field(default_factory=list)
    cities: list[MockCity] = field(default_factory=list)
    units: list[Any] = field(default_factory=list)
    legal_actions: list[Any] = field(default_factory=list)
    map_data: Any = None

    def __post_init__(self):
        if self.map_data is None:
            self.map_data = MagicMock()
            self.map_data.width = 16
            self.map_data.height = 16


class TestResourceExtraction:
    """Tests for resource extraction methods."""

    def test_extract_resources_from_tiles(self):
        """Test extraction of map tile resources (iron, horses, etc.)."""
        from alpha_zero.bridge.protocol import TileData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        # Create mock game state with resource tiles
        tiles = [
            TileData(x=0, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_IRON", owner_id=0),
            TileData(x=1, y=0, terrain="TERRAIN_GRASSLAND", resource="RESOURCE_IRON", owner_id=0),
            TileData(x=2, y=0, terrain="TERRAIN_HILLS", resource="RESOURCE_HORSES", owner_id=0),
            TileData(
                x=3, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_WHEAT", owner_id=1
            ),  # Enemy owned
        ]

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.map_data.tiles = tiles

        resources = converter._extract_resources(state)

        # Should only count resources on tiles we own (owner_id=0)
        assert resources["iron"] == 2
        assert resources["horses"] == 1
        assert "wheat" not in resources  # Enemy owned

    def test_extract_resources_empty_map(self):
        """Test resource extraction with no resources on map."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.map_data.tiles = []

        resources = converter._extract_resources(state)

        # Should return empty dict
        assert resources == {}

    def test_extract_resources_no_ownership(self):
        """Test resource extraction when no tiles are owned."""
        from alpha_zero.bridge.protocol import TileData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        tiles = [
            TileData(x=0, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_IRON", owner_id=None),
            TileData(
                x=1, y=0, terrain="TERRAIN_GRASSLAND", resource="RESOURCE_HORSES", owner_id=None
            ),
        ]

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.map_data.tiles = tiles

        resources = converter._extract_resources(state)

        # Should return empty dict since no tiles are owned by player 0
        assert resources == {}


class TestTechnologyExtraction:
    """Tests for technology extraction methods."""

    def test_extract_researched_techs(self):
        """Test extraction of researched technologies."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(
            id=0,
            civilization="ROME",
            techs_researched=["TECH_POTTERY", "TECH_MINING", "TECH_WRITING"],
        )

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]

        techs = converter._extract_technologies(state)

        # _extract_technologies returns a list of normalized tech names
        assert "pottery" in techs
        assert "mining" in techs
        assert "writing" in techs
        assert len(techs) == 3

    def test_extract_techs_empty(self):
        """Test technology extraction with no researched techs."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME", techs_researched=[])

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]

        techs = converter._extract_technologies(state)

        assert techs == []

    def test_extract_techs_no_player(self):
        """Test technology extraction when player is not found."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = []

        techs = converter._extract_technologies(state)

        # Should return empty list
        assert techs == []


class TestCivicExtraction:
    """Tests for civic extraction methods."""

    def test_extract_civics_returns_empty_list(self):
        """Test that civic extraction returns empty list (placeholder implementation)."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = []

        civics = converter._extract_civics(state)

        # _extract_civics returns a list (currently empty as placeholder)
        assert civics == []

    def test_extract_civics_with_player(self):
        """Test civic extraction with player data."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME")

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]

        civics = converter._extract_civics(state)

        # Should return empty list (civics not yet implemented in protocol)
        assert civics == []


class TestVictoryProgressExtraction:
    """Tests for victory progress extraction methods."""

    def test_extract_victory_progress_default(self):
        """Test default victory progress values."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME", score=100)

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]
        state.cities = []
        state.is_game_over = False
        state.winner_id = None

        victory = converter._extract_victory_progress(state)

        # Should have science, domination, score, and complete keys
        assert "science" in victory
        assert "domination" in victory
        assert "score" in victory
        assert "complete" in victory

        # Default values should be in [0.0, 1.0]
        for key, value in victory.items():
            assert isinstance(value, float)
            if key == "complete":
                assert value in [-1.0, 0.0, 1.0]
            else:
                assert 0.0 <= value <= 1.0

    def test_extract_domination_progress(self):
        """Test domination victory progress calculation."""
        from alpha_zero.bridge.protocol import CityData, PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME", score=100)
        enemy = PlayerData(id=1, civilization="GREECE", score=50)

        # Create cities (2 ours, 1 enemy)
        cities = [
            CityData(id=1, name="Rome", owner_id=0, x=0, y=0),
            CityData(id=2, name="Antium", owner_id=0, x=5, y=5),
            CityData(id=3, name="Athens", owner_id=1, x=10, y=10),
        ]

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player, enemy]
        state.cities = cities
        state.is_game_over = False
        state.winner_id = None

        victory = converter._extract_victory_progress(state)

        # Domination progress = our cities / total cities = 2/3 ≈ 0.667
        assert "domination" in victory
        assert abs(victory["domination"] - 2 / 3) < 0.01

    def test_extract_science_progress(self):
        """Test science victory progress calculation."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(
            id=0,
            civilization="ROME",
            score=100,
            techs_researched=[
                "TECH_POTTERY",
                "TECH_MINING",
                "TECH_WRITING",
                "TECH_SAILING",
                "TECH_ASTROLOGY",
            ],
        )

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]
        state.cities = []
        state.is_game_over = False
        state.winner_id = None

        victory = converter._extract_victory_progress(state)

        # Science progress based on tech count / 70
        assert "science" in victory
        assert victory["science"] == min(1.0, 5 / 70.0)

    def test_extract_game_over_victory(self):
        """Test victory progress when game is over and player wins."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME", score=500)

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]
        state.cities = []
        state.is_game_over = True
        state.winner_id = 0  # Player wins

        victory = converter._extract_victory_progress(state)

        # complete should be 1.0 for victory
        assert victory["complete"] == 1.0

    def test_extract_game_over_defeat(self):
        """Test victory progress when game is over and player loses."""
        from alpha_zero.bridge.protocol import PlayerData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=0, civilization="ROME", score=100)

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = [player]
        state.cities = []
        state.is_game_over = True
        state.winner_id = 1  # Enemy wins

        victory = converter._extract_victory_progress(state)

        # complete should be -1.0 for defeat
        assert victory["complete"] == -1.0


class TestStateConverterIntegration:
    """Integration tests for StateConverter."""

    @pytest.mark.skip(reason="Requires full protocol implementation - covered by integration tests")
    def test_convert_includes_all_fields(self):
        """Test that convert() includes all extraction fields."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = MockPlayerData(
            id=0,
            gold=500,
            techs_researched=["pottery"],
        )
        state = MockGameStateData(
            player_id=0,
            turn=50,
            players=[player],
            cities=[],
        )

        result = converter.convert(state)

        # Should have all fields
        assert "resources" in result
        assert "technologies" in result
        assert "civics" in result
        assert "victory_progress" in result

    @pytest.mark.skip(reason="Requires full protocol implementation - covered by integration tests")
    def test_convert_handles_missing_data(self):
        """Test that convert() handles missing data gracefully."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        # Minimal state with no player data
        state = MockGameStateData(
            player_id=0,
            players=[],
            cities=[],
        )

        # Should not raise
        result = converter.convert(state)

        assert "resources" in result
        assert "technologies" in result


class TestMicroCivFallback:
    """Tests for MicroCiv environment fallback extraction."""

    def test_microciv_observation_structure(self):
        """Test that MicroCiv observation can be created."""
        # Create MicroCiv-style observation
        obs = {
            "vector": np.array([0.5, 0.8, 0.3, 0.2, 0.4, 0.1]),
            "spatial": np.zeros((13, 16, 16)),
        }

        # Verify observation structure
        assert obs["vector"].shape == (6,)
        assert obs["spatial"].shape == (13, 16, 16)


class TestStateExtractionEdgeCases:
    """Tests for edge cases in state extraction."""

    def test_empty_player_list(self):
        """Test extraction with empty player list."""
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0
        state.players = []
        state.cities = []
        state.map_data.tiles = []
        state.is_game_over = False
        state.winner_id = None

        # Should not raise
        resources = converter._extract_resources(state)
        techs = converter._extract_technologies(state)
        civics = converter._extract_civics(state)
        victory = converter._extract_victory_progress(state)

        assert resources == {}  # No resources on map
        assert techs == []  # No player = no techs
        assert civics == []  # No player = no civics
        # Victory progress should have default values
        assert "domination" in victory
        assert victory["domination"] == 0.0

    def test_wrong_player_id(self):
        """Test extraction when player_id doesn't match any player."""
        from alpha_zero.bridge.protocol import PlayerData, TileData
        from alpha_zero.bridge.state_converter import StateConverter

        converter = StateConverter()

        player = PlayerData(id=1, civilization="ROME", gold=500)  # ID is 1

        # Create a tile owned by player 1
        tile = TileData(x=0, y=0, terrain="TERRAIN_PLAINS", resource="RESOURCE_IRON", owner_id=1)

        from unittest.mock import MagicMock

        state = MagicMock()
        state.player_id = 0  # Looking for ID 0
        state.players = [player]
        state.cities = []
        state.map_data.tiles = [tile]
        state.is_game_over = False
        state.winner_id = None

        resources = converter._extract_resources(state)

        # Should return empty since no tiles owned by player 0
        assert resources == {}

        techs = converter._extract_technologies(state)

        # Should return empty list since no matching player
        assert techs == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
