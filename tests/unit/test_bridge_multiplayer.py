"""Unit tests for multiplayer bridge enhancements (Phase 24A / Gap G12).

Tests cover:
- Player ID remapping (non-contiguous -> sequential)
- N-player state conversion
- Action throttling
- Multiplayer bridge configuration
- Unciv multiplayer save parsing (with mock save data)
- Backward compatibility (existing 2-player behavior)
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import FileBridge, MockFileBridge
from alpha_zero.bridge.state_converter import StateConverter

# =============================================================================
# Fixtures and helpers
# =============================================================================


def _make_raw_state(
    num_players: int = 2,
    player_ids: list[int] | None = None,
    current_player_id: int = 0,
) -> dict:
    """Build a minimal raw state dict compatible with GameStateData.from_dict.

    Args:
        num_players: Number of players to generate.
        player_ids: Explicit player IDs (may be non-contiguous).
        current_player_id: The player whose turn it is.

    Returns:
        Raw state dict.
    """
    if player_ids is None:
        player_ids = list(range(num_players))

    players = [
        {
            "id": pid,
            "civilization": f"Civ_{pid}",
            "gold": 100.0 + pid * 10,
            "science_per_turn": 5.0,
            "culture_per_turn": 3.0,
            "faith": 0.0,
            "score": 50 + pid * 5,
            "is_alive": True,
            "is_human": pid == current_player_id,
            "techs_researched": [],
            "civics_researched": [],
        }
        for pid in player_ids
    ]

    units = [
        {
            "id": 100 + idx,
            "unit_type": "UNIT_WARRIOR",
            "x": 5 + idx,
            "y": 5 + idx,
            "owner_id": pid,
            "health": 100,
            "max_health": 100,
            "combat_strength": 20,
            "moves_remaining": 2,
            "max_moves": 2,
        }
        for idx, pid in enumerate(player_ids)
    ]

    cities = [
        {
            "id": 200 + idx,
            "name": f"City_{pid}",
            "x": 3 + idx,
            "y": 3 + idx,
            "owner_id": pid,
            "population": 1,
            "defense_strength": 10,
            "production_progress": 0,
            "is_capital": True,
            "buildings": [],
            "available_production": [],
        }
        for idx, pid in enumerate(player_ids)
    ]

    return {
        "turn_id": 10,
        "player_id": current_player_id,
        "map": {
            "width": 44,
            "height": 26,
            "tiles": [
                {
                    "x": x,
                    "y": y,
                    "terrain": "TERRAIN_PLAINS",
                    "features": [],
                    "yields": {},
                    "resource": None,
                    "is_visible": True,
                    "is_revealed": True,
                    "owner_id": -1,
                }
                for x in range(4)
                for y in range(4)
            ],
        },
        "units": units,
        "cities": cities,
        "players": players,
        "legal_actions": [],
    }


def _make_state_message(raw_state: dict) -> dict:
    """Wrap a raw state in a StateMessage-compatible dict."""
    return raw_state


# =============================================================================
# BridgeConfig multiplayer tests
# =============================================================================


class TestBridgeConfigMultiplayer:
    """Tests for new multiplayer fields in BridgeConfig."""

    def test_default_values(self) -> None:
        """Test that new fields have correct defaults."""
        config = BridgeConfig()
        assert config.num_players == 2
        assert config.max_actions_per_second == 1.0
        assert config.multiplayer_mode is False
        assert config.player_id_mapping is None

    def test_custom_multiplayer_config(self) -> None:
        """Test setting multiplayer configuration."""
        mapping = {0: 0, 3: 1, 5: 2, 7: 3}
        config = BridgeConfig(
            num_players=4,
            max_actions_per_second=2.0,
            multiplayer_mode=True,
            player_id_mapping=mapping,
        )
        assert config.num_players == 4
        assert config.max_actions_per_second == 2.0
        assert config.multiplayer_mode is True
        assert config.player_id_mapping == mapping

    def test_num_players_validation_low(self) -> None:
        """Test that num_players < 2 is rejected."""
        with pytest.raises(ValueError, match="num_players"):
            BridgeConfig(num_players=1)

    def test_num_players_validation_high(self) -> None:
        """Test that num_players > 12 is rejected."""
        with pytest.raises(ValueError, match="num_players"):
            BridgeConfig(num_players=13)

    def test_max_actions_per_second_validation(self) -> None:
        """Test that negative max_actions_per_second is rejected."""
        with pytest.raises(ValueError, match="max_actions_per_second"):
            BridgeConfig(max_actions_per_second=-1.0)

    def test_max_actions_per_second_zero_allowed(self) -> None:
        """Test that 0.0 (disabled) is allowed."""
        config = BridgeConfig(max_actions_per_second=0.0)
        assert config.max_actions_per_second == 0.0

    def test_to_dict_includes_new_fields(self) -> None:
        """Test that to_dict includes multiplayer fields."""
        config = BridgeConfig(num_players=6, multiplayer_mode=True)
        d = config.to_dict()
        assert d["num_players"] == 6
        assert d["multiplayer_mode"] is True
        assert d["max_actions_per_second"] == 1.0
        assert d["player_id_mapping"] is None

    def test_from_env_multiplayer(self) -> None:
        """Test loading multiplayer config from environment."""
        with patch.dict(
            "os.environ",
            {
                "CIV6_BRIDGE_NUM_PLAYERS": "6",
                "CIV6_BRIDGE_MAX_ACTIONS_PER_SECOND": "0.5",
                "CIV6_BRIDGE_MULTIPLAYER_MODE": "true",
            },
        ):
            config = BridgeConfig.from_env()
            assert config.num_players == 6
            assert config.max_actions_per_second == 0.5
            assert config.multiplayer_mode is True


# =============================================================================
# StateConverter player ID remapping tests
# =============================================================================


class TestPlayerIdRemapping:
    """Tests for player ID remapping functionality."""

    def test_build_mapping_contiguous(self) -> None:
        """Test mapping with contiguous IDs (no change needed)."""
        mapping = StateConverter.build_player_id_mapping([0, 1, 2, 3])
        assert mapping == {0: 0, 1: 1, 2: 2, 3: 3}

    def test_build_mapping_non_contiguous(self) -> None:
        """Test mapping with non-contiguous IDs."""
        mapping = StateConverter.build_player_id_mapping([0, 1, 3, 5])
        assert mapping == {0: 0, 1: 1, 3: 2, 5: 3}

    def test_build_mapping_unsorted(self) -> None:
        """Test mapping with unsorted IDs."""
        mapping = StateConverter.build_player_id_mapping([5, 0, 3, 1])
        assert mapping == {0: 0, 1: 1, 3: 2, 5: 3}

    def test_build_mapping_duplicates(self) -> None:
        """Test mapping with duplicate IDs."""
        mapping = StateConverter.build_player_id_mapping([0, 1, 1, 3])
        assert mapping == {0: 0, 1: 1, 3: 2}

    def test_remap_player_ids_basic(self) -> None:
        """Test basic player ID remapping on a raw state dict."""
        raw_state = {
            "player_id": 5,
            "players": [
                {"id": 0, "civilization": "America"},
                {"id": 3, "civilization": "Egypt"},
                {"id": 5, "civilization": "China"},
            ],
            "units": [
                {"id": "u1", "owner_id": 0},
                {"id": "u2", "owner_id": 3},
                {"id": "u3", "owner_id": 5},
            ],
            "cities": [
                {"id": "c1", "owner_id": 0},
                {"id": "c2", "owner_id": 5},
            ],
            "winner_id": 5,
        }
        mapping = {0: 0, 3: 1, 5: 2}
        remapped = StateConverter._remap_player_ids(raw_state, mapping)

        # Player IDs remapped
        assert remapped["players"][0]["id"] == 0
        assert remapped["players"][1]["id"] == 1
        assert remapped["players"][2]["id"] == 2

        # Unit owner_ids remapped
        assert remapped["units"][0]["owner_id"] == 0
        assert remapped["units"][1]["owner_id"] == 1
        assert remapped["units"][2]["owner_id"] == 2

        # City owner_ids remapped
        assert remapped["cities"][0]["owner_id"] == 0
        assert remapped["cities"][1]["owner_id"] == 2

        # Top-level IDs remapped
        assert remapped["player_id"] == 2
        assert remapped["winner_id"] == 2

    def test_remap_preserves_original(self) -> None:
        """Test that remapping does not modify the original state."""
        raw_state = {
            "player_id": 5,
            "players": [{"id": 5, "civilization": "China"}],
            "units": [{"id": "u1", "owner_id": 5}],
            "cities": [{"id": "c1", "owner_id": 5}],
        }
        mapping = {5: 0}
        _ = StateConverter._remap_player_ids(raw_state, mapping)

        # Original unchanged
        assert raw_state["player_id"] == 5
        assert raw_state["players"][0]["id"] == 5

    def test_remap_none_winner(self) -> None:
        """Test remapping with winner_id=None."""
        raw_state = {
            "player_id": 0,
            "players": [{"id": 0}],
            "units": [],
            "cities": [],
            "winner_id": None,
        }
        mapping = {0: 0}
        remapped = StateConverter._remap_player_ids(raw_state, mapping)
        assert remapped["winner_id"] is None


# =============================================================================
# StateConverter N-player conversion tests
# =============================================================================


class TestNPlayerStateConversion:
    """Tests for N-player state conversion."""

    @pytest.mark.parametrize("num_players", [2, 4, 6])
    def test_convert_multiplayer_state(self, num_players: int) -> None:
        """Test multiplayer state conversion with different player counts."""
        raw_state = _make_raw_state(num_players=num_players)
        converter = StateConverter(board_size=16, num_players=num_players, create_encoder=False)

        result = converter.convert_multiplayer_state(raw_state, num_players=num_players)

        assert result["_num_players"] == num_players
        assert isinstance(result["_player_mapping"], dict)
        assert len(result["_player_mapping"]) == num_players
        assert result["turn"] == 10

    @pytest.mark.parametrize("num_players", [2, 4, 6])
    def test_convert_multiplayer_non_contiguous_ids(self, num_players: int) -> None:
        """Test conversion with non-contiguous player IDs."""
        # Use non-contiguous IDs
        player_ids = [i * 2 + 1 for i in range(num_players)]  # 1, 3, 5, ...
        raw_state = _make_raw_state(
            num_players=num_players,
            player_ids=player_ids,
            current_player_id=player_ids[0],
        )
        converter = StateConverter(board_size=16, num_players=num_players, create_encoder=False)

        result = converter.convert_multiplayer_state(raw_state)

        mapping = result["_player_mapping"]
        # Verify mapping is sequential
        assert sorted(mapping.values()) == list(range(num_players))
        assert result["_num_players"] == num_players

    def test_convert_multiplayer_with_explicit_mapping(self) -> None:
        """Test conversion with an explicit player mapping."""
        raw_state = _make_raw_state(
            num_players=3,
            player_ids=[0, 5, 10],
            current_player_id=0,
        )
        explicit_mapping = {0: 0, 5: 1, 10: 2}
        converter = StateConverter(board_size=16, num_players=3, create_encoder=False)

        result = converter.convert_multiplayer_state(
            raw_state, player_mapping=explicit_mapping
        )

        assert result["_player_mapping"] == explicit_mapping
        assert result["_num_players"] == 3

    def test_backward_compat_2_player(self) -> None:
        """Test that default 2-player conversion is unchanged."""
        raw_state = _make_raw_state(num_players=2)

        # Standard converter (2-player default)
        converter = StateConverter(board_size=16, create_encoder=False)
        result = converter.convert(raw_state)

        # Should still work as before
        assert result["current_player"] == 0
        assert result["turn"] == 10

        # Units should have player 0 and 1
        unit_players = {u["player"] for u in result["units"]}
        assert unit_players == {0, 1}

    def test_n_player_unit_ownership(self) -> None:
        """Test that N-player mode preserves per-player unit ownership."""
        num_players = 4
        raw_state = _make_raw_state(num_players=num_players)
        converter = StateConverter(
            board_size=16, num_players=num_players, create_encoder=False
        )

        result = converter.convert_multiplayer_state(raw_state)

        # Each player's unit should have distinct player index
        unit_players = sorted(u["player"] for u in result["units"])
        assert unit_players == list(range(num_players))

    def test_n_player_city_ownership(self) -> None:
        """Test that N-player mode preserves per-player city ownership."""
        num_players = 4
        raw_state = _make_raw_state(num_players=num_players)
        converter = StateConverter(
            board_size=16, num_players=num_players, create_encoder=False
        )

        result = converter.convert_multiplayer_state(raw_state)

        city_players = sorted(c["player"] for c in result["cities"])
        assert city_players == list(range(num_players))


# =============================================================================
# Action throttling tests
# =============================================================================


class TestActionThrottling:
    """Tests for action throttling in FileBridge."""

    @pytest.mark.asyncio
    async def test_throttle_disabled(self) -> None:
        """Test that throttling is disabled when rate is 0."""
        config = BridgeConfig(max_actions_per_second=0.0)
        bridge = MockFileBridge(config)
        await bridge.start()

        start = time.monotonic()
        for _ in range(5):
            await bridge._throttle_action()
        elapsed = time.monotonic() - start

        # Should complete almost instantly
        assert elapsed < 0.5
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_throttle_limits_rate(self) -> None:
        """Test that throttling enforces rate limit."""
        # 10 actions per second -> 0.1s between actions
        config = BridgeConfig(max_actions_per_second=10.0)
        bridge = MockFileBridge(config)
        await bridge.start()

        # Send 3 actions rapidly
        start = time.monotonic()
        # First action is instant (no previous action)
        await bridge._throttle_action()
        await bridge._throttle_action()
        await bridge._throttle_action()
        elapsed = time.monotonic() - start

        # Should take at least 0.2s for 3 actions at 10/s
        # (0.1s wait before 2nd, 0.1s wait before 3rd)
        assert elapsed >= 0.15  # Small margin for timing
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_throttle_via_send_command(self) -> None:
        """Test that FileBridge.send_command applies throttling.

        Note: MockFileBridge overrides send_command and skips throttling,
        so we test throttling integration via _throttle_action directly
        combined with the real FileBridge logic (using a temp directory).
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            config = BridgeConfig(
                comm_dir=Path(tmp),
                max_actions_per_second=10.0,
            )
            bridge = FileBridge(config)
            await bridge.start()

            from alpha_zero.bridge.protocol import ActionType, CommandMessage, LuaCommand

            cmd = CommandMessage(
                msg_id=1,
                commands=[LuaCommand(action_type=ActionType.END_TURN)],
            )

            start = time.monotonic()
            await bridge.send_command(cmd)
            await bridge.send_command(cmd)
            elapsed = time.monotonic() - start

            # Second command should be throttled (~0.1s gap)
            assert elapsed >= 0.08  # ~0.1s with margin
            await bridge.stop()


# =============================================================================
# Multiplayer state detection tests
# =============================================================================


class TestMultiplayerStateDetection:
    """Tests for multiplayer state detection in FileBridge."""

    def test_detect_2_player(self) -> None:
        """Test detecting 2-player game."""
        state_data = {
            "players": [
                {"id": 0, "civilization": "America"},
                {"id": 1, "civilization": "Egypt"},
            ]
        }
        assert FileBridge.detect_multiplayer_state(state_data) == 2

    def test_detect_4_player(self) -> None:
        """Test detecting 4-player game."""
        state_data = {
            "players": [{"id": i} for i in range(4)]
        }
        assert FileBridge.detect_multiplayer_state(state_data) == 4

    def test_detect_6_player(self) -> None:
        """Test detecting 6-player game."""
        state_data = {
            "players": [{"id": i} for i in range(6)]
        }
        assert FileBridge.detect_multiplayer_state(state_data) == 6

    def test_detect_from_nested_state(self) -> None:
        """Test detecting players from nested 'state' object."""
        state_data = {
            "state": {
                "players": [{"id": i} for i in range(3)]
            }
        }
        assert FileBridge.detect_multiplayer_state(state_data) == 3

    def test_detect_default_no_players(self) -> None:
        """Test default when no players list is present."""
        state_data = {"turn_id": 10}
        assert FileBridge.detect_multiplayer_state(state_data) == 2

    def test_detect_single_player_list(self) -> None:
        """Test default when only 1 player in list."""
        state_data = {"players": [{"id": 0}]}
        assert FileBridge.detect_multiplayer_state(state_data) == 2


# =============================================================================
# Unciv multiplayer save parsing tests
# =============================================================================


class TestUncivMultiplayerSaveParsing:
    """Tests for Unciv multiplayer save parsing."""

    def _make_unciv_save(self, num_players: int) -> dict:
        """Create a minimal Unciv-style save JSON."""
        civs = []
        for i in range(num_players):
            civ_name = f"TestCiv_{i}"
            civs.append({
                "civName": civ_name,
                "playerType": "Human" if i == 0 else "AI",
                "gold": 100 + i * 50,
                "isDefeated": False,
                "cities": [],
                "units": [],
                "tech": {"techsResearched": []},
                "policies": {"adoptedPolicies": []},
            })

        return {
            "version": {"number": "4.0.0"},
            "gameInfo": {
                "gameId": "test-mp-game",
                "difficulty": "Prince",
                "speed": "Standard",
            },
            "currentPlayer": "TestCiv_0",
            "turns": 25,
            "civilizations": civs,
            "tileMap": {"tileList": []},
        }

    @pytest.mark.parametrize("num_players", [2, 4, 6])
    def test_parse_multiplayer_save(self, num_players: int) -> None:
        """Test parsing multiplayer saves with varying player counts."""
        from alpha_zero.interfaces.unciv_interface import UncivInterface

        save_data = self._make_unciv_save(num_players)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(save_data, f)
            save_path = f.name

        try:
            interface = UncivInterface(board_size=16)
            result = interface.parse_multiplayer_save(save_path)

            assert result["num_players"] == num_players
            assert len(result["player_names"]) == num_players
            assert result["current_player_index"] == 0
            assert result["is_multiplayer"] == (num_players > 2)
            assert result["game_state"] is not None

            # Check player_id_mapping
            assert len(result["player_id_mapping"]) == num_players
            for i in range(num_players):
                assert i in result["player_id_mapping"]
                assert result["player_id_mapping"][i] == f"TestCiv_{i}"
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_parse_standard_2_player(self) -> None:
        """Test that 2-player save is not flagged as multiplayer."""
        from alpha_zero.interfaces.unciv_interface import UncivInterface

        save_data = self._make_unciv_save(2)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(save_data, f)
            save_path = f.name

        try:
            interface = UncivInterface(board_size=16)
            result = interface.parse_multiplayer_save(save_path)

            assert result["num_players"] == 2
            assert result["is_multiplayer"] is False
        finally:
            Path(save_path).unlink(missing_ok=True)

    def test_parse_current_player_detection(self) -> None:
        """Test that the current player is correctly detected."""
        from alpha_zero.interfaces.unciv_interface import UncivInterface

        save_data = self._make_unciv_save(4)
        save_data["currentPlayer"] = "TestCiv_2"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(save_data, f)
            save_path = f.name

        try:
            interface = UncivInterface(board_size=16)
            result = interface.parse_multiplayer_save(save_path)

            assert result["current_player_index"] == 2
            assert result["player_names"][2] == "TestCiv_2"
        finally:
            Path(save_path).unlink(missing_ok=True)


# =============================================================================
# Backward compatibility tests
# =============================================================================


class TestBackwardCompatibility:
    """Tests to verify existing 2-player behavior is preserved."""

    def test_bridge_config_defaults_unchanged(self) -> None:
        """Test that existing BridgeConfig defaults are unchanged."""
        config = BridgeConfig()
        assert config.poll_interval_ms == 10
        assert config.command_timeout_ms == 5000
        assert config.heartbeat_timeout_s == 2.0
        assert config.max_retries == 5
        assert config.board_size == 16
        assert config.log_level == "INFO"
        assert config.debug_mode is False

    def test_state_converter_default_num_players(self) -> None:
        """Test that StateConverter defaults to 2 players."""
        converter = StateConverter(board_size=16, create_encoder=False)
        assert converter.num_players == 2

    def test_2_player_convert_unchanged(self) -> None:
        """Test that 2-player convert() produces same results as before."""
        raw_state = _make_raw_state(num_players=2)
        converter = StateConverter(board_size=16, create_encoder=False)

        result = converter.convert(raw_state)

        # Standard 2-player behavior: current player is 0
        assert result["current_player"] == 0

        # Units belong to player 0 (self) or player 1 (opponent)
        for unit in result["units"]:
            assert unit["player"] in (0, 1)

    @pytest.mark.asyncio
    async def test_mock_bridge_backward_compat(self) -> None:
        """Test MockFileBridge still works with default config."""
        bridge = MockFileBridge()
        await bridge.start()

        state_data = {
            "turn_id": 100,
            "player_id": 0,
            "map": {"width": 44, "height": 26},
            "units": [],
            "cities": [],
            "legal_actions": [],
        }
        bridge.set_mock_state(state_data)

        state = await bridge.poll_state()
        assert state is not None
        assert state.turn_id == 100
        await bridge.stop()

    def test_file_bridge_has_throttle_attribute(self) -> None:
        """Test that FileBridge initializes throttle state."""
        bridge = FileBridge()
        assert hasattr(bridge, "_last_action_time")
        assert bridge._last_action_time == 0.0
