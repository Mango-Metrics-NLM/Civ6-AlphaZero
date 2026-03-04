"""Tests for alpha_zero.bridge.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from alpha_zero.bridge.config import (
    MAP_SIZES,
    BridgeConfig,
    MapSizeConfig,
    _get_default_comm_dir,
    get_map_config,
)


class TestGetDefaultCommDir:
    """Test _get_default_comm_dir function."""

    def test_returns_path(self) -> None:
        result = _get_default_comm_dir()
        assert isinstance(result, Path)

    def test_path_is_absolute(self) -> None:
        result = _get_default_comm_dir()
        assert result.is_absolute()


class TestBridgeConfigDefaults:
    """Test BridgeConfig default values."""

    def test_defaults(self) -> None:
        config = BridgeConfig()
        assert config.poll_interval_ms == 10
        assert config.command_timeout_ms == 5000
        assert config.heartbeat_timeout_s == 2.0
        assert config.max_retries == 5
        assert config.retry_delay_s == 1.0
        assert config.board_size == 16
        assert config.log_level == "INFO"
        assert config.debug_mode is False
        assert config.num_players == 2
        assert config.max_actions_per_second == 1.0
        assert config.multiplayer_mode is False
        assert config.player_id_mapping is None

    def test_custom_values(self) -> None:
        config = BridgeConfig(
            comm_dir=Path("/tmp/test"),
            poll_interval_ms=50,
            command_timeout_ms=10000,
            heartbeat_timeout_s=5.0,
            max_retries=10,
            retry_delay_s=2.0,
            board_size=32,
            log_level="DEBUG",
            debug_mode=True,
            num_players=6,
            max_actions_per_second=2.5,
            multiplayer_mode=True,
            player_id_mapping={0: 0, 1: 1, 2: 2},
        )
        assert config.poll_interval_ms == 50
        assert config.board_size == 32
        assert config.num_players == 6
        assert config.multiplayer_mode is True
        assert config.player_id_mapping == {0: 0, 1: 1, 2: 2}

    def test_string_comm_dir_converted_to_path(self) -> None:
        config = BridgeConfig(comm_dir="/tmp/test_bridge")  # type: ignore[arg-type]
        assert isinstance(config.comm_dir, Path)
        assert str(config.comm_dir) == "/tmp/test_bridge"


class TestBridgeConfigValidation:
    """Test BridgeConfig __post_init__ validation."""

    def test_poll_interval_too_low(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_ms"):
            BridgeConfig(poll_interval_ms=0)

    def test_command_timeout_too_low(self) -> None:
        with pytest.raises(ValueError, match="command_timeout_ms"):
            BridgeConfig(command_timeout_ms=50)

    def test_heartbeat_timeout_too_low(self) -> None:
        with pytest.raises(ValueError, match="heartbeat_timeout_s"):
            BridgeConfig(heartbeat_timeout_s=0.1)

    def test_max_retries_too_low(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            BridgeConfig(max_retries=0)

    def test_board_size_too_small(self) -> None:
        with pytest.raises(ValueError, match="board_size"):
            BridgeConfig(board_size=4)

    def test_board_size_too_large(self) -> None:
        with pytest.raises(ValueError, match="board_size"):
            BridgeConfig(board_size=256)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            BridgeConfig(log_level="VERBOSE")

    def test_num_players_too_low(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            BridgeConfig(num_players=1)

    def test_num_players_too_high(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            BridgeConfig(num_players=13)

    def test_negative_max_actions_per_second(self) -> None:
        with pytest.raises(ValueError, match="max_actions_per_second"):
            BridgeConfig(max_actions_per_second=-1.0)

    def test_zero_max_actions_per_second_allowed(self) -> None:
        config = BridgeConfig(max_actions_per_second=0.0)
        assert config.max_actions_per_second == 0.0

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            config = BridgeConfig(log_level=level)
            assert config.log_level == level

    def test_boundary_board_sizes(self) -> None:
        config_min = BridgeConfig(board_size=8)
        assert config_min.board_size == 8
        config_max = BridgeConfig(board_size=128)
        assert config_max.board_size == 128

    def test_boundary_num_players(self) -> None:
        config_min = BridgeConfig(num_players=2)
        assert config_min.num_players == 2
        config_max = BridgeConfig(num_players=12)
        assert config_max.num_players == 12


class TestBridgeConfigFromEnv:
    """Test BridgeConfig.from_env class method."""

    def test_defaults_from_empty_env(self) -> None:
        config = BridgeConfig.from_env()
        assert isinstance(config, BridgeConfig)

    def test_comm_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_COMM_DIR", "/tmp/custom_dir")
        config = BridgeConfig.from_env()
        assert config.comm_dir == Path("/tmp/custom_dir")

    def test_poll_interval_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_POLL_MS", "50")
        config = BridgeConfig.from_env()
        assert config.poll_interval_ms == 50

    def test_command_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_COMMAND_TIMEOUT_MS", "10000")
        config = BridgeConfig.from_env()
        assert config.command_timeout_ms == 10000

    def test_heartbeat_timeout_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_HEARTBEAT_TIMEOUT_S", "5.0")
        config = BridgeConfig.from_env()
        assert config.heartbeat_timeout_s == 5.0

    def test_max_retries_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_MAX_RETRIES", "10")
        config = BridgeConfig.from_env()
        assert config.max_retries == 10

    def test_retry_delay_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_RETRY_DELAY_S", "2.5")
        config = BridgeConfig.from_env()
        assert config.retry_delay_s == 2.5

    def test_board_size_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BOARD_SIZE", "32")
        config = BridgeConfig.from_env()
        assert config.board_size == 32

    def test_log_level_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_LOG_LEVEL", "debug")
        config = BridgeConfig.from_env()
        assert config.log_level == "DEBUG"

    def test_debug_mode_true_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_DEBUG_MODE", "1")
        config = BridgeConfig.from_env()
        assert config.debug_mode is True

    def test_debug_mode_false_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_DEBUG_MODE", "0")
        config = BridgeConfig.from_env()
        assert config.debug_mode is False

    def test_num_players_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_NUM_PLAYERS", "4")
        config = BridgeConfig.from_env()
        assert config.num_players == 4

    def test_max_actions_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_MAX_ACTIONS_PER_SECOND", "2.5")
        config = BridgeConfig.from_env()
        assert config.max_actions_per_second == 2.5

    def test_multiplayer_mode_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_MULTIPLAYER_MODE", "true")
        config = BridgeConfig.from_env()
        assert config.multiplayer_mode is True


class TestBridgeConfigProperties:
    """Test BridgeConfig file path properties."""

    def test_state_file(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        assert config.state_file == Path("/tmp/test/state.json")

    def test_commands_file(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        assert config.commands_file == Path("/tmp/test/commands.json")

    def test_ack_file(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        assert config.ack_file == Path("/tmp/test/ack.json")

    def test_errors_file(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        assert config.errors_file == Path("/tmp/test/errors.json")

    def test_heartbeat_file(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        assert config.heartbeat_file == Path("/tmp/test/heartbeat.txt")


class TestBridgeConfigMethods:
    """Test BridgeConfig methods."""

    def test_ensure_comm_dir(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "bridge_comm"
        config = BridgeConfig(comm_dir=test_dir)
        assert not test_dir.exists()
        config.ensure_comm_dir()
        assert test_dir.exists()

    def test_ensure_comm_dir_nested(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "a" / "b" / "c"
        config = BridgeConfig(comm_dir=test_dir)
        config.ensure_comm_dir()
        assert test_dir.exists()

    def test_to_dict(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["comm_dir"] == "/tmp/test"
        assert d["poll_interval_ms"] == 10
        assert d["board_size"] == 16
        assert d["num_players"] == 2
        assert d["multiplayer_mode"] is False
        assert d["player_id_mapping"] is None

    def test_to_dict_all_fields_present(self) -> None:
        config = BridgeConfig(comm_dir=Path("/tmp/test"))
        d = config.to_dict()
        expected_keys = {
            "comm_dir", "poll_interval_ms", "command_timeout_ms",
            "heartbeat_timeout_s", "max_retries", "retry_delay_s",
            "board_size", "log_level", "debug_mode", "num_players",
            "max_actions_per_second", "multiplayer_mode", "player_id_mapping",
        }
        assert set(d.keys()) == expected_keys


class TestMapSizeConfig:
    """Test MapSizeConfig dataclass."""

    def test_construction(self) -> None:
        config = MapSizeConfig("Test", 80, 50)
        assert config.name == "Test"
        assert config.width == 80
        assert config.height == 50

    def test_total_tiles(self) -> None:
        config = MapSizeConfig("Test", 80, 50)
        assert config.total_tiles == 4000

    def test_get_scale_factor(self) -> None:
        config = MapSizeConfig("Test", 80, 40)
        sx, sy = config.get_scale_factor(16)
        assert sx == pytest.approx(0.2)
        assert sy == pytest.approx(0.4)

    def test_get_scale_factor_square(self) -> None:
        config = MapSizeConfig("Square", 32, 32)
        sx, sy = config.get_scale_factor(16)
        assert sx == pytest.approx(0.5)
        assert sy == pytest.approx(0.5)


class TestMapSizes:
    """Test MAP_SIZES constant."""

    def test_has_all_standard_sizes(self) -> None:
        expected = {"duel", "tiny", "small", "standard", "large", "huge"}
        assert set(MAP_SIZES.keys()) == expected

    def test_all_values_are_map_size_config(self) -> None:
        for name, config in MAP_SIZES.items():
            assert isinstance(config, MapSizeConfig), f"{name} is not MapSizeConfig"

    def test_sizes_increase(self) -> None:
        sizes = ["duel", "tiny", "small", "standard", "large", "huge"]
        for i in range(len(sizes) - 1):
            c1 = MAP_SIZES[sizes[i]]
            c2 = MAP_SIZES[sizes[i + 1]]
            assert c2.width > c1.width, f"{sizes[i+1]} width should be > {sizes[i]}"
            assert c2.height > c1.height, f"{sizes[i+1]} height should be > {sizes[i]}"

    def test_duel_size(self) -> None:
        d = MAP_SIZES["duel"]
        assert d.width == 44
        assert d.height == 26

    def test_standard_size(self) -> None:
        s = MAP_SIZES["standard"]
        assert s.width == 84
        assert s.height == 54


class TestGetMapConfig:
    """Test get_map_config function."""

    def test_matches_standard_size(self) -> None:
        config = get_map_config(84, 54)
        assert config.name == "Standard"
        assert config.width == 84
        assert config.height == 54

    def test_matches_duel(self) -> None:
        config = get_map_config(44, 26)
        assert config.name == "Duel"

    def test_creates_custom_for_unknown(self) -> None:
        config = get_map_config(100, 100)
        assert "Custom" in config.name
        assert config.width == 100
        assert config.height == 100

    def test_custom_name_includes_dimensions(self) -> None:
        config = get_map_config(75, 45)
        assert config.name == "Custom_75x45"
