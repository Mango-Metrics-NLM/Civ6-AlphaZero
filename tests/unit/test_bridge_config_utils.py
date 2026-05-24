"""Comprehensive pytest tests for alpha_zero.bridge.config and alpha_zero.bridge.utils.

Coverage targets:
    - config.py  ≥ 85 %
    - utils.py   ≥ 85 %

Run with:
    pytest tests/unit/test_bridge_config_utils.py -v --cov=alpha_zero/bridge/config \
           --cov=alpha_zero/bridge/utils --cov-report=term-missing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from alpha_zero.bridge.config import (
    MAP_SIZES,
    BridgeConfig,
    MapSizeConfig,
    _get_default_comm_dir,
    get_map_config,
)
from alpha_zero.bridge.utils import extract_scalar_value

# ===========================================================================
# _get_default_comm_dir
# ===========================================================================


class TestGetDefaultCommDir:
    """Unit tests for the module-level _get_default_comm_dir helper."""

    def test_windows_prefers_onedrive_when_civ6_exists(self, tmp_path: Path) -> None:
        """On Windows, if the OneDrive Civ6 path exists, it should be preferred."""
        onedrive_docs = tmp_path / "OneDrive" / "Documents"
        civ6_path = onedrive_docs / "My Games" / "Sid Meier's Civilization VI"
        civ6_path.mkdir(parents=True)

        with (
            patch("os.name", "nt"),
            patch("alpha_zero.bridge.config.Path.home", return_value=tmp_path),
        ):
            result = _get_default_comm_dir()

        assert "AlphaZeroAI" in str(result)
        assert "OneDrive" in str(result)

    def test_windows_falls_back_to_documents_when_civ6_exists_there(
        self, tmp_path: Path
    ) -> None:
        """On Windows without OneDrive civ6 path, uses regular Documents if Civ6 is there."""
        docs = tmp_path / "Documents"
        civ6_path = docs / "My Games" / "Sid Meier's Civilization VI"
        civ6_path.mkdir(parents=True)
        # No OneDrive equivalent

        with (
            patch("os.name", "nt"),
            patch("alpha_zero.bridge.config.Path.home", return_value=tmp_path),
        ):
            result = _get_default_comm_dir()

        assert "AlphaZeroAI" in str(result)
        assert "Documents" in str(result)

    def test_windows_defaults_to_documents_when_neither_exists(
        self, tmp_path: Path
    ) -> None:
        """On Windows with no Civ6 folder anywhere, defaults to regular Documents."""
        with (
            patch("os.name", "nt"),
            patch("alpha_zero.bridge.config.Path.home", return_value=tmp_path),
        ):
            result = _get_default_comm_dir()

        assert "AlphaZeroAI" in str(result)

    def test_non_windows_returns_home_civ6_alphazero(self, tmp_path: Path) -> None:
        """On Linux/macOS, returns ~/.civ6_alphazero."""
        with (
            patch("os.name", "posix"),
            patch("alpha_zero.bridge.config.Path.home", return_value=tmp_path),
        ):
            result = _get_default_comm_dir()

        assert result == tmp_path / ".civ6_alphazero"


# ===========================================================================
# BridgeConfig – construction and defaults
# ===========================================================================


class TestBridgeConfigDefaults:
    """Verify default values are applied correctly."""

    def test_default_poll_interval(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.poll_interval_ms == 10

    def test_default_command_timeout(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.command_timeout_ms == 5000

    def test_default_heartbeat_timeout(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.heartbeat_timeout_s == 2.0

    def test_default_num_players(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.num_players == 2

    def test_default_max_actions_per_second(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.max_actions_per_second == 1.0

    def test_default_multiplayer_mode_false(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.multiplayer_mode is False

    def test_default_player_id_mapping_none(self) -> None:
        cfg = BridgeConfig(comm_dir=Path("/tmp/az"))
        assert cfg.player_id_mapping is None

    def test_string_comm_dir_converted_to_path(self) -> None:
        cfg = BridgeConfig(comm_dir="/tmp/az_str")  # type: ignore[arg-type]
        assert isinstance(cfg.comm_dir, Path)


# ===========================================================================
# BridgeConfig.__post_init__ – validation errors
# ===========================================================================


class TestBridgeConfigValidation:
    """Each __post_init__ guard should raise ValueError with a meaningful message."""

    _base: dict[str, Any] = dict(comm_dir=Path("/tmp/az"))

    def _make(self, **overrides: Any) -> BridgeConfig:
        return BridgeConfig(**{**self._base, **overrides})

    def test_poll_interval_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_ms"):
            self._make(poll_interval_ms=0)

    def test_poll_interval_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="poll_interval_ms"):
            self._make(poll_interval_ms=-5)

    def test_command_timeout_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="command_timeout_ms"):
            self._make(command_timeout_ms=99)

    def test_heartbeat_timeout_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="heartbeat_timeout_s"):
            self._make(heartbeat_timeout_s=0.4)

    def test_max_retries_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_retries"):
            self._make(max_retries=0)

    def test_board_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="board_size"):
            self._make(board_size=7)

    def test_board_size_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="board_size"):
            self._make(board_size=129)

    def test_board_size_boundary_low_valid(self) -> None:
        cfg = self._make(board_size=8)
        assert cfg.board_size == 8

    def test_board_size_boundary_high_valid(self) -> None:
        cfg = self._make(board_size=128)
        assert cfg.board_size == 128

    def test_invalid_log_level_raises(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            self._make(log_level="VERBOSE")

    def test_valid_log_levels_accepted(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            cfg = self._make(log_level=level)
            assert cfg.log_level == level

    def test_num_players_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            self._make(num_players=1)

    def test_num_players_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            self._make(num_players=13)

    def test_num_players_boundary_valid(self) -> None:
        cfg = self._make(num_players=12)
        assert cfg.num_players == 12

    def test_max_actions_per_second_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_actions_per_second"):
            self._make(max_actions_per_second=-0.1)

    def test_max_actions_per_second_zero_allowed(self) -> None:
        cfg = self._make(max_actions_per_second=0.0)
        assert cfg.max_actions_per_second == 0.0


# ===========================================================================
# BridgeConfig.from_env
# ===========================================================================


class TestBridgeConfigFromEnv:
    """from_env should map every documented environment variable."""

    def test_defaults_when_no_env(self, tmp_path: Path) -> None:
        env: dict[str, str] = {}
        with (
            patch.dict("os.environ", env, clear=True),
            patch("alpha_zero.bridge.config.Path.home", return_value=tmp_path),
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.poll_interval_ms == 10
        assert cfg.debug_mode is False
        assert cfg.multiplayer_mode is False

    def test_comm_dir_from_env(self, tmp_path: Path) -> None:
        with patch.dict("os.environ", {"CIV6_BRIDGE_COMM_DIR": str(tmp_path)}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.comm_dir == tmp_path

    def test_poll_ms_from_env(self) -> None:
        with patch.dict("os.environ", {"CIV6_BRIDGE_POLL_MS": "25"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.poll_interval_ms == 25

    def test_command_timeout_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_COMMAND_TIMEOUT_MS": "8000"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.command_timeout_ms == 8000

    def test_heartbeat_timeout_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_HEARTBEAT_TIMEOUT_S": "5.0"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.heartbeat_timeout_s == 5.0

    def test_max_retries_from_env(self) -> None:
        with patch.dict("os.environ", {"CIV6_BRIDGE_MAX_RETRIES": "10"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.max_retries == 10

    def test_retry_delay_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_RETRY_DELAY_S": "2.5"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.retry_delay_s == 2.5

    def test_board_size_from_env(self) -> None:
        with patch.dict("os.environ", {"CIV6_BOARD_SIZE": "32"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.board_size == 32

    def test_log_level_from_env_uppercase(self) -> None:
        with patch.dict("os.environ", {"CIV6_LOG_LEVEL": "debug"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.log_level == "DEBUG"

    def test_debug_mode_true_from_env_1(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "1"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is True

    def test_debug_mode_true_from_env_true(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "true"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is True

    def test_debug_mode_true_from_env_yes(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "yes"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is True

    def test_debug_mode_false_from_env_0(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "0"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is False

    def test_debug_mode_false_from_env_false(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "false"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is False

    def test_debug_mode_false_from_env_no(self) -> None:
        with patch.dict("os.environ", {"CIV6_DEBUG_MODE": "no"}, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.debug_mode is False

    def test_num_players_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_NUM_PLAYERS": "4"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.num_players == 4

    def test_max_actions_per_second_from_env(self) -> None:
        with patch.dict(
            "os.environ",
            {"CIV6_BRIDGE_MAX_ACTIONS_PER_SECOND": "2.5"},
            clear=False,
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.max_actions_per_second == 2.5

    def test_multiplayer_mode_true_from_env(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_MULTIPLAYER_MODE": "yes"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.multiplayer_mode is True

    def test_multiplayer_mode_false_from_env_no(self) -> None:
        with patch.dict(
            "os.environ", {"CIV6_BRIDGE_MULTIPLAYER_MODE": "no"}, clear=False
        ):
            cfg = BridgeConfig.from_env()
        assert cfg.multiplayer_mode is False

    def test_all_vars_together(self, tmp_path: Path) -> None:
        env = {
            "CIV6_BRIDGE_COMM_DIR": str(tmp_path),
            "CIV6_BRIDGE_POLL_MS": "15",
            "CIV6_BRIDGE_COMMAND_TIMEOUT_MS": "3000",
            "CIV6_BRIDGE_HEARTBEAT_TIMEOUT_S": "3.0",
            "CIV6_BRIDGE_MAX_RETRIES": "3",
            "CIV6_BRIDGE_RETRY_DELAY_S": "0.5",
            "CIV6_BOARD_SIZE": "16",
            "CIV6_LOG_LEVEL": "WARNING",
            "CIV6_DEBUG_MODE": "1",
            "CIV6_BRIDGE_NUM_PLAYERS": "2",
            "CIV6_BRIDGE_MAX_ACTIONS_PER_SECOND": "0.0",
            "CIV6_BRIDGE_MULTIPLAYER_MODE": "1",
        }
        with patch.dict("os.environ", env, clear=False):
            cfg = BridgeConfig.from_env()
        assert cfg.comm_dir == tmp_path
        assert cfg.poll_interval_ms == 15
        assert cfg.command_timeout_ms == 3000
        assert cfg.heartbeat_timeout_s == 3.0
        assert cfg.max_retries == 3
        assert cfg.retry_delay_s == 0.5
        assert cfg.board_size == 16
        assert cfg.log_level == "WARNING"
        assert cfg.debug_mode is True
        assert cfg.num_players == 2
        assert cfg.max_actions_per_second == 0.0
        assert cfg.multiplayer_mode is True


# ===========================================================================
# BridgeConfig – file properties
# ===========================================================================


class TestBridgeConfigFileProperties:
    """All path properties derive from comm_dir."""

    @pytest.fixture()
    def cfg(self, tmp_path: Path) -> BridgeConfig:
        return BridgeConfig(comm_dir=tmp_path)

    def test_state_file(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        assert cfg.state_file == tmp_path / "state.json"

    def test_commands_file(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        assert cfg.commands_file == tmp_path / "commands.json"

    def test_ack_file(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        assert cfg.ack_file == tmp_path / "ack.json"

    def test_errors_file(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        assert cfg.errors_file == tmp_path / "errors.json"

    def test_heartbeat_file(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        assert cfg.heartbeat_file == tmp_path / "heartbeat.txt"


# ===========================================================================
# BridgeConfig.ensure_comm_dir
# ===========================================================================


class TestBridgeConfigEnsureCommDir:
    """ensure_comm_dir should create the directory (and parents) if absent."""

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "comm"
        cfg = BridgeConfig(comm_dir=target)
        assert not target.exists()
        cfg.ensure_comm_dir()
        assert target.is_dir()

    def test_idempotent_when_dir_exists(self, tmp_path: Path) -> None:
        cfg = BridgeConfig(comm_dir=tmp_path)
        cfg.ensure_comm_dir()  # second call should not raise
        cfg.ensure_comm_dir()
        assert tmp_path.is_dir()


# ===========================================================================
# BridgeConfig.to_dict
# ===========================================================================


class TestBridgeConfigToDict:
    """to_dict should serialise every field; comm_dir as a string."""

    @pytest.fixture()
    def cfg(self, tmp_path: Path) -> BridgeConfig:
        return BridgeConfig(
            comm_dir=tmp_path,
            poll_interval_ms=20,
            command_timeout_ms=2000,
            heartbeat_timeout_s=3.0,
            max_retries=3,
            retry_delay_s=0.5,
            board_size=32,
            log_level="DEBUG",
            debug_mode=True,
            num_players=4,
            max_actions_per_second=2.0,
            multiplayer_mode=True,
            player_id_mapping={0: 0, 1: 1},
        )

    def test_returns_dict(self, cfg: BridgeConfig) -> None:
        assert isinstance(cfg.to_dict(), dict)

    def test_comm_dir_is_str(self, cfg: BridgeConfig, tmp_path: Path) -> None:
        d = cfg.to_dict()
        assert d["comm_dir"] == str(tmp_path)
        assert isinstance(d["comm_dir"], str)

    def test_all_fields_present(self, cfg: BridgeConfig) -> None:
        expected_keys = {
            "comm_dir",
            "poll_interval_ms",
            "command_timeout_ms",
            "heartbeat_timeout_s",
            "max_retries",
            "retry_delay_s",
            "board_size",
            "log_level",
            "debug_mode",
            "num_players",
            "max_actions_per_second",
            "multiplayer_mode",
            "player_id_mapping",
        }
        assert set(cfg.to_dict().keys()) == expected_keys

    def test_values_match_attributes(self, cfg: BridgeConfig) -> None:
        d = cfg.to_dict()
        assert d["poll_interval_ms"] == 20
        assert d["log_level"] == "DEBUG"
        assert d["debug_mode"] is True
        assert d["player_id_mapping"] == {0: 0, 1: 1}

    def test_none_player_id_mapping_preserved(self, tmp_path: Path) -> None:
        cfg = BridgeConfig(comm_dir=tmp_path)
        assert cfg.to_dict()["player_id_mapping"] is None


# ===========================================================================
# MapSizeConfig
# ===========================================================================


class TestMapSizeConfig:
    """Unit tests for the MapSizeConfig helper dataclass."""

    def test_total_tiles(self) -> None:
        m = MapSizeConfig("Test", 10, 5)
        assert m.total_tiles == 50

    def test_total_tiles_standard(self) -> None:
        standard = MAP_SIZES["standard"]
        assert standard.total_tiles == 84 * 54

    def test_get_scale_factor_returns_tuple(self) -> None:
        m = MapSizeConfig("Test", 20, 10)
        result = m.get_scale_factor(board_size=40)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_get_scale_factor_values(self) -> None:
        m = MapSizeConfig("Test", 84, 54)
        x_scale, y_scale = m.get_scale_factor(board_size=16)
        assert x_scale == pytest.approx(16 / 84)
        assert y_scale == pytest.approx(16 / 54)

    def test_get_scale_factor_square_map(self) -> None:
        m = MapSizeConfig("Square", 32, 32)
        x_scale, y_scale = m.get_scale_factor(board_size=16)
        assert x_scale == pytest.approx(0.5)
        assert y_scale == pytest.approx(0.5)


# ===========================================================================
# MAP_SIZES constant
# ===========================================================================


class TestMapSizesConstant:
    """Sanity-check that the MAP_SIZES dict has the expected entries."""

    @pytest.mark.parametrize(
        "key,width,height",
        [
            ("duel", 44, 26),
            ("tiny", 60, 38),
            ("small", 74, 46),
            ("standard", 84, 54),
            ("large", 96, 60),
            ("huge", 106, 66),
        ],
    )
    def test_known_map_dimensions(self, key: str, width: int, height: int) -> None:
        m = MAP_SIZES[key]
        assert m.width == width
        assert m.height == height


# ===========================================================================
# get_map_config
# ===========================================================================


class TestGetMapConfig:
    """get_map_config should return the known config or fabricate a custom one."""

    def test_known_standard(self) -> None:
        result = get_map_config(84, 54)
        assert result.name == "Standard"
        assert result is MAP_SIZES["standard"]

    def test_known_duel(self) -> None:
        result = get_map_config(44, 26)
        assert result.name == "Duel"

    def test_known_huge(self) -> None:
        result = get_map_config(106, 66)
        assert result.name == "Huge"

    def test_custom_dimensions(self) -> None:
        result = get_map_config(50, 30)
        assert result.name == "Custom_50x30"
        assert result.width == 50
        assert result.height == 30

    def test_custom_is_map_size_config_instance(self) -> None:
        result = get_map_config(1, 1)
        assert isinstance(result, MapSizeConfig)

    def test_near_known_dimensions_creates_custom(self) -> None:
        # 84x53 differs from standard (84x54) — should be custom
        result = get_map_config(84, 53)
        assert "Custom" in result.name


# ===========================================================================
# extract_scalar_value – None / missing
# ===========================================================================


class TestExtractScalarValueNone:
    """When value is None the default is returned."""

    def test_none_returns_default(self) -> None:
        assert extract_scalar_value(None) == 0.0

    def test_none_returns_custom_default(self) -> None:
        assert extract_scalar_value(None, default=-1.0) == -1.0

    def test_none_default_type_is_float(self) -> None:
        # Pass a float default; the function returns it unchanged as a float.
        result = extract_scalar_value(None, default=5.0)
        assert isinstance(result, float)
        assert result == 5.0


# ===========================================================================
# extract_scalar_value – plain Python scalars
# ===========================================================================


class TestExtractScalarValuePlainScalars:
    """Plain int/float/bool/str inputs (no reshape attribute)."""

    def test_plain_int(self) -> None:
        assert extract_scalar_value(42) == pytest.approx(42.0)

    def test_plain_float(self) -> None:
        assert extract_scalar_value(3.14) == pytest.approx(3.14)

    def test_plain_zero(self) -> None:
        assert extract_scalar_value(0) == pytest.approx(0.0)

    def test_plain_negative(self) -> None:
        assert extract_scalar_value(-7.5) == pytest.approx(-7.5)

    def test_non_numeric_string_returns_default(self) -> None:
        assert extract_scalar_value("not-a-number", default=-99.0) == -99.0

    def test_list_without_reshape_returns_default(self) -> None:
        # plain lists have no .reshape — should fall through to float() which
        # raises TypeError, so default is returned.
        assert extract_scalar_value([1, 2, 3], default=-1.0) == -1.0


# ===========================================================================
# extract_scalar_value – numpy arrays
# ===========================================================================


class TestExtractScalarValueNumpy:
    """Tests using numpy arrays to cover the array branch."""

    def test_empty_array_returns_default(self) -> None:
        arr = np.array([], dtype=np.float32)
        assert extract_scalar_value(arr, default=-1.0) == -1.0

    def test_scalar_array_size_1(self) -> None:
        arr = np.array([0.75], dtype=np.float32)
        result = extract_scalar_value(arr)
        assert result == pytest.approx(0.75, abs=1e-5)

    def test_2d_array_1x1(self) -> None:
        arr = np.array([[0.5]], dtype=np.float64)
        result = extract_scalar_value(arr)
        assert result == pytest.approx(0.5)

    def test_multi_element_returns_first(self) -> None:
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = extract_scalar_value(arr)
        assert result == pytest.approx(0.1, abs=1e-5)

    def test_multi_element_log_per_player_true_logs_debug(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        arr = np.array([0.9, 0.1], dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            result = extract_scalar_value(arr, log_per_player=True)
        assert result == pytest.approx(0.9, abs=1e-5)
        assert any(
            "per-player value tensor" in record.message for record in caplog.records
        )

    def test_multi_element_log_per_player_false_no_debug_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        arr = np.array([0.9, 0.1], dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            extract_scalar_value(arr, log_per_player=False)
        assert not any(
            "per-player value tensor" in record.message for record in caplog.records
        )

    def test_1d_shape_in_log_message(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        arr = np.array([0.6, 0.4], dtype=np.float32)
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            extract_scalar_value(arr, log_per_player=True)
        joined = " ".join(record.message for record in caplog.records)
        assert "(2,)" in joined

    def test_2d_nplayer_array(self) -> None:
        """(1, N) shaped value head returns first element."""
        arr = np.array([[0.7, 0.3]], dtype=np.float32)
        result = extract_scalar_value(arr)
        assert result == pytest.approx(0.7, abs=1e-5)


# ===========================================================================
# extract_scalar_value – torch-like objects (simulated without torch dep)
# ===========================================================================


class TestExtractScalarValueTorchLike:
    """Simulate a torch tensor using numpy + MagicMock.item() to cover the
    torch-specific branch (flat[0].item() path)."""

    @staticmethod
    def _make_torch_like(data: list[float]) -> MagicMock:
        """Return a MagicMock that behaves enough like a 1-D torch tensor."""
        arr = np.array(data, dtype=np.float32)
        mock_tensor = MagicMock()
        # reshape(-1) returns a mock flat tensor
        flat_mock = MagicMock()
        mock_tensor.reshape.return_value = flat_mock
        # numel() path – torch uses numel()
        flat_mock.numel.return_value = len(data)
        # flat[0] -> element mock with .item()
        elem_mock = MagicMock()
        elem_mock.item.return_value = float(arr[0])
        flat_mock.__getitem__ = MagicMock(return_value=elem_mock)
        flat_mock.size = len(data)  # also set .size for fallback
        mock_tensor.shape = tuple(arr.shape)
        return mock_tensor

    def test_single_element_torch_like(self) -> None:
        t = self._make_torch_like([0.42])
        result = extract_scalar_value(t)
        assert result == pytest.approx(0.42, abs=1e-4)

    def test_multi_element_torch_like_returns_first(self) -> None:
        t = self._make_torch_like([0.8, 0.2])
        result = extract_scalar_value(t)
        assert result == pytest.approx(0.8, abs=1e-4)

    def test_multi_element_torch_like_log_per_player(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        t = self._make_torch_like([0.6, 0.4])
        with caplog.at_level(logging.DEBUG, logger="alpha_zero.bridge.utils"):
            result = extract_scalar_value(t, log_per_player=True)
        assert result == pytest.approx(0.6, abs=1e-4)
        assert any(
            "per-player value tensor" in r.message for r in caplog.records
        )

    def test_empty_torch_like_returns_default(self) -> None:
        """A mock tensor where numel() returns 0 should yield the default."""
        mock_tensor = MagicMock()
        flat_mock = MagicMock()
        mock_tensor.reshape.return_value = flat_mock
        flat_mock.numel.return_value = 0
        flat_mock.size = 0

        result = extract_scalar_value(mock_tensor, default=-99.0)
        assert result == -99.0
