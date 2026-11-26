"""Tests for alpha_zero.local.gameplay_config, logging setup, and BridgeConfig.from_live_env."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

from alpha_zero import civ6_local
from alpha_zero.bridge.config import (
    DEFAULT_COMMAND_TRANSPORT,
    DEFAULT_TUNER_HOST,
    DEFAULT_TUNER_PORT,
    VALID_COMMAND_TRANSPORTS,
    BridgeConfig,
    get_default_civ6_log_path,
    get_default_comm_dir,
)
from alpha_zero.local import gameplay_config as gc

REPO_ROOT = Path(__file__).resolve().parents[2]


class TestGameplayConfigReexports:
    """Decomposing gameplay_config must preserve exact object identity on civ6_local."""

    def test_civ6_local_reexports_match_gameplay_config(self) -> None:
        assert civ6_local.Civ6LocalConfig is gc.Civ6LocalConfig
        assert civ6_local.search_mode_refusal is gc.search_mode_refusal
        assert civ6_local._get_default_bridge_ipc_dir is gc._get_default_bridge_ipc_dir
        assert civ6_local._get_default_civ6_log_path is gc._get_default_civ6_log_path
        assert civ6_local.VALID_MODES is gc.VALID_MODES
        assert civ6_local.VALID_LOG_LEVELS is gc.VALID_LOG_LEVELS
        assert civ6_local.VALID_DEVICES is gc.VALID_DEVICES
        assert civ6_local.ENV_PREFIX is gc.ENV_PREFIX
        assert civ6_local.SEARCH_MODE_GAP_ID == gc.SEARCH_MODE_GAP_ID == "GM-B5"
        assert civ6_local.SEARCH_TRANSLATOR_GAP_ID == gc.SEARCH_TRANSLATOR_GAP_ID == "GM-B6"
        assert civ6_local.DEFAULT_DEMO_MAX_UNITS == gc.DEFAULT_DEMO_MAX_UNITS
        assert civ6_local.DEFAULT_DEMO_MAX_CITIES == gc.DEFAULT_DEMO_MAX_CITIES
        assert civ6_local.DEFAULT_DEMO_NUM_PRODUCTION_ITEMS == gc.DEFAULT_DEMO_NUM_PRODUCTION_ITEMS

    def test_gameplay_config_import_avoids_torch_and_numpy(self) -> None:
        code = (
            "import sys\n"
            "import alpha_zero.local.gameplay_config as gc\n"
            "from alpha_zero.bridge.config import BridgeConfig\n"
            "bcfg = BridgeConfig.from_live_env()\n"
            "assert bcfg is not None\n"
            "assert str(gc._get_default_civ6_log_path()).endswith('Lua.log')\n"
            "assert str(gc._get_default_bridge_ipc_dir())\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] in ('torch', 'numpy'))\n"
            "print(heavy)\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip() == "[]"


class TestSetupLoggingLevelUpdate:
    """setup_logging() must apply config.log_level even after early initialization."""

    def test_subsequent_setup_logging_updates_root_level(self) -> None:
        root = logging.getLogger()
        previous = root.level
        try:
            civ6_local.setup_logging("INFO")
            assert root.level == logging.INFO
            civ6_local.setup_logging("DEBUG")
            assert root.level == logging.DEBUG
            civ6_local.setup_logging("WARNING")
            assert root.level == logging.WARNING
        finally:
            root.setLevel(previous)


class TestBridgeConfigLiveEnv:
    """Single-source transport configuration and path helpers."""

    def test_defaults_and_env_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in (
            "CIV6_BRIDGE_TRANSPORT",
            "CIV6_BRIDGE_TUNER_HOST",
            "CIV6_BRIDGE_TUNER_PORT",
            "CIV6_BRIDGE_COMM_DIR",
        ):
            monkeypatch.delenv(var, raising=False)

        default_cfg = BridgeConfig.from_live_env(comm_dir=tmp_path)
        assert default_cfg.command_transport == DEFAULT_COMMAND_TRANSPORT == "file"
        assert VALID_COMMAND_TRANSPORTS == ("file", "socket")
        assert default_cfg.tuner_host == DEFAULT_TUNER_HOST == "127.0.0.1"
        assert default_cfg.tuner_port == DEFAULT_TUNER_PORT == 4318
        assert default_cfg.comm_dir == tmp_path

        monkeypatch.setenv("CIV6_BRIDGE_TRANSPORT", "SOCKET")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_HOST", "127.0.0.2")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_PORT", "4399")
        env_dir = tmp_path / "env_comm"
        monkeypatch.setenv("CIV6_BRIDGE_COMM_DIR", str(env_dir))

        live_cfg = BridgeConfig.from_live_env(log_level="debug")
        assert live_cfg.command_transport == "socket"
        assert live_cfg.tuner_host == "127.0.0.2"
        assert live_cfg.tuner_port == 4399
        assert live_cfg.comm_dir == env_dir
        assert live_cfg.log_level == "DEBUG"

    def test_invalid_transport_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIV6_BRIDGE_TRANSPORT", "carrier_pigeon")
        with pytest.raises(ValueError, match="command_transport"):
            BridgeConfig.from_live_env(comm_dir=tmp_path)

    def test_public_path_helpers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        log_path = get_default_civ6_log_path()
        assert log_path.name == "Lua.log"
        assert log_path.parts[-4:] == (
            "Firaxis Games",
            "Sid Meier's Civilization VI",
            "Logs",
            "Lua.log",
        )
        comm_dir = get_default_comm_dir()
        assert comm_dir.name in ("AlphaZeroAI", ".civ6_alphazero")

    def test_resolve_live_paths_precedence_and_ignores_bridge_comm_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cli_ipc = tmp_path / "cli_ipc"
        env_ipc = tmp_path / "env_ipc"
        cli_log = tmp_path / "cli_Lua.log"
        env_log = tmp_path / "env_Lua.log"
        monkeypatch.setenv("CIV6_BRIDGE_COMM_DIR", str(tmp_path / "ignored_comm_dir"))
        monkeypatch.delenv("CIV6AZ_LOCAL_BRIDGE_IPC_DIR", raising=False)
        monkeypatch.delenv("CIV6AZ_LOCAL_LOG_PATH", raising=False)

        assert gc.resolve_live_bridge_ipc_dir() != tmp_path / "ignored_comm_dir"
        monkeypatch.setenv("CIV6AZ_LOCAL_BRIDGE_IPC_DIR", str(env_ipc))
        monkeypatch.setenv("CIV6AZ_LOCAL_LOG_PATH", str(env_log))
        assert gc.resolve_live_bridge_ipc_dir() == env_ipc
        assert gc.resolve_live_civ6_log_path() == env_log
        assert gc.resolve_live_bridge_ipc_dir(cli_ipc) == cli_ipc
        assert gc.resolve_live_civ6_log_path(cli_log) == cli_log

    def test_from_env_preserves_raw_posix_path_and_blank_live_env_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CIV6AZ_LOCAL_BRIDGE_IPC_DIR", "/custom/ipc")
        monkeypatch.setenv("CIV6AZ_LOCAL_LOG_PATH", "/custom/Lua.log")
        cfg = gc.Civ6LocalConfig.from_env()
        assert cfg.bridge_ipc_dir == "/custom/ipc"
        assert cfg.civ6_log_path == "/custom/Lua.log"

        monkeypatch.setenv("CIV6_BRIDGE_TRANSPORT", "   ")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_HOST", "")
        monkeypatch.setenv("CIV6_BRIDGE_TUNER_PORT", " ")
        live_cfg = BridgeConfig.from_live_env(comm_dir=tmp_path)
        assert live_cfg.command_transport == DEFAULT_COMMAND_TRANSPORT
        assert live_cfg.tuner_host == DEFAULT_TUNER_HOST
        assert live_cfg.tuner_port == DEFAULT_TUNER_PORT

        with pytest.raises(ValueError, match="tuner_host"):
            BridgeConfig(comm_dir=tmp_path, tuner_host="   ")
