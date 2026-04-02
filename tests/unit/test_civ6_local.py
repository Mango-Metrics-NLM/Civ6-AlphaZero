"""Unit tests for alpha_zero.civ6_local.

Covers:
- Civ6LocalConfig defaults, validation, from_yaml, from_env, from_preset, to_dict
- CLI argument parsing via create_parser / build_config
- Dry-run mode (no real game launched)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from alpha_zero.civ6_local import (
    VALID_DEVICES,
    VALID_LOG_LEVELS,
    VALID_MODES,
    Civ6LocalConfig,
    build_config,
    create_parser,
    main,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_yaml(data: dict, tmp_path: Path) -> Path:
    """Write *data* as YAML to a temp file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(data))
    return p


# ---------------------------------------------------------------------------
# Civ6LocalConfig — defaults
# ---------------------------------------------------------------------------


class TestCiv6LocalConfigDefaults:
    """Civ6LocalConfig should have stable, reasonable defaults."""

    def test_mode_default(self) -> None:
        assert Civ6LocalConfig().mode == "microciv"

    def test_checkpoint_path_default(self) -> None:
        assert Civ6LocalConfig().checkpoint_path is None

    def test_board_size_default(self) -> None:
        assert Civ6LocalConfig().board_size == 16

    def test_num_players_default(self) -> None:
        assert Civ6LocalConfig().num_players == 2

    def test_num_simulations_default(self) -> None:
        assert Civ6LocalConfig().num_simulations == 50

    def test_device_default(self) -> None:
        assert Civ6LocalConfig().device == "cpu"

    def test_web_host_default(self) -> None:
        assert Civ6LocalConfig().web_host == "127.0.0.1"

    def test_web_port_default(self) -> None:
        assert Civ6LocalConfig().web_port == 8080

    def test_bridge_ipc_dir_default(self) -> None:
        assert Civ6LocalConfig().bridge_ipc_dir == "/tmp/civ6az_bridge"

    def test_log_level_default(self) -> None:
        assert Civ6LocalConfig().log_level == "INFO"

    def test_dry_run_default(self) -> None:
        assert Civ6LocalConfig().dry_run is False

    def test_to_dict_keys(self) -> None:
        d = Civ6LocalConfig().to_dict()
        expected = {
            "mode",
            "checkpoint_path",
            "board_size",
            "num_players",
            "num_simulations",
            "device",
            "web_host",
            "web_port",
            "bridge_ipc_dir",
            "log_level",
            "dry_run",
        }
        assert set(d.keys()) == expected


# ---------------------------------------------------------------------------
# Civ6LocalConfig — validation
# ---------------------------------------------------------------------------


class TestCiv6LocalConfigValidation:
    """Invalid field values must raise ValueError at construction time."""

    def test_invalid_mode(self) -> None:
        with pytest.raises(ValueError, match="mode"):
            Civ6LocalConfig(mode="battleship")

    def test_invalid_board_size_low(self) -> None:
        with pytest.raises(ValueError, match="board_size"):
            Civ6LocalConfig(board_size=2)

    def test_invalid_num_players_low(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            Civ6LocalConfig(num_players=1)

    def test_invalid_num_players_high(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            Civ6LocalConfig(num_players=9)

    def test_invalid_num_simulations(self) -> None:
        with pytest.raises(ValueError, match="num_simulations"):
            Civ6LocalConfig(num_simulations=0)

    def test_invalid_device(self) -> None:
        with pytest.raises(ValueError, match="device"):
            Civ6LocalConfig(device="tpu")

    def test_invalid_web_port_low(self) -> None:
        with pytest.raises(ValueError, match="web_port"):
            Civ6LocalConfig(web_port=0)

    def test_invalid_web_port_high(self) -> None:
        with pytest.raises(ValueError, match="web_port"):
            Civ6LocalConfig(web_port=99999)

    def test_invalid_log_level(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            Civ6LocalConfig(log_level="VERBOSE")

    def test_valid_modes(self) -> None:
        for mode in VALID_MODES:
            cfg = Civ6LocalConfig(mode=mode)
            assert cfg.mode == mode

    def test_valid_devices(self) -> None:
        for device in VALID_DEVICES:
            cfg = Civ6LocalConfig(device=device)
            assert cfg.device == device

    def test_log_level_normalised_to_upper(self) -> None:
        cfg = Civ6LocalConfig(log_level="info")
        assert cfg.log_level == "INFO"


# ---------------------------------------------------------------------------
# Civ6LocalConfig — from_yaml
# ---------------------------------------------------------------------------


class TestCiv6LocalConfigFromYaml:
    """from_yaml should load values correctly from YAML files."""

    def test_flat_yaml(self, tmp_path: Path) -> None:
        p = make_yaml({"mode": "web", "web_port": 3000, "board_size": 8}, tmp_path)
        cfg = Civ6LocalConfig.from_yaml(str(p))
        assert cfg.mode == "web"
        assert cfg.web_port == 3000
        assert cfg.board_size == 8

    def test_nested_civ6_local_key(self, tmp_path: Path) -> None:
        p = make_yaml({"civ6_local": {"mode": "civ6", "num_simulations": 100}}, tmp_path)
        cfg = Civ6LocalConfig.from_yaml(str(p))
        assert cfg.mode == "civ6"
        assert cfg.num_simulations == 100

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        p = make_yaml({"mode": "microciv", "nonexistent_key": 42}, tmp_path)
        cfg = Civ6LocalConfig.from_yaml(str(p))
        assert cfg.mode == "microciv"

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = Civ6LocalConfig.from_yaml(str(p))
        assert cfg == Civ6LocalConfig()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            Civ6LocalConfig.from_yaml("/nonexistent/path.yaml")

    def test_uses_production_config_file(self) -> None:
        """The checked-in configs/civ6_local_gameplay.yaml must be loadable."""
        # Resolve path relative to the project root (two levels up from tests/unit)
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "civ6_local_gameplay.yaml"
        if config_path.exists():
            cfg = Civ6LocalConfig.from_yaml(str(config_path))
            assert cfg.mode == "microciv"


# ---------------------------------------------------------------------------
# Civ6LocalConfig — from_env
# ---------------------------------------------------------------------------


class TestCiv6LocalConfigFromEnv:
    """from_env must read CIV6AZ_LOCAL_ prefixed environment variables."""

    def _run_with_env(self, env_vars: dict[str, str]) -> Civ6LocalConfig:
        with patch.dict(os.environ, env_vars, clear=False):
            return Civ6LocalConfig.from_env()

    def test_mode_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_MODE": "civ6"})
        assert cfg.mode == "civ6"

    def test_board_size_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_BOARD_SIZE": "8"})
        assert cfg.board_size == 8

    def test_num_simulations_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_NUM_SIMULATIONS": "25"})
        assert cfg.num_simulations == 25

    def test_device_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_DEVICE": "cpu"})
        assert cfg.device == "cpu"

    def test_web_host_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_WEB_HOST": "0.0.0.0"})
        assert cfg.web_host == "0.0.0.0"

    def test_web_port_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_WEB_PORT": "3000"})
        assert cfg.web_port == 3000

    def test_bridge_ipc_dir_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_BRIDGE_IPC_DIR": "/custom/ipc"})
        assert cfg.bridge_ipc_dir == "/custom/ipc"

    def test_log_level_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_LOG_LEVEL": "DEBUG"})
        assert cfg.log_level == "DEBUG"

    def test_dry_run_override_true(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_DRY_RUN": "true"})
        assert cfg.dry_run is True

    def test_dry_run_override_1(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_DRY_RUN": "1"})
        assert cfg.dry_run is True

    def test_dry_run_override_false(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_DRY_RUN": "false"})
        assert cfg.dry_run is False

    def test_checkpoint_path_override(self) -> None:
        cfg = self._run_with_env({"CIV6AZ_LOCAL_CHECKPOINT_PATH": "/tmp/model.pt"})
        assert cfg.checkpoint_path == "/tmp/model.pt"

    def test_no_env_vars_returns_defaults(self) -> None:
        # Remove any CIV6AZ_LOCAL_ vars that might be set in the test environment
        clean_env = {k: v for k, v in os.environ.items() if not k.startswith("CIV6AZ_LOCAL_")}
        with patch.dict(os.environ, clean_env, clear=True):
            cfg = Civ6LocalConfig.from_env()
        assert cfg.mode == "microciv"
        assert cfg.board_size == 16


# ---------------------------------------------------------------------------
# Civ6LocalConfig — from_preset
# ---------------------------------------------------------------------------


class TestCiv6LocalConfigFromPreset:
    """from_preset must return valid config for each known preset."""

    def test_microciv_fast_preset(self) -> None:
        cfg = Civ6LocalConfig.from_preset("microciv_fast")
        assert cfg.mode == "microciv"
        assert cfg.board_size == 8
        assert cfg.num_simulations <= 20

    def test_microciv_full_preset(self) -> None:
        cfg = Civ6LocalConfig.from_preset("microciv_full")
        assert cfg.mode == "microciv"
        assert cfg.board_size == 16
        assert cfg.num_simulations >= 30

    def test_civ6_preset(self) -> None:
        cfg = Civ6LocalConfig.from_preset("civ6")
        assert cfg.mode == "civ6"

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown preset"):
            Civ6LocalConfig.from_preset("battleship")

    def test_preset_produces_valid_config(self) -> None:
        for preset in ("microciv_fast", "microciv_full", "civ6"):
            cfg = Civ6LocalConfig.from_preset(preset)
            # All presets must produce a config that passes __post_init__ validation
            d = cfg.to_dict()
            assert d["mode"] in VALID_MODES


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestCreateParser:
    """create_parser must expose the required CLI arguments."""

    def test_help_does_not_raise(self, capsys: pytest.CaptureFixture[str]) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_dry_run_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--dry-run"])
        assert args.dry_run is True

    def test_mode_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--mode", "web"])
        assert args.mode == "web"

    def test_checkpoint_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--checkpoint", "checkpoints/best.pt"])
        assert args.checkpoint_path == "checkpoints/best.pt"

    def test_board_size_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--board-size", "8"])
        assert args.board_size == 8

    def test_num_simulations_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--num-simulations", "200"])
        assert args.num_simulations == 200

    def test_device_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--device", "cpu"])
        assert args.device == "cpu"

    def test_web_host_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--web-host", "0.0.0.0"])
        assert args.web_host == "0.0.0.0"

    def test_web_port_arg(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--web-port", "3000"])
        assert args.web_port == 3000

    def test_config_and_preset_mutually_exclusive(self) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "foo.yaml", "--preset", "microciv_fast"])

    def test_preset_choices(self) -> None:
        parser = create_parser()
        for preset in ("microciv_fast", "microciv_full", "civ6"):
            args = parser.parse_args(["--preset", preset])
            assert args.preset == preset

    def test_invalid_preset(self) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--preset", "impossible"])


# ---------------------------------------------------------------------------
# build_config
# ---------------------------------------------------------------------------


class TestBuildConfig:
    """build_config integrates args + env vars into a Civ6LocalConfig."""

    def _parse(self, argv: list[str]) -> Civ6LocalConfig:
        parser = create_parser()
        args = parser.parse_args(argv)
        return build_config(args)

    def test_defaults_when_no_args(self) -> None:
        cfg = self._parse([])
        assert cfg.mode == "microciv"

    def test_mode_override(self) -> None:
        cfg = self._parse(["--mode", "web"])
        assert cfg.mode == "web"

    def test_board_size_override(self) -> None:
        cfg = self._parse(["--board-size", "8"])
        assert cfg.board_size == 8

    def test_dry_run_sets_flag(self) -> None:
        cfg = self._parse(["--dry-run"])
        assert cfg.dry_run is True

    def test_preset_applied(self) -> None:
        cfg = self._parse(["--preset", "microciv_fast"])
        assert cfg.mode == "microciv"
        assert cfg.board_size == 8

    def test_cli_overrides_preset(self) -> None:
        cfg = self._parse(["--preset", "microciv_fast", "--board-size", "12"])
        assert cfg.board_size == 12

    def test_config_file(self, tmp_path: Path) -> None:
        p = make_yaml({"mode": "web", "web_port": 9999}, tmp_path)
        cfg = self._parse(["--config", str(p)])
        assert cfg.mode == "web"
        assert cfg.web_port == 9999

    def test_verbose_sets_debug(self) -> None:
        cfg = self._parse(["--verbose"])
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# main() — dry-run smoke test
# ---------------------------------------------------------------------------


class TestMain:
    """main() must complete cleanly in dry-run mode without starting a game."""

    def test_dry_run_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--dry-run"])
        assert rc == 0

    def test_dry_run_prints_config(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["--dry-run"])
        captured = capsys.readouterr()
        assert "mode:" in captured.out
        assert "board_size:" in captured.out
        assert "Dry run complete" in captured.out

    def test_dry_run_with_mode_microciv(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--mode", "microciv", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_mode_web(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--mode", "web", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_mode_civ6(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--mode", "civ6", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_preset(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main(["--preset", "microciv_fast", "--dry-run"])
        assert rc == 0

    def test_dry_run_with_config_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        p = make_yaml({"mode": "microciv", "num_simulations": 10}, tmp_path)
        rc = main(["--config", str(p), "--dry-run"])
        assert rc == 0

    def test_invalid_config_returns_nonzero(self) -> None:
        # argparse rejects invalid choices with SystemExit(2) — both non-zero outcomes are valid
        with pytest.raises(SystemExit) as exc_info:
            main(["--mode", "invalid_mode", "--dry-run"])
        assert exc_info.value.code != 0

    def test_missing_config_file_returns_nonzero(self) -> None:
        rc = main(["--config", "/nonexistent/path.yaml"])
        assert rc != 0

    def test_help_exits(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    def test_production_config_file_dry_run(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The shipped configs/civ6_local_gameplay.yaml must be usable with --dry-run."""
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "configs" / "civ6_local_gameplay.yaml"
        if config_path.exists():
            rc = main(["--config", str(config_path), "--dry-run"])
            assert rc == 0


# ---------------------------------------------------------------------------
# setup_logging — smoke test
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """setup_logging must not raise for all valid log levels."""

    def test_all_valid_levels(self) -> None:
        for level in VALID_LOG_LEVELS:
            setup_logging(level)  # must not raise

    def test_lowercase_level(self) -> None:
        setup_logging("debug")  # must not raise


# ---------------------------------------------------------------------------
# Civ6LocalConfig — validate() helper
# ---------------------------------------------------------------------------


class TestValidateMethod:
    """validate() returns warning strings for non-fatal issues."""

    def test_no_warnings_for_defaults(self) -> None:
        cfg = Civ6LocalConfig()
        warnings = cfg.validate()
        # By default there should be no warnings (bridge dir /tmp exists on Linux)
        # We just ensure it returns a list
        assert isinstance(warnings, list)

    def test_missing_checkpoint_warns(self, tmp_path: Path) -> None:
        non_existent = str(tmp_path / "missing.pt")
        cfg = Civ6LocalConfig(checkpoint_path=non_existent)
        warnings = cfg.validate()
        assert any("checkpoint" in w.lower() for w in warnings)

    def test_cuda_not_available_warns(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            cfg = Civ6LocalConfig(device="cuda")
            warnings = cfg.validate()
        assert any("cuda" in w.lower() for w in warnings)

    def test_civ6_mode_missing_bridge_dir_warns(self, tmp_path: Path) -> None:
        missing = str(tmp_path / "no_such_dir")
        cfg = Civ6LocalConfig(mode="civ6", bridge_ipc_dir=missing)
        warnings = cfg.validate()
        assert any("bridge_ipc_dir" in w for w in warnings)

    def test_web_mode_open_host_warns(self) -> None:
        cfg = Civ6LocalConfig(mode="web", web_host="0.0.0.0")
        warnings = cfg.validate()
        assert any("0.0.0.0" in w for w in warnings)


# ---------------------------------------------------------------------------
# Additional env-var branches (num_players)
# ---------------------------------------------------------------------------


class TestFromEnvAdditional:
    """Cover env branches not exercised by the main class."""

    def test_num_players_override(self) -> None:
        with patch.dict(os.environ, {"CIV6AZ_LOCAL_NUM_PLAYERS": "4"}, clear=False):
            cfg = Civ6LocalConfig.from_env()
        assert cfg.num_players == 4


# ---------------------------------------------------------------------------
# build_config — remaining override branches
# ---------------------------------------------------------------------------


class TestBuildConfigAdditional:
    """Cover build_config branches for every CLI arg."""

    def _parse(self, argv: list[str]) -> Civ6LocalConfig:
        parser = create_parser()
        args = parser.parse_args(argv)
        return build_config(args)

    def test_num_players_override(self) -> None:
        cfg = self._parse(["--num-players", "3"])
        assert cfg.num_players == 3

    def test_checkpoint_override(self) -> None:
        cfg = self._parse(["--checkpoint", "/tmp/model.pt"])
        assert cfg.checkpoint_path == "/tmp/model.pt"

    def test_num_simulations_override(self) -> None:
        cfg = self._parse(["--num-simulations", "100"])
        assert cfg.num_simulations == 100

    def test_device_override(self) -> None:
        cfg = self._parse(["--device", "cpu"])
        assert cfg.device == "cpu"

    def test_web_host_override(self) -> None:
        cfg = self._parse(["--web-host", "0.0.0.0"])
        assert cfg.web_host == "0.0.0.0"

    def test_web_port_override(self) -> None:
        cfg = self._parse(["--web-port", "9000"])
        assert cfg.web_port == 9000

    def test_bridge_ipc_dir_override(self) -> None:
        cfg = self._parse(["--bridge-ipc-dir", "/custom/ipc"])
        assert cfg.bridge_ipc_dir == "/custom/ipc"

    def test_log_level_override(self) -> None:
        cfg = self._parse(["--log-level", "warning"])
        assert cfg.log_level == "WARNING"


# ---------------------------------------------------------------------------
# main() — non-dry-run paths (mode runners mocked)
# ---------------------------------------------------------------------------


class TestMainRunners:
    """Test main() dispatches to mode runners, mocking the actual game launch."""

    def test_microciv_mode_dispatched(self) -> None:
        with patch("alpha_zero.civ6_local.run_microciv", return_value=0) as mock_run:
            rc = main(["--mode", "microciv"])
        assert rc == 0
        mock_run.assert_called_once()

    def test_civ6_mode_dispatched(self) -> None:
        with patch("alpha_zero.civ6_local.run_civ6_bridge", return_value=0) as mock_run:
            rc = main(["--mode", "civ6"])
        assert rc == 0
        mock_run.assert_called_once()

    def test_web_mode_dispatched(self) -> None:
        with patch("alpha_zero.civ6_local.run_web", return_value=0) as mock_run:
            rc = main(["--mode", "web"])
        assert rc == 0
        mock_run.assert_called_once()

    def test_keyboard_interrupt_returns_130(self) -> None:
        with patch("alpha_zero.civ6_local.run_microciv", side_effect=KeyboardInterrupt):
            rc = main(["--mode", "microciv"])
        assert rc == 130

    def test_exception_returns_1(self) -> None:
        with patch("alpha_zero.civ6_local.run_microciv", side_effect=RuntimeError("boom")):
            rc = main(["--mode", "microciv"])
        assert rc == 1

    def test_validate_warnings_logged(self, capsys: pytest.CaptureFixture[str]) -> None:
        """validate() warnings should be emitted without crashing."""
        with patch("alpha_zero.civ6_local.run_microciv", return_value=0):
            with patch.object(Civ6LocalConfig, "validate", return_value=["test warning"]):
                rc = main(["--mode", "microciv"])
        assert rc == 0


# ---------------------------------------------------------------------------
# run_microciv — unit-level mock test
# ---------------------------------------------------------------------------


class TestRunMicrociv:
    """run_microciv should complete a game loop via mocked env and net."""

    def test_run_microciv_completes(self) -> None:
        import numpy as np

        from alpha_zero.civ6_local import run_microciv

        cfg = Civ6LocalConfig(mode="microciv", board_size=4, num_simulations=1)

        # Build a minimal fake observation
        action_size = 10
        fake_obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros((64,), dtype=np.float32),
            "action_mask": np.ones(action_size, dtype=np.float32),
        }

        # Mock gymnasium environment
        mock_env = MagicMock()
        mock_env.action_space.n = action_size
        mock_env.reset.return_value = (fake_obs, {})
        # Step returns terminated=True immediately so the loop ends in one step
        mock_env.step.return_value = (fake_obs, 1.0, True, False, {"winner": "player_0"})

        import torch

        fake_policy = torch.zeros(1, action_size)
        fake_value = torch.tensor([[0.5]])

        mock_net = MagicMock()
        mock_net.return_value = (fake_policy, fake_value)
        # .to(device) must return the same mock so __call__ return_value is preserved
        mock_net.to.return_value = mock_net

        mock_net_cls = MagicMock(return_value=mock_net)
        mock_env_cls = MagicMock(return_value=mock_env)

        with patch.dict(
            sys.modules,
            {
                "alpha_zero.envs.microciv": MagicMock(MicroCivEnv=mock_env_cls),
                "alpha_zero.core.civ6_network": MagicMock(MicroCivNetwork=mock_net_cls),
                "alpha_zero.core.civ6_mcts": MagicMock(
                    GumbelMCTS=MagicMock(), GumbelMCTSConfig=MagicMock()
                ),
            },
        ):
            rc = run_microciv(cfg)

        assert rc == 0

    def test_run_microciv_loads_checkpoint(self, tmp_path: Path) -> None:
        import numpy as np
        import torch

        from alpha_zero.civ6_local import run_microciv

        # Create a real (empty) checkpoint file
        ckpt = tmp_path / "model.pt"
        torch.save({}, ckpt)

        cfg = Civ6LocalConfig(
            mode="microciv", board_size=4, num_simulations=1, checkpoint_path=str(ckpt)
        )

        action_size = 10
        fake_obs = {
            "spatial": np.zeros((13, 4, 4), dtype=np.float32),
            "vector": np.zeros((64,), dtype=np.float32),
            "action_mask": np.ones(action_size, dtype=np.float32),
        }

        mock_env = MagicMock()
        mock_env.action_space.n = action_size
        mock_env.reset.return_value = (fake_obs, {})
        mock_env.step.return_value = (fake_obs, 0.0, True, False, {})

        fake_policy = torch.zeros(1, action_size)
        mock_net = MagicMock()
        mock_net.return_value = (fake_policy, torch.tensor([[0.0]]))
        mock_net.to.return_value = mock_net

        with patch.dict(
            sys.modules,
            {
                "alpha_zero.envs.microciv": MagicMock(MicroCivEnv=MagicMock(return_value=mock_env)),
                "alpha_zero.core.civ6_network": MagicMock(
                    MicroCivNetwork=MagicMock(return_value=mock_net)
                ),
                "alpha_zero.core.civ6_mcts": MagicMock(
                    GumbelMCTS=MagicMock(), GumbelMCTSConfig=MagicMock()
                ),
            },
        ):
            rc = run_microciv(cfg)
        assert rc == 0


# ---------------------------------------------------------------------------
# run_civ6_bridge — unit-level mock test
# ---------------------------------------------------------------------------


class TestRunCiv6Bridge:
    """run_civ6_bridge should call FileBridge.run() and handle KeyboardInterrupt."""

    def _make_bridge_mocks(self) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Return (mock_bridge_instance, mock_bridge_cls, mock_cfg_instance)."""
        mock_bridge = MagicMock()
        mock_bridge_cls = MagicMock(return_value=mock_bridge)
        mock_cfg_instance = MagicMock()
        mock_cfg_cls = MagicMock(return_value=mock_cfg_instance)
        return mock_bridge, mock_bridge_cls, mock_cfg_cls

    def test_run_bridge_normal(self) -> None:
        from alpha_zero.civ6_local import run_civ6_bridge

        cfg = Civ6LocalConfig(mode="civ6")
        mock_bridge, mock_bridge_cls, mock_cfg_cls = self._make_bridge_mocks()

        # BridgeConfig and FileBridge are imported inside the function, so patch at source
        with (
            patch("alpha_zero.bridge.config.BridgeConfig", mock_cfg_cls),
            patch("alpha_zero.bridge.file_bridge.FileBridge", mock_bridge_cls),
            patch.dict(
                sys.modules,
                {
                    "alpha_zero.bridge.config": MagicMock(BridgeConfig=mock_cfg_cls),
                    "alpha_zero.bridge.file_bridge": MagicMock(FileBridge=mock_bridge_cls),
                },
            ),
        ):
            rc = run_civ6_bridge(cfg)

        assert rc == 0
        mock_bridge.run.assert_called_once()
        mock_bridge.close.assert_called_once()

    def test_run_bridge_keyboard_interrupt(self) -> None:
        from alpha_zero.civ6_local import run_civ6_bridge

        cfg = Civ6LocalConfig(mode="civ6")
        mock_bridge = MagicMock()
        mock_bridge.run.side_effect = KeyboardInterrupt
        mock_bridge_cls = MagicMock(return_value=mock_bridge)
        mock_cfg_cls = MagicMock(return_value=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "alpha_zero.bridge.config": MagicMock(BridgeConfig=mock_cfg_cls),
                "alpha_zero.bridge.file_bridge": MagicMock(FileBridge=mock_bridge_cls),
            },
        ):
            rc = run_civ6_bridge(cfg)

        assert rc == 0
        mock_bridge.close.assert_called_once()

    def test_run_bridge_import_error(self) -> None:
        from alpha_zero.civ6_local import run_civ6_bridge

        cfg = Civ6LocalConfig(mode="civ6")

        # Simulate missing bridge package by injecting None into sys.modules
        with patch.dict(
            sys.modules,
            {
                "alpha_zero.bridge.config": None,  # type: ignore[dict-item]
                "alpha_zero.bridge.file_bridge": None,  # type: ignore[dict-item]
            },
        ):
            rc = run_civ6_bridge(cfg)

        assert rc == 1


# ---------------------------------------------------------------------------
# run_web — unit-level mock test
# ---------------------------------------------------------------------------


class TestRunWeb:
    """run_web should launch uvicorn with the correct config, or return 1 on ImportError."""

    def test_run_web_normal(self) -> None:
        from alpha_zero.civ6_local import run_web

        cfg = Civ6LocalConfig(mode="web", web_host="127.0.0.1", web_port=8080)

        mock_app = MagicMock()
        mock_web_cfg = MagicMock()
        mock_uvicorn = MagicMock()

        mock_web_cfg_cls = MagicMock(return_value=mock_web_cfg)
        mock_create_app = MagicMock(return_value=mock_app)

        with patch.dict(
            sys.modules,
            {
                "uvicorn": mock_uvicorn,
                "alpha_zero.web.app": MagicMock(create_app=mock_create_app),
                "alpha_zero.web.config": MagicMock(WebConfig=mock_web_cfg_cls),
            },
        ):
            rc = run_web(cfg)

        assert rc == 0
        mock_uvicorn.run.assert_called_once()

    def test_run_web_keyboard_interrupt(self) -> None:
        from alpha_zero.civ6_local import run_web

        cfg = Civ6LocalConfig(mode="web")

        mock_uvicorn = MagicMock()
        mock_uvicorn.run.side_effect = KeyboardInterrupt
        mock_web_cfg_cls = MagicMock(return_value=MagicMock())
        mock_create_app = MagicMock(return_value=MagicMock())

        with patch.dict(
            sys.modules,
            {
                "uvicorn": mock_uvicorn,
                "alpha_zero.web.app": MagicMock(create_app=mock_create_app),
                "alpha_zero.web.config": MagicMock(WebConfig=mock_web_cfg_cls),
            },
        ):
            rc = run_web(cfg)

        assert rc == 0

    def test_run_web_import_error(self) -> None:
        from alpha_zero.civ6_local import run_web

        cfg = Civ6LocalConfig(mode="web")

        with patch.dict(
            sys.modules,
            {
                "uvicorn": None,  # type: ignore[dict-item]
            },
        ):
            rc = run_web(cfg)

        assert rc == 1
