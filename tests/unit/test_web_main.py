"""Unit tests for web CLI entry point module (alpha_zero.web.__main__)."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.web.__main__ import build_config, main, parse_args

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_namespace(**kwargs: Any) -> argparse.Namespace:
    """Build an argparse.Namespace with sensible defaults, overridden by *kwargs*."""
    defaults: dict[str, Any] = {
        "config": None,
        "preset": None,
        "host": None,
        "port": None,
        "checkpoint": None,
        "opponent": None,
        "board_size": None,
        "debug": None,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _make_mock_config(**overrides: Any) -> MagicMock:
    """Return a MagicMock that behaves like a WebConfig instance."""
    defaults: dict[str, Any] = {
        "host": "0.0.0.0",  # noqa: S104
        "port": 8080,
        "debug": False,
        "log_level": "info",
        "board_size": 16,
        "default_opponent": "random",
        "checkpoint_path": None,
    }
    defaults.update(overrides)
    mock = MagicMock()
    mock.host = defaults["host"]
    mock.port = defaults["port"]
    mock.debug = defaults["debug"]
    mock.log_level = defaults["log_level"]
    mock.board_size = defaults["board_size"]
    mock.default_opponent = defaults["default_opponent"]
    mock.checkpoint_path = defaults["checkpoint_path"]
    mock.to_dict.return_value = dict(defaults)
    return mock


# ===========================================================================
# parse_args
# ===========================================================================


class TestParseArgsDefaults:
    """Tests for parse_args with no arguments or minimal arguments."""

    def test_no_arguments_returns_all_none_defaults(self) -> None:
        """Calling parse_args([]) should produce None for every optional arg."""
        args = parse_args([])
        assert args.config is None
        assert args.preset is None
        assert args.host is None
        assert args.port is None
        assert args.checkpoint is None
        assert args.opponent is None
        assert args.board_size is None
        assert args.debug is None
        assert args.dry_run is False

    def test_none_argv_delegates_to_sys_argv(self) -> None:
        """parse_args(None) should fall through to argparse's default (sys.argv)."""
        with patch.object(sys, "argv", ["prog"]):
            args = parse_args(None)
        assert args.dry_run is False


class TestParseArgsPreset:
    """Tests for --preset flag."""

    @pytest.mark.parametrize("preset", ["dev", "production"])
    def test_valid_presets_accepted(self, preset: str) -> None:
        args = parse_args(["--preset", preset])
        assert args.preset == preset

    def test_invalid_preset_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--preset", "unknown"])


class TestParseArgsHostPort:
    """Tests for --host and --port flags."""

    def test_host_parsed(self) -> None:
        args = parse_args(["--host", "127.0.0.1"])
        assert args.host == "127.0.0.1"

    def test_port_parsed_as_int(self) -> None:
        args = parse_args(["--port", "3000"])
        assert args.port == 3000
        assert isinstance(args.port, int)

    def test_invalid_port_type_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--port", "not_an_int"])


class TestParseArgsAllOptions:
    """Tests for combining every supported flag."""

    def test_all_options_parsed(self) -> None:
        argv = [
            "--config",
            "my.yaml",
            "--host",
            "10.0.0.1",
            "--port",
            "9090",
            "--checkpoint",
            "/models/best.pt",
            "--opponent",
            "neural",
            "--board-size",
            "12",
            "--debug",
            "--dry-run",
        ]
        args = parse_args(argv)
        assert args.config == "my.yaml"
        assert args.host == "10.0.0.1"
        assert args.port == 9090
        assert args.checkpoint == "/models/best.pt"
        assert args.opponent == "neural"
        assert args.board_size == 12
        assert args.debug is True
        assert args.dry_run is True

    @pytest.mark.parametrize("opponent", ["random", "heuristic", "neural"])
    def test_valid_opponent_choices(self, opponent: str) -> None:
        args = parse_args(["--opponent", opponent])
        assert args.opponent == opponent

    def test_invalid_opponent_raises_system_exit(self) -> None:
        with pytest.raises(SystemExit):
            parse_args(["--opponent", "godlike"])


class TestParseArgsDryRun:
    """Tests for the --dry-run flag."""

    def test_dry_run_default_is_false(self) -> None:
        args = parse_args([])
        assert args.dry_run is False

    def test_dry_run_flag_sets_true(self) -> None:
        args = parse_args(["--dry-run"])
        assert args.dry_run is True


# ===========================================================================
# build_config
# ===========================================================================


class TestBuildConfigFromPreset:
    """Tests for build_config when --preset is specified."""

    @pytest.mark.parametrize("preset", ["dev", "production"])
    @patch("alpha_zero.web.__main__.build_config.__module__", create=True)
    def test_preset_calls_from_preset(self, _: Any, preset: str) -> None:
        mock_config = _make_mock_config()
        args = _make_namespace(preset=preset)
        with patch("alpha_zero.web.config.WebConfig.from_preset", return_value=mock_config) as m:
            result = build_config(args)
        m.assert_called_once_with(preset)
        assert result is mock_config


class TestBuildConfigFromYaml:
    """Tests for build_config when --config is specified."""

    def test_config_file_calls_from_yaml(self) -> None:
        mock_config = _make_mock_config()
        args = _make_namespace(config="web.yaml")
        with patch("alpha_zero.web.config.WebConfig.from_yaml", return_value=mock_config) as m:
            result = build_config(args)
        m.assert_called_once_with("web.yaml")
        assert result is mock_config


class TestBuildConfigFromEnv:
    """Tests for build_config when neither --preset nor --config is given."""

    def test_fallback_calls_from_env(self) -> None:
        mock_config = _make_mock_config()
        args = _make_namespace()
        with patch("alpha_zero.web.config.WebConfig.from_env", return_value=mock_config) as m:
            result = build_config(args)
        m.assert_called_once()
        assert result is mock_config


class TestBuildConfigPrecedence:
    """Tests that --preset takes precedence over --config."""

    def test_preset_wins_over_config(self) -> None:
        mock_config = _make_mock_config()
        args = _make_namespace(preset="dev", config="ignored.yaml")
        with (
            patch("alpha_zero.web.config.WebConfig.from_preset", return_value=mock_config) as mp,
            patch("alpha_zero.web.config.WebConfig.from_yaml") as my,
        ):
            build_config(args)
        mp.assert_called_once_with("dev")
        my.assert_not_called()


class TestBuildConfigCLIOverrides:
    """Tests that CLI flags override the base config."""

    @pytest.mark.parametrize(
        "flag,value,config_key",
        [
            ("host", "192.168.1.1", "host"),
            ("port", 4000, "port"),
            ("checkpoint", "/ckpt.pt", "checkpoint_path"),
            ("opponent", "heuristic", "default_opponent"),
            ("board_size", 24, "board_size"),
        ],
    )
    def test_single_override_applied(self, flag: str, value: object, config_key: str) -> None:
        base_dict: dict[str, Any] = {
            "host": "0.0.0.0",  # noqa: S104
            "port": 8080,
            "debug": False,
            "log_level": "info",
        }
        mock_config = MagicMock()
        mock_config.to_dict.return_value = dict(base_dict)

        args = _make_namespace(**{flag: value})

        with (
            patch("alpha_zero.web.config.WebConfig.from_env", return_value=mock_config),
            patch("alpha_zero.web.config.WebConfig") as web_config_cls,
        ):
            web_config_cls.from_env = MagicMock(return_value=mock_config)
            # The actual constructor call is what we want to inspect
            build_config(args)

        # The constructor should have been called with the override applied
        call_kwargs = web_config_cls.call_args
        assert call_kwargs is not None
        passed_dict = call_kwargs[1] if call_kwargs[1] else call_kwargs[0][0]
        assert passed_dict[config_key] == value

    def test_debug_flag_sets_debug_and_log_level(self) -> None:
        base_dict: dict[str, Any] = {
            "host": "0.0.0.0",  # noqa: S104
            "port": 8080,
            "debug": False,
            "log_level": "info",
        }
        mock_config = MagicMock()
        mock_config.to_dict.return_value = dict(base_dict)

        args = _make_namespace(debug=True)

        with (
            patch("alpha_zero.web.config.WebConfig.from_env", return_value=mock_config),
            patch("alpha_zero.web.config.WebConfig") as web_config_cls,
        ):
            web_config_cls.from_env = MagicMock(return_value=mock_config)
            build_config(args)

        call_kwargs = web_config_cls.call_args
        assert call_kwargs is not None
        passed_dict = call_kwargs[1] if call_kwargs[1] else call_kwargs[0][0]
        assert passed_dict["debug"] is True
        assert passed_dict["log_level"] == "debug"

    def test_no_overrides_returns_original_config(self) -> None:
        """When no CLI overrides are provided, the original config is returned as-is."""
        mock_config = _make_mock_config()
        args = _make_namespace()
        with patch("alpha_zero.web.config.WebConfig.from_env", return_value=mock_config):
            result = build_config(args)
        # No overrides means to_dict / WebConfig() is never called
        mock_config.to_dict.assert_not_called()
        assert result is mock_config

    def test_multiple_overrides_applied_together(self) -> None:
        base_dict: dict[str, Any] = {
            "host": "0.0.0.0",  # noqa: S104
            "port": 8080,
            "debug": False,
            "log_level": "info",
        }
        mock_config = MagicMock()
        mock_config.to_dict.return_value = dict(base_dict)

        args = _make_namespace(host="10.0.0.1", port=9090, board_size=8)

        with (
            patch("alpha_zero.web.config.WebConfig.from_env", return_value=mock_config),
            patch("alpha_zero.web.config.WebConfig") as web_config_cls,
        ):
            web_config_cls.from_env = MagicMock(return_value=mock_config)
            build_config(args)

        call_kwargs = web_config_cls.call_args
        assert call_kwargs is not None
        passed_dict = call_kwargs[1] if call_kwargs[1] else call_kwargs[0][0]
        assert passed_dict["host"] == "10.0.0.1"
        assert passed_dict["port"] == 9090
        assert passed_dict["board_size"] == 8


# ===========================================================================
# main
# ===========================================================================


class TestMainDryRun:
    """Tests for main() with --dry-run."""

    def test_dry_run_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_config = _make_mock_config()
        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=True)
            # patch isinstance check inside main
            with patch("alpha_zero.web.config.WebConfig", new=type(mock_config)):
                exit_code = main([])
        assert exit_code == 0

    def test_dry_run_prints_config_keys(self, capsys: pytest.CaptureFixture[str]) -> None:
        config_dict = {"host": "0.0.0.0", "port": 8080}  # noqa: S104
        mock_config = _make_mock_config()
        mock_config.to_dict.return_value = config_dict
        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=True)
            with patch("alpha_zero.web.config.WebConfig", new=type(mock_config)):
                main([])
        captured = capsys.readouterr()
        assert "Configuration valid:" in captured.out
        assert "host: 0.0.0.0" in captured.out
        assert "port: 8080" in captured.out
        assert "Dry run complete" in captured.out


class TestMainConfigError:
    """Tests for main() when configuration fails."""

    @pytest.mark.parametrize(
        "exc_class,message",
        [
            (FileNotFoundError, "Config file not found: bad.yaml"),
            (ValueError, "Invalid board_size"),
        ],
    )
    def test_config_error_returns_one(
        self,
        exc_class: type[Exception],
        message: str,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with patch("alpha_zero.web.__main__.build_config", side_effect=exc_class(message)):
            exit_code = main(["--preset", "dev"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Configuration error:" in captured.err
        assert message in captured.err


class TestMainImportError:
    """Tests for main() when uvicorn or create_app is missing."""

    def test_import_error_returns_one(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_config = _make_mock_config()
        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            # Simulate ImportError when importing uvicorn
            with patch.dict("sys.modules", {"uvicorn": None}):
                # We need to make the import actually fail inside main()
                # Use a side_effect on the create_app import path
                import builtins

                real_import = builtins.__import__

                def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                    if name == "uvicorn":
                        raise ImportError("No module named 'uvicorn'")
                    return real_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=fake_import):
                    exit_code = main([])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Missing dependency:" in captured.err
        assert "pip install" in captured.err


class TestMainKeyboardInterrupt:
    """Tests for main() when user interrupts with Ctrl-C."""

    def test_keyboard_interrupt_returns_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        mock_config = _make_mock_config()
        mock_app = MagicMock()
        mock_uvicorn = MagicMock()
        mock_uvicorn.run.side_effect = KeyboardInterrupt

        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "alpha_zero.web.app":
                    mod = MagicMock()
                    mod.create_app = MagicMock(return_value=mock_app)
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                exit_code = main([])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Shutdown requested" in captured.out


class TestMainGeneralException:
    """Tests for main() when an unexpected exception occurs."""

    def test_general_exception_returns_one(self) -> None:
        mock_config = _make_mock_config()
        mock_uvicorn = MagicMock()
        mock_uvicorn.run.side_effect = RuntimeError("something broke")

        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "alpha_zero.web.app":
                    mod = MagicMock()
                    mod.create_app = MagicMock(return_value=MagicMock())
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                exit_code = main([])
        assert exit_code == 1


class TestMainNormalRun:
    """Tests for main() in the normal (non-dry-run) path."""

    def test_normal_run_calls_uvicorn_and_returns_zero(self) -> None:
        mock_config = _make_mock_config(host="127.0.0.1", port=3000, log_level="info", debug=False)
        mock_app = MagicMock()
        mock_uvicorn = MagicMock()

        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "alpha_zero.web.app":
                    mod = MagicMock()
                    mod.create_app = MagicMock(return_value=mock_app)
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                exit_code = main([])

        assert exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            mock_app,
            host="127.0.0.1",
            port=3000,
            log_level="info",
            reload=False,
        )

    def test_normal_run_passes_debug_reload_flag(self) -> None:
        mock_config = _make_mock_config(
            host="0.0.0.0",
            port=8080,
            log_level="debug",
            debug=True,  # noqa: S104
        )
        mock_app = MagicMock()
        mock_uvicorn = MagicMock()

        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "alpha_zero.web.app":
                    mod = MagicMock()
                    mod.create_app = MagicMock(return_value=mock_app)
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                exit_code = main([])

        assert exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            mock_app,
            host="0.0.0.0",  # noqa: S104
            port=8080,
            log_level="debug",
            reload=True,
        )


class TestMainLoggingSetup:
    """Tests that main() configures logging from config.log_level."""

    @pytest.mark.parametrize("level_str", ["info", "debug", "warning"])
    def test_logging_configured_from_config(self, level_str: str) -> None:
        import logging as _logging

        mock_config = _make_mock_config(log_level=level_str)
        mock_uvicorn = MagicMock()

        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.__main__.parse_args") as mock_parse,
            patch("logging.basicConfig") as mock_basic,
        ):
            mock_parse.return_value = _make_namespace(dry_run=False)
            import builtins

            real_import = builtins.__import__

            def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "uvicorn":
                    return mock_uvicorn
                if name == "alpha_zero.web.app":
                    mod = MagicMock()
                    mod.create_app = MagicMock(return_value=MagicMock())
                    return mod
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import):
                main([])

        expected_level = getattr(_logging, level_str.upper(), _logging.INFO)
        mock_basic.assert_called_once()
        call_kwargs = mock_basic.call_args[1]
        assert call_kwargs["level"] == expected_level


class TestMainIntegrationWithRealParseArgs:
    """Integration-style tests that exercise parse_args + main together."""

    def test_dry_run_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Run main with --dry-run and a real parse, mocking only build_config."""
        mock_config = _make_mock_config()
        with (
            patch("alpha_zero.web.__main__.build_config", return_value=mock_config),
            patch("alpha_zero.web.config.WebConfig", new=type(mock_config)),
        ):
            exit_code = main(["--dry-run"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Dry run complete" in captured.out

    def test_config_error_end_to_end(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Run main with --config pointing to missing file."""
        with patch(
            "alpha_zero.web.config.WebConfig.from_yaml",
            side_effect=FileNotFoundError("not found"),
        ):
            exit_code = main(["--config", "missing.yaml"])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Configuration error:" in captured.err
