"""Unit tests for alpha_zero.orchestrator.__main__ CLI module.

Tests argument parsing, logging setup, config loading, and main() entry point,
using mocks to avoid real training or filesystem I/O.

Notes on patching:
    OrchestratorConfig is imported inside load_config() (deferred import), so we
    must patch it at its source path: "alpha_zero.orchestrator.config.OrchestratorConfig".
    TrainingOrchestrator is imported inside main(), patched similarly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.orchestrator.__main__ import (
    create_parser,
    load_config,
    main,
    setup_logging,
)

# Patch targets for deferred imports inside __main__ functions
_CONFIG_PATH = "alpha_zero.orchestrator.config.OrchestratorConfig"
_ORCHESTRATOR_PATH = "alpha_zero.orchestrator.orchestrator.TrainingOrchestrator"


class TestCreateParser:
    """Tests for the create_parser function."""

    def test_returns_argument_parser(self) -> None:
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_prog_name(self) -> None:
        parser = create_parser()
        assert parser.prog == "alpha_zero.orchestrator"

    def test_dry_run_default_false(self) -> None:
        parser = create_parser()
        args = parser.parse_args([])
        assert args.dry_run is False

    def test_dry_run_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--dry_run"])
        assert args.dry_run is True

    def test_distributed_default_false(self) -> None:
        parser = create_parser()
        args = parser.parse_args([])
        assert args.distributed is False

    def test_distributed_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--distributed"])
        assert args.distributed is True

    def test_config_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--config", "configs/my.yaml"])
        assert args.config == "configs/my.yaml"

    def test_preset_fast(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--preset", "fast"])
        assert args.preset == "fast"

    def test_preset_balanced(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--preset", "balanced"])
        assert args.preset == "balanced"

    def test_preset_production(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--preset", "production"])
        assert args.preset == "production"

    def test_invalid_preset_raises(self) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--preset", "invalid"])

    def test_config_and_preset_mutually_exclusive(self) -> None:
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--config", "a.yaml", "--preset", "fast"])

    def test_iterations_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--iterations", "5000"])
        assert args.iterations == 5000

    def test_games_per_iteration_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--games-per-iteration", "25"])
        assert args.games_per_iteration == 25

    def test_experiment_name_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--experiment-name", "my_exp"])
        assert args.experiment_name == "my_exp"

    def test_checkpoint_dir_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--checkpoint-dir", "/tmp/ckpts"])
        assert args.checkpoint_dir == "/tmp/ckpts"

    def test_seed_argument(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--seed", "42"])
        assert args.seed == 42

    def test_no_wandb_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--no-wandb"])
        assert args.no_wandb is True

    def test_no_rag_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--no-rag"])
        assert args.no_rag is True

    def test_no_curriculum_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--no-curriculum"])
        assert args.no_curriculum is True

    def test_verbose_default_zero(self) -> None:
        parser = create_parser()
        args = parser.parse_args([])
        assert args.verbose == 0

    def test_verbose_once(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["-v"])
        assert args.verbose == 1

    def test_verbose_twice(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["-v", "-v"])
        assert args.verbose == 2

    def test_verbose_long_form(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["--verbose"])
        assert args.verbose == 1

    def test_quiet_flag(self) -> None:
        parser = create_parser()
        args = parser.parse_args(["-q"])
        assert args.quiet is True

    def test_quiet_default_false(self) -> None:
        parser = create_parser()
        args = parser.parse_args([])
        assert args.quiet is False

    def test_all_flags_combined(self) -> None:
        parser = create_parser()
        args = parser.parse_args(
            [
                "--preset",
                "fast",
                "--dry_run",
                "--iterations",
                "100",
                "--games-per-iteration",
                "10",
                "--seed",
                "7",
                "--no-wandb",
                "--no-rag",
                "--no-curriculum",
                "-v",
            ]
        )
        assert args.preset == "fast"
        assert args.dry_run is True
        assert args.iterations == 100
        assert args.games_per_iteration == 10
        assert args.seed == 7
        assert args.no_wandb is True
        assert args.no_rag is True
        assert args.no_curriculum is True
        assert args.verbose == 1

    def test_defaults_for_optional_overrides(self) -> None:
        parser = create_parser()
        args = parser.parse_args([])
        assert args.iterations is None
        assert args.games_per_iteration is None
        assert args.experiment_name is None
        assert args.checkpoint_dir is None
        assert args.seed is None
        assert args.config is None
        assert args.preset is None


class TestSetupLogging:
    """Tests for setup_logging — verifies level selection logic.

    setup_logging calls logging.basicConfig which is a no-op if root handlers
    already exist (pytest captures logs and installs handlers early).  We
    therefore patch basicConfig and inspect the ``level`` kwarg it would
    receive, which is the only observable output of the level-selection logic.
    """

    def _call_and_capture_level(self, verbose: int, quiet: bool) -> int:
        """Call setup_logging and return the level passed to basicConfig."""
        with patch("logging.basicConfig") as mock_basic:
            setup_logging(verbose=verbose, quiet=quiet)
            assert mock_basic.called, "basicConfig was not called"
            return mock_basic.call_args.kwargs["level"]

    def test_quiet_sets_error_level(self) -> None:
        level = self._call_and_capture_level(verbose=0, quiet=True)
        assert level == logging.ERROR

    def test_verbose_2_sets_debug(self) -> None:
        level = self._call_and_capture_level(verbose=2, quiet=False)
        assert level == logging.DEBUG

    def test_verbose_1_sets_info(self) -> None:
        level = self._call_and_capture_level(verbose=1, quiet=False)
        assert level == logging.INFO

    def test_verbose_0_sets_warning(self) -> None:
        level = self._call_and_capture_level(verbose=0, quiet=False)
        assert level == logging.WARNING

    def test_quiet_overrides_verbose(self) -> None:
        level = self._call_and_capture_level(verbose=2, quiet=True)
        assert level == logging.ERROR

    def test_verbose_3_plus_sets_debug(self) -> None:
        level = self._call_and_capture_level(verbose=3, quiet=False)
        assert level == logging.DEBUG


class TestLoadConfig:
    """Tests for load_config function."""

    def _make_args(self, **kwargs: object) -> argparse.Namespace:
        """Create a Namespace with sensible defaults."""
        defaults: dict[str, object] = {
            "config": None,
            "preset": None,
            "iterations": None,
            "games_per_iteration": None,
            "experiment_name": None,
            "checkpoint_dir": None,
            "seed": None,
            "no_wandb": False,
            "no_rag": False,
            "no_curriculum": False,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def _make_mock_config(self) -> MagicMock:
        mock_config = MagicMock()
        mock_config.apply_env_overrides.return_value = mock_config
        return mock_config

    def test_loads_default_config_when_no_args(self) -> None:
        args = self._make_args()
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
            MockConfig.assert_called_once_with()

    def test_loads_preset_config(self) -> None:
        args = self._make_args(preset="fast")
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.from_preset.return_value = mock_config
            load_config(args)
            MockConfig.from_preset.assert_called_once_with("fast")

    def test_loads_yaml_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("experiment_name: test\n")
        args = self._make_args(config=str(config_file))
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.from_yaml.return_value = mock_config
            load_config(args)
            MockConfig.from_yaml.assert_called_once_with(str(config_file))

    def test_exits_if_config_file_not_found(self, tmp_path: Path) -> None:
        nonexistent = str(tmp_path / "missing.yaml")
        args = self._make_args(config=nonexistent)
        with patch(_CONFIG_PATH):
            with pytest.raises(SystemExit) as exc_info:
                load_config(args)
            assert exc_info.value.code == 1

    def test_applies_iterations_override(self) -> None:
        args = self._make_args(iterations=500)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.training.num_iterations == 500

    def test_applies_games_per_iteration_override(self) -> None:
        args = self._make_args(games_per_iteration=20)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.training.games_per_iteration == 20

    def test_applies_experiment_name_override(self) -> None:
        args = self._make_args(experiment_name="my_experiment")
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.experiment_name == "my_experiment"

    def test_applies_checkpoint_dir_override(self) -> None:
        args = self._make_args(checkpoint_dir="/tmp/ckpts")
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.checkpoint_dir == "/tmp/ckpts"

    def test_applies_seed_override(self) -> None:
        args = self._make_args(seed=99)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.random_seed == 99

    def test_no_wandb_disables_wandb(self) -> None:
        args = self._make_args(no_wandb=True)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.wandb.enabled is False

    def test_no_rag_disables_rag(self) -> None:
        args = self._make_args(no_rag=True)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.rag.enabled is False

    def test_no_curriculum_disables_curriculum(self) -> None:
        args = self._make_args(no_curriculum=True)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        assert mock_config.curriculum.enabled is False

    def test_calls_apply_env_overrides(self) -> None:
        args = self._make_args()
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        mock_config.apply_env_overrides.assert_called_once()

    def test_no_override_when_iterations_is_none(self) -> None:
        args = self._make_args(iterations=None)
        mock_config = self._make_mock_config()
        with patch(_CONFIG_PATH) as MockConfig:
            MockConfig.return_value = mock_config
            load_config(args)
        # Attribute should remain as MagicMock default (not explicitly set to None)
        assert mock_config.training.num_iterations != 500


class TestMain:
    """Tests for the main() entry point.

    All heavy dependencies (OrchestratorConfig, TrainingOrchestrator) are
    patched at their source module paths to avoid real training or I/O.
    """

    def _make_mock_config(self, warnings: list[str] | None = None) -> MagicMock:
        mock_config = MagicMock()
        mock_config.validate.return_value = warnings or []
        mock_config.experiment_name = "test_exp"
        mock_config.apply_env_overrides.return_value = mock_config
        return mock_config

    def test_dry_run_returns_zero_on_success(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.dry_run.return_value = True

        with patch.object(sys, "argv", ["prog", "--preset", "fast", "--dry_run"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        assert result == 0

    def test_dry_run_returns_one_on_failure(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.dry_run.return_value = False

        with patch.object(sys, "argv", ["prog", "--preset", "fast", "--dry_run"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        assert result == 1

    def test_distributed_mode_returns_one(self) -> None:
        with patch.object(sys, "argv", ["prog", "--distributed"]):
            result = main()
        assert result == 1

    def test_run_mode_calls_orchestrator_run(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()

        with patch.object(sys, "argv", ["prog", "--preset", "fast"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        mock_orchestrator.run.assert_called_once()
        assert result == 0

    def test_keyboard_interrupt_returns_130(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = KeyboardInterrupt()

        with patch.object(sys, "argv", ["prog", "--preset", "fast"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        assert result == 130

    def test_exception_in_run_returns_one(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = RuntimeError("training boom")

        with patch.object(sys, "argv", ["prog", "--preset", "fast"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        assert result == 1

    def test_config_load_exception_returns_one(self) -> None:
        with patch.object(sys, "argv", ["prog", "--preset", "fast"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.side_effect = ValueError("bad preset config")
                result = main()
        assert result == 1

    def test_verbose_traceback_on_exception(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.run.side_effect = RuntimeError("boom")

        with patch.object(sys, "argv", ["prog", "--preset", "fast", "-v"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    with patch("traceback.print_exc") as mock_traceback:
                        result = main()
        assert result == 1
        mock_traceback.assert_called_once()

    def test_warnings_from_validate_are_logged(self) -> None:
        mock_config = self._make_mock_config(warnings=["something suboptimal"])
        mock_orchestrator = MagicMock()
        mock_orchestrator.dry_run.return_value = True

        with patch.object(sys, "argv", ["prog", "--preset", "fast", "--dry_run"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.from_preset.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        # Validate was called; warnings don't cause failure
        mock_config.validate.assert_called_once()
        assert result == 0

    def test_default_config_used_when_no_preset_or_file(self) -> None:
        mock_config = self._make_mock_config()
        mock_orchestrator = MagicMock()
        mock_orchestrator.dry_run.return_value = True

        with patch.object(sys, "argv", ["prog", "--dry_run"]):
            with patch(_CONFIG_PATH) as MockConfig:
                MockConfig.return_value = mock_config
                with patch(_ORCHESTRATOR_PATH) as MockOrch:
                    MockOrch.return_value = mock_orchestrator
                    result = main()
        MockConfig.assert_called_once_with()
        assert result == 0
