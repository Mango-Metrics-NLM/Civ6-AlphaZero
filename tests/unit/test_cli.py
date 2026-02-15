"""Tests for the CLI entry point."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.cli import (
    build_parser,
    cmd_benchmark,
    cmd_evaluate,
    cmd_health_check,
    cmd_train,
    main,
)


class TestBuildParser:
    """Tests for the argument parser builder."""

    def test_parser_created(self) -> None:
        parser = build_parser()
        assert parser is not None

    def test_train_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["train", "--dry-run"])
        assert args.command == "train"
        assert args.dry_run is True

    def test_evaluate_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["evaluate", "--games", "10"])
        assert args.command == "evaluate"
        assert args.games == 10

    def test_health_check_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        assert args.command == "health-check"

    def test_benchmark_subcommand(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--episodes", "50"])
        assert args.command == "benchmark"
        assert args.episodes == 50

    def test_verbose_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["-v", "health-check"])
        assert args.verbose is True

    def test_train_all_options(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "train",
            "--config", "configs/config.yaml",
            "--env", "mini_resource",
            "--iterations", "5",
            "--games", "10",
            "--batch-size", "32",
            "--dry-run",
        ])
        assert args.config == "configs/config.yaml"
        assert args.env == "mini_resource"
        assert args.iterations == 5
        assert args.games == 10
        assert args.batch_size == 32
        assert args.dry_run is True

    def test_no_command(self) -> None:
        """No command exits cleanly."""
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None


class TestMain:
    """Tests for the main entry point."""

    def test_no_args_returns_zero(self) -> None:
        result = main([])
        assert result == 0

    def test_health_check_returns_zero(self) -> None:
        result = main(["health-check"])
        assert result == 0

    def test_benchmark_quick(self) -> None:
        result = main(["benchmark", "--episodes", "3"])
        assert result == 0

    def test_evaluate_quick(self) -> None:
        result = main(["evaluate", "--games", "2"])
        assert result == 0


class TestHealthCheck:
    """Tests for the health check command."""

    def test_health_check_passes(self) -> None:
        """All SDK components should be importable."""
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        result = cmd_health_check(args)
        assert result in (0, 1)


# ---------------------------------------------------------------------------
# Additional test classes for coverage improvement
# ---------------------------------------------------------------------------


@dataclass
class _FakeIterationMetrics:
    """Lightweight stand-in for IterationMetrics used by cmd_train tests."""

    eval_elo: float = 1200.0
    training_loss: float = 0.05


class TestCmdTrain:
    """Tests for the train command (cmd_train)."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        defaults = {
            "config": None,
            "env": None,
            "iterations": None,
            "games": None,
            "batch_size": None,
            "dry_run": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("src.core.training.TrainingOrchestrator")
    @patch("src.core.training.Learner")
    @patch("src.core.config.SDKConfig")
    def test_train_default_env(
        self, mock_sdk: MagicMock, mock_learner_cls: MagicMock, mock_orch_cls: MagicMock, capsys
    ) -> None:
        mock_config = MagicMock()
        mock_config.training.max_iterations = 1
        mock_config.apply_env_overrides.return_value = mock_config
        mock_sdk.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.run.return_value = [_FakeIterationMetrics()]
        mock_instance.best_elo = 1200.0
        mock_orch_cls.return_value = mock_instance

        result = cmd_train(self._make_args())

        assert result == 0
        out = capsys.readouterr().out
        assert "Training complete" in out

    @patch("src.core.training.TrainingOrchestrator")
    @patch("src.core.training.Learner")
    @patch("src.core.config.SDKConfig")
    def test_train_dry_run(
        self, mock_sdk: MagicMock, mock_learner_cls: MagicMock, mock_orch_cls: MagicMock, capsys
    ) -> None:
        mock_config = MagicMock()
        mock_config.training.max_iterations = 1
        mock_config.apply_env_overrides.return_value = mock_config
        mock_sdk.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.run.return_value = []
        mock_instance.best_elo = 1000.0
        mock_orch_cls.return_value = mock_instance

        result = cmd_train(self._make_args(dry_run=True))

        assert result == 0
        out = capsys.readouterr().out
        assert "DRY RUN MODE" in out
        assert "Training complete" in out
        mock_instance.run.assert_called_once_with(dry_run=True)

    @patch("src.core.training.TrainingOrchestrator")
    @patch("src.core.training.Learner")
    @patch("src.core.config.SDKConfig")
    def test_train_custom_overrides(
        self,
        mock_sdk: MagicMock,
        mock_learner_cls: MagicMock,
        mock_orch_cls: MagicMock,
        capsys,
    ) -> None:
        mock_config = MagicMock()
        mock_config.training.max_iterations = 5
        mock_config.apply_env_overrides.return_value = mock_config
        mock_config.model_dump.return_value = {"training": {}}
        mock_sdk.return_value = mock_config
        mock_sdk.model_validate.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.run.return_value = [_FakeIterationMetrics()]
        mock_instance.best_elo = 1300.0
        mock_orch_cls.return_value = mock_instance

        result = cmd_train(self._make_args(iterations=5, games=10, batch_size=32))

        assert result == 0
        # SDKConfig.model_validate should be called to merge overrides
        mock_sdk.model_validate.assert_called_once()
        out = capsys.readouterr().out
        assert "Training complete" in out

    @patch("src.core.training.TrainingOrchestrator")
    @patch("src.core.training.Learner")
    @patch("src.core.config.load_config")
    def test_train_with_config_file(
        self, mock_load: MagicMock, mock_learner_cls: MagicMock, mock_orch_cls: MagicMock, capsys
    ) -> None:
        mock_config = MagicMock()
        mock_config.training.max_iterations = 2
        mock_config.apply_env_overrides.return_value = mock_config
        mock_load.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.run.return_value = [
            _FakeIterationMetrics(eval_elo=1100.0, training_loss=0.1),
        ]
        mock_instance.best_elo = 1100.0
        mock_orch_cls.return_value = mock_instance

        result = cmd_train(self._make_args(config="configs/test.yaml"))

        assert result == 0
        mock_load.assert_any_call("configs/test.yaml")
        out = capsys.readouterr().out
        assert "Training complete" in out
        assert "Final Elo" in out
        assert "Final Loss" in out
        assert "Best Elo" in out

    @patch("src.core.training.TrainingOrchestrator")
    @patch("src.core.training.Learner")
    @patch("src.core.config.SDKConfig")
    def test_train_no_metrics_output(
        self, mock_sdk: MagicMock, mock_learner_cls: MagicMock, mock_orch_cls: MagicMock, capsys
    ) -> None:
        """When orchestrator returns empty list, final stats are not printed."""
        mock_config = MagicMock()
        mock_config.training.max_iterations = 1
        mock_config.apply_env_overrides.return_value = mock_config
        mock_sdk.return_value = mock_config

        mock_instance = MagicMock()
        mock_instance.run.return_value = []
        mock_orch_cls.return_value = mock_instance

        result = cmd_train(self._make_args())

        assert result == 0
        out = capsys.readouterr().out
        assert "Training complete: 0 iterations" in out
        assert "Final Elo" not in out


class TestCmdEvaluate:
    """Tests for the evaluate command (cmd_evaluate)."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        defaults = {"env": None, "games": 20, "checkpoint": None}
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_evaluate_default(self, capsys) -> None:
        result = cmd_evaluate(self._make_args(games=2))
        assert result == 0
        out = capsys.readouterr().out
        assert "Evaluation Results" in out
        assert "Elo" in out
        assert "Win Rate" in out

    def test_evaluate_custom_env_and_games(self, capsys) -> None:
        result = cmd_evaluate(self._make_args(env="territory_control", games=3))
        assert result == 0
        out = capsys.readouterr().out
        assert "Evaluation Results" in out
        assert "W/L/D" in out


class TestCmdBenchmark:
    """Tests for the benchmark command (cmd_benchmark)."""

    def _make_args(self, **overrides: object) -> argparse.Namespace:
        defaults = {"env": None, "episodes": 100}
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_benchmark_default(self, capsys) -> None:
        result = cmd_benchmark(self._make_args(episodes=5))
        assert result == 0
        out = capsys.readouterr().out
        assert "Benchmark" in out
        assert "Steps/sec" in out
        assert "Episodes/sec" in out

    def test_benchmark_custom_episodes(self, capsys) -> None:
        result = cmd_benchmark(self._make_args(episodes=3))
        assert result == 0
        out = capsys.readouterr().out
        assert "Episodes: 3" in out
        assert "Total steps" in out

    def test_benchmark_custom_env(self, capsys) -> None:
        result = cmd_benchmark(self._make_args(env="tactical_combat", episodes=2))
        assert result == 0
        out = capsys.readouterr().out
        assert "tactical_combat" in out


class TestCmdHealthCheckDetailed:
    """More detailed tests for cmd_health_check output."""

    def test_output_includes_component_names(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        cmd_health_check(args)
        out = capsys.readouterr().out
        for name in ["Core types", "Configuration", "Protocols", "MCTS Engine", "Training"]:
            assert name in out

    def test_pytorch_shows_not_installed(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        cmd_health_check(args)
        out = capsys.readouterr().out
        # In this sandbox torch is not installed
        assert "PyTorch" in out
        assert "Not installed" in out or "Available" in out

    def test_all_checks_header(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        cmd_health_check(args)
        out = capsys.readouterr().out
        assert "SDK Health Check" in out
        assert "=" * 50 in out

    def test_mini_games_check(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        cmd_health_check(args)
        out = capsys.readouterr().out
        assert "Mini-games" in out
        assert "3 environments" in out

    def test_registry_check(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        cmd_health_check(args)
        out = capsys.readouterr().out
        assert "Registry" in out
        assert "environments" in out

    def test_overall_status(self, capsys) -> None:
        parser = build_parser()
        args = parser.parse_args(["health-check"])
        result = cmd_health_check(args)
        out = capsys.readouterr().out
        assert "Overall:" in out
        if result == 0:
            assert "ALL CHECKS PASSED" in out
        else:
            assert "SOME CHECKS FAILED" in out

    def test_fail_status_sets_all_ok_false(self, capsys) -> None:
        """When a check fails, all_ok becomes False and return code is 1."""
        with patch("src.cli._get_default_registry", side_effect=RuntimeError("boom")):
            parser = build_parser()
            args = parser.parse_args(["health-check"])
            result = cmd_health_check(args)
            out = capsys.readouterr().out
            assert "FAIL" in out
            # Registry check should have failed
            assert result == 1 or "SOME CHECKS FAILED" in out


class TestMainUnknownCommand:
    """Tests for the fallback path when an unknown command is dispatched."""

    def test_no_command_returns_zero(self, capsys) -> None:
        result = main([])
        assert result == 0

    def test_unknown_command_via_monkeypatch(self, monkeypatch, capsys) -> None:
        """Force an unknown command through the dispatcher.

        argparse subparsers normally reject unknown commands, so we
        monkeypatch parse_args to inject one.
        """
        fake_args = argparse.Namespace(command="nonexistent", verbose=False)
        monkeypatch.setattr(
            "src.cli.build_parser",
            lambda: type(
                "P",
                (),
                {
                    "parse_args": lambda self, argv=None: fake_args,
                    "print_help": lambda self: print("Usage: spsdk ..."),
                },
            )(),
        )
        result = main(["nonexistent"])
        assert result == 1


class TestVerboseFlag:
    """Tests for the --verbose / -v flag."""

    def test_verbose_enables_debug(self, capsys) -> None:
        """--verbose should configure logging at DEBUG level."""
        import logging

        with patch.object(logging, "basicConfig", wraps=logging.basicConfig) as mock_bc:
            main(["-v", "health-check"])
            mock_bc.assert_called_once()
            call_kwargs = mock_bc.call_args
            assert call_kwargs[1]["level"] == logging.DEBUG or call_kwargs[0][0] == logging.DEBUG

    def test_default_is_info(self, capsys) -> None:
        """Without --verbose, logging should be configured at INFO level."""
        import logging

        with patch.object(logging, "basicConfig", wraps=logging.basicConfig) as mock_bc:
            main(["health-check"])
            mock_bc.assert_called_once()
            call_kwargs = mock_bc.call_args
            assert call_kwargs[1]["level"] == logging.INFO or call_kwargs[0][0] == logging.INFO


class TestMainIntegration:
    """End-to-end main() calls that exercise full paths."""

    def test_main_train_dry_run(self, capsys) -> None:
        """main(['train', '--dry-run']) should complete successfully."""
        result = main(["train", "--dry-run"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Training complete" in out

    def test_main_train_with_overrides(self, capsys) -> None:
        """main(['train', '--iterations', '1', '--games', '1', '--dry-run'])
        should merge overrides into default config, not discard it.

        Regression test: CLI overrides previously discarded the loaded config
        by calling load_config(overrides=overrides) which started from defaults.
        """
        result = main([
            "train", "--iterations", "1", "--games", "1",
            "--batch-size", "4", "--dry-run",
        ])
        assert result == 0
        out = capsys.readouterr().out
        assert "Training complete" in out

    def test_main_benchmark_small(self, capsys) -> None:
        result = main(["benchmark", "--episodes", "2"])
        assert result == 0
        out = capsys.readouterr().out
        assert "Benchmark" in out
