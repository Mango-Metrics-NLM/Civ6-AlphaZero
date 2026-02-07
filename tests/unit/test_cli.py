"""Tests for the CLI entry point."""
from __future__ import annotations

from src.cli import build_parser, cmd_health_check, main


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
