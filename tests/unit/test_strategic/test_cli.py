"""Unit tests for the Strategic Planning SDK CLI entry point.

Tests the CLI module at alpha_zero.strategic.__main__, covering:
- Argument parser construction and parsing
- The ``list-envs`` subcommand with a mocked registry
- The ``info`` subcommand for both valid and unknown environments
- The ``plan`` subcommand in ``--dry_run`` mode
- Top-level ``--dry_run`` without a subcommand
- The ``main()`` dispatch function with various argument combinations
- Output-file serialization in ``cmd_plan``
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.strategic.__main__ import (
    cmd_info,
    cmd_list_envs,
    cmd_plan,
    create_parser,
    main,
)

# Patch targets: __main__.py uses module-level imports so we patch
# where the names are looked up (i.e. in the __main__ module namespace).
_MAIN = "alpha_zero.strategic.__main__"
_REGISTRY = f"{_MAIN}.EnvironmentRegistry"
_CONFIG = f"{_MAIN}.StrategicConfig"
_PLANNER = f"{_MAIN}.StrategicPlanner"


# ---------------------------------------------------------------------------
# Helpers / Stubs
# ---------------------------------------------------------------------------


@dataclass
class _FakeStateSpec:
    """Minimal state-spec stub returned by fake environments."""

    has_spatial: bool = False
    spatial_shape: tuple[int, ...] = (0, 0, 0)
    vector_dim: int = 32


@dataclass
class _FakeActionSpec:
    """Minimal action-spec stub returned by fake environments."""

    total_size: int = 60


@dataclass
class _FakeEnv:
    """Lightweight environment stub for ``cmd_info`` and ``cmd_plan`` tests."""

    num_objectives: int = 3
    num_agents: int = 1
    objective_names: list[str] = field(
        default_factory=lambda: ["efficiency", "equity", "cost"]
    )
    state_spec: _FakeStateSpec = field(default_factory=_FakeStateSpec)
    action_spec: _FakeActionSpec = field(default_factory=_FakeActionSpec)


@dataclass
class _FakePlanningResult:
    """Stub for PlanningResult returned by StrategicPlanner.plan."""

    num_strategies: int = 1
    objective_names: list[str] = field(
        default_factory=lambda: ["efficiency", "equity"]
    )
    elapsed_seconds: float = 0.42
    pareto_frontier: object = None
    best_trace: object = None

    def summary(self) -> str:
        return "Fake planning summary"


# ---------------------------------------------------------------------------
# TestCreateParser
# ---------------------------------------------------------------------------


class TestCreateParser:
    """Tests for the argument parser factory."""

    def test_parser_returns_argparse_parser(self) -> None:
        """create_parser returns an ArgumentParser instance."""
        import argparse

        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_plan_subcommand_defaults(self) -> None:
        """Parsing 'plan' with no extra flags yields expected defaults."""
        parser = create_parser()
        args = parser.parse_args(["plan"])

        assert args.command == "plan"
        assert args.env == "resource_allocation"
        assert args.preset == "fast"
        assert args.config is None
        assert args.episodes is None
        assert args.output is None
        assert args.dry_run is False
        assert args.seed is None

    def test_plan_subcommand_all_flags(self) -> None:
        """Parsing 'plan' with all optional flags captures each value."""
        parser = create_parser()
        args = parser.parse_args([
            "plan",
            "--env", "climate_negotiation",
            "--preset", "thorough",
            "--config", "/tmp/my_config.yaml",
            "--episodes", "50",
            "--output", "/tmp/results.json",
            "--seed", "123",
            "--dry_run",
        ])

        assert args.command == "plan"
        assert args.env == "climate_negotiation"
        assert args.preset == "thorough"
        assert args.config == "/tmp/my_config.yaml"
        assert args.episodes == 50
        assert args.output == "/tmp/results.json"
        assert args.seed == 123
        assert args.dry_run is True

    def test_plan_preset_choices(self) -> None:
        """An invalid --preset value causes a parser error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["plan", "--preset", "invalid_preset"])

    def test_list_envs_subcommand(self) -> None:
        """Parsing 'list-envs' sets the command correctly."""
        parser = create_parser()
        args = parser.parse_args(["list-envs"])

        assert args.command == "list-envs"

    def test_info_subcommand(self) -> None:
        """Parsing 'info <name>' sets command and env_name."""
        parser = create_parser()
        args = parser.parse_args(["info", "resource_allocation"])

        assert args.command == "info"
        assert args.env_name == "resource_allocation"

    def test_info_requires_env_name(self) -> None:
        """The 'info' subcommand requires a positional env_name argument."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["info"])

    def test_top_level_dry_run(self) -> None:
        """Top-level --dry_run flag is parsed without a subcommand."""
        parser = create_parser()
        args = parser.parse_args(["--dry_run"])

        assert args.dry_run is True
        assert args.command is None

    def test_verbose_flag_short(self) -> None:
        """Short -v flag enables verbose."""
        parser = create_parser()
        args = parser.parse_args(["-v"])

        assert args.verbose is True

    def test_verbose_flag_long(self) -> None:
        """Long --verbose flag enables verbose."""
        parser = create_parser()
        args = parser.parse_args(["--verbose"])

        assert args.verbose is True

    def test_no_subcommand_no_flags(self) -> None:
        """Parsing with no arguments yields command=None, dry_run=False."""
        parser = create_parser()
        args = parser.parse_args([])

        assert args.command is None
        assert args.dry_run is False


# ---------------------------------------------------------------------------
# TestCmdListEnvs
# ---------------------------------------------------------------------------


class TestCmdListEnvs:
    """Tests for the ``list-envs`` subcommand handler."""

    @patch(_REGISTRY)
    def test_lists_environments(
        self,
        mock_registry: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_list_envs prints each environment from the registry."""
        mock_registry.get_info.return_value = {
            "resource_allocation": "Allocate resources across regions.",
            "climate_negotiation": "Climate policy negotiation.",
        }

        cmd_list_envs()

        captured = capsys.readouterr().out
        assert "Available Strategic Environments:" in captured
        assert "resource_allocation" in captured
        assert "Allocate resources across regions." in captured
        assert "climate_negotiation" in captured
        assert "Climate policy negotiation." in captured
        assert "Total: 2 environments" in captured

    @patch(_REGISTRY)
    def test_empty_registry(
        self,
        mock_registry: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_list_envs handles an empty registry gracefully."""
        mock_registry.get_info.return_value = {}

        cmd_list_envs()

        captured = capsys.readouterr().out
        assert "Total: 0 environments" in captured


# ---------------------------------------------------------------------------
# TestCmdInfo
# ---------------------------------------------------------------------------


class TestCmdInfo:
    """Tests for the ``info`` subcommand handler."""

    @patch(_REGISTRY)
    def test_info_valid_environment(
        self,
        mock_registry: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_info prints detailed information for a known environment."""
        fake_env = _FakeEnv()
        mock_registry.is_registered.return_value = True
        mock_registry.create.return_value = fake_env
        mock_registry.get_description.return_value = "A test environment."

        cmd_info("test_env")

        captured = capsys.readouterr().out
        assert "Environment: test_env" in captured
        assert "A test environment." in captured
        assert "Objectives (3):" in captured
        assert "efficiency" in captured
        assert "equity" in captured
        assert "cost" in captured
        assert "Agents: 1" in captured
        assert "Vector dim: 32" in captured
        assert "60 actions" in captured

    @patch(_REGISTRY)
    def test_info_with_spatial(
        self,
        mock_registry: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_info prints spatial shape when the environment has spatial input."""
        fake_env = _FakeEnv(
            state_spec=_FakeStateSpec(has_spatial=True, spatial_shape=(13, 16, 16)),
        )
        mock_registry.is_registered.return_value = True
        mock_registry.create.return_value = fake_env
        mock_registry.get_description.return_value = "Spatial env."

        cmd_info("spatial_env")

        captured = capsys.readouterr().out
        assert "Spatial:" in captured
        assert "(13, 16, 16)" in captured

    @patch(_REGISTRY)
    def test_info_unknown_environment_exits(
        self,
        mock_registry: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_info exits with code 1 for an unknown environment."""
        mock_registry.is_registered.return_value = False
        mock_registry.list_environments.return_value = [
            "resource_allocation",
            "climate_negotiation",
        ]

        with pytest.raises(SystemExit) as exc_info:
            cmd_info("nonexistent_env")

        assert exc_info.value.code == 1

        captured = capsys.readouterr().out
        assert "Error" in captured
        assert "nonexistent_env" in captured
        assert "resource_allocation" in captured


# ---------------------------------------------------------------------------
# TestCmdPlan
# ---------------------------------------------------------------------------


class TestCmdPlan:
    """Tests for the ``plan`` subcommand handler."""

    @patch(_CONFIG)
    def test_dry_run_from_preset(
        self,
        mock_config_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan with --dry_run validates config and prints summary without planning."""
        mock_config = MagicMock()
        mock_config.mode = "plan"
        mock_config.planning.num_simulations = 10
        mock_config.planning.num_planning_iterations = 2
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="resource_allocation",
            seed=None,
            dry_run=True,
            episodes=None,
            output=None,
        )

        cmd_plan(args)

        captured = capsys.readouterr().out
        assert "Dry run" in captured
        assert "configuration validated successfully" in captured
        assert "fast" in captured
        assert "resource_allocation" in captured

        mock_config_class.from_preset.assert_called_once_with("fast")
        mock_config.validate.assert_called_once()

    @patch(_CONFIG)
    def test_dry_run_from_yaml(
        self,
        mock_config_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan with --config loads from YAML file instead of preset."""
        mock_config = MagicMock()
        mock_config.mode = "plan"
        mock_config.planning.num_simulations = 200
        mock_config.planning.num_planning_iterations = 10
        mock_config.validate.return_value = []
        mock_config_class.from_yaml.return_value = mock_config

        args = SimpleNamespace(
            preset="fast",
            config="/tmp/custom.yaml",
            env="resource_allocation",
            seed=None,
            dry_run=True,
            episodes=None,
            output=None,
        )

        cmd_plan(args)

        mock_config_class.from_yaml.assert_called_once_with("/tmp/custom.yaml")
        mock_config_class.from_preset.assert_not_called()

    @patch(_CONFIG)
    def test_dry_run_sets_seed(
        self,
        mock_config_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan sets random_seed on the config when --seed is provided."""
        mock_config = MagicMock()
        mock_config.mode = "plan"
        mock_config.planning.num_simulations = 10
        mock_config.planning.num_planning_iterations = 2
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="resource_allocation",
            seed=42,
            dry_run=True,
            episodes=None,
            output=None,
        )

        cmd_plan(args)

        assert mock_config.random_seed == 42

    @patch(_CONFIG)
    def test_dry_run_reports_validation_warnings(
        self,
        mock_config_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan logs validation warnings from config.validate()."""
        mock_config = MagicMock()
        mock_config.mode = "plan"
        mock_config.planning.num_simulations = 10
        mock_config.planning.num_planning_iterations = 2
        mock_config.validate.return_value = ["Warning: something looks odd"]
        mock_config_class.from_preset.return_value = mock_config

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="resource_allocation",
            seed=None,
            dry_run=True,
            episodes=None,
            output=None,
        )

        # Should not raise despite warnings
        cmd_plan(args)

        captured = capsys.readouterr().out
        assert "Dry run" in captured

    @patch(_PLANNER)
    @patch(_REGISTRY)
    @patch(_CONFIG)
    def test_plan_unknown_env_exits(
        self,
        mock_config_class: MagicMock,
        mock_registry: MagicMock,
        mock_planner_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan exits with code 1 when the environment is not registered."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        mock_registry.is_registered.return_value = False
        mock_registry.list_environments.return_value = ["resource_allocation"]

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="nonexistent_env",
            seed=None,
            dry_run=False,
            episodes=None,
            output=None,
        )

        with pytest.raises(SystemExit) as exc_info:
            cmd_plan(args)

        assert exc_info.value.code == 1
        captured = capsys.readouterr().out
        assert "Error" in captured
        assert "nonexistent_env" in captured

    @patch(_PLANNER)
    @patch(_REGISTRY)
    @patch(_CONFIG)
    def test_plan_runs_and_prints_summary(
        self,
        mock_config_class: MagicMock,
        mock_registry: MagicMock,
        mock_planner_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan creates a planner, runs planning, and prints the result summary."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        fake_env = _FakeEnv()
        mock_registry.is_registered.return_value = True
        mock_registry.create.return_value = fake_env

        fake_result = _FakePlanningResult()
        mock_planner_instance = MagicMock()
        mock_planner_instance.plan.return_value = fake_result
        mock_planner_class.return_value = mock_planner_instance

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="resource_allocation",
            seed=None,
            dry_run=False,
            episodes=10,
            output=None,
        )

        cmd_plan(args)

        mock_planner_instance.plan.assert_called_once_with(
            environment=fake_env,
            num_episodes=10,
        )
        captured = capsys.readouterr().out
        assert "Fake planning summary" in captured

    @patch(_PLANNER)
    @patch(_REGISTRY)
    @patch(_CONFIG)
    def test_plan_saves_output_json(
        self,
        mock_config_class: MagicMock,
        mock_registry: MagicMock,
        mock_planner_class: MagicMock,
        tmp_path: pytest.TempPathFactory,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_plan saves results to a JSON file when --output is provided."""
        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        fake_env = _FakeEnv()
        mock_registry.is_registered.return_value = True
        mock_registry.create.return_value = fake_env

        fake_result = _FakePlanningResult()
        mock_planner_instance = MagicMock()
        mock_planner_instance.plan.return_value = fake_result
        mock_planner_class.return_value = mock_planner_instance

        output_file = str(tmp_path / "results.json")

        args = SimpleNamespace(
            preset="fast",
            config=None,
            env="resource_allocation",
            seed=None,
            dry_run=False,
            episodes=None,
            output=output_file,
        )

        cmd_plan(args)

        # Verify JSON was written
        with open(output_file) as f:
            data = json.load(f)

        assert data["summary"] == "Fake planning summary"
        assert data["num_strategies"] == 1
        assert data["objective_names"] == ["efficiency", "equity"]
        assert data["elapsed_seconds"] == pytest.approx(0.42)
        assert data["pareto_frontier_size"] == 0

        captured = capsys.readouterr().out
        assert f"Results saved to: {output_file}" in captured


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the ``main()`` dispatch function."""

    @patch(f"{_MAIN}.cmd_list_envs")
    def test_dispatch_list_envs(self, mock_cmd: MagicMock) -> None:
        """main() dispatches to cmd_list_envs for 'list-envs' command."""
        with patch.object(sys, "argv", ["prog", "list-envs"]):
            main()

        mock_cmd.assert_called_once()

    @patch(f"{_MAIN}.cmd_info")
    def test_dispatch_info(self, mock_cmd: MagicMock) -> None:
        """main() dispatches to cmd_info for 'info' command."""
        with patch.object(sys, "argv", ["prog", "info", "resource_allocation"]):
            main()

        mock_cmd.assert_called_once_with("resource_allocation")

    @patch(f"{_MAIN}.cmd_plan")
    def test_dispatch_plan(self, mock_cmd: MagicMock) -> None:
        """main() dispatches to cmd_plan for 'plan' command."""
        with patch.object(sys, "argv", ["prog", "plan", "--dry_run"]):
            main()

        mock_cmd.assert_called_once()
        dispatched_args = mock_cmd.call_args[0][0]
        assert dispatched_args.command == "plan"
        assert dispatched_args.dry_run is True

    def test_top_level_dry_run(self, capsys: pytest.CaptureFixture) -> None:
        """main() with only --dry_run prints SDK status message."""
        with patch.object(sys, "argv", ["prog", "--dry_run"]):
            main()

        captured = capsys.readouterr().out
        assert "dry run OK" in captured

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture) -> None:
        """main() with no arguments prints help text (does not crash)."""
        with patch.object(sys, "argv", ["prog"]):
            main()

        captured = capsys.readouterr().out
        # argparse help contains the prog name or usage
        assert "strategic" in captured.lower() or "usage" in captured.lower()

    @patch(f"{_MAIN}.cmd_plan")
    def test_verbose_flag_sets_debug_logging(self, mock_cmd: MagicMock) -> None:
        """main() with -v configures DEBUG-level logging."""
        with patch.object(sys, "argv", ["prog", "-v", "plan", "--dry_run"]):
            with patch(f"{_MAIN}.logging") as mock_logging:
                mock_logging.DEBUG = 10
                mock_logging.INFO = 20
                main()

                mock_logging.basicConfig.assert_called_once()
                call_kwargs = mock_logging.basicConfig.call_args
                assert call_kwargs[1]["level"] == 10 or call_kwargs.kwargs["level"] == 10

    @patch(f"{_MAIN}.cmd_plan")
    def test_plan_passes_env_and_preset(self, mock_cmd: MagicMock) -> None:
        """main() forwards --env and --preset to the plan subcommand args."""
        with patch.object(sys, "argv", [
            "prog", "plan",
            "--env", "climate_negotiation",
            "--preset", "balanced",
        ]):
            main()

        dispatched_args = mock_cmd.call_args[0][0]
        assert dispatched_args.env == "climate_negotiation"
        assert dispatched_args.preset == "balanced"

    @patch(f"{_MAIN}.cmd_plan")
    def test_plan_passes_episodes_and_seed(self, mock_cmd: MagicMock) -> None:
        """main() forwards --episodes and --seed to the plan subcommand args."""
        with patch.object(sys, "argv", [
            "prog", "plan",
            "--episodes", "25",
            "--seed", "999",
        ]):
            main()

        dispatched_args = mock_cmd.call_args[0][0]
        assert dispatched_args.episodes == 25
        assert dispatched_args.seed == 999
