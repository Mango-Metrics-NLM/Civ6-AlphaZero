"""Tests for gap-fill components: planner.train(), RobustnessEvaluator, CLI train/evaluate.

Covers:
- StrategicPlanner.train() method
- RobustnessEvaluator class in scenarios.py
- CLI train and evaluate subcommands
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.envs.resource_allocation import (
    ResourceAllocationConfig,
    ResourceAllocationEnv,
)
from alpha_zero.strategic.planner import PlanningResult, StrategicPlanner
from alpha_zero.strategic.scenarios import (
    RobustnessEvaluator,
    Scenario,
    ScenarioEngine,
)
from alpha_zero.strategic.trace.decision_trace import DecisionTrace

# Patch targets for CLI
_MAIN = "alpha_zero.strategic.__main__"
_REGISTRY = f"{_MAIN}.EnvironmentRegistry"
_CONFIG = f"{_MAIN}.StrategicConfig"
_PLANNER = f"{_MAIN}.StrategicPlanner"
_SCENARIO = f"{_MAIN}.Scenario"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_env() -> ResourceAllocationEnv:
    """Minimal environment for fast tests."""
    return ResourceAllocationEnv(
        ResourceAllocationConfig(
            num_regions=2,
            num_resource_types=1,
            num_time_steps=3,
            total_budget=50.0,
            resource_names=["supplies"],
            region_names=["A", "B"],
            seed=42,
        )
    )


@pytest.fixture
def fast_config() -> StrategicConfig:
    """Minimal planner config for fast tests."""
    config = StrategicConfig.from_preset("fast")
    config.planning.num_simulations = 5
    config.planning.num_planning_iterations = 2
    config.planning.strategies_per_iteration = 2
    config.random_seed = 123
    return config


@pytest.fixture
def planner(fast_config: StrategicConfig) -> StrategicPlanner:
    """Planner with fast config."""
    return StrategicPlanner(fast_config)


@pytest.fixture
def three_scenarios() -> list[Scenario]:
    """Three scenarios with probabilities summing to 1."""
    return [
        Scenario(name="optimistic", parameters={"growth": 0.05}, probability=0.3),
        Scenario(name="pessimistic", parameters={"growth": 0.01}, probability=0.5),
        Scenario(name="neutral", parameters={"growth": 0.03}, probability=0.2),
    ]


@pytest.fixture
def engine_3(three_scenarios: list[Scenario]) -> ScenarioEngine:
    """Engine with three scenarios."""
    return ScenarioEngine(three_scenarios)


# ===========================================================================
# TestStrategicPlannerTrain
# ===========================================================================


class TestStrategicPlannerTrain:
    """Tests for StrategicPlanner.train() method."""

    def test_train_returns_planning_result(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() returns a PlanningResult with correct structure."""
        result = planner.train(environment=tiny_env)

        assert isinstance(result, PlanningResult)
        expected_episodes = (
            planner.config.planning.num_planning_iterations
            * planner.config.planning.strategies_per_iteration
        )
        assert result.total_episodes == expected_episodes
        assert result.elapsed_seconds > 0.0
        assert len(result.traces) == expected_episodes

    def test_train_respects_num_iterations(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() with explicit num_iterations overrides config."""
        result = planner.train(environment=tiny_env, num_iterations=1)

        expected = 1 * planner.config.planning.strategies_per_iteration
        assert result.total_episodes == expected

    def test_train_produces_finite_outcomes(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() produces finite non-NaN outcomes."""
        result = planner.train(environment=tiny_env, num_iterations=1)

        for trace in result.traces:
            assert np.all(np.isfinite(trace.final_outcomes))

    def test_train_builds_pareto_frontier(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() builds a Pareto frontier."""
        result = planner.train(environment=tiny_env)

        assert result.pareto_frontier is not None
        assert result.pareto_frontier.size >= 1

    def test_train_metadata_contains_mode(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() metadata includes 'mode': 'train'."""
        result = planner.train(environment=tiny_env, num_iterations=1)

        assert result.metadata.get("mode") == "train"
        assert "num_iterations" in result.metadata

    def test_train_traces_are_sorted(
        self, planner: StrategicPlanner, tiny_env: ResourceAllocationEnv
    ) -> None:
        """train() returns traces sorted by scalarized value (best first)."""
        result = planner.train(environment=tiny_env)

        if len(result.traces) >= 2:
            # Compare sums as a rough scalarization proxy
            sums = [float(np.sum(t.final_outcomes)) for t in result.traces]
            # Should be non-increasing (best first)
            for i in range(len(sums) - 1):
                assert sums[i] >= sums[i + 1] - 1e-6


# ===========================================================================
# TestRobustnessEvaluator
# ===========================================================================


class TestRobustnessEvaluator:
    """Tests for the RobustnessEvaluator class."""

    def test_evaluate_returns_expected_keys(
        self, engine_3: ScenarioEngine
    ) -> None:
        """evaluate() returns dict with all expected keys."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([0.9, 0.8]),
            "pessimistic": np.array([0.3, 0.2]),
            "neutral": np.array([0.6, 0.5]),
        }
        result = evaluator.evaluate(values)

        assert "robustness_score" in result
        assert "expected_outcomes" in result
        assert "worst_scenario" in result
        assert "best_scenario" in result
        assert "scenario_rankings" in result
        assert "vulnerability_threshold" in result

    def test_evaluate_identifies_best_and_worst(
        self, engine_3: ScenarioEngine
    ) -> None:
        """evaluate() correctly identifies best and worst scenarios."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([0.9, 0.8]),     # sum=1.7
            "pessimistic": np.array([0.1, 0.1]),     # sum=0.2
            "neutral": np.array([0.5, 0.5]),         # sum=1.0
        }
        result = evaluator.evaluate(values)

        assert result["best_scenario"] == "optimistic"
        assert result["worst_scenario"] == "pessimistic"

    def test_evaluate_robustness_score_in_range(
        self, engine_3: ScenarioEngine
    ) -> None:
        """evaluate() robustness_score is in [0, 1]."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([0.9, 0.8]),
            "pessimistic": np.array([0.3, 0.2]),
            "neutral": np.array([0.6, 0.5]),
        }
        result = evaluator.evaluate(values)

        assert 0.0 <= result["robustness_score"] <= 1.0

    def test_evaluate_identical_outcomes_perfectly_robust(
        self, engine_3: ScenarioEngine
    ) -> None:
        """Identical outcomes across scenarios yields robustness 1.0."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([0.5, 0.5]),
            "pessimistic": np.array([0.5, 0.5]),
            "neutral": np.array([0.5, 0.5]),
        }
        result = evaluator.evaluate(values)

        assert result["robustness_score"] == pytest.approx(1.0)

    def test_evaluate_scenario_rankings_ordered(
        self, engine_3: ScenarioEngine
    ) -> None:
        """scenario_rankings is ordered from best to worst."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([0.9, 0.8]),     # sum=1.7
            "pessimistic": np.array([0.1, 0.1]),     # sum=0.2
            "neutral": np.array([0.5, 0.5]),         # sum=1.0
        }
        result = evaluator.evaluate(values)

        rankings = result["scenario_rankings"]
        assert rankings[0] == "optimistic"
        assert rankings[-1] == "pessimistic"

    def test_evaluate_vulnerability_detection(
        self, engine_3: ScenarioEngine
    ) -> None:
        """vulnerability_threshold lists scenarios below 50% of best."""
        evaluator = RobustnessEvaluator(engine_3)
        values = {
            "optimistic": np.array([1.0, 1.0]),     # sum=2.0
            "pessimistic": np.array([0.1, 0.1]),     # sum=0.2 < 1.0 (50% of 2.0)
            "neutral": np.array([0.6, 0.6]),         # sum=1.2 >= 1.0
        }
        result = evaluator.evaluate(values)

        assert "pessimistic" in result["vulnerability_threshold"]
        assert "optimistic" not in result["vulnerability_threshold"]

    def test_engine_property(self, engine_3: ScenarioEngine) -> None:
        """engine property returns the underlying ScenarioEngine."""
        evaluator = RobustnessEvaluator(engine_3)
        assert evaluator.engine is engine_3

    def test_rank_strategies(self, engine_3: ScenarioEngine) -> None:
        """rank_strategies sorts strategies by robustness score."""
        evaluator = RobustnessEvaluator(engine_3)
        strategies = {
            "strategy_a": {
                "optimistic": np.array([0.9, 0.8]),
                "pessimistic": np.array([0.1, 0.1]),
                "neutral": np.array([0.5, 0.5]),
            },
            "strategy_b": {
                "optimistic": np.array([0.6, 0.6]),
                "pessimistic": np.array([0.5, 0.5]),
                "neutral": np.array([0.55, 0.55]),
            },
        }
        rankings = evaluator.rank_strategies(strategies)

        assert len(rankings) == 2
        # strategy_b is more robust (smaller gap between best/worst)
        assert rankings[0][0] == "strategy_b"
        assert rankings[1][0] == "strategy_a"
        # All scores in [0, 1]
        for _, score in rankings:
            assert 0.0 <= score <= 1.0


# ===========================================================================
# TestCLITrainCommand
# ===========================================================================


class TestCLITrainCommand:
    """Tests for the CLI train subcommand."""

    def test_train_parser_defaults(self) -> None:
        """train subparser has correct defaults."""
        from alpha_zero.strategic.__main__ import create_parser

        parser = create_parser()
        args = parser.parse_args(["train"])

        assert args.command == "train"
        assert args.env == "resource_allocation"
        assert args.preset == "fast"
        assert args.config is None
        assert args.iterations is None
        assert args.output is None
        assert args.dry_run is False
        assert args.seed is None

    def test_train_parser_all_flags(self) -> None:
        """train subparser parses all flags."""
        from alpha_zero.strategic.__main__ import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "train",
            "--env", "climate_negotiation",
            "--preset", "balanced",
            "--iterations", "50",
            "--output", "/tmp/out.json",
            "--dry_run",
            "--seed", "42",
        ])

        assert args.env == "climate_negotiation"
        assert args.preset == "balanced"
        assert args.iterations == 50
        assert args.output == "/tmp/out.json"
        assert args.dry_run is True
        assert args.seed == 42

    @patch(_CONFIG)
    def test_train_dry_run(
        self, mock_config_class: MagicMock, capsys: pytest.CaptureFixture
    ) -> None:
        """cmd_train with --dry_run validates config and prints summary."""
        from alpha_zero.strategic.__main__ import cmd_train

        mock_config = MagicMock()
        mock_config.planning.num_planning_iterations = 10
        mock_config.planning.strategies_per_iteration = 5
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        args = SimpleNamespace(
            preset="fast", env="test_env", config=None,
            iterations=None, output=None, dry_run=True, seed=None,
        )
        cmd_train(args)

        captured = capsys.readouterr().out
        assert "Dry run" in captured
        assert "test_env" in captured

    @patch(_PLANNER)
    @patch(_REGISTRY)
    @patch(_CONFIG)
    def test_train_runs(
        self,
        mock_config_class: MagicMock,
        mock_registry: MagicMock,
        mock_planner_class: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_train creates planner, runs training, prints summary."""
        from alpha_zero.strategic.__main__ import cmd_train

        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        mock_registry.is_registered.return_value = True
        mock_env = MagicMock()
        mock_env.objective_names = ["a", "b"]
        mock_registry.create.return_value = mock_env

        fake_result = MagicMock()
        fake_result.summary.return_value = "Training complete"
        mock_planner_class.return_value.train.return_value = fake_result

        args = SimpleNamespace(
            preset="fast", env="test_env", config=None,
            iterations=5, output=None, dry_run=False, seed=None,
        )
        cmd_train(args)

        captured = capsys.readouterr().out
        assert "Training complete" in captured
        mock_planner_class.return_value.train.assert_called_once()

    @patch(f"{_MAIN}.cmd_train")
    def test_main_dispatch_train(self, mock_cmd: MagicMock) -> None:
        """main() dispatches to cmd_train for 'train' command."""
        from alpha_zero.strategic.__main__ import main

        with patch.object(sys, "argv", ["prog", "train", "--dry_run"]):
            main()

        mock_cmd.assert_called_once()


# ===========================================================================
# TestCLIEvaluateCommand
# ===========================================================================


class TestCLIEvaluateCommand:
    """Tests for the CLI evaluate subcommand."""

    def test_evaluate_parser(self) -> None:
        """evaluate subparser parses required and optional flags."""
        from alpha_zero.strategic.__main__ import create_parser

        parser = create_parser()
        args = parser.parse_args([
            "evaluate",
            "--scenarios", "/tmp/scenarios.yaml",
            "--env", "supply_chain",
            "--episodes-per-scenario", "3",
            "--output", "/tmp/eval.json",
            "--dry_run",
        ])

        assert args.command == "evaluate"
        assert args.scenarios == "/tmp/scenarios.yaml"
        assert args.env == "supply_chain"
        assert args.episodes_per_scenario == 3
        assert args.output == "/tmp/eval.json"
        assert args.dry_run is True

    def test_evaluate_requires_scenarios(self) -> None:
        """evaluate subparser requires --scenarios flag."""
        from alpha_zero.strategic.__main__ import create_parser

        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])

    @patch(f"{_MAIN}._load_scenarios")
    @patch(_CONFIG)
    def test_evaluate_dry_run(
        self,
        mock_config_class: MagicMock,
        mock_load_scenarios: MagicMock,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cmd_evaluate with --dry_run validates and prints summary."""
        from alpha_zero.strategic.__main__ import cmd_evaluate

        mock_config = MagicMock()
        mock_config.validate.return_value = []
        mock_config_class.from_preset.return_value = mock_config

        mock_load_scenarios.return_value = [
            Scenario(name="high", probability=0.6),
            Scenario(name="low", probability=0.4),
        ]

        args = SimpleNamespace(
            preset="fast", env="test_env", config=None,
            scenarios="/tmp/s.yaml", episodes_per_scenario=1,
            output=None, dry_run=True, seed=None,
        )
        cmd_evaluate(args)

        captured = capsys.readouterr().out
        assert "Dry run" in captured
        assert "Scenarios: 2" in captured
        assert "high" in captured

    @patch(f"{_MAIN}.cmd_evaluate")
    def test_main_dispatch_evaluate(self, mock_cmd: MagicMock) -> None:
        """main() dispatches to cmd_evaluate for 'evaluate' command."""
        from alpha_zero.strategic.__main__ import main

        with patch.object(
            sys, "argv",
            ["prog", "evaluate", "--scenarios", "/tmp/s.yaml", "--dry_run"],
        ):
            main()

        mock_cmd.assert_called_once()

    def test_load_scenarios_from_yaml(self, tmp_path: pytest.TempPathFactory) -> None:
        """_load_scenarios reads scenarios from a YAML file."""
        from alpha_zero.strategic.__main__ import _load_scenarios

        yaml_content = """
scenarios:
  - name: high_growth
    parameters:
      rate: 0.05
    probability: 0.6
    description: High growth scenario
  - name: low_growth
    parameters:
      rate: 0.01
    probability: 0.4
"""
        scenario_file = tmp_path / "scenarios.yaml"
        scenario_file.write_text(yaml_content)

        scenarios = _load_scenarios(str(scenario_file))

        assert len(scenarios) == 2
        assert scenarios[0].name == "high_growth"
        assert scenarios[0].probability == pytest.approx(0.6)
        assert scenarios[0].parameters == {"rate": 0.05}
        assert scenarios[1].name == "low_growth"
