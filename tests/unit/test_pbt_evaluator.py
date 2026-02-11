"""Unit tests for PBTEvaluator — cross-agent evaluation bridge.

Tests cover:
  - Matchup generation (round_robin, top_k_vs_best)
  - ELO fitness computation
  - Edge cases: single agent, all draws, all wins
  - ArenaRunner protocol compliance
  - EvaluationReport structure
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.evaluator import (
    EvaluationReport,
    MatchResult,
    PBTEvaluator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pbt_config() -> PBTConfig:
    """Minimal PBT config for evaluation tests."""
    return PBTConfig(
        population_size=4,
        evaluation_games_per_match=10,
        evaluation_strategy="round_robin",
        elo_initial_rating=1200.0,
        elo_k_factor=32.0,
    )


@pytest.fixture
def agents() -> list[AgentState]:
    """Create a test population of 4 agents."""
    return [
        AgentState(agent_id="a1", elo_rating=1200.0),
        AgentState(agent_id="a2", elo_rating=1200.0),
        AgentState(agent_id="a3", elo_rating=1200.0),
        AgentState(agent_id="a4", elo_rating=1200.0),
    ]


def _make_arena_runner(
    win_rate: float = 0.6,
    draw_rate: float = 0.1,
    loss_rate: float = 0.3,
) -> Any:
    """Create an arena runner that returns fixed results."""

    def runner(
        agent_new: AgentState,
        agent_old: AgentState,
        num_games: int,
    ) -> MatchResult:
        return MatchResult(
            agent_id_new=agent_new.agent_id,
            agent_id_old=agent_old.agent_id,
            win_rate=win_rate,
            draw_rate=draw_rate,
            loss_rate=loss_rate,
            total_games=num_games,
        )

    return runner


# ---------------------------------------------------------------------------
# MatchResult tests
# ---------------------------------------------------------------------------


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_score_new_win(self) -> None:
        r = MatchResult("a", "b", win_rate=1.0, draw_rate=0.0, loss_rate=0.0, total_games=10)
        assert r.score_new == pytest.approx(1.0)
        assert r.score_old == pytest.approx(0.0)

    def test_score_draw(self) -> None:
        r = MatchResult("a", "b", win_rate=0.0, draw_rate=1.0, loss_rate=0.0, total_games=10)
        assert r.score_new == pytest.approx(0.5)
        assert r.score_old == pytest.approx(0.5)

    def test_score_mixed(self) -> None:
        r = MatchResult("a", "b", win_rate=0.6, draw_rate=0.2, loss_rate=0.2, total_games=10)
        assert r.score_new == pytest.approx(0.7)
        assert r.score_old == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Matchup generation tests
# ---------------------------------------------------------------------------


class TestMatchupGeneration:
    """Tests for matchup strategy selection."""

    def test_round_robin_4_agents(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        evaluator = PBTEvaluator(pbt_config)
        matchups = evaluator._generate_matchups(agents)
        assert len(matchups) == 6  # C(4,2)

    def test_round_robin_pairs_are_unique(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        evaluator = PBTEvaluator(pbt_config)
        matchups = evaluator._generate_matchups(agents)
        pair_ids = {(a.agent_id, b.agent_id) for a, b in matchups}
        assert len(pair_ids) == 6

    def test_top_k_vs_best(self, agents: list[AgentState]) -> None:
        agents[2].elo_rating = 1500.0
        config = PBTConfig(population_size=4, evaluation_strategy="top_k_vs_best")
        evaluator = PBTEvaluator(config)
        matchups = evaluator._generate_matchups(agents)
        assert len(matchups) == 3
        for _, agent_old in matchups:
            assert agent_old.agent_id == "a3"

    def test_round_robin_2_agents(self, pbt_config: PBTConfig) -> None:
        agents_2 = [AgentState(agent_id="x1"), AgentState(agent_id="x2")]
        evaluator = PBTEvaluator(pbt_config)
        matchups = evaluator._generate_matchups(agents_2)
        assert len(matchups) == 1


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluatePopulation:
    """Tests for the full evaluate_population flow."""

    def test_basic_evaluation(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        runner = _make_arena_runner(win_rate=0.8, draw_rate=0.0, loss_rate=0.2)
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        report = evaluator.evaluate_population(agents, generation=1)

        assert isinstance(report, EvaluationReport)
        assert report.generation == 1
        assert len(report.matchups) == 6
        assert len(report.elo_ratings) == 4
        assert len(report.fitness_scores) == 4

    def test_agents_get_updated(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        runner = _make_arena_runner(win_rate=0.7, draw_rate=0.1, loss_rate=0.2)
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        evaluator.evaluate_population(agents, generation=1)

        for agent in agents:
            assert agent.elo_rating != 1200.0
            assert 0.0 <= agent.fitness_score <= 1.0
            assert agent.fitness_score != 0.5  # Asymmetric runner should produce non-0.5 fitness

    def test_all_draws_equal_fitness(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        runner = _make_arena_runner(win_rate=0.0, draw_rate=1.0, loss_rate=0.0)
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        report = evaluator.evaluate_population(agents, generation=1)

        fitness_vals = list(report.fitness_scores.values())
        assert all(f == pytest.approx(0.5, abs=0.01) for f in fitness_vals)

    def test_clear_winner_highest_fitness(self, pbt_config: PBTConfig) -> None:
        agents_3 = [
            AgentState(agent_id="winner"),
            AgentState(agent_id="loser1"),
            AgentState(agent_id="loser2"),
        ]

        def asymmetric_runner(
            agent_new: AgentState, agent_old: AgentState, num_games: int
        ) -> MatchResult:
            if agent_new.agent_id == "winner":
                return MatchResult("winner", agent_old.agent_id, 1.0, 0.0, 0.0, num_games)
            elif agent_old.agent_id == "winner":
                return MatchResult(agent_new.agent_id, "winner", 0.0, 0.0, 1.0, num_games)
            else:
                return MatchResult(agent_new.agent_id, agent_old.agent_id, 0.5, 0.0, 0.5, num_games)

        evaluator = PBTEvaluator(pbt_config, arena_runner=asymmetric_runner)
        report = evaluator.evaluate_population(agents_3, generation=1)

        assert report.elo_ratings["winner"] > report.elo_ratings["loser1"]
        assert report.elo_ratings["winner"] > report.elo_ratings["loser2"]
        assert report.fitness_scores["winner"] == pytest.approx(1.0)

    def test_no_arena_runner_raises(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        evaluator = PBTEvaluator(pbt_config)
        with pytest.raises(ValueError, match="No arena_runner"):
            evaluator.evaluate_population(agents, generation=1)

    def test_override_arena_runner(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        default_runner = _make_arena_runner(win_rate=0.5)
        override_runner = _make_arena_runner(win_rate=0.9)
        evaluator = PBTEvaluator(pbt_config, arena_runner=default_runner)
        report = evaluator.evaluate_population(agents, generation=1, arena_runner=override_runner)
        assert len(report.matchups) == 6

    def test_games_played_propagated(self, pbt_config: PBTConfig, agents: list[AgentState]) -> None:
        """Verify that agent.games_played is updated after evaluation."""
        games_per_match = pbt_config.evaluation_games_per_match
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)

        # Initial games
        for a in agents:
            a.games_played = 5

        evaluator.evaluate_population(agents, generation=1)

        # In round-robin with 4 agents, each plays 3 matches
        expected_increase = games_per_match * 3
        for agent in agents:
            assert agent.games_played == 5 + expected_increase


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests."""

    def test_single_agent_skips_evaluation(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_make_arena_runner())
        report = evaluator.evaluate_population([AgentState(agent_id="solo")], generation=1)
        assert len(report.matchups) == 0

    def test_empty_population(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_make_arena_runner())
        report = evaluator.evaluate_population([], generation=1)
        assert len(report.matchups) == 0

    def test_normalise_ratings_all_equal(self) -> None:
        result = PBTEvaluator._normalise_ratings({"a": 1200.0, "b": 1200.0, "c": 1200.0})
        assert all(v == pytest.approx(0.5) for v in result.values())

    def test_normalise_ratings_empty(self) -> None:
        assert PBTEvaluator._normalise_ratings({}) == {}

    def test_normalise_ratings_spread(self) -> None:
        result = PBTEvaluator._normalise_ratings({"low": 1000.0, "mid": 1200.0, "high": 1400.0})
        assert result["low"] == pytest.approx(0.0)
        assert result["mid"] == pytest.approx(0.5)
        assert result["high"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Test that PBTConfig evaluation fields work correctly."""

    def test_config_defaults(self) -> None:
        config = PBTConfig()
        assert config.evaluation_strategy == "round_robin"
        assert config.evaluation_games_per_match == 10
        assert config.elo_initial_rating == 1200.0
        assert config.elo_k_factor == 32.0

    def test_agent_state_elo_field(self) -> None:
        assert AgentState(agent_id="test").elo_rating == 1200.0

    def test_agent_state_elo_serialization(self) -> None:
        agent = AgentState(agent_id="test", elo_rating=1500.0)
        d = agent.to_dict()
        assert d["elo_rating"] == 1500.0
        assert AgentState.from_dict(d).elo_rating == 1500.0

    def test_agent_state_backwards_compatible(self) -> None:
        agent = AgentState.from_dict({"agent_id": "old", "fitness_score": 0.7})
        assert agent.elo_rating == 1200.0

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluation_strategy"):
            PBTConfig(evaluation_strategy="invalid")

    def test_invalid_games_per_match_raises(self) -> None:
        with pytest.raises(ValueError, match="evaluation_games_per_match"):
            PBTConfig(evaluation_games_per_match=0)

    def test_pbt_config_serialization(self) -> None:
        config = PBTConfig(evaluation_strategy="top_k_vs_best")
        d = config.to_dict()
        assert d["evaluation_strategy"] == "top_k_vs_best"
        assert "elo_k_factor" in d


# ---------------------------------------------------------------------------
# Baseline benchmarking tests
# ---------------------------------------------------------------------------


class TestBaselineBenchmarking:
    """Tests for PBTEvaluator baseline benchmark features."""

    def test_baseline_results_field_default(self) -> None:
        report = EvaluationReport(generation=1)
        assert report.baseline_results == []

    def test_baseline_results_field_populated(self) -> None:
        result = MatchResult("a1", "baseline_random", 0.8, 0.1, 0.1, 10)
        report = EvaluationReport(generation=1, baseline_results=[result])
        assert len(report.baseline_results) == 1

    def test_run_baseline_benchmarks_at_interval(self) -> None:
        baselines = [AgentState(agent_id="baseline_random")]
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner(win_rate=0.7, draw_rate=0.1, loss_rate=0.2)
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baselines,
            baseline_interval=5,
        )

        agents = [AgentState(agent_id=f"a{i}") for i in range(4)]

        # gen=1: should NOT run
        results = evaluator.run_baseline_benchmarks(agents, generation=1)
        assert results == []

        # gen=5: SHOULD run (4 agents x 1 baseline = 4 matches)
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        assert len(results) == 4

        # gen=10: SHOULD run
        results = evaluator.run_baseline_benchmarks(agents, generation=10)
        assert len(results) == 4

    def test_run_baseline_benchmarks_no_baselines(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config, arena_runner=runner, baseline_agents=[], baseline_interval=5,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        assert results == []

    def test_run_baseline_benchmarks_interval_zero(self) -> None:
        baselines = [AgentState(agent_id="baseline_random")]
        config = PBTConfig(population_size=4, baseline_interval=0)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config, arena_runner=runner, baseline_agents=baselines, baseline_interval=0,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        assert results == []

    def test_baseline_does_not_affect_elo(self) -> None:
        baselines = [AgentState(agent_id="baseline_random", elo_rating=1200.0)]
        config = PBTConfig(population_size=4, baseline_interval=1)
        runner = _make_arena_runner(win_rate=1.0, draw_rate=0.0, loss_rate=0.0)
        evaluator = PBTEvaluator(
            config, arena_runner=runner, baseline_agents=baselines, baseline_interval=1,
        )

        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
        ]

        evaluator.evaluate_population(agents, generation=1)

        # Baseline ELO should not have been modified
        assert baselines[0].elo_rating == 1200.0

    def test_evaluate_population_includes_baseline_results(self) -> None:
        baselines = [AgentState(agent_id="baseline_random")]
        config = PBTConfig(
            population_size=4, baseline_interval=1, evaluation_games_per_match=10,
        )
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config, arena_runner=runner, baseline_agents=baselines, baseline_interval=1,
        )

        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
            AgentState(agent_id="a3", elo_rating=1200.0),
        ]

        report = evaluator.evaluate_population(agents, generation=1)
        assert len(report.baseline_results) == 3  # 3 agents x 1 baseline

    def test_baseline_interval_from_config(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=7)
        evaluator = PBTEvaluator(config)
        assert evaluator._baseline_interval == 7
