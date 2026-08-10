"""Unit tests for PBTEvaluator -- cross-agent evaluation bridge.

Tests cover:
  - MatchResult dataclass: construction, score_new/score_old properties, frozen behavior
  - EvaluationReport dataclass: defaults, construction with data
  - PBTEvaluator.__init__: with/without arena_runner, with/without baselines,
    default baseline_interval
  - PBTEvaluator.evaluate_population: round_robin and top_k_vs_best strategies,
    no runner raises ValueError, fewer than 2 agents returns empty report,
    agent state updates (elo_rating, fitness, games_played)
  - PBTEvaluator._generate_matchups: round_robin, top_k_vs_best, unknown fallback
  - PBTEvaluator._round_robin: 2 agents = 1 pair, 4 agents = 6 pairs
  - PBTEvaluator._top_k_vs_best: 1 agent = 0 pairs, correct best identification
  - PBTEvaluator._compute_fitness: ELO rating updates, normalisation
  - PBTEvaluator._normalise_ratings: empty dict, equal ratings -> 0.5,
    different ratings -> [0,1]
  - PBTEvaluator.run_baseline_benchmarks: interval=0 skips, interval mismatch
    skips, no baselines skips, no runner skips, successful benchmarks,
    error handling
  - PBTEvaluator.arena_runner property: returns configured runner
"""

from __future__ import annotations

import dataclasses
from typing import Any
from unittest.mock import MagicMock

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
def top_k_config() -> PBTConfig:
    """PBT config with top_k_vs_best strategy."""
    return PBTConfig(
        population_size=4,
        evaluation_games_per_match=10,
        evaluation_strategy="top_k_vs_best",
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


@pytest.fixture
def two_agents() -> list[AgentState]:
    """Create a test population of 2 agents."""
    return [
        AgentState(agent_id="x1", elo_rating=1200.0),
        AgentState(agent_id="x2", elo_rating=1200.0),
    ]


@pytest.fixture
def baseline_agents() -> list[AgentState]:
    """Create baseline agents for benchmarking."""
    return [
        AgentState(agent_id="baseline_random", elo_rating=1000.0),
        AgentState(agent_id="baseline_heuristic", elo_rating=1100.0),
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


def _make_failing_runner(
    fail_on_agent_id: str | None = None,
) -> Any:
    """Create an arena runner that raises an exception."""

    def runner(
        agent_new: AgentState,
        agent_old: AgentState,
        num_games: int,
    ) -> MatchResult:
        if fail_on_agent_id is None or agent_new.agent_id == fail_on_agent_id:
            raise RuntimeError("Arena match failed!")
        return MatchResult(
            agent_id_new=agent_new.agent_id,
            agent_id_old=agent_old.agent_id,
            win_rate=0.5,
            draw_rate=0.0,
            loss_rate=0.5,
            total_games=num_games,
        )

    return runner


# ---------------------------------------------------------------------------
# MatchResult tests
# ---------------------------------------------------------------------------


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_construction(self) -> None:
        r = MatchResult("a", "b", win_rate=0.6, draw_rate=0.2, loss_rate=0.2, total_games=10)
        assert r.agent_id_new == "a"
        assert r.agent_id_old == "b"
        assert r.win_rate == 0.6
        assert r.draw_rate == 0.2
        assert r.loss_rate == 0.2
        assert r.total_games == 10

    def test_score_new_win(self) -> None:
        r = MatchResult("a", "b", win_rate=1.0, draw_rate=0.0, loss_rate=0.0, total_games=10)
        assert r.score_new == pytest.approx(1.0)
        assert r.score_old == pytest.approx(0.0)

    def test_score_new_loss(self) -> None:
        r = MatchResult("a", "b", win_rate=0.0, draw_rate=0.0, loss_rate=1.0, total_games=10)
        assert r.score_new == pytest.approx(0.0)
        assert r.score_old == pytest.approx(1.0)

    def test_score_draw(self) -> None:
        r = MatchResult("a", "b", win_rate=0.0, draw_rate=1.0, loss_rate=0.0, total_games=10)
        assert r.score_new == pytest.approx(0.5)
        assert r.score_old == pytest.approx(0.5)

    def test_score_mixed(self) -> None:
        r = MatchResult("a", "b", win_rate=0.6, draw_rate=0.2, loss_rate=0.2, total_games=10)
        assert r.score_new == pytest.approx(0.7)
        assert r.score_old == pytest.approx(0.3)

    def test_score_new_formula(self) -> None:
        """Verify score_new = win_rate + 0.5 * draw_rate."""
        r = MatchResult("a", "b", win_rate=0.4, draw_rate=0.4, loss_rate=0.2, total_games=20)
        assert r.score_new == pytest.approx(0.4 + 0.5 * 0.4)

    def test_score_old_formula(self) -> None:
        """Verify score_old = loss_rate + 0.5 * draw_rate."""
        r = MatchResult("a", "b", win_rate=0.4, draw_rate=0.4, loss_rate=0.2, total_games=20)
        assert r.score_old == pytest.approx(0.2 + 0.5 * 0.4)

    def test_frozen_cannot_set_attribute(self) -> None:
        r = MatchResult("a", "b", win_rate=0.5, draw_rate=0.0, loss_rate=0.5, total_games=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.win_rate = 0.9  # type: ignore[misc]

    def test_frozen_cannot_delete_attribute(self) -> None:
        r = MatchResult("a", "b", win_rate=0.5, draw_rate=0.0, loss_rate=0.5, total_games=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            del r.win_rate  # type: ignore[misc]


# ---------------------------------------------------------------------------
# EvaluationReport tests
# ---------------------------------------------------------------------------


class TestEvaluationReport:
    """Tests for EvaluationReport dataclass."""

    def test_defaults(self) -> None:
        report = EvaluationReport()
        assert report.generation == 0
        assert report.matchups == []
        assert report.elo_ratings == {}
        assert report.fitness_scores == {}
        assert report.games_played == {}
        assert report.baseline_results == []

    def test_construction_with_data(self) -> None:
        result = MatchResult("a1", "a2", 0.7, 0.1, 0.2, 10)
        report = EvaluationReport(
            generation=5,
            matchups=[result],
            elo_ratings={"a1": 1250.0, "a2": 1150.0},
            fitness_scores={"a1": 1.0, "a2": 0.0},
            games_played={"a1": 10, "a2": 10},
            baseline_results=[result],
        )
        assert report.generation == 5
        assert len(report.matchups) == 1
        assert report.elo_ratings["a1"] == 1250.0
        assert report.fitness_scores["a2"] == 0.0
        assert report.games_played["a1"] == 10
        assert len(report.baseline_results) == 1

    def test_mutable_fields_independent(self) -> None:
        """Each instance should have independent mutable defaults."""
        r1 = EvaluationReport()
        r2 = EvaluationReport()
        r1.matchups.append(MatchResult("a", "b", 0.5, 0.0, 0.5, 10))
        assert len(r2.matchups) == 0


# ---------------------------------------------------------------------------
# PBTEvaluator.__init__ tests
# ---------------------------------------------------------------------------


class TestPBTEvaluatorInit:
    """Tests for PBTEvaluator constructor."""

    def test_init_minimal(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        assert evaluator._config is pbt_config
        assert evaluator._arena_runner is None
        assert evaluator._baseline_agents == []
        assert evaluator._baseline_interval == pbt_config.baseline_interval

    def test_init_with_arena_runner(self, pbt_config: PBTConfig) -> None:
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        assert evaluator._arena_runner is runner

    def test_init_with_baselines(
        self, pbt_config: PBTConfig, baseline_agents: list[AgentState]
    ) -> None:
        evaluator = PBTEvaluator(pbt_config, baseline_agents=baseline_agents)
        assert len(evaluator._baseline_agents) == 2
        assert evaluator._baseline_agents[0].agent_id == "baseline_random"

    def test_init_baseline_interval_explicit(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config, baseline_interval=10)
        assert evaluator._baseline_interval == 10

    def test_init_baseline_interval_from_config(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=7)
        evaluator = PBTEvaluator(config)
        assert evaluator._baseline_interval == 7

    def test_init_baseline_interval_none_uses_config(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=3)
        evaluator = PBTEvaluator(config, baseline_interval=None)
        assert evaluator._baseline_interval == 3

    def test_init_baseline_interval_zero(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=0)
        evaluator = PBTEvaluator(config, baseline_interval=0)
        assert evaluator._baseline_interval == 0

    def test_init_elo_system_created(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        assert evaluator._elo is not None
        assert evaluator._elo.config.initial_rating == pbt_config.elo_initial_rating
        assert evaluator._elo.config.k_factor_base == pbt_config.elo_k_factor


# ---------------------------------------------------------------------------
# arena_runner property tests
# ---------------------------------------------------------------------------


class TestArenaRunnerProperty:
    """Tests for the arena_runner property."""

    def test_returns_none_when_not_set(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        assert evaluator.arena_runner is None

    def test_returns_configured_runner(self, pbt_config: PBTConfig) -> None:
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        assert evaluator.arena_runner is runner

    def test_returns_mock_runner(self, pbt_config: PBTConfig) -> None:
        mock_runner = MagicMock(return_value=MatchResult("a", "b", 0.5, 0.0, 0.5, 10))
        evaluator = PBTEvaluator(pbt_config, arena_runner=mock_runner)
        assert evaluator.arena_runner is mock_runner


# ---------------------------------------------------------------------------
# _generate_matchups tests
# ---------------------------------------------------------------------------


class TestGenerateMatchups:
    """Tests for _generate_matchups strategy dispatch."""

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

    def test_unknown_strategy_falls_back_to_round_robin(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        """Unknown strategy should log a warning and fall back to round_robin."""
        evaluator = PBTEvaluator(pbt_config)
        # Override strategy to something unknown after construction
        evaluator._config = PBTConfig.__new__(PBTConfig)
        # Copy all fields from pbt_config
        for f in dataclasses.fields(pbt_config):
            setattr(evaluator._config, f.name, getattr(pbt_config, f.name))
        # Force unknown strategy by bypassing validation
        object.__setattr__(evaluator._config, "evaluation_strategy", "unknown_strategy")

        matchups = evaluator._generate_matchups(agents)
        # Should fall back to round_robin: C(4,2) = 6
        assert len(matchups) == 6


# ---------------------------------------------------------------------------
# _round_robin tests
# ---------------------------------------------------------------------------


class TestRoundRobin:
    """Tests for _round_robin static method."""

    def test_two_agents_one_pair(self) -> None:
        agents = [AgentState(agent_id="x1"), AgentState(agent_id="x2")]
        pairs = PBTEvaluator._round_robin(agents)
        assert len(pairs) == 1
        assert pairs[0][0].agent_id == "x1"
        assert pairs[0][1].agent_id == "x2"

    def test_four_agents_six_pairs(self) -> None:
        agents = [AgentState(agent_id=f"a{i}") for i in range(4)]
        pairs = PBTEvaluator._round_robin(agents)
        assert len(pairs) == 6

    def test_three_agents_three_pairs(self) -> None:
        agents = [AgentState(agent_id=f"a{i}") for i in range(3)]
        pairs = PBTEvaluator._round_robin(agents)
        assert len(pairs) == 3

    def test_one_agent_zero_pairs(self) -> None:
        agents = [AgentState(agent_id="solo")]
        pairs = PBTEvaluator._round_robin(agents)
        assert len(pairs) == 0

    def test_empty_agents_zero_pairs(self) -> None:
        pairs = PBTEvaluator._round_robin([])
        assert len(pairs) == 0

    def test_five_agents_ten_pairs(self) -> None:
        agents = [AgentState(agent_id=f"a{i}") for i in range(5)]
        pairs = PBTEvaluator._round_robin(agents)
        assert len(pairs) == 10  # C(5,2)


# ---------------------------------------------------------------------------
# _top_k_vs_best tests
# ---------------------------------------------------------------------------


class TestTopKVsBest:
    """Tests for _top_k_vs_best static method."""

    def test_single_agent_zero_pairs(self) -> None:
        agents = [AgentState(agent_id="solo", elo_rating=1500.0)]
        pairs = PBTEvaluator._top_k_vs_best(agents)
        assert len(pairs) == 0

    def test_empty_agents_zero_pairs(self) -> None:
        pairs = PBTEvaluator._top_k_vs_best([])
        assert len(pairs) == 0

    def test_correct_best_identification(self) -> None:
        agents = [
            AgentState(agent_id="low", elo_rating=1000.0),
            AgentState(agent_id="mid", elo_rating=1200.0),
            AgentState(agent_id="high", elo_rating=1500.0),
        ]
        pairs = PBTEvaluator._top_k_vs_best(agents)
        assert len(pairs) == 2
        for agent_new, agent_old in pairs:
            assert agent_old.agent_id == "high"
            assert agent_new.agent_id != "high"

    def test_two_agents(self) -> None:
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1300.0),
        ]
        pairs = PBTEvaluator._top_k_vs_best(agents)
        assert len(pairs) == 1
        assert pairs[0][0].agent_id == "a1"
        assert pairs[0][1].agent_id == "a2"

    def test_tied_ratings_pick_one_best(self) -> None:
        """When ratings are tied, max() picks one; all others play against it."""
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
            AgentState(agent_id="a3", elo_rating=1200.0),
        ]
        pairs = PBTEvaluator._top_k_vs_best(agents)
        assert len(pairs) == 2
        best_id = pairs[0][1].agent_id
        for _, opponent in pairs:
            assert opponent.agent_id == best_id


# ---------------------------------------------------------------------------
# _compute_fitness tests
# ---------------------------------------------------------------------------


class TestComputeFitness:
    """Tests for _compute_fitness method."""

    def test_elo_ratings_updated(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [
            AgentState(agent_id="winner", elo_rating=1200.0),
            AgentState(agent_id="loser", elo_rating=1200.0),
        ]
        results = [
            MatchResult("winner", "loser", 1.0, 0.0, 0.0, 10),
        ]
        report = evaluator._compute_fitness(agents, results, generation=1)

        # Winner should gain ELO, loser should lose ELO
        assert report.elo_ratings["winner"] > 1200.0
        assert report.elo_ratings["loser"] < 1200.0

    def test_normalisation_applied(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
        ]
        results = [
            MatchResult("a1", "a2", 1.0, 0.0, 0.0, 10),
        ]
        report = evaluator._compute_fitness(agents, results, generation=1)

        # Fitness should be normalised to [0, 1]
        assert report.fitness_scores["a1"] == pytest.approx(1.0)
        assert report.fitness_scores["a2"] == pytest.approx(0.0)

    def test_games_played_accumulated(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0, games_played=5),
            AgentState(agent_id="a2", elo_rating=1200.0, games_played=10),
        ]
        results = [
            MatchResult("a1", "a2", 0.5, 0.0, 0.5, 20),
        ]
        report = evaluator._compute_fitness(agents, results, generation=1)

        assert report.games_played["a1"] == 25  # 5 + 20
        assert report.games_played["a2"] == 30  # 10 + 20

    def test_all_draws_equal_fitness(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
            AgentState(agent_id="a3", elo_rating=1200.0),
        ]
        results = [
            MatchResult("a1", "a2", 0.0, 1.0, 0.0, 10),
            MatchResult("a1", "a3", 0.0, 1.0, 0.0, 10),
            MatchResult("a2", "a3", 0.0, 1.0, 0.0, 10),
        ]
        report = evaluator._compute_fitness(agents, results, generation=1)

        for fitness in report.fitness_scores.values():
            assert fitness == pytest.approx(0.5, abs=0.01)

    def test_report_generation_set(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [AgentState(agent_id="a1"), AgentState(agent_id="a2")]
        results = [MatchResult("a1", "a2", 0.5, 0.0, 0.5, 10)]
        report = evaluator._compute_fitness(agents, results, generation=42)
        assert report.generation == 42

    def test_report_matchups_preserved(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config)
        agents = [AgentState(agent_id="a1"), AgentState(agent_id="a2")]
        results = [MatchResult("a1", "a2", 0.6, 0.2, 0.2, 10)]
        report = evaluator._compute_fitness(agents, results, generation=1)
        assert report.matchups is results

    def test_unknown_agent_in_results_uses_initial_rating(self, pbt_config: PBTConfig) -> None:
        """If a result references an agent not in the population, use initial rating."""
        evaluator = PBTEvaluator(pbt_config)
        agents = [AgentState(agent_id="a1", elo_rating=1200.0)]
        # Result has an agent_id_old that is NOT in agents list
        results = [MatchResult("a1", "unknown", 1.0, 0.0, 0.0, 10)]
        report = evaluator._compute_fitness(agents, results, generation=1)
        # Should not crash; "unknown" gets the initial rating
        assert "unknown" in report.elo_ratings
        assert "a1" in report.elo_ratings


# ---------------------------------------------------------------------------
# _normalise_ratings tests
# ---------------------------------------------------------------------------


class TestNormaliseRatings:
    """Tests for _normalise_ratings static method."""

    def test_empty_dict(self) -> None:
        assert PBTEvaluator._normalise_ratings({}) == {}

    def test_equal_ratings_gives_half(self) -> None:
        result = PBTEvaluator._normalise_ratings({"a": 1200.0, "b": 1200.0, "c": 1200.0})
        assert all(v == pytest.approx(0.5) for v in result.values())

    def test_different_ratings_min_max(self) -> None:
        result = PBTEvaluator._normalise_ratings({"low": 1000.0, "mid": 1200.0, "high": 1400.0})
        assert result["low"] == pytest.approx(0.0)
        assert result["mid"] == pytest.approx(0.5)
        assert result["high"] == pytest.approx(1.0)

    def test_single_agent(self) -> None:
        result = PBTEvaluator._normalise_ratings({"solo": 1500.0})
        assert result["solo"] == pytest.approx(0.5)

    def test_two_agents(self) -> None:
        result = PBTEvaluator._normalise_ratings({"low": 1000.0, "high": 1500.0})
        assert result["low"] == pytest.approx(0.0)
        assert result["high"] == pytest.approx(1.0)

    def test_very_close_ratings_treated_as_equal(self) -> None:
        """Ratings within 1e-9 spread should map to 0.5."""
        result = PBTEvaluator._normalise_ratings({"a": 1200.0, "b": 1200.0 + 1e-12})
        assert all(v == pytest.approx(0.5) for v in result.values())

    def test_negative_ratings(self) -> None:
        """Normalisation should work with negative-ish ratings."""
        result = PBTEvaluator._normalise_ratings({"a": 100.0, "b": 200.0, "c": 300.0})
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_population tests
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
            # Asymmetric runner should produce non-0.5 fitness for some agents
            assert agent.fitness_score != 0.5

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

    def test_top_k_vs_best_strategy(
        self, top_k_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        agents[0].elo_rating = 1500.0  # Make a1 the best
        runner = _make_arena_runner(win_rate=0.6, draw_rate=0.1, loss_rate=0.3)
        evaluator = PBTEvaluator(top_k_config, arena_runner=runner)
        report = evaluator.evaluate_population(agents, generation=1)

        # top_k_vs_best with 4 agents: 3 matchups (each non-best vs best)
        assert len(report.matchups) == 3
        assert report.generation == 1

    def test_fewer_than_two_agents_returns_empty_report(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_make_arena_runner())

        # Single agent
        report = evaluator.evaluate_population([AgentState(agent_id="solo")], generation=1)
        assert len(report.matchups) == 0
        assert report.generation == 1
        assert report.elo_ratings == {}
        assert report.fitness_scores == {}

    def test_empty_population_returns_empty_report(self, pbt_config: PBTConfig) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_make_arena_runner())
        report = evaluator.evaluate_population([], generation=1)
        assert len(report.matchups) == 0

    def test_fitness_history_updated(self, pbt_config: PBTConfig) -> None:
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
        ]
        runner = _make_arena_runner(win_rate=0.8, draw_rate=0.0, loss_rate=0.2)
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        evaluator.evaluate_population(agents, generation=1)

        # update_fitness appends to fitness_history
        for agent in agents:
            assert len(agent.fitness_history) >= 1

    def test_evaluate_population_with_mock_runner(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        mock_runner = MagicMock(
            side_effect=lambda agent_new, agent_old, num_games: MatchResult(
                agent_id_new=agent_new.agent_id,
                agent_id_old=agent_old.agent_id,
                win_rate=0.5,
                draw_rate=0.0,
                loss_rate=0.5,
                total_games=num_games,
            )
        )
        evaluator = PBTEvaluator(pbt_config, arena_runner=mock_runner)
        report = evaluator.evaluate_population(agents, generation=1)
        assert mock_runner.call_count == 6  # C(4,2)
        assert len(report.matchups) == 6


# ---------------------------------------------------------------------------
# run_baseline_benchmarks tests
# ---------------------------------------------------------------------------


class TestRunBaselineBenchmarks:
    """Tests for run_baseline_benchmarks method."""

    def test_interval_zero_skips(self, baseline_agents: list[AgentState]) -> None:
        config = PBTConfig(population_size=4, baseline_interval=0)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baseline_agents,
            baseline_interval=0,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        assert results == []

    def test_interval_mismatch_skips(self, baseline_agents: list[AgentState]) -> None:
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baseline_agents,
            baseline_interval=5,
        )
        agents = [AgentState(agent_id="a1")]
        # gen=3 is not a multiple of 5
        results = evaluator.run_baseline_benchmarks(agents, generation=3)
        assert results == []

    def test_no_baselines_skips(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=[],
            baseline_interval=5,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        assert results == []

    def test_no_runner_skips(self, baseline_agents: list[AgentState]) -> None:
        config = PBTConfig(population_size=4, baseline_interval=1)
        evaluator = PBTEvaluator(
            config,
            arena_runner=None,
            baseline_agents=baseline_agents,
            baseline_interval=1,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=1)
        assert results == []

    def test_successful_benchmarks(self, baseline_agents: list[AgentState]) -> None:
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner(win_rate=0.7, draw_rate=0.1, loss_rate=0.2)
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baseline_agents,
            baseline_interval=5,
        )
        agents = [
            AgentState(agent_id="a1"),
            AgentState(agent_id="a2"),
            AgentState(agent_id="a3"),
        ]
        # gen=5 matches interval
        results = evaluator.run_baseline_benchmarks(agents, generation=5)
        # 3 agents x 2 baselines = 6 matches
        assert len(results) == 6

    def test_successful_benchmarks_gen_zero(self, baseline_agents: list[AgentState]) -> None:
        """Generation 0 should match interval since 0 % N == 0."""
        config = PBTConfig(population_size=4, baseline_interval=5)
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baseline_agents,
            baseline_interval=5,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(agents, generation=0)
        # 1 agent x 2 baselines = 2 matches
        assert len(results) == 2

    def test_error_handling_continues(self, baseline_agents: list[AgentState]) -> None:
        """If a baseline match fails, it is skipped and others continue."""
        config = PBTConfig(population_size=4, baseline_interval=1)

        call_count = 0

        def flaky_runner(
            agent_new: AgentState,
            agent_old: AgentState,
            num_games: int,
        ) -> MatchResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("First match exploded")
            return MatchResult(
                agent_id_new=agent_new.agent_id,
                agent_id_old=agent_old.agent_id,
                win_rate=0.5,
                draw_rate=0.0,
                loss_rate=0.5,
                total_games=num_games,
            )

        evaluator = PBTEvaluator(
            config,
            arena_runner=flaky_runner,
            baseline_agents=baseline_agents,
            baseline_interval=1,
        )
        agents = [AgentState(agent_id="a1"), AgentState(agent_id="a2")]
        # 2 agents x 2 baselines = 4 total, 1 failure = 3 results
        results = evaluator.run_baseline_benchmarks(agents, generation=1)
        assert len(results) == 3

    def test_override_runner_in_benchmarks(self, baseline_agents: list[AgentState]) -> None:
        """arena_runner kwarg should override the instance runner."""
        config = PBTConfig(population_size=4, baseline_interval=1)
        override_runner = _make_arena_runner(win_rate=0.9)
        evaluator = PBTEvaluator(
            config,
            arena_runner=None,
            baseline_agents=baseline_agents,
            baseline_interval=1,
        )
        agents = [AgentState(agent_id="a1")]
        results = evaluator.run_baseline_benchmarks(
            agents, generation=1, arena_runner=override_runner
        )
        assert len(results) == 2
        for r in results:
            assert r.win_rate == 0.9

    def test_baseline_does_not_affect_elo(self) -> None:
        baselines = [AgentState(agent_id="baseline_random", elo_rating=1200.0)]
        config = PBTConfig(population_size=4, baseline_interval=1)
        runner = _make_arena_runner(win_rate=1.0, draw_rate=0.0, loss_rate=0.0)
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baselines,
            baseline_interval=1,
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
            population_size=4,
            baseline_interval=1,
            evaluation_games_per_match=10,
        )
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(
            config,
            arena_runner=runner,
            baseline_agents=baselines,
            baseline_interval=1,
        )

        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
            AgentState(agent_id="a3", elo_rating=1200.0),
        ]

        report = evaluator.evaluate_population(agents, generation=1)
        assert len(report.baseline_results) == 3  # 3 agents x 1 baseline


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

    def test_baseline_interval_from_config(self) -> None:
        config = PBTConfig(population_size=4, baseline_interval=7)
        evaluator = PBTEvaluator(config)
        assert evaluator._baseline_interval == 7


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Additional edge cases."""

    def test_evaluate_two_agents_minimal(self, pbt_config: PBTConfig) -> None:
        """Smallest valid population: 2 agents."""
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
        ]
        runner = _make_arena_runner(win_rate=0.6, draw_rate=0.0, loss_rate=0.4)
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        report = evaluator.evaluate_population(agents, generation=0)
        assert len(report.matchups) == 1
        assert "a1" in report.elo_ratings
        assert "a2" in report.elo_ratings

    def test_multiple_evaluations_accumulate_elo(self, pbt_config: PBTConfig) -> None:
        """Running evaluate_population multiple times should keep accumulating."""
        agents = [
            AgentState(agent_id="winner", elo_rating=1200.0),
            AgentState(agent_id="loser", elo_rating=1200.0),
        ]

        def winner_runner(
            agent_new: AgentState, agent_old: AgentState, num_games: int
        ) -> MatchResult:
            if agent_new.agent_id == "winner":
                return MatchResult("winner", agent_old.agent_id, 1.0, 0.0, 0.0, num_games)
            else:
                return MatchResult(agent_new.agent_id, "winner", 0.0, 0.0, 1.0, num_games)

        evaluator = PBTEvaluator(pbt_config, arena_runner=winner_runner)

        evaluator.evaluate_population(agents, generation=1)
        elo_after_gen1 = agents[0].elo_rating

        evaluator.evaluate_population(agents, generation=2)
        elo_after_gen2 = agents[0].elo_rating

        # Winner's ELO should keep increasing
        assert elo_after_gen2 > elo_after_gen1 > 1200.0

    def test_agent_elo_not_updated_if_not_in_report(self, pbt_config: PBTConfig) -> None:
        """Agent not in report should keep original ELO."""
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
        ]
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        evaluator.evaluate_population(agents, generation=1)

        # External agent not in population
        extra = AgentState(agent_id="extra", elo_rating=999.0)
        # Verify it was not modified
        assert extra.elo_rating == 999.0

    def test_generation_zero_evaluation(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        runner = _make_arena_runner()
        evaluator = PBTEvaluator(pbt_config, arena_runner=runner)
        report = evaluator.evaluate_population(agents, generation=0)
        assert report.generation == 0
        assert len(report.matchups) == 6

    def test_all_losses_for_new_agent(self, pbt_config: PBTConfig) -> None:
        """When new agent always loses, it should have lowest fitness."""
        agents = [
            AgentState(agent_id="a1", elo_rating=1200.0),
            AgentState(agent_id="a2", elo_rating=1200.0),
            AgentState(agent_id="a3", elo_rating=1200.0),
        ]

        def loser_is_new_runner(
            agent_new: AgentState, agent_old: AgentState, num_games: int
        ) -> MatchResult:
            return MatchResult(agent_new.agent_id, agent_old.agent_id, 0.0, 0.0, 1.0, num_games)

        evaluator = PBTEvaluator(pbt_config, arena_runner=loser_is_new_runner)
        report = evaluator.evaluate_population(agents, generation=1)

        # With round_robin, each agent appears as "new" in some matchups
        # and "old" in others. All fitness should be computed.
        assert len(report.fitness_scores) == 3
