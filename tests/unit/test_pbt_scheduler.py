"""Tests for PBT scheduler and selection strategies.

Tests cover:
- ScheduleType enum values
- SchedulerState dataclass defaults and tracking
- Selection strategies (Truncation, Tournament, Roulette, Rank, SteadyState)
- Selection strategy factory function
- EvaluationResult dataclass and properties
- FitnessEvaluator with callbacks
- PBTScheduler scheduling logic, recording, and selection
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from alpha_zero.pbt.config import AgentState, PBTConfig, SelectionMethod
from alpha_zero.pbt.scheduler import (
    EvaluationResult,
    FitnessEvaluator,
    PBTScheduler,
    RankSelection,
    RouletteSelection,
    SchedulerState,
    ScheduleType,
    SteadyStateSelection,
    TournamentSelection,
    TruncationSelection,
    create_selection_strategy,
)

# =============================================================================
# Constants
# =============================================================================

SEED = 42


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create seeded random number generator for reproducibility."""
    return np.random.default_rng(SEED)


@pytest.fixture
def pbt_config() -> PBTConfig:
    """Create minimal PBT configuration for testing."""
    return PBTConfig(
        population_size=8,
        evaluation_interval=100,
        evaluation_games=5,
        selection_method="truncation",
        truncation_fraction=0.25,
        tournament_size=3,
        checkpoint_interval=500,
    )


@pytest.fixture
def sample_agents() -> list[AgentState]:
    """Create sample agents with varying fitness scores."""
    agents = []
    for i in range(8):
        agent = AgentState(
            agent_id=f"agent_{i}",
            hyperparameters={"learning_rate": 0.001 * (i + 1)},
            fitness_score=0.1 * (i + 1),  # Fitness: 0.1, 0.2, ..., 0.8
        )
        # Add fitness history for smoothed fitness calculations
        for _ in range(5):
            agent.fitness_history.append(agent.fitness_score)
        agents.append(agent)
    return agents


@pytest.fixture
def evaluation_results(sample_agents: list[AgentState]) -> list[EvaluationResult]:
    """Create sample evaluation results."""
    return [
        EvaluationResult(
            agent_id=agent.agent_id,
            fitness=agent.fitness_score,
            games_played=10,
            wins=int(agent.fitness_score * 10),
            losses=10 - int(agent.fitness_score * 10),
        )
        for agent in sample_agents
    ]


# =============================================================================
# ScheduleType Tests
# =============================================================================


class TestScheduleType:
    """Tests for ScheduleType enum."""

    def test_schedule_type_values(self) -> None:
        """Test all schedule type values exist."""
        assert ScheduleType.FIXED_INTERVAL == "fixed_interval"
        assert ScheduleType.ADAPTIVE == "adaptive"
        assert ScheduleType.EPISODE_BASED == "episode_based"
        assert ScheduleType.FITNESS_BASED == "fitness_based"

    def test_schedule_type_from_string(self) -> None:
        """Test creating schedule type from string."""
        assert ScheduleType("fixed_interval") == ScheduleType.FIXED_INTERVAL
        assert ScheduleType("adaptive") == ScheduleType.ADAPTIVE

    def test_schedule_type_is_string(self) -> None:
        """Test schedule type can be used as string."""
        schedule = ScheduleType.FIXED_INTERVAL
        assert isinstance(schedule.value, str)
        assert schedule.value == "fixed_interval"


# =============================================================================
# SchedulerState Tests
# =============================================================================


class TestSchedulerState:
    """Tests for SchedulerState dataclass."""

    def test_default_values(self) -> None:
        """Test default state values."""
        state = SchedulerState()

        assert state.last_evaluation_step == 0
        assert state.last_exploit_explore_step == 0
        assert state.evaluation_count == 0
        assert state.exploit_explore_count == 0
        assert state.fitness_plateau_count == 0
        assert state.best_fitness_seen == float("-inf")

    def test_custom_values(self) -> None:
        """Test custom state initialization."""
        state = SchedulerState(
            last_evaluation_step=100,
            last_exploit_explore_step=50,
            evaluation_count=5,
            exploit_explore_count=3,
            fitness_plateau_count=2,
            best_fitness_seen=0.85,
        )

        assert state.last_evaluation_step == 100
        assert state.last_exploit_explore_step == 50
        assert state.evaluation_count == 5
        assert state.exploit_explore_count == 3
        assert state.fitness_plateau_count == 2
        assert state.best_fitness_seen == 0.85

    def test_state_mutability(self) -> None:
        """Test state can be modified."""
        state = SchedulerState()

        state.last_evaluation_step = 200
        state.evaluation_count = 10
        state.best_fitness_seen = 0.9

        assert state.last_evaluation_step == 200
        assert state.evaluation_count == 10
        assert state.best_fitness_seen == 0.9


# =============================================================================
# Selection Strategy Tests
# =============================================================================


class TestTruncationSelection:
    """Tests for TruncationSelection strategy."""

    def test_selects_top_performers(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test truncation selects highest fitness agents."""
        strategy = TruncationSelection()
        selected = strategy.select(sample_agents, num_to_select=2, rng=rng)

        assert len(selected) == 2
        # Should select agents with highest fitness (0.8 and 0.7)
        selected_ids = {a.agent_id for a in selected}
        assert "agent_7" in selected_ids
        assert "agent_6" in selected_ids

    def test_selects_correct_count(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test truncation selects requested count."""
        strategy = TruncationSelection()

        for count in [1, 3, 5]:
            selected = strategy.select(sample_agents, num_to_select=count, rng=rng)
            assert len(selected) == count

    def test_ordering_by_smoothed_fitness(self, rng: np.random.Generator) -> None:
        """Test selection uses smoothed fitness."""
        agents = [
            AgentState(agent_id="a", fitness_score=0.9, fitness_history=[0.5, 0.5]),
            AgentState(agent_id="b", fitness_score=0.3, fitness_history=[0.8, 0.8]),
        ]
        # Agent 'b' has higher smoothed fitness due to history

        strategy = TruncationSelection()
        selected = strategy.select(agents, num_to_select=1, rng=rng)

        # Should select agent with higher smoothed fitness
        assert selected[0].agent_id == "b"


class TestTournamentSelection:
    """Tests for TournamentSelection strategy."""

    def test_default_tournament_size(self) -> None:
        """Test default tournament size."""
        strategy = TournamentSelection()
        assert strategy.tournament_size == 3

    def test_custom_tournament_size(self) -> None:
        """Test custom tournament size."""
        strategy = TournamentSelection(tournament_size=5)
        assert strategy.tournament_size == 5

    def test_selects_correct_count(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test tournament selects requested count."""
        strategy = TournamentSelection(tournament_size=3)

        selected = strategy.select(sample_agents, num_to_select=4, rng=rng)
        assert len(selected) == 4

    def test_tournament_selection_favors_fit(self, sample_agents: list[AgentState]) -> None:
        """Test tournament selection tends to favor fitter agents."""
        rng = np.random.default_rng(SEED)
        strategy = TournamentSelection(tournament_size=3)

        # Run multiple selections and count how often top agents are selected
        top_agent_ids = {"agent_7", "agent_6", "agent_5"}
        top_count = 0
        total_selections = 100

        for _ in range(total_selections):
            selected = strategy.select(sample_agents, num_to_select=1, rng=rng)
            if selected[0].agent_id in top_agent_ids:
                top_count += 1

        # Top agents should be selected more often than bottom
        assert top_count > total_selections * 0.3

    def test_tournament_handles_small_population(self, rng: np.random.Generator) -> None:
        """Test tournament handles population smaller than tournament size."""
        agents = [
            AgentState(agent_id="a", fitness_score=0.8),
            AgentState(agent_id="b", fitness_score=0.3),
        ]
        strategy = TournamentSelection(tournament_size=5)

        # Should not raise even though tournament_size > len(agents)
        selected = strategy.select(agents, num_to_select=1, rng=rng)
        assert len(selected) == 1


class TestRouletteSelection:
    """Tests for RouletteSelection strategy."""

    def test_selects_correct_count(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test roulette selects requested count."""
        strategy = RouletteSelection()

        selected = strategy.select(sample_agents, num_to_select=3, rng=rng)
        assert len(selected) == 3

    def test_selection_proportional_to_fitness(self, sample_agents: list[AgentState]) -> None:
        """Test selection probability proportional to fitness."""
        rng = np.random.default_rng(SEED)
        strategy = RouletteSelection()

        # Count selections over many iterations
        selection_counts: dict[str, int] = {a.agent_id: 0 for a in sample_agents}

        for _ in range(500):
            selected = strategy.select(sample_agents, num_to_select=1, rng=rng)
            selection_counts[selected[0].agent_id] += 1

        # Higher fitness agents should be selected more often
        assert selection_counts["agent_7"] > selection_counts["agent_0"]

    def test_handles_zero_fitness(self, rng: np.random.Generator) -> None:
        """Test roulette handles agents with near-zero fitness."""
        agents = [
            AgentState(agent_id="a", fitness_score=0.0),
            AgentState(agent_id="b", fitness_score=0.0),
            AgentState(agent_id="c", fitness_score=0.5),
        ]
        # Add fitness history
        for agent in agents:
            agent.fitness_history = [agent.fitness_score]

        strategy = RouletteSelection()
        # Should not raise due to min(0.01, ...) in implementation
        selected = strategy.select(agents, num_to_select=2, rng=rng)
        assert len(selected) == 2


class TestRankSelection:
    """Tests for RankSelection strategy."""

    def test_default_selection_pressure(self) -> None:
        """Test default selection pressure."""
        strategy = RankSelection()
        assert strategy.selection_pressure == 2.0

    def test_custom_selection_pressure(self) -> None:
        """Test custom selection pressure."""
        strategy = RankSelection(selection_pressure=1.5)
        assert strategy.selection_pressure == 1.5

    def test_selects_correct_count(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test rank selection selects requested count."""
        strategy = RankSelection()

        selected = strategy.select(sample_agents, num_to_select=3, rng=rng)
        assert len(selected) == 3

    def test_rank_based_selection(self, sample_agents: list[AgentState]) -> None:
        """Test selection is based on rank not raw fitness."""
        rng = np.random.default_rng(SEED)
        strategy = RankSelection(selection_pressure=2.0)

        selection_counts: dict[str, int] = {a.agent_id: 0 for a in sample_agents}

        for _ in range(500):
            selected = strategy.select(sample_agents, num_to_select=1, rng=rng)
            selection_counts[selected[0].agent_id] += 1

        # Top-ranked agents should be selected more often
        assert selection_counts["agent_7"] > selection_counts["agent_0"]

    def test_single_agent_no_division_by_zero(self, rng: np.random.Generator) -> None:
        """Test RankSelection handles single agent (n=1) without division by zero (Phase 3 fix)."""
        agent = AgentState(agent_id="solo", fitness_score=0.5)
        agent.fitness_history = [0.5]

        strategy = RankSelection()

        # Should not raise ZeroDivisionError
        selected = strategy.select([agent], num_to_select=1, rng=rng)

        assert len(selected) == 1
        assert selected[0].agent_id == "solo"

    def test_empty_population_returns_empty(self, rng: np.random.Generator) -> None:
        """Test RankSelection handles empty population (Phase 3 fix)."""
        strategy = RankSelection()

        selected = strategy.select([], num_to_select=5, rng=rng)

        assert len(selected) == 0

    def test_num_to_select_larger_than_population(self, rng: np.random.Generator) -> None:
        """Test RankSelection handles num_to_select > population size (Phase 3 fix)."""
        agents = [
            AgentState(agent_id="a", fitness_score=0.8),
            AgentState(agent_id="b", fitness_score=0.3),
        ]
        for agent in agents:
            agent.fitness_history = [agent.fitness_score]

        strategy = RankSelection()

        # Request more than available - should return all available
        selected = strategy.select(agents, num_to_select=5, rng=rng)

        assert len(selected) == 2  # Only 2 agents available

    def test_two_agents_no_division_by_zero(self, rng: np.random.Generator) -> None:
        """Test RankSelection handles two agents correctly."""
        agents = [
            AgentState(agent_id="a", fitness_score=0.8),
            AgentState(agent_id="b", fitness_score=0.3),
        ]
        for agent in agents:
            agent.fitness_history = [agent.fitness_score]

        strategy = RankSelection()

        # Should work with n=2 (division by n-1 = 1)
        selected = strategy.select(agents, num_to_select=1, rng=rng)

        assert len(selected) == 1


class TestSteadyStateSelection:
    """Tests for SteadyStateSelection strategy."""

    def test_selects_worst_performers(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test steady-state selects lowest fitness agents."""
        strategy = SteadyStateSelection()
        selected = strategy.select(sample_agents, num_to_select=2, rng=rng)

        assert len(selected) == 2
        # Should select agents with lowest fitness (0.1 and 0.2)
        selected_ids = {a.agent_id for a in selected}
        assert "agent_0" in selected_ids
        assert "agent_1" in selected_ids

    def test_selects_correct_count(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test steady-state selects requested count."""
        strategy = SteadyStateSelection()

        for count in [1, 3, 5]:
            selected = strategy.select(sample_agents, num_to_select=count, rng=rng)
            assert len(selected) == count


# =============================================================================
# Selection Strategy Factory Tests
# =============================================================================


class TestCreateSelectionStrategy:
    """Tests for create_selection_strategy factory function."""

    def test_create_truncation(self) -> None:
        """Test creating truncation strategy."""
        strategy = create_selection_strategy(SelectionMethod.TRUNCATION)
        assert isinstance(strategy, TruncationSelection)

    def test_create_tournament_default(self) -> None:
        """Test creating tournament strategy with defaults."""
        strategy = create_selection_strategy(SelectionMethod.TOURNAMENT)
        assert isinstance(strategy, TournamentSelection)
        assert strategy.tournament_size == 3

    def test_create_tournament_custom_size(self) -> None:
        """Test creating tournament strategy with custom size."""
        strategy = create_selection_strategy(SelectionMethod.TOURNAMENT, tournament_size=5)
        assert isinstance(strategy, TournamentSelection)
        assert strategy.tournament_size == 5

    def test_create_roulette(self) -> None:
        """Test creating roulette strategy."""
        strategy = create_selection_strategy(SelectionMethod.ROULETTE)
        assert isinstance(strategy, RouletteSelection)

    def test_create_rank_default(self) -> None:
        """Test creating rank strategy with defaults."""
        strategy = create_selection_strategy(SelectionMethod.RANK)
        assert isinstance(strategy, RankSelection)
        assert strategy.selection_pressure == 2.0

    def test_create_rank_custom_pressure(self) -> None:
        """Test creating rank strategy with custom pressure."""
        strategy = create_selection_strategy(SelectionMethod.RANK, selection_pressure=1.8)
        assert isinstance(strategy, RankSelection)
        assert strategy.selection_pressure == 1.8


# =============================================================================
# EvaluationResult Tests
# =============================================================================


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self) -> None:
        """Test creating evaluation result."""
        result = EvaluationResult(
            agent_id="agent_1",
            fitness=0.75,
            games_played=20,
            wins=15,
            losses=3,
            draws=2,
        )

        assert result.agent_id == "agent_1"
        assert result.fitness == 0.75
        assert result.games_played == 20
        assert result.wins == 15
        assert result.losses == 3
        assert result.draws == 2
        assert result.timestamp is not None

    def test_default_values(self) -> None:
        """Test default values for optional fields."""
        result = EvaluationResult(agent_id="test", fitness=0.5)

        assert result.games_played == 0
        assert result.wins == 0
        assert result.losses == 0
        assert result.draws == 0
        assert result.details == {}

    def test_win_rate_property(self) -> None:
        """Test win_rate calculation."""
        result = EvaluationResult(
            agent_id="test",
            fitness=0.6,
            games_played=20,
            wins=12,
            losses=8,
        )

        assert result.win_rate == 0.6

    def test_win_rate_zero_games(self) -> None:
        """Test win_rate with no games played."""
        result = EvaluationResult(agent_id="test", fitness=0.0, games_played=0)
        assert result.win_rate == 0.0

    def test_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = EvaluationResult(
            agent_id="agent_1",
            fitness=0.8,
            games_played=10,
            wins=8,
            losses=1,
            draws=1,
            details={"avg_moves": 45},
        )

        result_dict = result.to_dict()

        assert result_dict["agent_id"] == "agent_1"
        assert result_dict["fitness"] == 0.8
        assert result_dict["games_played"] == 10
        assert result_dict["wins"] == 8
        assert result_dict["losses"] == 1
        assert result_dict["draws"] == 1
        assert result_dict["win_rate"] == 0.8
        assert result_dict["details"]["avg_moves"] == 45
        assert "timestamp" in result_dict


# =============================================================================
# FitnessEvaluator Tests
# =============================================================================


class TestFitnessEvaluator:
    """Tests for FitnessEvaluator class."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        evaluator = FitnessEvaluator()

        assert evaluator.games_per_evaluation == 20
        assert evaluator.rng is not None
        assert evaluator._play_game_fn is None
        assert evaluator._get_agent_policy_fn is None

    def test_initialization_custom(self, rng: np.random.Generator) -> None:
        """Test custom initialization."""
        evaluator = FitnessEvaluator(games_per_evaluation=50, rng=rng)

        assert evaluator.games_per_evaluation == 50
        assert evaluator.rng is rng

    def test_set_callbacks(self) -> None:
        """Test setting evaluation callbacks."""
        evaluator = FitnessEvaluator()

        play_fn = MagicMock()
        policy_fn = MagicMock()

        evaluator.set_callbacks(play_game_fn=play_fn, get_agent_policy_fn=policy_fn)

        assert evaluator._play_game_fn is play_fn
        assert evaluator._get_agent_policy_fn is policy_fn

    def test_evaluate_agent_no_callback(self) -> None:
        """Test evaluate_agent returns dummy result without callbacks."""
        evaluator = FitnessEvaluator()
        agent = AgentState(agent_id="test_agent")

        result = evaluator.evaluate_agent(agent)

        assert result.agent_id == "test_agent"
        assert result.fitness == 0.5
        assert result.games_played == 0

    def test_evaluate_agent_with_game_results(self) -> None:
        """Test evaluate_agent with mock game function."""
        evaluator = FitnessEvaluator(games_per_evaluation=10)

        # Mock game that agent wins 7/10 times
        game_count = 0

        def mock_play_game(policy1: Any, policy2: Any) -> tuple[float, float]:
            nonlocal game_count
            game_count += 1
            if game_count <= 7:
                return (1.0, 0.0)  # Agent wins
            return (0.0, 1.0)  # Agent loses

        evaluator.set_callbacks(play_game_fn=mock_play_game)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        assert result.games_played == 10
        assert result.wins == 7
        assert result.losses == 3
        assert result.fitness == 0.7

    def test_evaluate_agent_with_draws(self) -> None:
        """Test evaluate_agent handling draws."""
        evaluator = FitnessEvaluator(games_per_evaluation=10)

        game_results = [(1.0, 0.0)] * 5 + [(0.5, 0.5)] * 3 + [(0.0, 1.0)] * 2
        result_idx = 0

        def mock_play_game(policy1: Any, policy2: Any) -> tuple[float, float]:
            nonlocal result_idx
            result = game_results[result_idx]
            result_idx += 1
            return result

        evaluator.set_callbacks(play_game_fn=mock_play_game)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        assert result.wins == 5
        assert result.draws == 3
        assert result.losses == 2
        # Fitness = (wins + 0.5 * draws) / games = (5 + 1.5) / 10 = 0.65
        assert result.fitness == pytest.approx(0.65)

    def test_evaluate_agent_handles_game_errors(self) -> None:
        """Test evaluate_agent handles game exceptions."""
        evaluator = FitnessEvaluator(games_per_evaluation=5)

        call_count = 0

        def mock_play_game(policy1: Any, policy2: Any) -> tuple[float, float]:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Game crashed")
            return (1.0, 0.0)

        evaluator.set_callbacks(play_game_fn=mock_play_game)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        # Should have 4 successful games (one failed)
        assert result.games_played == 4

    def test_evaluate_population(
        self, sample_agents: list[AgentState], rng: np.random.Generator
    ) -> None:
        """Test evaluating entire population."""
        evaluator = FitnessEvaluator(games_per_evaluation=3, rng=rng)

        def mock_play_game(policy1: Any, policy2: Any) -> tuple[float, float]:
            return (0.5, 0.5)  # All draws

        evaluator.set_callbacks(play_game_fn=mock_play_game)

        results = evaluator.evaluate_population(sample_agents[:3])

        assert len(results) == 3
        assert all(r.games_played == 3 for r in results)


# =============================================================================
# PBTScheduler Tests
# =============================================================================


class TestPBTScheduler:
    """Tests for PBTScheduler class."""

    def test_initialization(self, pbt_config: PBTConfig, rng: np.random.Generator) -> None:
        """Test scheduler initialization."""
        scheduler = PBTScheduler(config=pbt_config, rng=rng)

        assert scheduler.config is pbt_config
        assert scheduler.rng is rng
        assert isinstance(scheduler._state, SchedulerState)
        assert scheduler._selection_strategy is not None
        assert scheduler._evaluator is not None

    def test_initialization_creates_selection_strategy(self, rng: np.random.Generator) -> None:
        """Test scheduler creates correct selection strategy."""
        config = PBTConfig(selection_method="tournament", tournament_size=5)
        scheduler = PBTScheduler(config=config, rng=rng)

        assert isinstance(scheduler._selection_strategy, TournamentSelection)
        assert scheduler._selection_strategy.tournament_size == 5

    def test_should_evaluate_initial(self, pbt_config: PBTConfig) -> None:
        """Test should_evaluate at start of training."""
        scheduler = PBTScheduler(config=pbt_config)

        # At step 0, last_evaluation_step is 0, so no evaluation needed yet
        assert scheduler.should_evaluate(0) is False
        assert scheduler.should_evaluate(50) is False
        assert scheduler.should_evaluate(100) is True

    def test_should_evaluate_after_recording(self, pbt_config: PBTConfig) -> None:
        """Test should_evaluate respects recorded evaluations."""
        scheduler = PBTScheduler(config=pbt_config)

        # Record evaluation at step 100
        scheduler.record_evaluation(100, [])

        assert scheduler.should_evaluate(100) is False
        assert scheduler.should_evaluate(150) is False
        assert scheduler.should_evaluate(200) is True

    def test_should_exploit_explore_requires_evaluation(self, pbt_config: PBTConfig) -> None:
        """Test should_exploit_explore requires prior evaluation."""
        scheduler = PBTScheduler(config=pbt_config)

        # Should not exploit/explore before any evaluations
        assert scheduler.should_exploit_explore(100) is False
        assert scheduler.should_exploit_explore(200) is False

        # After evaluation, exploit/explore should be allowed
        scheduler.record_evaluation(100, [])
        assert scheduler.should_exploit_explore(200) is True

    def test_should_checkpoint(self, pbt_config: PBTConfig) -> None:
        """Test should_checkpoint logic."""
        scheduler = PBTScheduler(config=pbt_config)

        assert scheduler.should_checkpoint(current_step=0, last_checkpoint_step=0) is False
        assert scheduler.should_checkpoint(current_step=400, last_checkpoint_step=0) is False
        assert scheduler.should_checkpoint(current_step=500, last_checkpoint_step=0) is True
        assert scheduler.should_checkpoint(current_step=600, last_checkpoint_step=500) is False
        assert scheduler.should_checkpoint(current_step=1000, last_checkpoint_step=500) is True

    def test_record_evaluation_updates_state(
        self, pbt_config: PBTConfig, evaluation_results: list[EvaluationResult]
    ) -> None:
        """Test record_evaluation updates scheduler state."""
        scheduler = PBTScheduler(config=pbt_config)

        initial_state = scheduler.get_state()
        assert initial_state.evaluation_count == 0
        assert initial_state.last_evaluation_step == 0

        scheduler.record_evaluation(100, evaluation_results)

        state = scheduler.get_state()
        assert state.evaluation_count == 1
        assert state.last_evaluation_step == 100
        assert state.best_fitness_seen == 0.8  # Max fitness in results

    def test_record_evaluation_tracks_plateau(self, pbt_config: PBTConfig) -> None:
        """Test record_evaluation tracks fitness plateau."""
        scheduler = PBTScheduler(config=pbt_config)

        # First evaluation sets best fitness
        results_high = [EvaluationResult(agent_id="a", fitness=0.8)]
        scheduler.record_evaluation(100, results_high)
        assert scheduler.get_state().fitness_plateau_count == 0

        # Same or lower fitness increments plateau count
        results_same = [EvaluationResult(agent_id="a", fitness=0.8)]
        scheduler.record_evaluation(200, results_same)
        assert scheduler.get_state().fitness_plateau_count == 1

        results_lower = [EvaluationResult(agent_id="a", fitness=0.7)]
        scheduler.record_evaluation(300, results_lower)
        assert scheduler.get_state().fitness_plateau_count == 2

        # Higher fitness resets plateau count
        results_higher = [EvaluationResult(agent_id="a", fitness=0.9)]
        scheduler.record_evaluation(400, results_higher)
        assert scheduler.get_state().fitness_plateau_count == 0

    def test_record_exploit_explore(self, pbt_config: PBTConfig) -> None:
        """Test record_exploit_explore updates state."""
        scheduler = PBTScheduler(config=pbt_config)

        scheduler.record_exploit_explore(150)

        state = scheduler.get_state()
        assert state.exploit_explore_count == 1
        assert state.last_exploit_explore_step == 150

    def test_select_for_exploitation(
        self, pbt_config: PBTConfig, sample_agents: list[AgentState]
    ) -> None:
        """Test select_for_exploitation uses selection strategy."""
        scheduler = PBTScheduler(config=pbt_config)

        selected = scheduler.select_for_exploitation(sample_agents, num_to_select=2)

        assert len(selected) == 2
        # With truncation, should select top fitness
        selected_ids = {a.agent_id for a in selected}
        assert "agent_7" in selected_ids

    def test_select_for_replacement(
        self, pbt_config: PBTConfig, sample_agents: list[AgentState]
    ) -> None:
        """Test select_for_replacement selects worst agents."""
        scheduler = PBTScheduler(config=pbt_config)

        to_replace = scheduler.select_for_replacement(sample_agents, num_to_replace=2)

        assert len(to_replace) == 2
        # Should select worst fitness agents
        replaced_ids = {a.agent_id for a in to_replace}
        assert "agent_0" in replaced_ids
        assert "agent_1" in replaced_ids

    def test_get_evaluator(self, pbt_config: PBTConfig) -> None:
        """Test get_evaluator returns evaluator instance."""
        scheduler = PBTScheduler(config=pbt_config)

        evaluator = scheduler.get_evaluator()

        assert isinstance(evaluator, FitnessEvaluator)
        assert evaluator.games_per_evaluation == pbt_config.evaluation_games

    def test_get_state(self, pbt_config: PBTConfig) -> None:
        """Test get_state returns current state."""
        scheduler = PBTScheduler(config=pbt_config)

        state = scheduler.get_state()

        assert isinstance(state, SchedulerState)
        assert state.evaluation_count == 0

    def test_is_plateau_detected(self, pbt_config: PBTConfig) -> None:
        """Test is_plateau_detected with custom threshold."""
        scheduler = PBTScheduler(config=pbt_config)

        # Initially no plateau
        assert scheduler.is_plateau_detected(threshold=5) is False

        # Simulate plateau by directly modifying state
        scheduler._state.fitness_plateau_count = 5
        assert scheduler.is_plateau_detected(threshold=5) is True
        assert scheduler.is_plateau_detected(threshold=10) is False

    def test_reset_state(self, pbt_config: PBTConfig) -> None:
        """Test reset_state clears scheduler state."""
        scheduler = PBTScheduler(config=pbt_config)

        # Modify state
        scheduler._state.evaluation_count = 10
        scheduler._state.best_fitness_seen = 0.9
        scheduler._state.fitness_plateau_count = 5

        scheduler.reset_state()

        state = scheduler.get_state()
        assert state.evaluation_count == 0
        assert state.best_fitness_seen == float("-inf")
        assert state.fitness_plateau_count == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestSchedulerIntegration:
    """Integration tests for scheduler workflow."""

    def test_full_training_cycle(
        self,
        pbt_config: PBTConfig,
        sample_agents: list[AgentState],
        rng: np.random.Generator,
    ) -> None:
        """Test complete training cycle with scheduler."""
        scheduler = PBTScheduler(config=pbt_config, rng=rng)

        # Step 0: No actions needed
        assert scheduler.should_evaluate(0) is False
        assert scheduler.should_exploit_explore(0) is False
        assert scheduler.should_checkpoint(0, 0) is False

        # Step 100: Evaluation needed
        assert scheduler.should_evaluate(100) is True

        # Perform evaluation
        results = [
            EvaluationResult(agent_id=a.agent_id, fitness=a.fitness_score) for a in sample_agents
        ]
        scheduler.record_evaluation(100, results)

        # After evaluation, exploit/explore becomes available
        # Note: exploit/explore is based on steps since last_exploit_explore_step (0)
        # At step 100, steps_since_ee = 100 >= evaluation_interval (100)
        assert scheduler.should_exploit_explore(100) is True

        # Perform exploit/explore
        top_agents = scheduler.select_for_exploitation(sample_agents, num_to_select=2)
        worst_agents = scheduler.select_for_replacement(sample_agents, num_to_replace=2)

        assert len(top_agents) == 2
        assert len(worst_agents) == 2

        # Top and worst should be different agents
        top_ids = {a.agent_id for a in top_agents}
        worst_ids = {a.agent_id for a in worst_agents}
        assert top_ids.isdisjoint(worst_ids)

        scheduler.record_exploit_explore(200)

        # Verify state after cycle
        state = scheduler.get_state()
        assert state.evaluation_count == 1
        assert state.exploit_explore_count == 1

    def test_multiple_evaluation_cycles(
        self, pbt_config: PBTConfig, rng: np.random.Generator
    ) -> None:
        """Test multiple evaluation cycles track fitness correctly."""
        scheduler = PBTScheduler(config=pbt_config, rng=rng)

        fitnesses = [0.5, 0.6, 0.65, 0.65, 0.65]

        for i, fitness in enumerate(fitnesses):
            step = (i + 1) * 100
            results = [EvaluationResult(agent_id="agent_0", fitness=fitness)]
            scheduler.record_evaluation(step, results)

        state = scheduler.get_state()
        assert state.evaluation_count == 5
        assert state.best_fitness_seen == 0.65
        assert state.fitness_plateau_count == 2  # Last two were plateaus


# =============================================================================
# Additional FitnessEvaluator Tests
# =============================================================================


class TestFitnessEvaluatorAdvanced:
    """Advanced tests for FitnessEvaluator with policy callbacks and opponents."""

    def test_evaluate_agent_with_policy_fn(self) -> None:
        """Test evaluate_agent uses get_agent_policy_fn when set."""
        evaluator = FitnessEvaluator(games_per_evaluation=5)

        policy_fn = MagicMock(return_value="mock_policy")

        def mock_play(p1: Any, p2: Any) -> tuple[float, float]:
            return (1.0, 0.0)

        evaluator.set_callbacks(play_game_fn=mock_play, get_agent_policy_fn=policy_fn)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        # policy_fn should have been called for the agent
        policy_fn.assert_called()
        assert result.wins == 5
        assert result.fitness == 1.0

    def test_evaluate_agent_with_opponent_agents(self) -> None:
        """Test evaluate_agent with explicit opponent agents."""
        evaluator = FitnessEvaluator(games_per_evaluation=5, rng=np.random.default_rng(42))

        policy_fn = MagicMock(return_value="mock_policy")

        def mock_play(p1: Any, p2: Any) -> tuple[float, float]:
            return (1.0, 0.0)

        evaluator.set_callbacks(play_game_fn=mock_play, get_agent_policy_fn=policy_fn)

        agent = AgentState(agent_id="test_agent")
        opponents = [AgentState(agent_id=f"opp_{i}") for i in range(3)]

        result = evaluator.evaluate_agent(agent, opponent_agents=opponents)

        assert result.wins == 5
        # policy_fn should have been called for agent + opponents
        assert policy_fn.call_count >= 5

    def test_evaluate_agent_with_opponent_policy(self) -> None:
        """Test evaluate_agent with explicit opponent policy function."""
        evaluator = FitnessEvaluator(games_per_evaluation=3)

        def mock_play(p1: Any, p2: Any) -> tuple[float, float]:
            return (0.0, 1.0)  # Agent always loses

        evaluator.set_callbacks(play_game_fn=mock_play)

        agent = AgentState(agent_id="test_agent")
        opponent_policy = MagicMock()

        result = evaluator.evaluate_agent(agent, opponent_policy=opponent_policy)

        assert result.losses == 3
        assert result.fitness == 0.0

    def test_evaluate_agent_all_games_fail(self) -> None:
        """Test evaluate_agent when all games throw exceptions."""
        evaluator = FitnessEvaluator(games_per_evaluation=5)

        def always_fail(p1: Any, p2: Any) -> tuple[float, float]:
            raise RuntimeError("Game crash")

        evaluator.set_callbacks(play_game_fn=always_fail)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        assert result.games_played == 0
        assert result.fitness == 0.0

    def test_evaluate_population_empty_list(self) -> None:
        """Test evaluating empty population."""
        evaluator = FitnessEvaluator(games_per_evaluation=3)

        def mock_play(p1: Any, p2: Any) -> tuple[float, float]:
            return (0.5, 0.5)

        evaluator.set_callbacks(play_game_fn=mock_play)

        results = evaluator.evaluate_population([])
        assert len(results) == 0

    def test_evaluate_agent_total_reward_tracked(self) -> None:
        """Test that total reward is tracked in evaluation details."""
        evaluator = FitnessEvaluator(games_per_evaluation=3)

        rewards = [1.0, 0.5, 0.0]
        game_idx = 0

        def mock_play(p1: Any, p2: Any) -> tuple[float, float]:
            nonlocal game_idx
            r = rewards[game_idx]
            game_idx += 1
            return (r, 1.0 - r)

        evaluator.set_callbacks(play_game_fn=mock_play)

        agent = AgentState(agent_id="test_agent")
        result = evaluator.evaluate_agent(agent)

        assert result.details["total_reward"] == pytest.approx(1.5)


# =============================================================================
# Additional Selection Strategy Edge Cases
# =============================================================================


class TestSelectionEdgeCases:
    """Additional edge case tests for selection strategies."""

    def test_truncation_all_same_fitness(self, rng: np.random.Generator) -> None:
        """Test truncation when all agents have same fitness."""
        agents = [
            AgentState(agent_id=f"agent_{i}", fitness_score=0.5)
            for i in range(5)
        ]
        for a in agents:
            a.fitness_history = [0.5]

        strategy = TruncationSelection()
        selected = strategy.select(agents, num_to_select=2, rng=rng)
        assert len(selected) == 2

    def test_steady_state_all_same_fitness(self, rng: np.random.Generator) -> None:
        """Test steady state when all agents have same fitness."""
        agents = [
            AgentState(agent_id=f"agent_{i}", fitness_score=0.5)
            for i in range(5)
        ]
        for a in agents:
            a.fitness_history = [0.5]

        strategy = SteadyStateSelection()
        selected = strategy.select(agents, num_to_select=2, rng=rng)
        assert len(selected) == 2

    def test_roulette_single_agent(self, rng: np.random.Generator) -> None:
        """Test roulette with single agent."""
        agents = [AgentState(agent_id="solo", fitness_score=0.8)]
        agents[0].fitness_history = [0.8]

        strategy = RouletteSelection()
        selected = strategy.select(agents, num_to_select=1, rng=rng)
        assert len(selected) == 1
        assert selected[0].agent_id == "solo"

    def test_tournament_single_agent(self, rng: np.random.Generator) -> None:
        """Test tournament with single agent."""
        agents = [AgentState(agent_id="solo", fitness_score=0.8)]
        agents[0].fitness_history = [0.8]

        strategy = TournamentSelection(tournament_size=3)
        selected = strategy.select(agents, num_to_select=1, rng=rng)
        assert len(selected) == 1
        assert selected[0].agent_id == "solo"


# =============================================================================
# Record Empty Evaluation
# =============================================================================


class TestSchedulerEmptyResults:
    """Tests for scheduler with empty evaluation results."""

    def test_record_evaluation_empty_results(self, pbt_config: PBTConfig) -> None:
        """Test recording evaluation with empty results list."""
        scheduler = PBTScheduler(config=pbt_config)

        scheduler.record_evaluation(100, [])

        state = scheduler.get_state()
        assert state.evaluation_count == 1
        assert state.best_fitness_seen == float("-inf")

    def test_record_evaluation_single_result(self, pbt_config: PBTConfig) -> None:
        """Test recording evaluation with single result."""
        scheduler = PBTScheduler(config=pbt_config)

        results = [EvaluationResult(agent_id="a", fitness=0.42)]
        scheduler.record_evaluation(100, results)

        state = scheduler.get_state()
        assert state.best_fitness_seen == 0.42
        assert state.fitness_plateau_count == 0


# =============================================================================
# Scheduler With Different Selection Methods
# =============================================================================


class TestSchedulerSelectionMethods:
    """Test PBTScheduler with different selection method configurations."""

    @pytest.mark.parametrize("method", ["truncation", "tournament", "roulette", "rank"])
    def test_scheduler_with_selection_method(
        self,
        method: str,
        sample_agents: list[AgentState],
    ) -> None:
        """Test scheduler works with all selection methods."""
        config = PBTConfig(
            population_size=8,
            selection_method=method,
            tournament_size=3,
            evaluation_interval=100,
        )
        scheduler = PBTScheduler(config=config, rng=np.random.default_rng(42))

        selected = scheduler.select_for_exploitation(sample_agents, num_to_select=2)
        assert len(selected) == 2
