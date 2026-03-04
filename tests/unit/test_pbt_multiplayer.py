"""Unit tests for PBT-Multiplayer Integration (Phase 21).

Tests cover:
  - NPlayerMatchResult creation and fields
  - MultiplayerPBTEvaluator matchup creation and population evaluation
  - PolicyDiversityTracker action recording and diversity computation
  - DiversityMetrics fields
  - Diversity bonus calculation
  - PBTConfig multiplayer fields
  - PopulationManager.evolve_multiplayer()
  - Parametrised tests for different num_players
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.diversity_tracker import DiversityMetrics, PolicyDiversityTracker
from alpha_zero.pbt.multiplayer_evaluator import (
    MultiplayerPBTEvaluator,
    NPlayerArenaRunner,
    NPlayerMatchResult,
)
from alpha_zero.pbt.population import PopulationManager

# ---------------------------------------------------------------------------
# Mock arena runner
# ---------------------------------------------------------------------------


class MockNPlayerArenaRunner:
    """Mock arena runner that produces deterministic N-player results."""

    def __init__(self, winner_index: int = 0) -> None:
        """Initialize with a fixed winner index (within the group).

        Args:
            winner_index: Index in the agent_ids list that wins.
        """
        self._winner_index = winner_index
        self.call_count = 0
        self.last_agent_ids: list[str] | None = None

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        self.call_count += 1
        self.last_agent_ids = agent_ids
        results: list[NPlayerMatchResult] = []
        n = len(agent_ids)

        for _ in range(num_games):
            winner_idx = min(self._winner_index, n - 1)
            rankings: dict[str, int] = {}
            scores: dict[str, float] = {}
            for i, aid in enumerate(agent_ids):
                if i == winner_idx:
                    rankings[aid] = 1
                    scores[aid] = 1.0
                else:
                    rankings[aid] = 2
                    scores[aid] = 0.0

            results.append(
                NPlayerMatchResult(
                    agent_ids=list(agent_ids),
                    rankings=rankings,
                    scores=scores,
                    game_length=50,
                )
            )
        return results


class RotatingWinnerRunner:
    """Mock runner where the winner rotates each game."""

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        results: list[NPlayerMatchResult] = []
        n = len(agent_ids)
        for g in range(num_games):
            winner_idx = g % n
            rankings: dict[str, int] = {}
            scores: dict[str, float] = {}
            for i, aid in enumerate(agent_ids):
                if i == winner_idx:
                    rankings[aid] = 1
                    scores[aid] = 1.0
                else:
                    rankings[aid] = 2
                    scores[aid] = 0.0
            results.append(
                NPlayerMatchResult(
                    agent_ids=list(agent_ids),
                    rankings=rankings,
                    scores=scores,
                    game_length=30,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agents_6() -> list[AgentState]:
    """Create a population of 6 agents."""
    return [AgentState(agent_id=f"agent_{i}") for i in range(6)]


@pytest.fixture
def agents_4() -> list[AgentState]:
    """Create a population of 4 agents."""
    return [AgentState(agent_id=f"a{i}") for i in range(4)]


@pytest.fixture
def mock_runner() -> MockNPlayerArenaRunner:
    """Create a mock arena runner with first agent as winner."""
    return MockNPlayerArenaRunner(winner_index=0)


@pytest.fixture
def rotating_runner() -> RotatingWinnerRunner:
    """Create a rotating winner runner."""
    return RotatingWinnerRunner()


@pytest.fixture
def diversity_tracker() -> PolicyDiversityTracker:
    """Create a diversity tracker with action space size 10."""
    return PolicyDiversityTracker(action_space_size=10)


@pytest.fixture
def multiplayer_config() -> PBTConfig:
    """Create a PBT config with multiplayer enabled."""
    return PBTConfig(
        population_size=6,
        multiplayer_mode=True,
        num_players=3,
        diversity_weight=0.1,
        evaluation_games_per_match=4,
    )


# ---------------------------------------------------------------------------
# NPlayerMatchResult tests
# ---------------------------------------------------------------------------


class TestNPlayerMatchResult:
    """Tests for NPlayerMatchResult dataclass."""

    def test_creation_basic(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 2, "c": 3},
            scores={"a": 10.0, "b": 5.0, "c": 2.0},
            game_length=100,
        )
        assert result.agent_ids == ["a", "b", "c"]
        assert result.rankings["a"] == 1
        assert result.scores["a"] == 10.0
        assert result.game_length == 100

    def test_default_game_length(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["x", "y"],
            rankings={"x": 1, "y": 2},
            scores={"x": 1.0, "y": 0.0},
        )
        assert result.game_length == 0

    def test_frozen(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={"a": 1.0, "b": 0.0},
        )
        with pytest.raises(AttributeError):
            result.game_length = 999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# MultiplayerPBTEvaluator matchup tests
# ---------------------------------------------------------------------------


class TestMultiplayerMatchupCreation:
    """Tests for matchup creation in MultiplayerPBTEvaluator."""

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_matchups_cover_all_agents(
        self, num_players: int, agents_6: list[AgentState]
    ) -> None:
        evaluator = MultiplayerPBTEvaluator(
            num_players=num_players, games_per_eval=2
        )
        agent_ids = [a.agent_id for a in agents_6]
        matchups = evaluator._create_matchups(agent_ids)

        # Every agent should appear in at least one matchup
        seen = set()
        for group in matchups:
            seen.update(group)
        assert seen == set(agent_ids)

    def test_matchup_group_size(self, agents_6: list[AgentState]) -> None:
        evaluator = MultiplayerPBTEvaluator(num_players=3, games_per_eval=2)
        agent_ids = [a.agent_id for a in agents_6]
        matchups = evaluator._create_matchups(agent_ids)

        for group in matchups:
            assert len(group) == 3

    def test_matchups_all_unique(self, agents_4: list[AgentState]) -> None:
        evaluator = MultiplayerPBTEvaluator(num_players=2, games_per_eval=1)
        agent_ids = [a.agent_id for a in agents_4]
        matchups = evaluator._create_matchups(agent_ids)

        # With 4 agents choosing 2, C(4,2)=6 unique pairs
        group_tuples = [tuple(sorted(g)) for g in matchups]
        assert len(group_tuples) == len(set(group_tuples))

    def test_matchups_empty_when_too_few_agents(self) -> None:
        evaluator = MultiplayerPBTEvaluator(num_players=4, games_per_eval=1)
        matchups = evaluator._create_matchups(["a", "b"])
        assert matchups == []

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_equal_games_per_agent(
        self, num_players: int, agents_6: list[AgentState]
    ) -> None:
        """All agents should appear in a similar number of matchups."""
        evaluator = MultiplayerPBTEvaluator(
            num_players=num_players, games_per_eval=1
        )
        agent_ids = [a.agent_id for a in agents_6]
        matchups = evaluator._create_matchups(agent_ids)

        counts: dict[str, int] = dict.fromkeys(agent_ids, 0)
        for group in matchups:
            for aid in group:
                counts[aid] += 1

        # All agents should appear in the same number of matchups
        # (since we use all combinations of size num_players from 6 agents,
        # each agent appears in C(5, num_players-1) matchups).
        values = list(counts.values())
        assert min(values) == max(values)


# ---------------------------------------------------------------------------
# MultiplayerPBTEvaluator evaluation tests
# ---------------------------------------------------------------------------


class TestMultiplayerEvaluation:
    """Tests for evaluate_population in MultiplayerPBTEvaluator."""

    def test_basic_evaluation(
        self,
        agents_6: list[AgentState],
        mock_runner: MockNPlayerArenaRunner,
    ) -> None:
        evaluator = MultiplayerPBTEvaluator(
            num_players=3, games_per_eval=2
        )
        fitness = evaluator.evaluate_population(agents_6, mock_runner)

        assert len(fitness) == 6
        for aid in [a.agent_id for a in agents_6]:
            assert aid in fitness
            assert 0.0 <= fitness[aid] <= 1.0

    def test_clear_winner_highest_fitness(
        self,
        agents_4: list[AgentState],
    ) -> None:
        """Agent at index 0 always wins -> should have highest fitness."""
        runner = MockNPlayerArenaRunner(winner_index=0)
        evaluator = MultiplayerPBTEvaluator(
            num_players=2, games_per_eval=4
        )
        fitness = evaluator.evaluate_population(agents_4, runner)

        # a0 should be the highest or tied for highest
        max_fitness = max(fitness.values())
        assert fitness["a0"] == pytest.approx(max_fitness, abs=0.01)

    def test_rotating_winner_balanced(
        self,
        agents_4: list[AgentState],
        rotating_runner: RotatingWinnerRunner,
    ) -> None:
        """With rotating winners, all agents should have similar fitness."""
        evaluator = MultiplayerPBTEvaluator(
            num_players=2, games_per_eval=100
        )
        fitness = evaluator.evaluate_population(agents_4, rotating_runner)

        values = list(fitness.values())
        # With enough games and rotation, should be roughly equal
        assert max(values) - min(values) < 0.5

    def test_too_few_agents_returns_uniform(self) -> None:
        runner = MockNPlayerArenaRunner()
        evaluator = MultiplayerPBTEvaluator(num_players=4, games_per_eval=1)
        agents = [AgentState(agent_id="solo")]
        fitness = evaluator.evaluate_population(agents, runner)
        assert fitness["solo"] == pytest.approx(0.5)

    def test_runner_is_called(
        self,
        agents_4: list[AgentState],
        mock_runner: MockNPlayerArenaRunner,
    ) -> None:
        evaluator = MultiplayerPBTEvaluator(
            num_players=2, games_per_eval=1
        )
        evaluator.evaluate_population(agents_4, mock_runner)
        assert mock_runner.call_count > 0

    @pytest.mark.parametrize("num_players", [2, 3, 4])
    def test_various_num_players(
        self, num_players: int, agents_6: list[AgentState]
    ) -> None:
        runner = MockNPlayerArenaRunner(winner_index=0)
        evaluator = MultiplayerPBTEvaluator(
            num_players=num_players, games_per_eval=2
        )
        fitness = evaluator.evaluate_population(agents_6, runner)
        assert len(fitness) == 6
        assert all(0.0 <= v <= 1.0 for v in fitness.values())


# ---------------------------------------------------------------------------
# MultiplayerPBTEvaluator validation tests
# ---------------------------------------------------------------------------


class TestMultiplayerEvaluatorValidation:
    """Tests for constructor validation."""

    def test_invalid_num_players(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            MultiplayerPBTEvaluator(num_players=1)

    def test_invalid_games_per_eval(self) -> None:
        with pytest.raises(ValueError, match="games_per_eval"):
            MultiplayerPBTEvaluator(games_per_eval=0)

    def test_properties(self) -> None:
        evaluator = MultiplayerPBTEvaluator(num_players=3, games_per_eval=5)
        assert evaluator.num_players == 3
        assert evaluator.games_per_eval == 5


# ---------------------------------------------------------------------------
# PolicyDiversityTracker tests
# ---------------------------------------------------------------------------


class TestPolicyDiversityTracker:
    """Tests for PolicyDiversityTracker."""

    def test_record_actions(self, diversity_tracker: PolicyDiversityTracker) -> None:
        diversity_tracker.record_actions("agent_1", [0, 1, 2, 0, 1])
        assert "agent_1" in diversity_tracker.tracked_agents

    def test_record_multiple_agents(
        self, diversity_tracker: PolicyDiversityTracker
    ) -> None:
        diversity_tracker.record_actions("a1", [0, 1])
        diversity_tracker.record_actions("a2", [3, 4])
        assert len(diversity_tracker.tracked_agents) == 2

    def test_accumulates_actions(
        self, diversity_tracker: PolicyDiversityTracker
    ) -> None:
        diversity_tracker.record_actions("a1", [0, 0, 0])
        diversity_tracker.record_actions("a1", [1, 1])
        dist = diversity_tracker._get_distribution("a1")
        # 3 zeros and 2 ones out of 5 total
        assert dist[0] == pytest.approx(0.6)
        assert dist[1] == pytest.approx(0.4)

    def test_out_of_range_actions_ignored(
        self, diversity_tracker: PolicyDiversityTracker
    ) -> None:
        diversity_tracker.record_actions("a1", [0, 100, -1, 5])
        dist = diversity_tracker._get_distribution("a1")
        total = dist.sum()
        assert total == pytest.approx(1.0)
        # Only action 0 and 5 should be counted
        assert dist[0] > 0
        assert dist[5] > 0
        assert dist[1] == 0

    def test_unknown_agent_uniform_distribution(
        self, diversity_tracker: PolicyDiversityTracker
    ) -> None:
        dist = diversity_tracker._get_distribution("unknown")
        assert dist.sum() == pytest.approx(1.0)
        assert np.allclose(dist, 1.0 / 10)

    def test_reset(self, diversity_tracker: PolicyDiversityTracker) -> None:
        diversity_tracker.record_actions("a1", [0, 1])
        diversity_tracker.reset()
        assert len(diversity_tracker.tracked_agents) == 0

    def test_invalid_action_space_size(self) -> None:
        with pytest.raises(ValueError, match="action_space_size"):
            PolicyDiversityTracker(action_space_size=0)


# ---------------------------------------------------------------------------
# DiversityMetrics tests
# ---------------------------------------------------------------------------


class TestDiversityMetrics:
    """Tests for DiversityMetrics dataclass."""

    def test_default_values(self) -> None:
        metrics = DiversityMetrics()
        assert metrics.action_entropy == 0.0
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0

    def test_custom_values(self) -> None:
        metrics = DiversityMetrics(
            action_entropy=1.5,
            pairwise_kl_divergence=0.8,
            behavioral_diversity=0.6,
        )
        assert metrics.action_entropy == 1.5
        assert metrics.pairwise_kl_divergence == 0.8
        assert metrics.behavioral_diversity == 0.6


# ---------------------------------------------------------------------------
# Diversity computation tests
# ---------------------------------------------------------------------------


class TestDiversityComputation:
    """Tests for compute_diversity() method."""

    def test_no_agents_returns_zeros(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        metrics = tracker.compute_diversity()
        assert metrics.action_entropy == 0.0
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0

    def test_single_agent_zero_kl(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        tracker.record_actions("a1", [0, 1, 2, 3, 4])
        metrics = tracker.compute_diversity()
        assert metrics.action_entropy > 0
        assert metrics.pairwise_kl_divergence == 0.0

    def test_identical_agents_zero_kl(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        tracker.record_actions("a1", [0, 1, 2, 3, 4])
        tracker.record_actions("a2", [0, 1, 2, 3, 4])
        metrics = tracker.compute_diversity()
        assert metrics.pairwise_kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_diverse_agents_positive_kl(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=10)
        # Agent 1 only uses low actions
        tracker.record_actions("a1", [0, 0, 0, 1, 1])
        # Agent 2 only uses high actions
        tracker.record_actions("a2", [8, 8, 8, 9, 9])
        metrics = tracker.compute_diversity()
        assert metrics.pairwise_kl_divergence > 0.5
        assert metrics.behavioral_diversity > 0

    def test_entropy_increases_with_spread(self) -> None:
        tracker1 = PolicyDiversityTracker(action_space_size=10)
        tracker1.record_actions("a1", [0, 0, 0, 0, 0])  # concentrated
        metrics1 = tracker1.compute_diversity()

        tracker2 = PolicyDiversityTracker(action_space_size=10)
        tracker2.record_actions("a1", [0, 1, 2, 3, 4])  # spread out
        metrics2 = tracker2.compute_diversity()

        assert metrics2.action_entropy > metrics1.action_entropy


# ---------------------------------------------------------------------------
# Diversity bonus tests
# ---------------------------------------------------------------------------


class TestDiversityBonus:
    """Tests for get_diversity_bonus() method."""

    def test_no_bonus_single_agent(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        tracker.record_actions("a1", [0, 1, 2])
        bonus = tracker.get_diversity_bonus("a1", weight=0.1)
        assert bonus == 0.0

    def test_no_bonus_unknown_agent(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        tracker.record_actions("a1", [0, 1])
        tracker.record_actions("a2", [3, 4])
        bonus = tracker.get_diversity_bonus("unknown", weight=0.1)
        assert bonus == 0.0

    def test_bonus_positive_for_diverse_agent(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=10)
        # Two agents with similar behaviour
        tracker.record_actions("a1", [0, 0, 0, 1, 1])
        tracker.record_actions("a2", [0, 0, 1, 1, 1])
        # One agent with very different behaviour
        tracker.record_actions("a3", [8, 8, 8, 9, 9])

        bonus_a1 = tracker.get_diversity_bonus("a1", weight=0.1)
        bonus_a3 = tracker.get_diversity_bonus("a3", weight=0.1)

        # a3 should get a higher bonus (more different from mean)
        assert bonus_a3 > bonus_a1
        assert bonus_a3 > 0

    def test_bonus_scales_with_weight(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=10)
        tracker.record_actions("a1", [0, 0, 0])
        tracker.record_actions("a2", [9, 9, 9])

        bonus_low = tracker.get_diversity_bonus("a1", weight=0.01)
        bonus_high = tracker.get_diversity_bonus("a1", weight=1.0)

        assert bonus_high > bonus_low

    def test_bonus_non_negative(self) -> None:
        tracker = PolicyDiversityTracker(action_space_size=5)
        tracker.record_actions("a1", [0, 1, 2])
        tracker.record_actions("a2", [3, 4])
        tracker.record_actions("a3", [0, 0])

        for aid in ["a1", "a2", "a3"]:
            bonus = tracker.get_diversity_bonus(aid, weight=0.5)
            assert bonus >= 0.0


# ---------------------------------------------------------------------------
# PBTConfig multiplayer fields tests
# ---------------------------------------------------------------------------


class TestPBTConfigMultiplayer:
    """Tests for multiplayer fields in PBTConfig."""

    def test_default_multiplayer_mode_false(self) -> None:
        config = PBTConfig()
        assert config.multiplayer_mode is False

    def test_default_diversity_weight(self) -> None:
        config = PBTConfig()
        assert config.diversity_weight == 0.1

    def test_default_num_players(self) -> None:
        config = PBTConfig()
        assert config.num_players == 2

    def test_multiplayer_config_creation(self) -> None:
        config = PBTConfig(
            multiplayer_mode=True,
            diversity_weight=0.2,
            num_players=4,
        )
        assert config.multiplayer_mode is True
        assert config.diversity_weight == 0.2
        assert config.num_players == 4

    def test_invalid_num_players(self) -> None:
        with pytest.raises(ValueError, match="num_players"):
            PBTConfig(num_players=1)

    def test_invalid_diversity_weight(self) -> None:
        with pytest.raises(ValueError, match="diversity_weight"):
            PBTConfig(diversity_weight=-0.1)

    def test_to_dict_includes_multiplayer_fields(self) -> None:
        config = PBTConfig(multiplayer_mode=True, num_players=4, diversity_weight=0.3)
        d = config.to_dict()
        assert d["multiplayer_mode"] is True
        assert d["num_players"] == 4
        assert d["diversity_weight"] == 0.3

    def test_backwards_compatible(self) -> None:
        """Existing configs without multiplayer fields should still work."""
        config = PBTConfig()
        assert config.multiplayer_mode is False
        assert config.num_players == 2


# ---------------------------------------------------------------------------
# PopulationManager.evolve_multiplayer() tests
# ---------------------------------------------------------------------------


class TestEvolveMultiplayer:
    """Tests for PopulationManager.evolve_multiplayer()."""

    def test_basic_evolve(self, multiplayer_config: PBTConfig) -> None:
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        manager.initialize()

        runner = MockNPlayerArenaRunner(winner_index=0)
        changes = manager.evolve_multiplayer(arena_runner=runner)

        # Should have replaced some agents
        assert isinstance(changes, list)
        # Generation should have incremented
        assert manager.generation >= 1

    def test_evolve_with_diversity_tracker(
        self, multiplayer_config: PBTConfig
    ) -> None:
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        manager.initialize()

        tracker = PolicyDiversityTracker(action_space_size=10)
        for aid in manager.get_agent_ids():
            # Record some actions to give the tracker data
            tracker.record_actions(aid, list(range(10)))

        runner = MockNPlayerArenaRunner(winner_index=0)
        changes = manager.evolve_multiplayer(
            arena_runner=runner, diversity_tracker=tracker
        )
        assert isinstance(changes, list)

    def test_evolve_empty_population(self, multiplayer_config: PBTConfig) -> None:
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        # Don't initialize -> empty population
        runner = MockNPlayerArenaRunner()
        changes = manager.evolve_multiplayer(arena_runner=runner)
        assert changes == []

    def test_evolve_updates_fitness(self, multiplayer_config: PBTConfig) -> None:
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        manager.initialize()

        runner = MockNPlayerArenaRunner(winner_index=0)
        manager.evolve_multiplayer(arena_runner=runner)

        # Check that at least some agents have non-zero fitness
        all_agents = manager.get_all_agents()
        assert any(a.fitness_score != 0.0 for a in all_agents)

    def test_evolve_preserves_population_size(
        self, multiplayer_config: PBTConfig
    ) -> None:
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        manager.initialize()
        initial_size = manager.population_size

        runner = MockNPlayerArenaRunner()
        manager.evolve_multiplayer(arena_runner=runner)

        assert manager.population_size == initial_size

    def test_evolve_without_diversity_tracker(
        self, multiplayer_config: PBTConfig
    ) -> None:
        """Should work fine without a diversity tracker."""
        manager = PopulationManager(multiplayer_config, rng=np.random.default_rng(42))
        manager.initialize()

        runner = MockNPlayerArenaRunner()
        changes = manager.evolve_multiplayer(arena_runner=runner, diversity_tracker=None)
        assert isinstance(changes, list)


# ---------------------------------------------------------------------------
# NPlayerArenaRunner protocol tests
# ---------------------------------------------------------------------------


class TestNPlayerArenaRunnerProtocol:
    """Tests for NPlayerArenaRunner protocol compliance."""

    def test_mock_runner_is_protocol_compliant(self) -> None:
        runner = MockNPlayerArenaRunner()
        assert isinstance(runner, NPlayerArenaRunner)

    def test_rotating_runner_is_protocol_compliant(self) -> None:
        runner = RotatingWinnerRunner()
        assert isinstance(runner, NPlayerArenaRunner)


# ---------------------------------------------------------------------------
# Scoring fallback tests
# ---------------------------------------------------------------------------


class TestScoringFallback:
    """Tests for the linear ranking fallback scoring."""

    def test_linear_ranking_2_players(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b"],
            rankings={"a": 1, "b": 2},
            scores={"a": 1.0, "b": 0.0},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["a"] == pytest.approx(1.0)
        assert values["b"] == pytest.approx(-1.0)

    def test_linear_ranking_4_players(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c", "d"],
            rankings={"a": 1, "b": 2, "c": 3, "d": 4},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        assert values["a"] == pytest.approx(1.0)
        assert values["d"] == pytest.approx(-1.0)
        assert values["b"] > values["c"]

    def test_linear_ranking_tied(self) -> None:
        result = NPlayerMatchResult(
            agent_ids=["a", "b", "c"],
            rankings={"a": 1, "b": 1, "c": 1},
            scores={},
        )
        values = MultiplayerPBTEvaluator._linear_ranking_fallback(result)
        # All tied at rank 1 -> all get value 1.0
        assert values["a"] == values["b"] == values["c"]


# ---------------------------------------------------------------------------
# Normalisation tests
# ---------------------------------------------------------------------------


class TestNormalisation:
    """Tests for score normalisation."""

    def test_normalise_spread(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores(
            {"a": -1.0, "b": 0.0, "c": 1.0}
        )
        assert result["a"] == pytest.approx(0.0)
        assert result["b"] == pytest.approx(0.5)
        assert result["c"] == pytest.approx(1.0)

    def test_normalise_all_equal(self) -> None:
        result = MultiplayerPBTEvaluator._normalise_scores(
            {"a": 0.5, "b": 0.5, "c": 0.5}
        )
        assert all(v == pytest.approx(0.5) for v in result.values())

    def test_normalise_empty(self) -> None:
        assert MultiplayerPBTEvaluator._normalise_scores({}) == {}
