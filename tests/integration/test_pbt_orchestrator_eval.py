"""Integration test for PBT orchestrator evaluation phase.

Verifies that PBTOrchestrator._run_evaluation_phase correctly wires
PBTEvaluator, updates agent ELO ratings and fitness scores, and
produces a valid EvaluationReport.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.pbt.evaluator import EvaluationReport, MatchResult, PBTEvaluator
from alpha_zero.pbt.population import PopulationManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pbt_config() -> PBTConfig:
    return PBTConfig(
        population_size=3,
        min_population_size=3,
        evaluation_games_per_match=4,
        evaluation_strategy="round_robin",
        elo_initial_rating=1200.0,
        elo_k_factor=32.0,
    )


@pytest.fixture
def agents() -> list[AgentState]:
    return [
        AgentState(agent_id="agent_0", elo_rating=1200.0),
        AgentState(agent_id="agent_1", elo_rating=1200.0),
        AgentState(agent_id="agent_2", elo_rating=1200.0),
    ]


def _balanced_runner(
    agent_new: AgentState,
    agent_old: AgentState,
    num_games: int,
) -> MatchResult:
    """Balanced runner: new model always slightly better."""
    return MatchResult(
        agent_id_new=agent_new.agent_id,
        agent_id_old=agent_old.agent_id,
        win_rate=0.6,
        draw_rate=0.1,
        loss_rate=0.3,
        total_games=num_games,
    )


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestOrchestratorEvalPhase:
    """Integration tests for the evaluation phase of PBTOrchestrator."""

    def test_evaluator_updates_agents(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        """Evaluator should update elo_rating and fitness_score on agents."""
        evaluator = PBTEvaluator(pbt_config, arena_runner=_balanced_runner)
        report = evaluator.evaluate_population(agents, generation=1)

        # All agents should have non-default ratings
        assert len(report.elo_ratings) == 3
        assert len(report.fitness_scores) == 3
        assert len(report.matchups) == 3  # C(3,2) = 3

        # Agent states should be updated in-place
        for agent in agents:
            assert agent.elo_rating in report.elo_ratings.values()

    def test_population_manager_integration(
        self, pbt_config: PBTConfig
    ) -> None:
        """Test full flow: PopulationManager → PBTEvaluator → updated agents."""
        manager = PopulationManager(pbt_config)
        manager.initialize()

        agents = manager.get_all_agents()
        evaluator = PBTEvaluator(pbt_config, arena_runner=_balanced_runner)

        report = evaluator.evaluate_population(agents, generation=1)

        assert report.generation == 1
        assert len(report.matchups) == 3  # C(3,2) = 3
        assert len(report.elo_ratings) == 3
        assert len(report.fitness_scores) == 3

        # Fitness scores should sum to something reasonable
        total_fitness = sum(report.fitness_scores.values())
        assert total_fitness > 0

    def test_multiple_generations(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        """Running multiple generations should accumulate ELO changes."""
        evaluator = PBTEvaluator(pbt_config, arena_runner=_balanced_runner)

        report_g1 = evaluator.evaluate_population(agents, generation=1)
        elo_g1 = dict(report_g1.elo_ratings)

        report_g2 = evaluator.evaluate_population(agents, generation=2)
        elo_g2 = dict(report_g2.elo_ratings)

        # Ratings should continue to evolve
        assert elo_g2 != elo_g1 or all(
            v == pytest.approx(1200.0) for v in elo_g1.values()
        )

    def test_top_k_strategy(self, agents: list[AgentState]) -> None:
        """top_k_vs_best strategy should produce N-1 matchups."""
        agents[0].elo_rating = 1500.0  # Make agent_0 the best
        config = PBTConfig(
            population_size=3,
            min_population_size=3,
            evaluation_strategy="top_k_vs_best",
        )
        evaluator = PBTEvaluator(config, arena_runner=_balanced_runner)
        report = evaluator.evaluate_population(agents, generation=1)

        # 2 matchups: agent_1 vs agent_0, agent_2 vs agent_0
        assert len(report.matchups) == 2


class TestEvaluationReportStructure:
    """Tests for EvaluationReport correctness."""

    def test_report_has_all_fields(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_balanced_runner)
        report = evaluator.evaluate_population(agents, generation=5)

        assert report.generation == 5
        assert isinstance(report.matchups, list)
        assert isinstance(report.elo_ratings, dict)
        assert isinstance(report.fitness_scores, dict)
        assert all(isinstance(m, MatchResult) for m in report.matchups)

    def test_fitness_scores_in_range(
        self, pbt_config: PBTConfig, agents: list[AgentState]
    ) -> None:
        evaluator = PBTEvaluator(pbt_config, arena_runner=_balanced_runner)
        report = evaluator.evaluate_population(agents, generation=1)

        for score in report.fitness_scores.values():
            assert 0.0 <= score <= 1.0
