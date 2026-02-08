"""PBT Cross-Agent Evaluation Bridge.

Provides the evaluation layer for Population-Based Training by running
arena matches between agents and computing ELO-based fitness scores.

Supports two evaluation strategies:
  - ``round_robin``: Every agent plays against every other agent.
  - ``top_k_vs_best``: Each agent plays against the current best agent.

Usage::

    from alpha_zero.pbt.evaluator import PBTEvaluator

    evaluator = PBTEvaluator(pbt_config)
    evaluator.evaluate_population(agents, arena_runner=my_runner)
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from alpha_zero.pbt.config import AgentState, PBTConfig
from alpha_zero.tournament.config import ELOConfig
from alpha_zero.tournament.elo import ELORatingSystem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols & types
# ---------------------------------------------------------------------------


class ArenaRunner(Protocol):
    """Protocol for running arena matches between two agents.

    Any callable matching this signature can be used, enabling both
    local evaluation (via ``run_arena``) and cloud evaluation (via
    ``VertexEvaluationJob``).
    """

    def __call__(
        self,
        agent_new: AgentState,
        agent_old: AgentState,
        num_games: int,
    ) -> MatchResult:
        ...


@dataclass(frozen=True)
class MatchResult:
    """Outcome of an arena match between two agents.

    Attributes:
        agent_id_new: ID of the challenger agent.
        agent_id_old: ID of the baseline agent.
        win_rate: Win rate from challenger's perspective (0.0-1.0).
        draw_rate: Draw rate (0.0-1.0).
        loss_rate: Loss rate from challenger's perspective (0.0-1.0).
        total_games: Number of games played.
    """

    agent_id_new: str
    agent_id_old: str
    win_rate: float
    draw_rate: float
    loss_rate: float
    total_games: int

    @property
    def score_new(self) -> float:
        """Score for the new agent: 1.0 per win, 0.5 per draw."""
        return self.win_rate + 0.5 * self.draw_rate

    @property
    def score_old(self) -> float:
        """Score for the old agent: 1.0 per loss, 0.5 per draw."""
        return self.loss_rate + 0.5 * self.draw_rate


@dataclass
class EvaluationReport:
    """Summary of a full population evaluation round.

    Attributes:
        generation: PBT generation number.
        matchups: List of match results.
        elo_ratings: Mapping of agent_id -> updated ELO rating.
        fitness_scores: Mapping of agent_id -> derived fitness score.
    """

    generation: int = 0
    matchups: list[MatchResult] = field(default_factory=list)
    elo_ratings: dict[str, float] = field(default_factory=dict)
    fitness_scores: dict[str, float] = field(default_factory=dict)
    games_played: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PBT Evaluator
# ---------------------------------------------------------------------------


class PBTEvaluator:
    """Bridge between PBT orchestrator and ELO-based evaluation.

    Orchestrates cross-agent arena matches and converts the outcomes
    into normalised fitness scores using the ELO rating system.

    Args:
        config: PBT configuration (supplies evaluation_strategy,
            evaluation_games_per_match, elo_* settings).
        arena_runner: Optional callable that executes a single match.
            Defaults to *None* (must be supplied before calling
            ``evaluate_population``).
    """

    def __init__(
        self,
        config: PBTConfig,
        arena_runner: ArenaRunner | Callable[..., MatchResult] | None = None,
    ) -> None:
        self._config = config
        self._arena_runner = arena_runner

        elo_config = ELOConfig(
            initial_rating=config.elo_initial_rating,
            k_factor_base=config.elo_k_factor,
        )
        self._elo = ELORatingSystem(elo_config)

        logger.info(
            "PBTEvaluator initialized: strategy=%s, games_per_match=%d, "
            "elo_initial=%.0f, elo_k=%.1f",
            config.evaluation_strategy,
            config.evaluation_games_per_match,
            config.elo_initial_rating,
            config.elo_k_factor,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        agents: Sequence[AgentState],
        *,
        generation: int = 0,
        arena_runner: ArenaRunner | Callable[..., MatchResult] | None = None,
    ) -> EvaluationReport:
        """Run cross-agent evaluation and update fitness scores.

        Args:
            agents: Population of agents to evaluate.
            generation: Current PBT generation.
            arena_runner: Override arena runner for this call.

        Returns:
            EvaluationReport with results and updated ratings.

        Raises:
            ValueError: If no arena_runner is available.
        """
        runner = arena_runner or self._arena_runner
        if runner is None:
            raise ValueError(
                "No arena_runner provided. Supply one via constructor "
                "or the arena_runner keyword argument."
            )

        if len(agents) < 2:
            logger.warning(
                "Population has fewer than 2 agents (%d); "
                "skipping evaluation.",
                len(agents),
            )
            return EvaluationReport(generation=generation)

        matchups = self._generate_matchups(agents)
        logger.info(
            "Generation %d: running %d matchups (%d games each)",
            generation,
            len(matchups),
            self._config.evaluation_games_per_match,
        )

        results: list[MatchResult] = []
        for agent_new, agent_old in matchups:
            result = runner(
                agent_new=agent_new,
                agent_old=agent_old,
                num_games=self._config.evaluation_games_per_match,
            )
            results.append(result)
            logger.debug(
                "  %s vs %s: win=%.2f draw=%.2f loss=%.2f",
                result.agent_id_new,
                result.agent_id_old,
                result.win_rate,
                result.draw_rate,
                result.loss_rate,
            )

        # Compute ELO-based fitness
        report = self._compute_fitness(agents, results, generation)

        # Update agent state
        for agent in agents:
            if agent.agent_id in report.elo_ratings:
                agent.elo_rating = report.elo_ratings[agent.agent_id]
            if agent.agent_id in report.fitness_scores:
                agent.update_fitness(report.fitness_scores[agent.agent_id])
            if agent.agent_id in report.games_played:
                agent.games_played = report.games_played[agent.agent_id]

        logger.info(
            "Generation %d evaluation complete: %d matchups, "
            "ratings=[%s]",
            generation,
            len(results),
            ", ".join(
                f"{aid}={r:.0f}"
                for aid, r in sorted(
                    report.elo_ratings.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ),
        )

        return report

    # ------------------------------------------------------------------
    # Matchup strategies
    # ------------------------------------------------------------------

    def _generate_matchups(
        self, agents: Sequence[AgentState]
    ) -> list[tuple[AgentState, AgentState]]:
        """Generate matchups based on the configured evaluation strategy.

        Args:
            agents: Population of agents.

        Returns:
            List of (challenger, baseline) pairs.
        """
        strategy = self._config.evaluation_strategy

        if strategy == "round_robin":
            return self._round_robin(agents)
        elif strategy == "top_k_vs_best":
            return self._top_k_vs_best(agents)
        else:
            logger.warning(
                "Unknown strategy '%s'; falling back to round_robin.",
                strategy,
            )
            return self._round_robin(agents)

    @staticmethod
    def _round_robin(
        agents: Sequence[AgentState],
    ) -> list[tuple[AgentState, AgentState]]:
        """All unique pairs — every agent plays every other agent once."""
        return list(itertools.combinations(agents, 2))

    @staticmethod
    def _top_k_vs_best(
        agents: Sequence[AgentState],
    ) -> list[tuple[AgentState, AgentState]]:
        """Each agent plays against the current highest-rated agent.

        The best agent is determined by ``elo_rating``. If the best
        agent is the only agent, returns an empty list.
        """
        if len(agents) < 2:
            return []

        best = max(agents, key=lambda a: a.elo_rating)
        return [
            (agent, best)
            for agent in agents
            if agent.agent_id != best.agent_id
        ]

    # ------------------------------------------------------------------
    # ELO-based fitness computation
    # ------------------------------------------------------------------

    def _compute_fitness(
        self,
        agents: Sequence[AgentState],
        results: list[MatchResult],
        generation: int,
    ) -> EvaluationReport:
        """Convert match results into ELO ratings and normalised fitness.

        The normalisation maps ELO ratings to [0, 1] using min-max
        scaling across the current population.

        Args:
            agents: Population of agents.
            results: Match results.
            generation: Current generation number.

        Returns:
            EvaluationReport with ratings and fitness scores.
        """
        # Initialise ratings from current agent state
        ratings: dict[str, float] = {
            a.agent_id: a.elo_rating for a in agents
        }
        games_played: dict[str, int] = {
            a.agent_id: a.games_played for a in agents
        }

        # Process each match through the ELO system
        for result in results:
            r_new = ratings.get(
                result.agent_id_new, self._config.elo_initial_rating
            )
            r_old = ratings.get(
                result.agent_id_old, self._config.elo_initial_rating
            )
            gp_new = games_played.get(result.agent_id_new, 0)
            gp_old = games_played.get(result.agent_id_old, 0)

            update1, update2 = self._elo.process_match(
                player1_id=result.agent_id_new,
                player2_id=result.agent_id_old,
                player1_rating=r_new,
                player2_rating=r_old,
                score1=result.score_new,
                score2=result.score_old,
                games_played1=gp_new,
                games_played2=gp_old,
            )

            ratings[result.agent_id_new] = update1.rating_after
            ratings[result.agent_id_old] = update2.rating_after
            games_played[result.agent_id_new] = gp_new + result.total_games
            games_played[result.agent_id_old] = gp_old + result.total_games

        # Normalise ratings to [0, 1] fitness
        fitness = self._normalise_ratings(ratings)

        return EvaluationReport(
            generation=generation,
            matchups=results,
            elo_ratings=ratings,
            fitness_scores=fitness,
            games_played=games_played,
        )

    @staticmethod
    def _normalise_ratings(ratings: dict[str, float]) -> dict[str, float]:
        """Min-max normalise ELO ratings to [0, 1].

        If all ratings are equal, returns 0.5 for each agent.

        Args:
            ratings: Mapping of agent_id -> ELO rating.

        Returns:
            Mapping of agent_id -> normalised fitness in [0, 1].
        """
        if not ratings:
            return {}

        values = list(ratings.values())
        min_r = min(values)
        max_r = max(values)
        spread = max_r - min_r

        if spread < 1e-9:
            return dict.fromkeys(ratings, 0.5)

        return {
            aid: (r - min_r) / spread
            for aid, r in ratings.items()
        }
