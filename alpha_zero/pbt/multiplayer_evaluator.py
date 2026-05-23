"""N-player evaluation for population-based training.

Extends PBT evaluation to support N-player tournaments where multiple
agents compete simultaneously, rather than pairwise matches. Uses ELO
ratings and multiplayer scoring to derive fitness scores.

Usage::

    from alpha_zero.pbt.multiplayer_evaluator import (
        MultiplayerPBTEvaluator,
        NPlayerMatchResult,
    )

    evaluator = MultiplayerPBTEvaluator(num_players=4, games_per_eval=10)
    fitness = evaluator.evaluate_population(agents, arena_runner=runner)
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from alpha_zero.pbt.config import AgentState

logger = logging.getLogger(__name__)

# Graceful imports for tournament/multiplayer modules
try:
    from alpha_zero.tournament.config import ELOConfig
    from alpha_zero.tournament.elo import ELORatingSystem

    _HAS_TOURNAMENT = True
except ImportError:  # pragma: no cover
    _HAS_TOURNAMENT = False

try:
    from alpha_zero.multiplayer.scoring import ScoringMethod, compute_player_values

    _HAS_MULTIPLAYER = True
except ImportError:  # pragma: no cover
    _HAS_MULTIPLAYER = False


@dataclass(frozen=True)
class NPlayerMatchResult:
    """Result of an N-player match.

    Attributes:
        agent_ids: Ordered list of agent IDs that participated.
        rankings: Mapping of agent_id -> rank (1 = first place).
        scores: Mapping of agent_id -> game score.
        game_length: Number of turns in the game.
    """

    agent_ids: list[str]
    rankings: dict[str, int]
    scores: dict[str, float]
    game_length: int = 0


@runtime_checkable
class NPlayerArenaRunner(Protocol):
    """Protocol for running N-player matches between agents."""

    def run_nplayer_match(
        self,
        agent_ids: list[str],
        num_games: int,
    ) -> list[NPlayerMatchResult]:
        """Run N-player games between the given agents.

        Args:
            agent_ids: List of agent IDs to include in the match.
            num_games: Number of games to play.

        Returns:
            List of NPlayerMatchResult, one per game.
        """
        ...


class MultiplayerPBTEvaluator:
    """Evaluates PBT population fitness via N-player tournaments.

    Creates balanced matchups of ``num_players`` agents drawn from the
    population, runs games via an :class:`NPlayerArenaRunner`, and
    aggregates results into fitness scores using ELO ratings or
    linear-ranking scoring.

    Args:
        num_players: Number of players per game.
        games_per_eval: Number of games per matchup group.
        scoring_method: Scoring method for converting rankings to values.
            One of ``"linear_ranking"``, ``"winner_take_all"``, or
            ``"proportional"``.
        elo_initial_rating: Starting ELO rating for new agents.
        elo_k_factor: K-factor for ELO updates.
    """

    def __init__(
        self,
        num_players: int = 4,
        games_per_eval: int = 10,
        scoring_method: str = "linear_ranking",
        elo_initial_rating: float = 1200.0,
        elo_k_factor: float = 32.0,
    ) -> None:
        if num_players < 2:
            raise ValueError(f"num_players must be >= 2, got {num_players}")
        if games_per_eval < 1:
            raise ValueError(f"games_per_eval must be >= 1, got {games_per_eval}")

        self._num_players = num_players
        self._games_per_eval = games_per_eval
        self._scoring_method = scoring_method
        self._elo_initial_rating = elo_initial_rating
        self._elo_k_factor = elo_k_factor

        # Initialise ELO system if available
        self._elo: ELORatingSystem | None = None
        if _HAS_TOURNAMENT:
            elo_config = ELOConfig(
                initial_rating=elo_initial_rating,
                k_factor_base=elo_k_factor,
            )
            self._elo = ELORatingSystem(elo_config)

        logger.info(
            "MultiplayerPBTEvaluator initialized: num_players=%d, games_per_eval=%d, scoring=%s",
            num_players,
            games_per_eval,
            scoring_method,
        )

    @property
    def num_players(self) -> int:
        """Number of players per game."""
        return self._num_players

    @property
    def games_per_eval(self) -> int:
        """Number of games per matchup group."""
        return self._games_per_eval

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_population(
        self,
        agents: list[AgentState],
        arena_runner: NPlayerArenaRunner,
    ) -> dict[str, float]:
        """Evaluate all agents via round-robin N-player tournaments.

        Creates matchup groups, runs games, and aggregates results into
        normalised fitness scores.

        Args:
            agents: Population of agents to evaluate.
            arena_runner: Runner that executes N-player matches.

        Returns:
            Mapping of agent_id -> fitness score in [0, 1].

        Raises:
            ValueError: If population is smaller than ``num_players``.
        """
        if len(agents) < self._num_players:
            logger.warning(
                "Population size %d < num_players %d; returning uniform fitness.",
                len(agents),
                self._num_players,
            )
            return {a.agent_id: 0.5 for a in agents}

        agent_ids = [a.agent_id for a in agents]
        matchups = self._create_matchups(agent_ids)

        logger.info(
            "Running %d matchup groups (%d games each, %d players per game)",
            len(matchups),
            self._games_per_eval,
            self._num_players,
        )

        # Accumulate scores across all games
        total_scores: dict[str, float] = dict.fromkeys(agent_ids, 0.0)
        game_counts: dict[str, int] = dict.fromkeys(agent_ids, 0)

        for group in matchups:
            results = arena_runner.run_nplayer_match(
                agent_ids=group,
                num_games=self._games_per_eval,
            )

            for result in results:
                game_values = self._score_result(result)
                for aid, value in game_values.items():
                    total_scores[aid] += value
                    game_counts[aid] += 1

        # Compute average scores
        avg_scores: dict[str, float] = {}
        for aid in agent_ids:
            count = game_counts[aid]
            avg_scores[aid] = total_scores[aid] / count if count > 0 else 0.0

        # Normalise to [0, 1]
        fitness = self._normalise_scores(avg_scores)

        logger.info(
            "Evaluation complete: %d matchups, fitness=[%s]",
            len(matchups),
            ", ".join(
                f"{aid}={f:.3f}"
                for aid, f in sorted(fitness.items(), key=lambda x: x[1], reverse=True)
            ),
        )

        return fitness

    # ------------------------------------------------------------------
    # Matchup creation
    # ------------------------------------------------------------------

    def _create_matchups(self, agent_ids: list[str]) -> list[list[str]]:
        """Create balanced N-player matchups covering all agents.

        Uses combinations to create all possible groups of
        ``num_players`` agents. For large populations where the number
        of combinations is excessive, a subset is sampled to keep
        evaluation time bounded.

        Args:
            agent_ids: List of agent IDs in the population.

        Returns:
            List of agent-ID groups, each of length ``num_players``.
        """
        n = len(agent_ids)
        k = self._num_players

        if n < k:
            return []

        all_combos = list(itertools.combinations(agent_ids, k))
        all_groups = [list(c) for c in all_combos]

        # Cap the number of matchups to avoid combinatorial explosion.
        # Each agent should appear in a reasonable number of matchups.
        max_matchups = max(n * 3, 20)
        if len(all_groups) > max_matchups:
            rng = np.random.default_rng()
            indices = rng.choice(len(all_groups), size=max_matchups, replace=False)
            all_groups = [all_groups[i] for i in sorted(indices)]

        return all_groups

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_result(self, result: NPlayerMatchResult) -> dict[str, float]:
        """Convert a single match result into per-agent values.

        Uses the multiplayer scoring module if available, otherwise
        falls back to a simple linear-ranking computation.

        Args:
            result: Match result to score.

        Returns:
            Mapping of agent_id -> value in [-1, 1].
        """
        n = len(result.agent_ids)

        if _HAS_MULTIPLAYER:
            # Map string agent_ids to integer player indices
            id_to_idx = {aid: i for i, aid in enumerate(result.agent_ids)}
            int_rankings = {id_to_idx[aid]: rank for aid, rank in result.rankings.items()}

            try:
                method = ScoringMethod(self._scoring_method)
            except ValueError:
                method = ScoringMethod.LINEAR_RANKING

            int_scores: dict[int, float] | None = None
            if result.scores:
                int_scores = {id_to_idx[aid]: s for aid, s in result.scores.items()}

            int_values = compute_player_values(
                rankings=int_rankings,
                num_players=n,
                method=method,
                player_scores=int_scores,
            )

            # Map back to string agent_ids
            idx_to_id = {i: aid for aid, i in id_to_idx.items()}
            return {idx_to_id[i]: v for i, v in int_values.items()}

        # Fallback: simple linear ranking
        return self._linear_ranking_fallback(result)

    @staticmethod
    def _linear_ranking_fallback(result: NPlayerMatchResult) -> dict[str, float]:
        """Simple linear-ranking scoring fallback.

        Maps rank 1 -> +1.0, rank N -> -1.0.

        Args:
            result: Match result.

        Returns:
            Mapping of agent_id -> value in [-1, 1].
        """
        n = len(result.agent_ids)
        max_rank = n  # ranks are 1-indexed
        values: dict[str, float] = {}
        for aid, rank in result.rankings.items():
            if max_rank > 1:
                values[aid] = 1.0 - 2.0 * ((rank - 1) / (max_rank - 1))
            else:
                values[aid] = 0.0
        return values

    @staticmethod
    def _normalise_scores(scores: dict[str, float]) -> dict[str, float]:
        """Min-max normalise scores to [0, 1].

        If all scores are equal, returns 0.5 for each agent.

        Args:
            scores: Mapping of agent_id -> raw score.

        Returns:
            Mapping of agent_id -> normalised fitness in [0, 1].
        """
        if not scores:
            return {}

        values = list(scores.values())
        min_s = min(values)
        max_s = max(values)
        spread = max_s - min_s

        if spread < 1e-9:
            return dict.fromkeys(scores, 0.5)

        return {aid: (s - min_s) / spread for aid, s in scores.items()}
