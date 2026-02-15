"""Model evaluation with Elo rating calculation.

Provides evaluation of trained policies by playing games against
baselines (random, previous best) and computing Elo ratings.

Layer 3 -- depends on types, config, logging.
"""
from __future__ import annotations

__all__ = ["EloRating", "EvalResult", "Evaluator", "random_eval_fn"]

import logging
import math
import random as stdlib_random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.config import MCTSConfig
from src.core.constants import (
    DEFAULT_ELO_K_FACTOR,
    DEFAULT_MAX_STEPS,
    ELO_BASE,
    ELO_SCALE,
    INITIAL_ELO_RATING,
)
from src.core.protocols import StrategicEnvironment
from src.core.types import GameOutcome

logger = logging.getLogger(__name__)

# Type for evaluation function: (spatial, vector, action_mask) -> (policy, value)
EvalFn = Callable[
    [NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]],
    tuple[NDArray[np.float32], float],
]


@dataclass(slots=True)
class EloRating:
    """Elo rating tracker with update logic."""

    rating: float = INITIAL_ELO_RATING
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    k_factor: float = DEFAULT_ELO_K_FACTOR
    history: list[float] = field(default_factory=list)

    def expected_score(self, opponent_rating: float) -> float:
        """Calculate expected score against an opponent."""
        return 1.0 / (1.0 + math.pow(ELO_BASE, (opponent_rating - self.rating) / ELO_SCALE))

    def update(self, actual_score: float, opponent_rating: float) -> float:
        """Update rating after a game.

        Args:
            actual_score: 1.0 for win, 0.5 for draw, 0.0 for loss.
            opponent_rating: Opponent's Elo rating.

        Returns:
            The updated rating.
        """
        expected = self.expected_score(opponent_rating)
        self.rating += self.k_factor * (actual_score - expected)
        self.games_played += 1
        self.history.append(self.rating)

        if actual_score > 0.5:
            self.wins += 1
        elif actual_score < 0.5:
            self.losses += 1
        else:
            self.draws += 1

        return self.rating

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played


@dataclass(slots=True)
class EvalResult:
    """Result of an evaluation run."""

    elo: float
    win_rate: float
    games_played: int
    wins: int
    losses: int
    draws: int
    opponent_type: str
    opponent_elo: float
    details: dict[str, Any] = field(default_factory=dict)


def random_eval_fn(
    spatial: NDArray[np.float32],
    vector: NDArray[np.float32],
    action_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], float]:
    """Random baseline evaluation function."""
    valid = action_mask.astype(np.float32)
    total = valid.sum()
    if total > 0:
        policy = valid / total
    else:
        policy = np.ones_like(valid) / len(valid)
    return policy, 0.0


class Evaluator:
    """Evaluates trained policies against baselines and computes Elo ratings.

    The evaluator plays a set of games between the candidate policy and
    a baseline policy, tracking outcomes to compute Elo ratings.

    Args:
        mcts_config: MCTS configuration for candidate's search.
        num_eval_games: Number of games per evaluation round.
        baseline_elo: Starting Elo for the baseline opponent.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        num_eval_games: int = 20,
        baseline_elo: float = INITIAL_ELO_RATING,
        seed: int | None = None,
    ) -> None:
        self._mcts_config = mcts_config
        self._num_eval_games = num_eval_games
        self._baseline_elo = baseline_elo
        self._rng = stdlib_random.Random(seed)  # noqa: S311
        self._candidate_elo = EloRating()

    @property
    def candidate_elo(self) -> EloRating:
        """Current Elo rating of the candidate."""
        return self._candidate_elo

    @property
    def num_eval_games(self) -> int:
        return self._num_eval_games

    def evaluate(
        self,
        candidate_fn: EvalFn,
        env_factory: Callable[[], StrategicEnvironment],
        baseline_fn: EvalFn | None = None,
        baseline_type: str = "random",
    ) -> EvalResult:
        """Run an evaluation round.

        Args:
            candidate_fn: The candidate's evaluation function.
            env_factory: Factory creating environment instances.
            baseline_fn: Baseline evaluation function.
                If None, uses random_eval_fn.
            baseline_type: Label for the baseline (for logging).

        Returns:
            EvalResult with Elo, win rate, and game outcomes.
        """
        if baseline_fn is None:
            baseline_fn = random_eval_fn

        wins = 0
        losses = 0
        draws = 0
        outcomes: list[GameOutcome] = []

        logger.debug(
            "Starting evaluation: %d games vs %s (baseline_elo=%.1f)",
            self._num_eval_games,
            baseline_type,
            self._baseline_elo,
        )

        for game_idx in range(self._num_eval_games):
            # Alternate who plays first
            candidate_is_p1 = game_idx % 2 == 0
            seed = self._rng.randint(0, 2**31 - 1)

            outcome = self._play_eval_game(
                candidate_fn=candidate_fn,
                baseline_fn=baseline_fn,
                env_factory=env_factory,
                candidate_is_p1=candidate_is_p1,
                seed=seed,
            )
            outcomes.append(outcome)

            if outcome == GameOutcome.win:
                wins += 1
                score = 1.0
            elif outcome == GameOutcome.loss:
                losses += 1
                score = 0.0
            else:
                draws += 1
                score = 0.5

            self._candidate_elo.update(score, self._baseline_elo)

        result = EvalResult(
            elo=self._candidate_elo.rating,
            win_rate=wins / self._num_eval_games if self._num_eval_games > 0 else 0.0,
            games_played=self._num_eval_games,
            wins=wins,
            losses=losses,
            draws=draws,
            opponent_type=baseline_type,
            opponent_elo=self._baseline_elo,
        )

        logger.info(
            "Evaluation complete: Elo=%.1f W/L/D=%d/%d/%d vs %s (Elo=%.1f)",
            result.elo,
            wins,
            losses,
            draws,
            baseline_type,
            self._baseline_elo,
        )
        return result

    def _play_eval_game(
        self,
        candidate_fn: EvalFn,
        baseline_fn: EvalFn,
        env_factory: Callable[[], StrategicEnvironment],
        candidate_is_p1: bool,
        seed: int,
    ) -> GameOutcome:
        """Play a single evaluation game between candidate and baseline.

        Uses simplified action selection (greedy from policy) for speed.

        Args:
            candidate_fn: Candidate's evaluation function.
            baseline_fn: Baseline's evaluation function.
            env_factory: Factory to create a fresh environment.
            candidate_is_p1: Whether the candidate plays as player 1.
            seed: Random seed for the environment.

        Returns:
            GameOutcome from the candidate's perspective.
        """
        env = env_factory()
        env.reset(seed=seed)
        logger.debug(
            "Eval game: candidate_is_p1=%s, seed=%d", candidate_is_p1, seed,
        )

        step = 0
        max_steps = DEFAULT_MAX_STEPS

        while not env.is_terminal() and step < max_steps:
            obs = env.get_observation()
            spatial = obs["spatial"]
            vector = obs["vector"]
            action_mask = env.get_valid_actions()

            # Determine whose turn it is (alternating)
            is_candidate_turn = (step % 2 == 0) == candidate_is_p1
            eval_fn = candidate_fn if is_candidate_turn else baseline_fn

            # Get policy and select action (greedy)
            policy, _value = eval_fn(spatial, vector, action_mask)

            # Mask invalid actions and select
            masked_policy = policy * action_mask.astype(np.float32)
            total = masked_policy.sum()
            if total > 0:
                masked_policy /= total
                action = int(np.argmax(masked_policy))
            else:
                # Fallback: random valid action
                valid_actions = np.where(action_mask)[0]
                if len(valid_actions) > 0:
                    action = int(self._rng.choice(valid_actions))
                else:
                    break

            _obs, reward, terminated, _truncated, _info = env.step(action)
            step += 1

            if terminated:
                # Interpret reward from candidate's perspective
                if candidate_is_p1:
                    if reward > 0:
                        return GameOutcome.win
                    elif reward < 0:
                        return GameOutcome.loss
                else:
                    # If candidate is p2, positive reward means p1 won (candidate lost)
                    if reward > 0:
                        return GameOutcome.loss
                    elif reward < 0:
                        return GameOutcome.win
                return GameOutcome.draw

        return GameOutcome.draw
