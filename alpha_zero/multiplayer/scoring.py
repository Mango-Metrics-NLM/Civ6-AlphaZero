"""Scoring and ranking utilities for N-player games.

Converts game outcomes (winner, scores, eliminations) into training
values for each player. Supports multiple scoring methods to handle
different game result types.

Example:
    >>> rankings = compute_rankings(winner=0, num_players=4)
    >>> rankings
    {0: 0, 1: 1, 2: 1, 3: 1}
    >>> values = compute_player_values(rankings, num_players=4, method=ScoringMethod.LINEAR_RANKING)
    >>> values[0]  # Winner
    1.0
    >>> values[1]  # Others tied for 2nd
    -0.333...
"""

from __future__ import annotations

import enum
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ScoringMethod(enum.Enum):
    """Methods for converting game outcomes to training values.

    Attributes:
        LINEAR_RANKING: Linear interpolation from +1 (1st) to -1 (last).
        WINNER_TAKE_ALL: Winner gets +1, all others get -1.
        PROPORTIONAL: Values proportional to in-game scores.
    """

    LINEAR_RANKING = "linear_ranking"
    WINNER_TAKE_ALL = "winner_take_all"
    PROPORTIONAL = "proportional"


def compute_rankings(
    *,
    winner: int | None = None,
    player_scores: dict[int, float] | None = None,
    eliminated: set[int] | None = None,
    num_players: int,
) -> dict[int, int]:
    """Compute player rankings from game outcome.

    Rankings are 0-indexed: 0 = 1st place, num_players-1 = last place.
    Ties share the same rank.

    Args:
        winner: ID of winning player, or None for draw.
        player_scores: Per-player scores for ranking (higher = better).
        eliminated: Set of eliminated player IDs (ranked last).
        num_players: Total number of players.

    Returns:
        Dict mapping player_id -> rank (0 = 1st place).

    Examples:
        >>> compute_rankings(winner=2, num_players=4)
        {0: 1, 1: 1, 2: 0, 3: 1}
        >>> compute_rankings(player_scores={0: 100, 1: 50, 2: 75}, num_players=3)
        {0: 0, 1: 2, 2: 1}
    """
    if num_players < 2:
        msg = f"num_players must be >= 2, got {num_players}"
        raise ValueError(msg)

    eliminated = eliminated or set()
    player_ids = list(range(num_players))

    # If we have scores, rank by scores (descending)
    if player_scores is not None:
        # Separate eliminated from active players
        active = [p for p in player_ids if p not in eliminated]
        sorted_active = sorted(active, key=lambda p: player_scores.get(p, 0), reverse=True)

        rankings: dict[int, int] = {}
        current_rank = 0
        for i, player in enumerate(sorted_active):
            if i > 0 and player_scores.get(player, 0) < player_scores.get(
                sorted_active[i - 1], 0
            ):
                current_rank = i
            rankings[player] = current_rank

        # Eliminated players get worst rank
        eliminated_rank = len(active)
        for player in eliminated:
            rankings[player] = eliminated_rank

        return rankings

    # If we have a winner, winner is 1st, others share 2nd
    if winner is not None:
        if winner < 0 or winner >= num_players:
            msg = f"winner {winner} out of range [0, {num_players})"
            raise ValueError(msg)

        rankings = {}
        non_eliminated = [p for p in player_ids if p not in eliminated and p != winner]
        rankings[winner] = 0

        # Non-eliminated losers share 2nd place
        for player in non_eliminated:
            rankings[player] = 1

        # Eliminated players get last
        eliminated_rank = max(2, len(non_eliminated) + 1)
        for player in eliminated:
            rankings[player] = eliminated_rank

        return rankings

    # Draw: everyone shares rank 0
    return {p: 0 for p in player_ids}


def compute_player_values(
    rankings: dict[int, int],
    num_players: int,
    method: ScoringMethod = ScoringMethod.LINEAR_RANKING,
    player_scores: dict[int, float] | None = None,
    draw_value: float = 0.0,
    elimination_penalty: float = 0.0,
) -> dict[int, float]:
    """Convert rankings to training values for each player.

    Args:
        rankings: Dict mapping player_id -> rank (0 = 1st place).
        num_players: Total number of players.
        method: Scoring method to use.
        player_scores: Raw scores for PROPORTIONAL method.
        draw_value: Value for all players on a draw.
        elimination_penalty: Extra penalty subtracted from eliminated players.

    Returns:
        Dict mapping player_id -> value in [-1, 1].

    Examples:
        >>> rankings = {0: 0, 1: 1, 2: 2}
        >>> compute_player_values(rankings, num_players=3)
        {0: 1.0, 1: 0.0, 2: -1.0}
        >>> compute_player_values(rankings, num_players=3, method=ScoringMethod.WINNER_TAKE_ALL)
        {0: 1.0, 1: -1.0, 2: -1.0}
    """
    if num_players < 2:
        msg = f"num_players must be >= 2, got {num_players}"
        raise ValueError(msg)

    # Check if it's a draw (all players share rank 0)
    if all(r == 0 for r in rankings.values()):
        return {p: draw_value for p in rankings}

    if method == ScoringMethod.LINEAR_RANKING:
        return _linear_ranking_values(rankings, num_players, elimination_penalty)
    elif method == ScoringMethod.WINNER_TAKE_ALL:
        return _winner_take_all_values(rankings, num_players, elimination_penalty)
    elif method == ScoringMethod.PROPORTIONAL:
        return _proportional_values(
            rankings, num_players, player_scores, elimination_penalty
        )
    else:
        msg = f"Unknown scoring method: {method}"
        raise ValueError(msg)


def _linear_ranking_values(
    rankings: dict[int, int],
    num_players: int,
    elimination_penalty: float,
) -> dict[int, float]:
    """Linear interpolation: 1st place = +1, last place = -1.

    For N players: value = 1 - 2 * (rank / (N - 1))
    This maps rank 0 -> +1.0 and rank N-1 -> -1.0.
    """
    max_rank = num_players - 1
    values: dict[int, float] = {}
    for player, rank in rankings.items():
        clamped_rank = min(rank, max_rank)
        value = 1.0 - 2.0 * (clamped_rank / max_rank) if max_rank > 0 else 0.0
        # Apply elimination penalty
        if rank >= num_players - 1 and elimination_penalty > 0:
            value = max(-1.0, value - elimination_penalty)
        values[player] = value
    return values


def _winner_take_all_values(
    rankings: dict[int, int],
    num_players: int,
    elimination_penalty: float,
) -> dict[int, float]:
    """Winner gets +1, all others get -1."""
    values: dict[int, float] = {}
    for player, rank in rankings.items():
        if rank == 0:
            values[player] = 1.0
        else:
            value = -1.0
            if rank >= num_players - 1 and elimination_penalty > 0:
                value = max(-1.0, value - elimination_penalty)
            values[player] = value
    return values


def _proportional_values(
    rankings: dict[int, int],
    num_players: int,
    player_scores: dict[int, float] | None,
    elimination_penalty: float,
) -> dict[int, float]:
    """Values proportional to normalized scores.

    Scores are normalized to [-1, 1] range using min-max scaling.
    Falls back to linear ranking if no scores provided.
    """
    if player_scores is None:
        logger.warning("No player_scores for proportional scoring; falling back to linear_ranking")
        return _linear_ranking_values(rankings, num_players, elimination_penalty)

    scores = [player_scores.get(p, 0.0) for p in range(num_players)]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score

    values: dict[int, float] = {}
    for player in range(num_players):
        score = player_scores.get(player, 0.0)
        if score_range > 0:
            # Normalize to [0, 1] then scale to [-1, 1]
            normalized = (score - min_score) / score_range
            value = 2.0 * normalized - 1.0
        else:
            value = 0.0  # All scores equal -> draw
        if rankings.get(player, 0) >= num_players - 1 and elimination_penalty > 0:
            value = max(-1.0, value - elimination_penalty)
        values[player] = value

    return values


def extract_game_result(
    info: dict[str, Any],
    num_players: int,
) -> dict[int, float]:
    """Extract player values from environment info dict.

    Convenience function that combines compute_rankings and
    compute_player_values for typical game-over info dicts.

    Args:
        info: Environment info dict from step() or get_info().
        num_players: Total number of players.

    Returns:
        Dict mapping player_id -> value in [-1, 1].
    """
    winner = info.get("winner")
    player_scores = info.get("player_scores")
    eliminated = info.get("eliminated")
    if isinstance(eliminated, (list, tuple)):
        eliminated = set(eliminated)

    rankings = compute_rankings(
        winner=winner,
        player_scores=player_scores,
        eliminated=eliminated,
        num_players=num_players,
    )

    method_str = info.get("scoring_method", "linear_ranking")
    try:
        method = ScoringMethod(method_str)
    except ValueError:
        method = ScoringMethod.LINEAR_RANKING

    return compute_player_values(
        rankings=rankings,
        num_players=num_players,
        method=method,
        player_scores=player_scores,
    )
