"""ELO Rating System implementation for tournament matchmaking.

This module provides a complete ELO rating system with support for:
- Standard ELO calculations
- K-factor adjustment based on games played
- Rating floor and ceiling enforcement
- Match result processing

Usage:
    from alpha_zero.tournament.elo import ELORatingSystem
    from alpha_zero.tournament.config import ELOConfig

    config = ELOConfig()
    system = ELORatingSystem(config)

    # Calculate rating changes
    change1, change2 = system.calculate_rating_changes(
        rating1=1500, rating2=1600, score1=1.0, score2=0.0
    )
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from alpha_zero.tournament.config import (
    ELOConfig,
    MatchOutcome,
    PlayerStats,
    TournamentConfig,
)

logger = logging.getLogger(__name__)


def calculate_expected_score(rating_a: float, rating_b: float) -> float:
    """Calculate expected score for player A against player B.

    Uses the standard ELO formula:
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Rating of player A
        rating_b: Rating of player B

    Returns:
        Expected score for player A (between 0 and 1).
    """
    return 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))


def calculate_rating_change(
    rating: float,
    expected_score: float,
    actual_score: float,
    k_factor: float,
) -> float:
    """Calculate rating change for a single game.

    Uses the formula: Change = K * (S - E)
    Where K is the K-factor, S is actual score, E is expected score.

    Args:
        rating: Current rating
        expected_score: Expected score based on ratings
        actual_score: Actual score (1 for win, 0.5 for draw, 0 for loss)
        k_factor: K-factor for this player

    Returns:
        Rating change (positive or negative).
    """
    return k_factor * (actual_score - expected_score)


@dataclass
class RatingUpdate:
    """Represents a rating update for a player.

    Attributes:
        player_id: Player identifier
        rating_before: Rating before the update
        rating_after: Rating after the update
        rating_change: Change in rating
        expected_score: Expected score based on opponent
        actual_score: Actual score achieved
        k_factor: K-factor used
        opponent_rating: Opponent's rating
        timestamp: When update occurred
    """

    player_id: str
    rating_before: float
    rating_after: float
    rating_change: float
    expected_score: float
    actual_score: float
    k_factor: float
    opponent_rating: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "rating_before": self.rating_before,
            "rating_after": self.rating_after,
            "rating_change": self.rating_change,
            "expected_score": self.expected_score,
            "actual_score": self.actual_score,
            "k_factor": self.k_factor,
            "opponent_rating": self.opponent_rating,
            "timestamp": self.timestamp,
        }


class ELORatingSystem:
    """Complete ELO rating system for tournament matchmaking.

    This class handles all rating calculations and updates, including:
    - Expected score calculation
    - Rating change computation
    - K-factor adjustment
    - Rating bounds enforcement

    Example:
        >>> config = ELOConfig()
        >>> system = ELORatingSystem(config)
        >>>
        >>> # Process a match result
        >>> update1, update2 = system.process_match(
        ...     player1_id="alice",
        ...     player2_id="bob",
        ...     player1_rating=1500,
        ...     player2_rating=1600,
        ...     score1=1.0,
        ...     score2=0.0,
        ... )
    """

    def __init__(self, config: ELOConfig | None = None) -> None:
        """Initialize ELO rating system.

        Args:
            config: ELO configuration. Uses defaults if None.
        """
        self.config = config or ELOConfig()
        self._update_history: list[RatingUpdate] = []

        logger.info(
            f"ELORatingSystem initialized: initial_rating={self.config.initial_rating}, "
            f"k_factor_base={self.config.k_factor_base}"
        )

    def calculate_expected_scores(self, rating1: float, rating2: float) -> tuple[float, float]:
        """Calculate expected scores for both players.

        Args:
            rating1: Player 1's rating
            rating2: Player 2's rating

        Returns:
            Tuple of (expected_score1, expected_score2).
        """
        expected1 = calculate_expected_score(rating1, rating2)
        expected2 = 1.0 - expected1  # Expected scores sum to 1
        return expected1, expected2

    def calculate_rating_changes(
        self,
        rating1: float,
        rating2: float,
        score1: float,
        score2: float,
        games_played1: int = 0,
        games_played2: int = 0,
        is_provisional1: bool = True,
        is_provisional2: bool = True,
    ) -> tuple[float, float]:
        """Calculate rating changes for both players.

        Args:
            rating1: Player 1's rating
            rating2: Player 2's rating
            score1: Player 1's actual score (1=win, 0.5=draw, 0=loss)
            score2: Player 2's actual score
            games_played1: Player 1's games played (for K-factor)
            games_played2: Player 2's games played (for K-factor)
            is_provisional1: Whether player 1 is provisional
            is_provisional2: Whether player 2 is provisional

        Returns:
            Tuple of (change1, change2) rating changes.
        """
        # Calculate expected scores
        expected1, expected2 = self.calculate_expected_scores(rating1, rating2)

        # Get K-factors
        k1 = self.config.get_k_factor(games_played1, is_provisional1)
        k2 = self.config.get_k_factor(games_played2, is_provisional2)

        # Calculate changes
        change1 = calculate_rating_change(rating1, expected1, score1, k1)
        change2 = calculate_rating_change(rating2, expected2, score2, k2)

        return change1, change2

    def apply_rating_bounds(self, rating: float) -> float:
        """Apply rating floor and ceiling.

        Args:
            rating: Raw rating value

        Returns:
            Rating clamped to valid range.
        """
        return max(self.config.rating_floor, min(self.config.rating_ceiling, rating))

    def process_match(
        self,
        player1_id: str,
        player2_id: str,
        player1_rating: float,
        player2_rating: float,
        score1: float,
        score2: float,
        games_played1: int = 0,
        games_played2: int = 0,
    ) -> tuple[RatingUpdate, RatingUpdate]:
        """Process a complete match and return rating updates.

        Args:
            player1_id: Player 1's identifier
            player2_id: Player 2's identifier
            player1_rating: Player 1's current rating
            player2_rating: Player 2's current rating
            score1: Player 1's score
            score2: Player 2's score
            games_played1: Player 1's total games
            games_played2: Player 2's total games

        Returns:
            Tuple of RatingUpdate objects for each player.
        """
        # Calculate changes
        change1, change2 = self.calculate_rating_changes(
            player1_rating,
            player2_rating,
            score1,
            score2,
            games_played1,
            games_played2,
        )

        # Apply bounds to new ratings
        new_rating1 = self.apply_rating_bounds(player1_rating + change1)
        new_rating2 = self.apply_rating_bounds(player2_rating + change2)

        # Recalculate actual changes after bounds
        actual_change1 = new_rating1 - player1_rating
        actual_change2 = new_rating2 - player2_rating

        # Get expected scores for logging
        expected1, expected2 = self.calculate_expected_scores(player1_rating, player2_rating)

        # Create update objects
        update1 = RatingUpdate(
            player_id=player1_id,
            rating_before=player1_rating,
            rating_after=new_rating1,
            rating_change=actual_change1,
            expected_score=expected1,
            actual_score=score1,
            k_factor=self.config.get_k_factor(games_played1),
            opponent_rating=player2_rating,
        )

        update2 = RatingUpdate(
            player_id=player2_id,
            rating_before=player2_rating,
            rating_after=new_rating2,
            rating_change=actual_change2,
            expected_score=expected2,
            actual_score=score2,
            k_factor=self.config.get_k_factor(games_played2),
            opponent_rating=player1_rating,
        )

        # Record updates
        self._update_history.append(update1)
        self._update_history.append(update2)

        logger.debug(
            f"Match processed: {player1_id} vs {player2_id}, "
            f"score={score1}-{score2}, "
            f"changes={actual_change1:+.1f}/{actual_change2:+.1f}"
        )

        return update1, update2

    def process_game_result(
        self,
        player1: PlayerStats,
        player2: PlayerStats,
        outcome: MatchOutcome,
        config: TournamentConfig | None = None,
    ) -> tuple[RatingUpdate, RatingUpdate]:
        """Process a game result and update player stats.

        Args:
            player1: Player 1's stats
            player2: Player 2's stats
            outcome: Outcome from player 1's perspective
            config: Tournament config for tier updates

        Returns:
            Tuple of RatingUpdate objects.
        """
        # Convert outcome to scores
        if outcome == MatchOutcome.WIN:
            score1, score2 = 1.0, 0.0
        elif outcome == MatchOutcome.LOSS:
            score1, score2 = 0.0, 1.0
        else:  # DRAW
            score1, score2 = self.config.draw_value, self.config.draw_value

        # Process match
        update1, update2 = self.process_match(
            player1.player_id,
            player2.player_id,
            player1.rating,
            player2.rating,
            score1,
            score2,
            player1.games_played,
            player2.games_played,
        )

        # Update player stats
        tier_config = config.get_tier_for_rating(update1.rating_after) if config else None
        player1.record_game(outcome, update1.rating_change, tier_config)

        # Reverse outcome for player 2
        if outcome == MatchOutcome.WIN:
            outcome2 = MatchOutcome.LOSS
        elif outcome == MatchOutcome.LOSS:
            outcome2 = MatchOutcome.WIN
        else:
            outcome2 = MatchOutcome.DRAW

        tier_config2 = config.get_tier_for_rating(update2.rating_after) if config else None
        player2.record_game(outcome2, update2.rating_change, tier_config2)

        return update1, update2

    def get_win_probability(self, rating1: float, rating2: float) -> float:
        """Get win probability for player 1 against player 2.

        This is the expected score, which represents the probability
        of winning plus half the probability of drawing.

        Args:
            rating1: Player 1's rating
            rating2: Player 2's rating

        Returns:
            Probability between 0 and 1.
        """
        return calculate_expected_score(rating1, rating2)

    def get_rating_needed_for_probability(
        self, opponent_rating: float, win_probability: float
    ) -> float:
        """Calculate rating needed to achieve a certain win probability.

        Args:
            opponent_rating: Opponent's rating
            win_probability: Desired win probability (0-1)

        Returns:
            Required rating.
        """
        if win_probability <= 0 or win_probability >= 1:
            raise ValueError("Win probability must be between 0 and 1")

        # Inverse of expected score formula
        # E = 1 / (1 + 10^((R_B - R_A) / 400))
        # Solving for R_A:
        # R_A = R_B - 400 * log10(1/E - 1)
        return opponent_rating - 400 * math.log10(1.0 / win_probability - 1.0)

    def get_update_history(
        self, player_id: str | None = None, limit: int = 100
    ) -> list[RatingUpdate]:
        """Get recent rating updates.

        Args:
            player_id: Filter by player ID (None for all)
            limit: Maximum updates to return

        Returns:
            List of RatingUpdate objects.
        """
        if player_id:
            updates = [u for u in self._update_history if u.player_id == player_id]
        else:
            updates = self._update_history

        return updates[-limit:]

    def clear_history(self) -> None:
        """Clear update history."""
        self._update_history.clear()

    def simulate_rating_change(
        self,
        your_rating: float,
        opponent_rating: float,
        outcome: MatchOutcome,
        games_played: int = 0,
    ) -> float:
        """Simulate rating change without recording.

        Args:
            your_rating: Your current rating
            opponent_rating: Opponent's rating
            outcome: Expected outcome
            games_played: Your games played

        Returns:
            Simulated rating change.
        """
        if outcome == MatchOutcome.WIN:
            score = 1.0
        elif outcome == MatchOutcome.LOSS:
            score = 0.0
        else:
            score = self.config.draw_value

        expected = calculate_expected_score(your_rating, opponent_rating)
        k_factor = self.config.get_k_factor(games_played)

        return calculate_rating_change(your_rating, expected, score, k_factor)


class RatingLeaderboard:
    """Maintains a sorted leaderboard of player ratings.

    Example:
        >>> leaderboard = RatingLeaderboard()
        >>> leaderboard.update("alice", 1600)
        >>> leaderboard.update("bob", 1550)
        >>> top = leaderboard.get_top(10)
    """

    def __init__(self) -> None:
        """Initialize empty leaderboard."""
        self._ratings: dict[str, float] = {}

    def update(self, player_id: str, rating: float) -> None:
        """Update a player's rating.

        Args:
            player_id: Player identifier
            rating: New rating
        """
        self._ratings[player_id] = rating

    def remove(self, player_id: str) -> None:
        """Remove a player from the leaderboard.

        Args:
            player_id: Player to remove
        """
        self._ratings.pop(player_id, None)

    def get_rank(self, player_id: str) -> int | None:
        """Get a player's rank (1-indexed).

        Args:
            player_id: Player to look up

        Returns:
            Rank or None if not found.
        """
        if player_id not in self._ratings:
            return None

        sorted_players = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        for i, (pid, _) in enumerate(sorted_players):
            if pid == player_id:
                return i + 1
        return None

    def get_top(self, n: int = 10) -> list[tuple[str, float, int]]:
        """Get top N players.

        Args:
            n: Number of players to return

        Returns:
            List of (player_id, rating, rank) tuples.
        """
        sorted_players = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(sorted_players[:n])]

    def get_rating(self, player_id: str) -> float | None:
        """Get a player's rating.

        Args:
            player_id: Player to look up

        Returns:
            Rating or None if not found.
        """
        return self._ratings.get(player_id)

    def get_players_in_range(self, min_rating: float, max_rating: float) -> list[tuple[str, float]]:
        """Get players within a rating range.

        Args:
            min_rating: Minimum rating
            max_rating: Maximum rating

        Returns:
            List of (player_id, rating) tuples.
        """
        return [
            (pid, rating)
            for pid, rating in self._ratings.items()
            if min_rating <= rating <= max_rating
        ]

    @property
    def size(self) -> int:
        """Get number of players on leaderboard."""
        return len(self._ratings)

    def __len__(self) -> int:
        """Get number of players."""
        return len(self._ratings)
