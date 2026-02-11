"""League and tier system for tournament progression.

This module provides tier-based progression with promotion/demotion,
seasons, and leaderboard management.

Usage:
    from alpha_zero.tournament.league import LeagueManager
    from alpha_zero.tournament.config import TournamentConfig

    config = TournamentConfig()
    league = LeagueManager(config)

    # Check player's tier
    tier = league.get_player_tier(player)

    # Process tier changes after match
    league.process_match_result(player1, player2, result)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from alpha_zero.tournament.config import (
    MatchResult,
    PlayerStats,
    TierConfig,
    TournamentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TierProgress:
    """Tracks a player's progress within a tier.

    Attributes:
        tier_name: Current tier name
        wins_in_tier: Wins in current tier
        losses_in_tier: Losses in current tier
        streak: Current win/loss streak (positive=wins, negative=losses)
        at_promotion_threshold: Whether at promotion series
        at_demotion_threshold: Whether at demotion risk
        promotion_wins: Wins in promotion series
        promotion_losses: Losses in promotion series
    """

    tier_name: str = "Bronze"
    wins_in_tier: int = 0
    losses_in_tier: int = 0
    streak: int = 0
    at_promotion_threshold: bool = False
    at_demotion_threshold: bool = False
    promotion_wins: int = 0
    promotion_losses: int = 0

    @property
    def games_in_tier(self) -> int:
        """Total games played in current tier."""
        return self.wins_in_tier + self.losses_in_tier

    @property
    def win_rate_in_tier(self) -> float:
        """Win rate in current tier."""
        if self.games_in_tier == 0:
            return 0.0
        return self.wins_in_tier / self.games_in_tier

    def record_win(self) -> None:
        """Record a win."""
        self.wins_in_tier += 1
        if self.streak > 0:
            self.streak += 1
        else:
            self.streak = 1

        if self.at_promotion_threshold:
            self.promotion_wins += 1

    def record_loss(self) -> None:
        """Record a loss."""
        self.losses_in_tier += 1
        if self.streak < 0:
            self.streak -= 1
        else:
            self.streak = -1

        if self.at_promotion_threshold:
            self.promotion_losses += 1

    def reset_for_new_tier(self, tier_name: str) -> None:
        """Reset progress for a new tier."""
        self.tier_name = tier_name
        self.wins_in_tier = 0
        self.losses_in_tier = 0
        self.at_promotion_threshold = False
        self.at_demotion_threshold = False
        self.promotion_wins = 0
        self.promotion_losses = 0
        # Keep streak

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier_name": self.tier_name,
            "wins_in_tier": self.wins_in_tier,
            "losses_in_tier": self.losses_in_tier,
            "streak": self.streak,
            "at_promotion_threshold": self.at_promotion_threshold,
            "at_demotion_threshold": self.at_demotion_threshold,
            "promotion_wins": self.promotion_wins,
            "promotion_losses": self.promotion_losses,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TierProgress:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SeasonInfo:
    """Information about a tournament season.

    Attributes:
        season_id: Unique season identifier
        name: Season name
        start_date: Season start date
        end_date: Optional end date
        is_active: Whether season is currently active
        rewards: Rewards for different tier placements
    """

    season_id: str
    name: str = "Season 1"
    start_date: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    end_date: str | None = None
    is_active: bool = True
    rewards: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "season_id": self.season_id,
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "is_active": self.is_active,
            "rewards": self.rewards,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SeasonInfo:
        """Create from dictionary."""
        return cls(**data)


class LeagueManager:
    """Manages tier-based league progression.

    This class handles:
    - Tier assignment based on rating
    - Promotion and demotion logic
    - Season management
    - Tier statistics

    Example:
        >>> config = TournamentConfig()
        >>> league = LeagueManager(config)
        >>>
        >>> # Get player tier
        >>> tier = league.get_tier_for_rating(1600)
        >>> print(tier.name)  # "Gold"
        >>>
        >>> # Process tier changes
        >>> promoted, demoted = league.update_player_tier(player)
    """

    def __init__(self, config: TournamentConfig) -> None:
        """Initialize league manager.

        Args:
            config: Tournament configuration
        """
        self.config = config
        self._player_progress: dict[str, TierProgress] = {}
        self._tier_populations: dict[str, set[str]] = defaultdict(set)
        self._current_season: SeasonInfo | None = None
        self._seasons: list[SeasonInfo] = []

        logger.info(f"LeagueManager initialized with {len(config.tiers)} tiers")

    def get_tier_for_rating(self, rating: float) -> TierConfig | None:
        """Get the tier for a given rating.

        Args:
            rating: Player rating

        Returns:
            TierConfig for the appropriate tier, or None if not found.
        """
        return self.config.get_tier_for_rating(rating)

    def get_player_tier(self, player: PlayerStats) -> TierConfig | None:
        """Get a player's current tier.

        Args:
            player: Player stats

        Returns:
            TierConfig for player's tier, or None if not found.
        """
        return self.get_tier_for_rating(player.rating)

    def _get_default_tier(self) -> TierConfig:
        """Get the default (lowest) tier.

        Returns:
            First tier in configuration, or a default Bronze tier.
        """
        if self.config.tiers:
            return self.config.tiers[0]
        return TierConfig(name="Bronze", rating_min=0, rating_max=1200)

    def get_player_progress(self, player_id: str) -> TierProgress:
        """Get a player's tier progress.

        Args:
            player_id: Player identifier

        Returns:
            TierProgress for the player.
        """
        if player_id not in self._player_progress:
            self._player_progress[player_id] = TierProgress()
        return self._player_progress[player_id]

    def register_player(self, player: PlayerStats) -> TierProgress:
        """Register a player in the league.

        Args:
            player: Player to register

        Returns:
            Initial tier progress.
        """
        tier = self.get_player_tier(player) or self._get_default_tier()
        progress = TierProgress(tier_name=tier.name)

        self._player_progress[player.player_id] = progress
        self._tier_populations[tier.name].add(player.player_id)

        logger.debug(f"Registered {player.player_id} in {tier.name}")
        return progress

    def update_player_tier(
        self,
        player: PlayerStats,
        won_last_game: bool,
    ) -> tuple[bool, bool]:
        """Update a player's tier based on their rating and performance.

        Args:
            player: Player stats
            won_last_game: Whether player won their last game

        Returns:
            Tuple of (was_promoted, was_demoted).
        """
        progress = self.get_player_progress(player.player_id)
        current_tier = self.config.get_tier_by_name(progress.tier_name) or self._get_default_tier()
        new_tier = self.get_tier_for_rating(player.rating) or self._get_default_tier()

        # Record game result
        if won_last_game:
            progress.record_win()
        else:
            progress.record_loss()

        was_promoted = False
        was_demoted = False

        # Check for tier change based on rating
        if new_tier.name != current_tier.name:
            tier_index_old = self._get_tier_index(current_tier.name)
            tier_index_new = self._get_tier_index(new_tier.name)

            if tier_index_new > tier_index_old:
                # Promotion
                was_promoted = self._process_promotion(player, progress, new_tier)
            else:
                # Demotion
                was_demoted = self._process_demotion(player, progress, new_tier)

        # Update promotion/demotion thresholds based on win rate
        self._update_thresholds(progress, current_tier)

        return was_promoted, was_demoted

    def _get_tier_index(self, tier_name: str) -> int:
        """Get index of tier (higher = better)."""
        for i, tier in enumerate(self.config.tiers):
            if tier.name == tier_name:
                return i
        return 0

    def _process_promotion(
        self,
        player: PlayerStats,
        progress: TierProgress,
        new_tier: TierConfig,
    ) -> bool:
        """Process a promotion.

        Args:
            player: Player being promoted
            progress: Player's tier progress
            new_tier: New tier

        Returns:
            True if promotion occurred.
        """
        old_tier = progress.tier_name

        # Remove from old tier
        self._tier_populations[old_tier].discard(player.player_id)

        # Add to new tier
        progress.reset_for_new_tier(new_tier.name)
        self._tier_populations[new_tier.name].add(player.player_id)

        logger.info(f"{player.player_id} promoted from {old_tier} to {new_tier.name}")
        return True

    def _process_demotion(
        self,
        player: PlayerStats,
        progress: TierProgress,
        new_tier: TierConfig,
    ) -> bool:
        """Process a demotion.

        Args:
            player: Player being demoted
            progress: Player's tier progress
            new_tier: New tier

        Returns:
            True if demotion occurred.
        """
        old_tier = progress.tier_name

        # Can't demote from lowest tier
        if self._get_tier_index(old_tier) == 0:
            return False

        # Remove from old tier
        self._tier_populations[old_tier].discard(player.player_id)

        # Add to new tier
        progress.reset_for_new_tier(new_tier.name)
        self._tier_populations[new_tier.name].add(player.player_id)

        logger.info(f"{player.player_id} demoted from {old_tier} to {new_tier.name}")
        return True

    def _update_thresholds(
        self,
        progress: TierProgress,
        tier: TierConfig,
    ) -> None:
        """Update promotion/demotion thresholds based on performance."""
        if progress.games_in_tier < 10:
            # Need minimum games before threshold updates
            return

        win_rate = progress.win_rate_in_tier

        # Check promotion threshold
        if win_rate >= tier.promotion_threshold:
            progress.at_promotion_threshold = True
        else:
            progress.at_promotion_threshold = False
            progress.promotion_wins = 0
            progress.promotion_losses = 0

        # Check demotion threshold
        progress.at_demotion_threshold = win_rate < tier.demotion_threshold

    def process_match_result(
        self,
        player1: PlayerStats,
        player2: PlayerStats,
        result: MatchResult,
    ) -> dict[str, tuple[bool, bool]]:
        """Process tier changes after a match.

        Args:
            player1: First player
            player2: Second player
            result: Match result

        Returns:
            Dictionary mapping player_id to (was_promoted, was_demoted).
        """
        changes = {}

        # Determine winners based on winner_id or score comparison
        if result.winner_id is not None:
            p1_won = result.winner_id == player1.player_id
            p2_won = result.winner_id == player2.player_id
        else:
            # Fallback to score comparison when winner_id is None (e.g., draws or timed games)
            p1_score = getattr(result, "player1_score", None)
            p2_score = getattr(result, "player2_score", None)
            if p1_score is not None and p2_score is not None:
                p1_won = p1_score > p2_score
                p2_won = p2_score > p1_score
            else:
                # True draw - neither wins
                p1_won = False
                p2_won = False

        changes[player1.player_id] = self.update_player_tier(player1, p1_won)
        changes[player2.player_id] = self.update_player_tier(player2, p2_won)

        return changes

    def get_tier_statistics(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all tiers.

        Returns:
            Dictionary mapping tier name to statistics.
        """
        stats = {}
        for tier in self.config.tiers:
            population = self._tier_populations.get(tier.name, set())
            stats[tier.name] = {
                "name": tier.name,
                "rating_min": tier.rating_min,
                "rating_max": tier.rating_max,
                "population": len(population),
                "color": tier.color,
            }
        return stats

    def get_tier_leaderboard(
        self,
        tier_name: str,
        players: dict[str, PlayerStats],
        limit: int = 10,
    ) -> list[tuple[str, float, int]]:
        """Get leaderboard for a specific tier.

        Args:
            tier_name: Tier to get leaderboard for
            players: Dictionary of player_id to PlayerStats
            limit: Maximum entries to return

        Returns:
            List of (player_id, rating, rank) tuples.
        """
        tier_players = self._tier_populations.get(tier_name, set())
        tier_player_stats = [(pid, players[pid].rating) for pid in tier_players if pid in players]

        # Sort by rating descending
        tier_player_stats.sort(key=lambda x: x[1], reverse=True)

        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(tier_player_stats[:limit])]

    def start_season(
        self,
        season_id: str,
        name: str = "New Season",
        reset_ratings: bool = False,
        soft_reset_factor: float = 0.5,
    ) -> SeasonInfo:
        """Start a new season.

        Args:
            season_id: Unique season identifier
            name: Season name
            reset_ratings: Whether to reset ratings
            soft_reset_factor: Factor for soft reset (0=full reset, 1=no change)

        Returns:
            SeasonInfo for new season.
        """
        # End current season if active
        if self._current_season and self._current_season.is_active:
            self._current_season.is_active = False
            self._current_season.end_date = datetime.now(UTC).isoformat()

        # Create new season
        season = SeasonInfo(
            season_id=season_id,
            name=name,
            is_active=True,
        )

        self._current_season = season
        self._seasons.append(season)

        # Reset tier progress
        for progress in self._player_progress.values():
            progress.wins_in_tier = 0
            progress.losses_in_tier = 0
            progress.at_promotion_threshold = False
            progress.at_demotion_threshold = False
            progress.promotion_wins = 0
            progress.promotion_losses = 0

        logger.info(f"Started season: {name}")
        return season

    def end_season(self) -> SeasonInfo | None:
        """End the current season.

        Returns:
            Ended SeasonInfo or None.
        """
        if not self._current_season:
            return None

        self._current_season.is_active = False
        self._current_season.end_date = datetime.now(UTC).isoformat()

        ended_season = self._current_season
        self._current_season = None

        logger.info(f"Ended season: {ended_season.name}")
        return ended_season

    def get_current_season(self) -> SeasonInfo | None:
        """Get the current active season."""
        return self._current_season

    def get_season_history(self) -> list[SeasonInfo]:
        """Get all past seasons."""
        return [s for s in self._seasons if not s.is_active]

    def get_global_statistics(self) -> dict[str, Any]:
        """Get global league statistics.

        Returns:
            Dictionary with global stats.
        """
        total_players = sum(len(pop) for pop in self._tier_populations.values())

        tier_distribution = {
            tier.name: len(self._tier_populations.get(tier.name, set()))
            for tier in self.config.tiers
        }

        return {
            "total_players": total_players,
            "tier_distribution": tier_distribution,
            "num_tiers": len(self.config.tiers),
            "season_active": self._current_season is not None,
            "current_season": (self._current_season.name if self._current_season else None),
        }
