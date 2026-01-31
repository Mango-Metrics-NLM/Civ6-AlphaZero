"""Matchmaking system for tournament opponent selection.

This module provides queue-based matchmaking with rating proximity
and various opponent selection strategies.

Usage:
    from alpha_zero.tournament.matchmaking import MatchMaker
    from alpha_zero.tournament.config import TournamentConfig

    config = TournamentConfig()
    matchmaker = MatchMaker(config)

    # Add player to queue
    matchmaker.add_to_queue(player)

    # Find a match
    match = matchmaker.find_match(player)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from alpha_zero.tournament.config import (
    MatchmakingConfig,
    PlayerStats,
    TournamentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class QueueEntry:
    """Entry in the matchmaking queue.

    Attributes:
        player: Player stats
        joined_at: When player joined queue
        rating_range: Current acceptable rating range
        recent_opponents: Recent opponent IDs to avoid
        priority: Queue priority (higher = earlier matching)
    """

    player: PlayerStats
    joined_at: float = field(default_factory=time.time)
    rating_range: float = 200.0
    recent_opponents: list[str] = field(default_factory=list)
    priority: int = 0

    @property
    def wait_time(self) -> float:
        """Get time spent waiting in queue."""
        return time.time() - self.joined_at

    def is_compatible_with(
        self,
        other: QueueEntry,
        config: MatchmakingConfig,
    ) -> bool:
        """Check if this entry can be matched with another.

        Args:
            other: Other queue entry
            config: Matchmaking configuration

        Returns:
            True if compatible.
        """
        # Check rating range (use wider range of the two)
        max_range = max(self.rating_range, other.rating_range)
        rating_diff = abs(self.player.rating - other.player.rating)

        if rating_diff > max_range:
            return False

        # Check recent opponents
        if other.player.player_id in self.recent_opponents:
            return False
        if self.player.player_id in other.recent_opponents:
            return False

        return True

    def expand_range(self, config: MatchmakingConfig) -> None:
        """Expand rating range based on wait time."""
        self.rating_range = config.get_rating_range(self.wait_time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player.player_id,
            "rating": self.player.rating,
            "joined_at": self.joined_at,
            "wait_time": self.wait_time,
            "rating_range": self.rating_range,
            "priority": self.priority,
        }


@dataclass
class PendingMatch:
    """A pending match waiting to be confirmed/played.

    Attributes:
        match_id: Unique match identifier
        player1: First player
        player2: Second player
        created_at: When match was created
        expires_at: When match expires if not started
        status: Current status
    """

    match_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    player1: PlayerStats | None = None
    player2: PlayerStats | None = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 60)
    status: str = "pending"

    @property
    def is_expired(self) -> bool:
        """Check if match has expired."""
        return time.time() > self.expires_at

    @property
    def is_ready(self) -> bool:
        """Check if both players are present."""
        return self.player1 is not None and self.player2 is not None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "player1_id": self.player1.player_id if self.player1 else None,
            "player2_id": self.player2.player_id if self.player2 else None,
            "created_at": self.created_at,
            "status": self.status,
        }


class MatchMaker:
    """Matchmaking system for finding opponents.

    This class manages a queue of players looking for matches and
    pairs them based on rating proximity and other criteria.

    Example:
        >>> config = TournamentConfig()
        >>> matchmaker = MatchMaker(config)
        >>>
        >>> # Add player to queue
        >>> matchmaker.add_to_queue(player)
        >>>
        >>> # Find match
        >>> match = matchmaker.find_match(player)
        >>> if match:
        ...     # Start game
        ...     pass
    """

    def __init__(self, config: TournamentConfig):
        """Initialize matchmaker.

        Args:
            config: Tournament configuration
        """
        self.config = config
        self.matchmaking_config = config.matchmaking

        self._queue: dict[str, QueueEntry] = {}
        self._pending_matches: dict[str, PendingMatch] = {}
        self._recent_matches: deque[str] = deque(maxlen=100)

        # Callbacks
        self._on_match_found: Callable[[PendingMatch], None] | None = None

        logger.info("MatchMaker initialized")

    def add_to_queue(
        self,
        player: PlayerStats,
        recent_opponents: list[str] | None = None,
        priority: int = 0,
    ) -> QueueEntry:
        """Add a player to the matchmaking queue.

        Args:
            player: Player to add
            recent_opponents: Recent opponent IDs to avoid
            priority: Queue priority

        Returns:
            QueueEntry for the player.
        """
        entry = QueueEntry(
            player=player,
            rating_range=self.matchmaking_config.rating_range_initial,
            recent_opponents=recent_opponents or [],
            priority=priority,
        )

        self._queue[player.player_id] = entry

        logger.debug(
            f"Added {player.player_id} to queue "
            f"(rating={player.rating}, queue_size={len(self._queue)})"
        )

        return entry

    def remove_from_queue(self, player_id: str) -> bool:
        """Remove a player from the queue.

        Args:
            player_id: Player to remove

        Returns:
            True if player was in queue.
        """
        if player_id in self._queue:
            del self._queue[player_id]
            logger.debug(f"Removed {player_id} from queue")
            return True
        return False

    def is_in_queue(self, player_id: str) -> bool:
        """Check if player is in queue.

        Args:
            player_id: Player to check

        Returns:
            True if in queue.
        """
        return player_id in self._queue

    def get_queue_position(self, player_id: str) -> int | None:
        """Get player's position in queue.

        Args:
            player_id: Player to check

        Returns:
            Position (1-indexed) or None if not in queue.
        """
        if player_id not in self._queue:
            return None

        # Sort by priority and wait time
        sorted_entries = sorted(
            self._queue.values(),
            key=lambda e: (-e.priority, e.joined_at),
        )

        for i, entry in enumerate(sorted_entries):
            if entry.player.player_id == player_id:
                return i + 1

        return None

    def find_match(self, player_id: str) -> PendingMatch | None:
        """Try to find a match for a player.

        Args:
            player_id: Player looking for match

        Returns:
            PendingMatch if found, None otherwise.
        """
        if player_id not in self._queue:
            return None

        entry = self._queue[player_id]

        # Expand rating range based on wait time
        entry.expand_range(self.matchmaking_config)

        # Find compatible opponents
        candidates = []
        for other_id, other_entry in self._queue.items():
            if other_id == player_id:
                continue

            if entry.is_compatible_with(other_entry, self.matchmaking_config):
                # Score based on rating difference (prefer closer)
                rating_diff = abs(entry.player.rating - other_entry.player.rating)
                score = 1000 - rating_diff

                # Bonus for similar games played
                if self.matchmaking_config.prefer_similar_games_played:
                    games_diff = abs(entry.player.games_played - other_entry.player.games_played)
                    score += max(0, 100 - games_diff)

                candidates.append((other_entry, score))

        if not candidates:
            return None

        # Select best candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        opponent_entry = candidates[0][0]

        # Create pending match
        match = PendingMatch(
            player1=entry.player,
            player2=opponent_entry.player,
        )

        # Remove both from queue
        self.remove_from_queue(player_id)
        self.remove_from_queue(opponent_entry.player.player_id)

        # Store pending match
        self._pending_matches[match.match_id] = match

        logger.info(
            f"Match found: {entry.player.player_id} vs {opponent_entry.player.player_id} "
            f"(ratings: {entry.player.rating} vs {opponent_entry.player.rating})"
        )

        # Fire callback
        if self._on_match_found:
            self._on_match_found(match)

        return match

    def process_queue(self) -> list[PendingMatch]:
        """Process the entire queue and create matches.

        Returns:
            List of created matches.
        """
        matches = []
        processed = set()

        # Sort entries by priority and wait time
        sorted_entries = sorted(
            self._queue.values(),
            key=lambda e: (-e.priority, e.joined_at),
        )

        for entry in sorted_entries:
            if entry.player.player_id in processed:
                continue

            match = self.find_match(entry.player.player_id)
            if match:
                matches.append(match)
                processed.add(match.player1.player_id)
                processed.add(match.player2.player_id)

        return matches

    def get_pending_match(self, match_id: str) -> PendingMatch | None:
        """Get a pending match by ID.

        Args:
            match_id: Match identifier

        Returns:
            PendingMatch or None.
        """
        return self._pending_matches.get(match_id)

    def confirm_match(self, match_id: str) -> bool:
        """Confirm a pending match.

        Args:
            match_id: Match to confirm

        Returns:
            True if confirmed successfully.
        """
        match = self._pending_matches.get(match_id)
        if not match or not match.is_ready:
            return False

        match.status = "confirmed"
        self._recent_matches.append(match_id)

        logger.debug(f"Match {match_id} confirmed")
        return True

    def cancel_match(self, match_id: str, requeue: bool = True) -> bool:
        """Cancel a pending match.

        Args:
            match_id: Match to cancel
            requeue: Whether to add players back to queue

        Returns:
            True if cancelled successfully.
        """
        match = self._pending_matches.pop(match_id, None)
        if not match:
            return False

        match.status = "cancelled"

        if requeue:
            if match.player1:
                self.add_to_queue(match.player1)
            if match.player2:
                self.add_to_queue(match.player2)

        logger.debug(f"Match {match_id} cancelled (requeue={requeue})")
        return True

    def cleanup_expired(self) -> list[str]:
        """Clean up expired pending matches.

        Returns:
            List of expired match IDs.
        """
        expired = []
        for match_id, match in list(self._pending_matches.items()):
            if match.is_expired:
                self.cancel_match(match_id, requeue=True)
                expired.append(match_id)

        return expired

    def set_on_match_found(self, callback: Callable[[PendingMatch], None]) -> None:
        """Set callback for when match is found.

        Args:
            callback: Function to call with PendingMatch
        """
        self._on_match_found = callback

    def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics.

        Returns:
            Dictionary with queue stats.
        """
        if not self._queue:
            return {
                "queue_size": 0,
                "pending_matches": len(self._pending_matches),
            }

        ratings = [e.player.rating for e in self._queue.values()]
        wait_times = [e.wait_time for e in self._queue.values()]

        return {
            "queue_size": len(self._queue),
            "pending_matches": len(self._pending_matches),
            "avg_rating": sum(ratings) / len(ratings),
            "min_rating": min(ratings),
            "max_rating": max(ratings),
            "avg_wait_time": sum(wait_times) / len(wait_times),
            "max_wait_time": max(wait_times),
        }

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)


class OpponentSelector:
    """Selects opponents for AI training matches.

    This class provides methods to select appropriate opponents
    for training based on various strategies.
    """

    def __init__(
        self,
        player_pool: list[PlayerStats] | None = None,
        config: TournamentConfig | None = None,
    ):
        """Initialize opponent selector.

        Args:
            player_pool: Pool of available opponents
            config: Tournament configuration
        """
        self.player_pool = player_pool or []
        self.config = config or TournamentConfig()

    def set_player_pool(self, players: list[PlayerStats]) -> None:
        """Set the player pool.

        Args:
            players: List of available players
        """
        self.player_pool = players

    def select_by_rating_proximity(
        self,
        target_rating: float,
        num_opponents: int = 1,
        exclude_ids: set[str] | None = None,
    ) -> list[PlayerStats]:
        """Select opponents closest to target rating.

        Args:
            target_rating: Target rating to match
            num_opponents: Number of opponents to select
            exclude_ids: Player IDs to exclude

        Returns:
            List of selected opponents.
        """
        exclude_ids = exclude_ids or set()

        candidates = [p for p in self.player_pool if p.player_id not in exclude_ids]

        if not candidates:
            return []

        # Sort by rating difference
        candidates.sort(key=lambda p: abs(p.rating - target_rating))

        return candidates[:num_opponents]

    def select_by_tier(
        self,
        tier_name: str,
        num_opponents: int = 1,
        exclude_ids: set[str] | None = None,
    ) -> list[PlayerStats]:
        """Select opponents from a specific tier.

        Args:
            tier_name: Name of tier
            num_opponents: Number to select
            exclude_ids: Player IDs to exclude

        Returns:
            List of selected opponents.
        """
        exclude_ids = exclude_ids or set()

        candidates = [
            p for p in self.player_pool if p.tier == tier_name and p.player_id not in exclude_ids
        ]

        if len(candidates) <= num_opponents:
            return candidates

        # Random selection
        import random

        return random.sample(candidates, num_opponents)

    def select_diverse(
        self,
        num_opponents: int,
        target_rating: float | None = None,
        exclude_ids: set[str] | None = None,
    ) -> list[PlayerStats]:
        """Select diverse opponents across rating range.

        Args:
            num_opponents: Number to select
            target_rating: Optional center rating
            exclude_ids: Player IDs to exclude

        Returns:
            List of selected opponents.
        """
        exclude_ids = exclude_ids or set()

        candidates = [p for p in self.player_pool if p.player_id not in exclude_ids]

        if len(candidates) <= num_opponents:
            return candidates

        # Select from different rating bands
        candidates.sort(key=lambda p: p.rating)
        step = len(candidates) / num_opponents

        selected = []
        for i in range(num_opponents):
            idx = int(i * step)
            if idx < len(candidates):
                selected.append(candidates[idx])

        return selected

    def select_challenging(
        self,
        player_rating: float,
        challenge_factor: float = 1.2,
        num_opponents: int = 1,
        exclude_ids: set[str] | None = None,
    ) -> list[PlayerStats]:
        """Select slightly stronger opponents.

        Args:
            player_rating: Player's rating
            challenge_factor: How much stronger (1.0 = same, 1.2 = 20% stronger)
            num_opponents: Number to select
            exclude_ids: IDs to exclude

        Returns:
            List of challenging opponents.
        """
        target_rating = player_rating * challenge_factor

        exclude_ids = exclude_ids or set()
        candidates = [
            p
            for p in self.player_pool
            if p.player_id not in exclude_ids
            and p.rating >= player_rating  # Must be at least equal
        ]

        if not candidates:
            return []

        # Sort by closeness to target
        candidates.sort(key=lambda p: abs(p.rating - target_rating))

        return candidates[:num_opponents]
