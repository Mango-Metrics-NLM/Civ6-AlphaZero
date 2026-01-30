"""Match history persistence and statistics.

This module provides SQLite-based persistence for match history,
player statistics, and tournament data.

Usage:
    from alpha_zero.tournament.history import MatchHistory

    history = MatchHistory("tournament.db")

    # Record a match
    history.record_match(match_result)

    # Query match history
    matches = history.get_player_matches("player1", limit=10)

    # Get statistics
    stats = history.get_player_statistics("player1")
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from alpha_zero.tournament.config import (
    MatchResult,
    TournamentConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MatchRecord:
    """A single match record for persistence.

    Attributes:
        match_id: Unique match identifier
        player1_id: First player ID
        player2_id: Second player ID
        player1_rating_before: P1 rating before match
        player2_rating_before: P2 rating before match
        player1_rating_after: P1 rating after match
        player2_rating_after: P2 rating after match
        outcome: Match outcome
        timestamp: When match occurred
        game_length: Number of turns/moves
        metadata: Additional match data
    """

    match_id: str
    player1_id: str
    player2_id: str
    player1_rating_before: float
    player2_rating_before: float
    player1_rating_after: float
    player2_rating_after: float
    outcome: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    game_length: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_match_result(
        cls,
        result: MatchResult,
        player1_rating_after: float,
        player2_rating_after: float,
    ) -> MatchRecord:
        """Create from MatchResult.

        Args:
            result: Match result
            player1_rating_after: P1 rating after match
            player2_rating_after: P2 rating after match

        Returns:
            MatchRecord instance.
        """
        # Determine outcome for player1
        outcome = result.get_outcome_for_player(result.player1_id)

        return cls(
            match_id=result.match_id,
            player1_id=result.player1_id,
            player2_id=result.player2_id,
            player1_rating_before=result.player1_rating_before,
            player2_rating_before=result.player2_rating_before,
            player1_rating_after=player1_rating_after,
            player2_rating_after=player2_rating_after,
            outcome=outcome.value,
            timestamp=result.timestamp,
            game_length=int(result.duration_seconds),
            metadata={"game_details": result.game_details} if result.game_details else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "player1_id": self.player1_id,
            "player2_id": self.player2_id,
            "player1_rating_before": self.player1_rating_before,
            "player2_rating_before": self.player2_rating_before,
            "player1_rating_after": self.player1_rating_after,
            "player2_rating_after": self.player2_rating_after,
            "outcome": self.outcome,
            "timestamp": self.timestamp,
            "game_length": self.game_length,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatchRecord:
        """Create from dictionary."""
        return cls(**data)

    @classmethod
    def from_row(cls, row: tuple) -> MatchRecord:
        """Create from database row."""
        return cls(
            match_id=row[0],
            player1_id=row[1],
            player2_id=row[2],
            player1_rating_before=row[3],
            player2_rating_before=row[4],
            player1_rating_after=row[5],
            player2_rating_after=row[6],
            outcome=row[7],
            timestamp=row[8],
            game_length=row[9] if len(row) > 9 else 0,
            metadata=json.loads(row[10]) if len(row) > 10 and row[10] else {},
        )


@dataclass
class PlayerStatistics:
    """Aggregated player statistics.

    Attributes:
        player_id: Player identifier
        total_games: Total games played
        wins: Total wins
        losses: Total losses
        draws: Total draws
        current_rating: Current rating
        peak_rating: Highest rating achieved
        lowest_rating: Lowest rating achieved
        avg_game_length: Average game length
        current_streak: Current win/loss streak
        longest_win_streak: Longest win streak
        longest_loss_streak: Longest loss streak
        last_played: Last game timestamp
    """

    player_id: str
    total_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    current_rating: float = 1500.0
    peak_rating: float = 1500.0
    lowest_rating: float = 1500.0
    avg_game_length: float = 0.0
    current_streak: int = 0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    last_played: str | None = None

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    @property
    def rating_volatility(self) -> float:
        """Calculate rating volatility (peak - lowest)."""
        return self.peak_rating - self.lowest_rating

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "total_games": self.total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "current_rating": self.current_rating,
            "peak_rating": self.peak_rating,
            "lowest_rating": self.lowest_rating,
            "avg_game_length": self.avg_game_length,
            "current_streak": self.current_streak,
            "longest_win_streak": self.longest_win_streak,
            "longest_loss_streak": self.longest_loss_streak,
            "last_played": self.last_played,
        }


class MatchHistory:
    """SQLite-based match history persistence.

    This class manages persistent storage of match records
    and provides query capabilities for statistics.

    Example:
        >>> history = MatchHistory("tournament.db")
        >>>
        >>> # Record matches
        >>> history.record_match(match_result, new_rating1, new_rating2)
        >>>
        >>> # Query history
        >>> recent = history.get_player_matches("player1", limit=10)
        >>> stats = history.get_player_statistics("player1")
    """

    def __init__(
        self,
        database_path: str | Path = "tournament.db",
        config: TournamentConfig | None = None,
    ):
        """Initialize match history.

        Args:
            database_path: Path to SQLite database
            config: Optional tournament configuration
        """
        self.database_path = Path(database_path)
        self.config = config or TournamentConfig()

        # Ensure directory exists
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()
        logger.info(f"MatchHistory initialized at {self.database_path}")

    @contextmanager
    def _get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection context manager."""
        conn = sqlite3.connect(self.database_path)
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Matches table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    player1_id TEXT NOT NULL,
                    player2_id TEXT NOT NULL,
                    player1_rating_before REAL NOT NULL,
                    player2_rating_before REAL NOT NULL,
                    player1_rating_after REAL NOT NULL,
                    player2_rating_after REAL NOT NULL,
                    outcome TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    game_length INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)

            # Players table for persistent stats
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS players (
                    player_id TEXT PRIMARY KEY,
                    current_rating REAL DEFAULT 1500.0,
                    peak_rating REAL DEFAULT 1500.0,
                    lowest_rating REAL DEFAULT 1500.0,
                    total_games INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    draws INTEGER DEFAULT 0,
                    current_streak INTEGER DEFAULT 0,
                    longest_win_streak INTEGER DEFAULT 0,
                    longest_loss_streak INTEGER DEFAULT 0,
                    last_played TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_player1
                ON matches(player1_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_player2
                ON matches(player2_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_matches_timestamp
                ON matches(timestamp)
            """)

    def record_match(
        self,
        result: MatchResult,
        player1_rating_after: float,
        player2_rating_after: float,
    ) -> MatchRecord:
        """Record a match result.

        Args:
            result: Match result
            player1_rating_after: P1 rating after match
            player2_rating_after: P2 rating after match

        Returns:
            Created MatchRecord.
        """
        record = MatchRecord.from_match_result(result, player1_rating_after, player2_rating_after)

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Insert match record
            cursor.execute(
                """
                INSERT INTO matches (
                    match_id, player1_id, player2_id,
                    player1_rating_before, player2_rating_before,
                    player1_rating_after, player2_rating_after,
                    outcome, timestamp, game_length, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.match_id,
                    record.player1_id,
                    record.player2_id,
                    record.player1_rating_before,
                    record.player2_rating_before,
                    record.player1_rating_after,
                    record.player2_rating_after,
                    record.outcome,
                    record.timestamp,
                    record.game_length,
                    json.dumps(record.metadata),
                ),
            )

            # Update player statistics
            self._update_player_stats(
                cursor,
                record.player1_id,
                player1_rating_after,
                record.outcome == "win",
                record.outcome == "draw",
                record.timestamp,
            )

            # Determine opponent outcome
            if record.outcome == "win":
                p2_won = False
                p2_draw = False
            elif record.outcome == "loss":
                p2_won = True
                p2_draw = False
            else:
                p2_won = False
                p2_draw = True

            self._update_player_stats(
                cursor,
                record.player2_id,
                player2_rating_after,
                p2_won,
                p2_draw,
                record.timestamp,
            )

        logger.debug(f"Recorded match {record.match_id}")
        return record

    def _update_player_stats(
        self,
        cursor: sqlite3.Cursor,
        player_id: str,
        new_rating: float,
        won: bool,
        drew: bool,
        timestamp: str,
    ) -> None:
        """Update player statistics after a match."""
        now = datetime.utcnow().isoformat()

        # Check if player exists
        cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()

        if row is None:
            # Create new player record
            cursor.execute(
                """
                INSERT INTO players (
                    player_id, current_rating, peak_rating, lowest_rating,
                    total_games, wins, losses, draws,
                    current_streak, longest_win_streak, longest_loss_streak,
                    last_played, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    player_id,
                    new_rating,
                    new_rating,
                    new_rating,
                    1,
                    1 if won else 0,
                    0 if won or drew else 1,
                    1 if drew else 0,
                    1 if won else (-1 if not drew else 0),
                    1 if won else 0,
                    0 if won or drew else 1,
                    timestamp,
                    now,
                    now,
                ),
            )
        else:
            # Update existing player
            current_streak = row[8]
            longest_win_streak = row[9]
            longest_loss_streak = row[10]

            if won:
                new_streak = max(1, current_streak + 1) if current_streak > 0 else 1
                longest_win_streak = max(longest_win_streak, new_streak)
            elif drew:
                new_streak = 0
            else:
                new_streak = min(-1, current_streak - 1) if current_streak < 0 else -1
                longest_loss_streak = max(longest_loss_streak, abs(new_streak))

            cursor.execute(
                """
                UPDATE players SET
                    current_rating = ?,
                    peak_rating = MAX(peak_rating, ?),
                    lowest_rating = MIN(lowest_rating, ?),
                    total_games = total_games + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    draws = draws + ?,
                    current_streak = ?,
                    longest_win_streak = ?,
                    longest_loss_streak = ?,
                    last_played = ?,
                    updated_at = ?
                WHERE player_id = ?
                """,
                (
                    new_rating,
                    new_rating,
                    new_rating,
                    1 if won else 0,
                    0 if won or drew else 1,
                    1 if drew else 0,
                    new_streak,
                    longest_win_streak,
                    longest_loss_streak,
                    timestamp,
                    now,
                    player_id,
                ),
            )

    def get_match(self, match_id: str) -> MatchRecord | None:
        """Get a specific match by ID.

        Args:
            match_id: Match identifier

        Returns:
            MatchRecord or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM matches WHERE match_id = ?", (match_id,))
            row = cursor.fetchone()

            if row:
                return MatchRecord.from_row(row)
            return None

    def get_player_matches(
        self,
        player_id: str,
        limit: int = 50,
        offset: int = 0,
        opponent_id: str | None = None,
    ) -> list[MatchRecord]:
        """Get match history for a player.

        Args:
            player_id: Player to get history for
            limit: Maximum matches to return
            offset: Number of matches to skip
            opponent_id: Filter to matches against specific opponent

        Returns:
            List of MatchRecord objects.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if opponent_id:
                cursor.execute(
                    """
                    SELECT * FROM matches
                    WHERE (player1_id = ? AND player2_id = ?)
                       OR (player1_id = ? AND player2_id = ?)
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (player_id, opponent_id, opponent_id, player_id, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM matches
                    WHERE player1_id = ? OR player2_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ? OFFSET ?
                    """,
                    (player_id, player_id, limit, offset),
                )

            return [MatchRecord.from_row(row) for row in cursor.fetchall()]

    def get_player_statistics(self, player_id: str) -> PlayerStatistics | None:
        """Get statistics for a player.

        Args:
            player_id: Player to get stats for

        Returns:
            PlayerStatistics or None.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM players WHERE player_id = ?", (player_id,))
            row = cursor.fetchone()

            if row is None:
                return None

            # Calculate average game length
            cursor.execute(
                """
                SELECT AVG(game_length) FROM matches
                WHERE player1_id = ? OR player2_id = ?
                """,
                (player_id, player_id),
            )
            avg_length = cursor.fetchone()[0] or 0.0

            return PlayerStatistics(
                player_id=row[0],
                current_rating=row[1],
                peak_rating=row[2],
                lowest_rating=row[3],
                total_games=row[4],
                wins=row[5],
                losses=row[6],
                draws=row[7],
                current_streak=row[8],
                longest_win_streak=row[9],
                longest_loss_streak=row[10],
                last_played=row[11],
                avg_game_length=avg_length,
            )

    def get_head_to_head(
        self,
        player1_id: str,
        player2_id: str,
    ) -> dict[str, Any]:
        """Get head-to-head statistics between two players.

        Args:
            player1_id: First player
            player2_id: Second player

        Returns:
            Dictionary with head-to-head stats.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get all matches between players
            cursor.execute(
                """
                SELECT * FROM matches
                WHERE (player1_id = ? AND player2_id = ?)
                   OR (player1_id = ? AND player2_id = ?)
                ORDER BY timestamp
                """,
                (player1_id, player2_id, player2_id, player1_id),
            )
            matches = [MatchRecord.from_row(row) for row in cursor.fetchall()]

            if not matches:
                return {
                    "total_games": 0,
                    "player1_wins": 0,
                    "player2_wins": 0,
                    "draws": 0,
                }

            p1_wins = 0
            p2_wins = 0
            draws = 0

            for match in matches:
                if match.player1_id == player1_id:
                    if match.outcome == "win":
                        p1_wins += 1
                    elif match.outcome == "loss":
                        p2_wins += 1
                    else:
                        draws += 1
                else:
                    if match.outcome == "win":
                        p2_wins += 1
                    elif match.outcome == "loss":
                        p1_wins += 1
                    else:
                        draws += 1

            return {
                "total_games": len(matches),
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "draws": draws,
                "player1_win_rate": p1_wins / len(matches) if matches else 0,
                "player2_win_rate": p2_wins / len(matches) if matches else 0,
            }

    def get_recent_matches(
        self,
        limit: int = 20,
        tier: str | None = None,
    ) -> list[MatchRecord]:
        """Get recent matches across all players.

        Args:
            limit: Maximum matches to return
            tier: Optional tier filter (requires metadata)

        Returns:
            List of recent MatchRecord objects.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            if tier:
                cursor.execute(
                    """
                    SELECT * FROM matches
                    WHERE metadata LIKE ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (f'%"tier": "{tier}"%', limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM matches
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (limit,),
                )

            return [MatchRecord.from_row(row) for row in cursor.fetchall()]

    def get_leaderboard(
        self,
        limit: int = 100,
        min_games: int = 10,
    ) -> list[tuple[str, float, int, float]]:
        """Get global leaderboard.

        Args:
            limit: Maximum entries to return
            min_games: Minimum games required

        Returns:
            List of (player_id, rating, games, win_rate) tuples.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT player_id, current_rating, total_games,
                       CAST(wins AS REAL) / total_games as win_rate
                FROM players
                WHERE total_games >= ?
                ORDER BY current_rating DESC
                LIMIT ?
                """,
                (min_games, limit),
            )

            return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

    def get_rating_history(
        self,
        player_id: str,
        limit: int = 100,
    ) -> list[tuple[str, float]]:
        """Get rating history for a player.

        Args:
            player_id: Player to get history for
            limit: Maximum entries

        Returns:
            List of (timestamp, rating) tuples.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT timestamp,
                       CASE WHEN player1_id = ? THEN player1_rating_after
                            ELSE player2_rating_after END as rating
                FROM matches
                WHERE player1_id = ? OR player2_id = ?
                ORDER BY timestamp
                LIMIT ?
                """,
                (player_id, player_id, player_id, limit),
            )

            return [(row[0], row[1]) for row in cursor.fetchall()]

    def get_global_statistics(self) -> dict[str, Any]:
        """Get global tournament statistics.

        Returns:
            Dictionary with global stats.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM matches")
            total_matches = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM players")
            total_players = cursor.fetchone()[0]

            cursor.execute("SELECT AVG(current_rating) FROM players")
            avg_rating = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT MAX(peak_rating) FROM players")
            highest_rating = cursor.fetchone()[0] or 0.0

            cursor.execute("SELECT AVG(game_length) FROM matches")
            avg_game_length = cursor.fetchone()[0] or 0.0

            return {
                "total_matches": total_matches,
                "total_players": total_players,
                "avg_rating": avg_rating,
                "highest_rating_ever": highest_rating,
                "avg_game_length": avg_game_length,
            }

    def delete_player(self, player_id: str) -> bool:
        """Delete a player and their match history.

        Args:
            player_id: Player to delete

        Returns:
            True if player was deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "DELETE FROM matches WHERE player1_id = ? OR player2_id = ?",
                (player_id, player_id),
            )

            cursor.execute("DELETE FROM players WHERE player_id = ?", (player_id,))

            return cursor.rowcount > 0

    def clear_all(self) -> None:
        """Clear all data from the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM matches")
            cursor.execute("DELETE FROM players")

        logger.warning("Cleared all match history data")

    def export_to_json(self, output_path: str | Path) -> None:
        """Export all data to JSON.

        Args:
            output_path: Path to write JSON file
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT * FROM matches")
            matches = [MatchRecord.from_row(row).to_dict() for row in cursor.fetchall()]

            cursor.execute("SELECT * FROM players")
            players = []
            for row in cursor.fetchall():
                players.append(
                    {
                        "player_id": row[0],
                        "current_rating": row[1],
                        "peak_rating": row[2],
                        "lowest_rating": row[3],
                        "total_games": row[4],
                        "wins": row[5],
                        "losses": row[6],
                        "draws": row[7],
                    }
                )

        data = {
            "matches": matches,
            "players": players,
            "exported_at": datetime.utcnow().isoformat(),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(matches)} matches to {output_path}")
