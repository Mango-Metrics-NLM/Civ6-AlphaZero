"""Tournament orchestration module for running tournaments.

This module provides the tournament orchestrator that coordinates
all tournament components: brackets, matchmaking, ELO ratings,
match history, and league management.

Usage:
    from alpha_zero.tournament.orchestrator import TournamentOrchestrator
    from alpha_zero.tournament.config import TournamentConfig

    config = TournamentConfig()
    orchestrator = TournamentOrchestrator(config)

    # Create and run a tournament
    bracket = orchestrator.create_bracket(
        players=["player1", "player2", "player3", "player4"],
        format=TournamentFormat.ROUND_ROBIN,
    )
    result = orchestrator.run_tournament(bracket)
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from alpha_zero.tournament.config import (
    MatchResult,
    PlayerStats,
    TournamentConfig,
)
from alpha_zero.tournament.elo import ELORatingSystem
from alpha_zero.tournament.history import MatchHistory
from alpha_zero.tournament.league import LeagueManager
from alpha_zero.tournament.matchmaking import MatchMaker

logger = logging.getLogger(__name__)


class TournamentFormat(str, Enum):
    """Tournament format types."""

    ROUND_ROBIN = "round_robin"
    SINGLE_ELIMINATION = "single_elimination"
    DOUBLE_ELIMINATION = "double_elimination"
    SWISS = "swiss"


@dataclass
class Matchup:
    """A single matchup in a tournament.

    Attributes:
        player_a: First player ID
        player_b: Second player ID
        round_num: Round number (0-indexed)
        result: Match result (None if not played yet)
    """

    player_a: str
    player_b: str
    round_num: int
    result: MatchResult | None = None

    @property
    def is_complete(self) -> bool:
        """Check if matchup has been played."""
        return self.result is not None

    @property
    def winner_id(self) -> str | None:
        """Get winner ID if match is complete."""
        if self.result is None:
            return None
        return self.result.winner_id

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_a": self.player_a,
            "player_b": self.player_b,
            "round_num": self.round_num,
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass
class Bracket:
    """Tournament bracket structure.

    Attributes:
        format: Tournament format
        rounds: List of rounds, each containing list of matchups
        players: List of player IDs in bracket
        seeding: Dictionary mapping player ID to seed number
        current_round: Current round number (0-indexed)
        results: Dictionary mapping round number to results
    """

    format: TournamentFormat
    rounds: list[list[Matchup]]
    players: list[str]
    seeding: dict[str, int]
    current_round: int = 0
    results: dict[int, list[MatchResult]] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if bracket is complete."""
        return self.current_round >= len(self.rounds)

    @property
    def total_rounds(self) -> int:
        """Get total number of rounds."""
        return len(self.rounds)

    @property
    def total_matches(self) -> int:
        """Get total number of matches."""
        return sum(len(round_matchups) for round_matchups in self.rounds)

    @property
    def matches_complete(self) -> int:
        """Get number of completed matches."""
        return sum(len(results) for results in self.results.values())

    def get_player_record(self, player_id: str) -> tuple[int, int, int]:
        """Get player's win-loss-draw record.

        Args:
            player_id: Player to get record for

        Returns:
            Tuple of (wins, losses, draws).
        """
        wins = 0
        losses = 0
        draws = 0

        for round_results in self.results.values():
            for result in round_results:
                if result.player1_id == player_id or result.player2_id == player_id:
                    if result.is_draw:
                        draws += 1
                    elif result.winner_id == player_id:
                        wins += 1
                    else:
                        losses += 1

        return wins, losses, draws

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.format.value,
            "players": self.players,
            "seeding": self.seeding,
            "current_round": self.current_round,
            "total_rounds": self.total_rounds,
            "total_matches": self.total_matches,
            "matches_complete": self.matches_complete,
        }


@dataclass
class PlayerStanding:
    """Player's standing in a tournament.

    Attributes:
        player_id: Player identifier
        rank: Current rank (1-indexed)
        wins: Number of wins
        losses: Number of losses
        draws: Number of draws
        elo_rating: Current ELO rating
        points: Tournament points (wins + 0.5 * draws)
    """

    player_id: str
    rank: int
    wins: int
    losses: int
    draws: int
    elo_rating: float
    points: float

    @property
    def total_games(self) -> int:
        """Get total games played."""
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "rank": self.rank,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "elo_rating": self.elo_rating,
            "points": self.points,
            "total_games": self.total_games,
            "win_rate": self.win_rate,
        }


@dataclass
class TournamentResult:
    """Complete tournament result.

    Attributes:
        bracket: Final bracket state
        winner: Winner's player ID
        final_standings: List of player standings
        total_games: Total games played
        duration_seconds: Tournament duration
        elo_changes: Dictionary of rating changes per player
    """

    bracket: Bracket
    winner: str
    final_standings: list[PlayerStanding]
    total_games: int
    duration_seconds: float
    elo_changes: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "format": self.bracket.format.value,
            "winner": self.winner,
            "total_games": self.total_games,
            "duration_seconds": self.duration_seconds,
            "standings": [s.to_dict() for s in self.final_standings],
            "elo_changes": self.elo_changes,
        }


class TournamentOrchestrator:
    """Orchestrates complete tournament execution.

    This class coordinates all tournament components:
    - Bracket creation and management
    - Match execution via arena runner
    - ELO rating updates
    - Match history persistence
    - League tier management
    - Matchmaking for Swiss format

    Example:
        >>> config = TournamentConfig()
        >>> orchestrator = TournamentOrchestrator(config)
        >>>
        >>> # Create round-robin tournament
        >>> bracket = orchestrator.create_bracket(
        ...     players=["player1", "player2", "player3", "player4"],
        ...     format=TournamentFormat.ROUND_ROBIN,
        ... )
        >>>
        >>> # Run tournament
        >>> result = orchestrator.run_tournament(bracket)
        >>> print(f"Winner: {result.winner}")
    """

    def __init__(
        self,
        *,
        config: TournamentConfig | None = None,
        elo_system: ELORatingSystem | None = None,
        match_history: MatchHistory | None = None,
        league_manager: LeagueManager | None = None,
        matchmaker: MatchMaker | None = None,
        arena_runner: Callable[[str, str, int], MatchResult] | None = None,
        games_per_match: int = 10,
    ) -> None:
        """Initialize tournament orchestrator.

        Args:
            config: Tournament configuration (default: TournamentConfig())
            elo_system: ELO rating system (default: created from config)
            match_history: Match history persistence (default: created from config)
            league_manager: League manager (default: created from config)
            matchmaker: Matchmaker for Swiss format (default: created from config)
            arena_runner: Callable that runs matches (player_a, player_b, num_games) -> result
            games_per_match: Default games per match
        """
        self.config = config or TournamentConfig()
        self.elo_system = elo_system or ELORatingSystem(self.config.elo)
        self.match_history = match_history or MatchHistory(
            self.config.database_path, self.config
        )
        self.league_manager = league_manager or LeagueManager(self.config)
        self.matchmaker = matchmaker or MatchMaker(self.config)
        self.arena_runner = arena_runner
        self.games_per_match = games_per_match

        # Track player stats
        self._player_stats: dict[str, PlayerStats] = {}

        logger.info(
            f"TournamentOrchestrator initialized: "
            f"games_per_match={games_per_match}, "
            f"has_arena_runner={arena_runner is not None}"
        )

    def _get_or_create_player_stats(self, player_id: str) -> PlayerStats:
        """Get or create player stats.

        Args:
            player_id: Player identifier

        Returns:
            PlayerStats for the player.
        """
        if player_id not in self._player_stats:
            # Try to load from history first
            stats = self.match_history.get_player_statistics(player_id)
            if stats:
                self._player_stats[player_id] = PlayerStats(
                    player_id=player_id,
                    rating=stats.current_rating,
                    games_played=stats.total_games,
                    wins=stats.wins,
                    losses=stats.losses,
                    draws=stats.draws,
                )
            else:
                # Create new
                self._player_stats[player_id] = PlayerStats(player_id=player_id)

        return self._player_stats[player_id]

    def create_bracket(
        self,
        players: list[str],
        *,
        format: TournamentFormat = TournamentFormat.ROUND_ROBIN,
        seeding: dict[str, int] | None = None,
    ) -> Bracket:
        """Create a tournament bracket.

        Args:
            players: List of player IDs
            format: Tournament format
            seeding: Optional seeding (player_id -> seed number)

        Returns:
            Created Bracket.
        """
        if len(players) < 2:
            raise ValueError("Need at least 2 players for a tournament")

        # Default seeding by player order
        if seeding is None:
            seeding = {player: i + 1 for i, player in enumerate(players)}

        # Create rounds based on format
        if format == TournamentFormat.ROUND_ROBIN:
            rounds = self._create_round_robin_rounds(players)
        elif format == TournamentFormat.SINGLE_ELIMINATION:
            rounds = self._create_single_elimination_rounds(players, seeding)
        elif format == TournamentFormat.DOUBLE_ELIMINATION:
            rounds = self._create_double_elimination_rounds(players, seeding)
        elif format == TournamentFormat.SWISS:
            # Swiss format generates rounds dynamically
            rounds = []
        else:
            raise ValueError(f"Unknown tournament format: {format}")

        bracket = Bracket(
            format=format,
            rounds=rounds,
            players=players,
            seeding=seeding,
        )

        logger.info(
            f"Created {format.value} bracket: "
            f"{len(players)} players, {bracket.total_rounds} rounds"
        )

        return bracket

    def _create_round_robin_rounds(self, players: list[str]) -> list[list[Matchup]]:
        """Create round-robin schedule (all pairs play once)."""
        rounds = []
        n = len(players)

        # Use circle method for round-robin scheduling
        if n % 2 == 1:
            players = [*players, "BYE"]
            n += 1

        for round_num in range(n - 1):
            round_matchups = []
            for i in range(n // 2):
                player_a = players[i]
                player_b = players[n - 1 - i]

                if player_a != "BYE" and player_b != "BYE":
                    round_matchups.append(Matchup(player_a, player_b, round_num))

            # Rotate players (keep first fixed)
            players = [players[0], players[-1], *players[1:-1]]
            rounds.append(round_matchups)

        return rounds

    def _create_single_elimination_rounds(
        self, players: list[str], seeding: dict[str, int]
    ) -> list[list[Matchup]]:
        """Create single-elimination bracket."""
        # Sort by seeding
        sorted_players = sorted(players, key=lambda p: seeding.get(p, 999))

        # Pad to power of 2
        n = len(sorted_players)
        bracket_size = 1
        while bracket_size < n:
            bracket_size *= 2

        # Add byes
        byes_needed = bracket_size - n
        sorted_players.extend([f"BYE_{i}" for i in range(byes_needed)])

        # Create initial round
        rounds = []
        current_players = sorted_players
        round_num = 0

        while len(current_players) > 1:
            round_matchups = []
            for i in range(0, len(current_players), 2):
                player_a = current_players[i]
                player_b = current_players[i + 1]

                # Skip if bye
                if not player_a.startswith("BYE_") and not player_b.startswith("BYE_"):
                    round_matchups.append(Matchup(player_a, player_b, round_num))

            rounds.append(round_matchups)
            current_players = current_players[: len(current_players) // 2]
            round_num += 1

        return rounds

    def _create_double_elimination_rounds(
        self, players: list[str], seeding: dict[str, int]
    ) -> list[list[Matchup]]:
        """Create double-elimination bracket (simplified - winners bracket only)."""
        # For simplicity, just do single elimination
        # Full double elimination requires tracking losers bracket
        logger.warning("Double elimination simplified to single elimination")
        return self._create_single_elimination_rounds(players, seeding)

    def run_tournament(self, bracket: Bracket) -> TournamentResult:
        """Run a complete tournament.

        Args:
            bracket: Tournament bracket

        Returns:
            TournamentResult with final standings.
        """
        start_time = time.time()
        total_games = 0

        logger.info(f"Starting {bracket.format.value} tournament with {len(bracket.players)} players")

        # Run each round
        while not bracket.is_complete:
            if bracket.format == TournamentFormat.SWISS:
                # Generate next round dynamically
                self._generate_swiss_round(bracket)

            round_results = self.run_round(bracket, bracket.current_round)
            bracket.results[bracket.current_round] = round_results
            total_games += len(round_results)

            # Advance to next round
            if bracket.format in (
                TournamentFormat.SINGLE_ELIMINATION,
                TournamentFormat.DOUBLE_ELIMINATION,
            ):
                bracket = self.advance_bracket(bracket, round_results)
            else:
                bracket.current_round += 1

        # Calculate final standings
        standings = self.get_standings(bracket)

        # Determine winner
        winner = standings[0].player_id if standings else ""

        duration = time.time() - start_time

        result = TournamentResult(
            bracket=bracket,
            winner=winner,
            final_standings=standings,
            total_games=total_games,
            duration_seconds=duration,
            elo_changes={
                pid: stats.rating - self.config.elo.initial_rating
                for pid, stats in self._player_stats.items()
            },
        )

        logger.info(
            f"Tournament complete: winner={winner}, "
            f"games={total_games}, duration={duration:.1f}s"
        )

        return result

    def run_round(self, bracket: Bracket, round_num: int) -> list[MatchResult]:
        """Run a single tournament round.

        Args:
            bracket: Tournament bracket
            round_num: Round number to run

        Returns:
            List of MatchResult from this round.
        """
        if round_num >= len(bracket.rounds):
            return []

        matchups = bracket.rounds[round_num]
        results = []

        logger.info(f"Running round {round_num + 1}/{bracket.total_rounds} ({len(matchups)} matches)")

        for matchup in matchups:
            result = self.run_match(matchup.player_a, matchup.player_b)
            matchup.result = result
            results.append(result)

        return results

    def run_match(
        self,
        player_a: str,
        player_b: str,
        *,
        num_games: int | None = None,
    ) -> MatchResult:
        """Run a match between two players.

        Args:
            player_a: First player ID
            player_b: Second player ID
            num_games: Number of games to play (default: self.games_per_match)

        Returns:
            MatchResult from the match.
        """
        if num_games is None:
            num_games = self.games_per_match

        # Get player stats
        stats_a = self._get_or_create_player_stats(player_a)
        stats_b = self._get_or_create_player_stats(player_b)

        # Run match via arena runner
        if self.arena_runner is not None:
            result = self.arena_runner(player_a, player_b, num_games)
        else:
            # Create mock result
            result = MatchResult(
                match_id=str(uuid.uuid4())[:8],
                player1_id=player_a,
                player2_id=player_b,
                player1_score=0.0,
                player2_score=0.0,
                player1_rating_before=stats_a.rating,
                player2_rating_before=stats_b.rating,
            )

        # Update ratings
        update_a, update_b = self.elo_system.process_match(
            player_a,
            player_b,
            stats_a.rating,
            stats_b.rating,
            result.player1_score,
            result.player2_score,
            stats_a.games_played,
            stats_b.games_played,
        )

        # Update player stats
        stats_a.rating = update_a.rating_after
        stats_a.games_played += num_games
        stats_b.rating = update_b.rating_after
        stats_b.games_played += num_games

        # Record in history
        self.match_history.record_match(
            result,
            update_a.rating_after,
            update_b.rating_after,
        )

        logger.debug(
            f"Match complete: {player_a} vs {player_b}, "
            f"score={result.player1_score}-{result.player2_score}"
        )

        return result

    def get_standings(self, bracket: Bracket) -> list[PlayerStanding]:
        """Calculate current tournament standings.

        Args:
            bracket: Tournament bracket

        Returns:
            List of PlayerStanding sorted by rank.
        """
        standings_data = []

        for player_id in bracket.players:
            wins, losses, draws = bracket.get_player_record(player_id)
            points = wins + 0.5 * draws

            stats = self._get_or_create_player_stats(player_id)

            standings_data.append(
                {
                    "player_id": player_id,
                    "wins": wins,
                    "losses": losses,
                    "draws": draws,
                    "points": points,
                    "elo_rating": stats.rating,
                }
            )

        # Sort by points (descending), then by rating
        standings_data.sort(key=lambda x: (-x["points"], -x["elo_rating"]))

        # Create standings with ranks
        standings = []
        for rank, data in enumerate(standings_data, 1):
            standings.append(
                PlayerStanding(
                    player_id=data["player_id"],
                    rank=rank,
                    wins=data["wins"],
                    losses=data["losses"],
                    draws=data["draws"],
                    elo_rating=data["elo_rating"],
                    points=data["points"],
                )
            )

        return standings

    def advance_bracket(
        self,
        bracket: Bracket,
        round_results: list[MatchResult],
    ) -> Bracket:
        """Advance bracket to next round (for elimination formats).

        Args:
            bracket: Current bracket
            round_results: Results from current round

        Returns:
            Updated bracket.
        """
        if bracket.format not in (
            TournamentFormat.SINGLE_ELIMINATION,
            TournamentFormat.DOUBLE_ELIMINATION,
        ):
            # Non-elimination formats don't advance
            bracket.current_round += 1
            return bracket

        # Get winners from current round
        winners = []
        for result in round_results:
            if result.winner_id:
                winners.append(result.winner_id)

        # Create next round matchups
        if len(winners) > 1 and bracket.current_round + 1 < len(bracket.rounds):
            next_round_matchups = []
            for i in range(0, len(winners), 2):
                if i + 1 < len(winners):
                    next_round_matchups.append(
                        Matchup(winners[i], winners[i + 1], bracket.current_round + 1)
                    )

            # Update bracket
            if bracket.current_round + 1 < len(bracket.rounds):
                bracket.rounds[bracket.current_round + 1] = next_round_matchups

        bracket.current_round += 1
        return bracket

    def _generate_swiss_round(self, bracket: Bracket) -> None:
        """Generate next round for Swiss format."""
        # Get current standings
        standings = self.get_standings(bracket)

        # Clear matchmaker queue
        for player in bracket.players:
            self.matchmaker.remove_from_queue(player)

        # Add players to matchmaker
        for standing in standings:
            stats = self._get_or_create_player_stats(standing.player_id)
            self.matchmaker.add_to_queue(stats)

        # Generate matches
        matches = self.matchmaker.process_queue()

        # Create matchups
        round_matchups = []
        for match in matches:
            if match.player1 and match.player2:
                round_matchups.append(
                    Matchup(
                        match.player1.player_id,
                        match.player2.player_id,
                        bracket.current_round,
                    )
                )

        # Add round to bracket
        if bracket.current_round < len(bracket.rounds):
            bracket.rounds[bracket.current_round] = round_matchups
        else:
            bracket.rounds.append(round_matchups)

        logger.debug(f"Generated Swiss round {bracket.current_round + 1}: {len(round_matchups)} matches")
