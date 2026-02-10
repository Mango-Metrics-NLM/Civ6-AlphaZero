"""Tournament W&B Visualization Module.

This module provides Weights & Biases integration for tournament metrics,
including match results, ELO leaderboards, round summaries, and head-to-head
matchup visualizations.

All visualization methods are no-ops when the underlying WandbLogger is not
initialized, ensuring graceful degradation when W&B is disabled.

Usage:
    from alpha_zero.utils.wandb_logger import WandbLogger
    from alpha_zero.tournament.wandb_dashboards import TournamentWandbLogger

    wandb_logger = WandbLogger.from_env()
    wandb_logger.init()

    tournament_logger = TournamentWandbLogger(wandb_logger)
    tournament_logger.define_tournament_charts()

    # Log match result
    tournament_logger.log_match_result(match, tournament_id="season_1")

    # Log round summary
    tournament_logger.log_round_results(round_num=5, results=match_list)

    # Log final standings
    tournament_logger.log_tournament_summary(tournament_result)

    # Log ELO leaderboard
    tournament_logger.log_elo_leaderboard(leaderboard)

Example:
    >>> wandb_logger = WandbLogger(WandbConfig(enabled=True))
    >>> wandb_logger.init()
    >>> tournament_logger = TournamentWandbLogger(wandb_logger)
    >>> tournament_logger.define_tournament_charts()
    >>> tournament_logger.log_match_result(
    ...     match=MatchResult(...),
    ...     tournament_id="round_robin_2026",
    ... )
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from alpha_zero.tournament.config import MatchResult
from alpha_zero.tournament.elo import RatingLeaderboard
from alpha_zero.utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


class TournamentWandbLogger:
    """Wraps WandbLogger for tournament-specific visualizations.

    This class provides specialized logging methods for tournament metrics,
    including match results, round summaries, ELO ratings, and head-to-head
    matrices. All methods degrade gracefully when W&B is not initialized.

    Attributes:
        wandb_logger: Underlying WandbLogger instance.

    Example:
        >>> wandb_logger = WandbLogger.from_env()
        >>> wandb_logger.init()
        >>> tournament_logger = TournamentWandbLogger(wandb_logger)
        >>> tournament_logger.define_tournament_charts()
    """

    def __init__(self, wandb_logger: WandbLogger) -> None:
        """Initialize tournament W&B logger.

        Args:
            wandb_logger: Configured WandbLogger instance.
        """
        self.wandb_logger = wandb_logger

    def define_tournament_charts(self) -> None:
        """Register tournament metrics with W&B.

        Defines the following metrics under the `tournament/` namespace:
        - tournament/round: Current tournament round (x-axis)
        - tournament/match_count: Number of matches played
        - tournament/avg_game_length: Average game duration
        - tournament/elo_spread: Rating spread (max - min)
        - tournament/upset_rate: Rate of lower-rated player wins

        Should be called after `wandb_logger.init()`.
        """
        if not self.wandb_logger.is_initialized:
            return

        try:
            # Define round as the primary step metric for tournaments
            self.wandb_logger.define_metric("tournament/round")

            # Match statistics
            self.wandb_logger.define_metric(
                "tournament/match_count",
                step_metric="tournament/round",
                summary="last",
            )
            self.wandb_logger.define_metric(
                "tournament/avg_game_length",
                step_metric="tournament/round",
                summary="mean",
            )

            # ELO spread and dynamics
            self.wandb_logger.define_metric(
                "tournament/elo_spread",
                step_metric="tournament/round",
                summary="last",
            )
            self.wandb_logger.define_metric(
                "tournament/upset_rate",
                step_metric="tournament/round",
                summary="mean",
            )

            logger.info("Defined tournament WandB charts")

        except Exception as e:
            logger.warning(f"Failed to define tournament charts: {e}")

    def log_match_result(
        self,
        match: MatchResult,
        *,
        tournament_id: str = "",
        step: int | None = None,
    ) -> None:
        """Log individual match result.

        Logs a single match result with metadata including ratings, score,
        and duration. No-op when W&B is not initialized.

        Args:
            match: Match result to log.
            tournament_id: Tournament identifier (optional metadata).
            step: Step counter (defaults to internal counter).
        """
        if not self.wandb_logger.is_initialized:
            return

        data: dict[str, Any] = {
            "tournament/match/player1_rating_before": match.player1_rating_before,
            "tournament/match/player2_rating_before": match.player2_rating_before,
            "tournament/match/player1_rating_change": match.player1_rating_change,
            "tournament/match/player2_rating_change": match.player2_rating_change,
            "tournament/match/player1_score": match.player1_score,
            "tournament/match/player2_score": match.player2_score,
            "tournament/match/duration_seconds": match.duration_seconds,
            "tournament/match/is_draw": 1 if match.is_draw else 0,
        }

        if tournament_id:
            data["tournament/match/tournament_id"] = tournament_id

        # Calculate rating delta (magnitude of surprise)
        rating_diff = abs(match.player1_rating_before - match.player2_rating_before)
        data["tournament/match/rating_difference"] = rating_diff

        # Detect upsets (lower-rated player wins)
        is_upset = False
        if match.winner_id is not None:
            if match.winner_id == match.player1_id:
                is_upset = match.player1_rating_before < match.player2_rating_before
            elif match.winner_id == match.player2_id:
                is_upset = match.player2_rating_before < match.player1_rating_before

        data["tournament/match/is_upset"] = 1 if is_upset else 0

        self.wandb_logger.log(data, step=step)

    def log_round_results(
        self,
        round_num: int,
        results: list[MatchResult],
        *,
        step: int | None = None,
    ) -> None:
        """Log summary statistics for a tournament round.

        Aggregates match results for a complete round and logs summary metrics
        including average game length, upset rate, and total matches.

        Args:
            round_num: Round number.
            results: List of match results from this round.
            step: Step counter (defaults to round_num).
        """
        if not self.wandb_logger.is_initialized:
            return

        if not results:
            logger.warning(f"No results for round {round_num}")
            return

        # Aggregate statistics
        total_matches = len(results)
        total_duration = sum(m.duration_seconds for m in results)
        avg_duration = total_duration / total_matches if total_matches > 0 else 0.0

        # Calculate upsets
        upsets = 0
        for match in results:
            if match.winner_id is None:
                continue
            if match.winner_id == match.player1_id:
                if match.player1_rating_before < match.player2_rating_before:
                    upsets += 1
            elif match.winner_id == match.player2_id:
                if match.player2_rating_before < match.player1_rating_before:
                    upsets += 1

        upset_rate = upsets / total_matches if total_matches > 0 else 0.0

        # Calculate rating changes
        total_rating_changes = sum(
            abs(m.player1_rating_change) + abs(m.player2_rating_change) for m in results
        )
        avg_rating_change = total_rating_changes / (2 * total_matches) if total_matches > 0 else 0.0

        # Count draws
        draws = sum(1 for m in results if m.is_draw)
        draw_rate = draws / total_matches if total_matches > 0 else 0.0

        data: dict[str, Any] = {
            "tournament/round": round_num,
            "tournament/match_count": total_matches,
            "tournament/avg_game_length": avg_duration,
            "tournament/upset_rate": upset_rate,
            "tournament/draw_rate": draw_rate,
            "tournament/avg_rating_change": avg_rating_change,
        }

        # Use round_num as step if not specified
        if step is None:
            step = round_num

        self.wandb_logger.log(data, step=step)
        logger.debug(f"Logged round {round_num} results: {total_matches} matches")

    def log_tournament_summary(
        self,
        result: Any,
        *,
        step: int | None = None,
    ) -> None:
        """Log final tournament standings.

        Logs the complete tournament result including final rankings, total games,
        and winner information. The `result` parameter is expected to be a
        TournamentResult from the orchestrator.

        Args:
            result: Tournament result object (from orchestrator).
            step: Step counter.
        """
        if not self.wandb_logger.is_initialized:
            return

        # Handle various result formats (duck typing for cross-module compatibility)
        data: dict[str, Any] = {}

        # Support both TournamentResult.winner and generic winner_id
        winner = getattr(result, "winner", None) or getattr(result, "winner_id", None)
        if winner is not None:
            data["tournament/summary/winner"] = winner

        if hasattr(result, "total_games"):
            data["tournament/summary/total_games"] = result.total_games

        if hasattr(result, "total_rounds"):
            data["tournament/summary/total_rounds"] = result.total_rounds

        if hasattr(result, "duration_seconds"):
            data["tournament/summary/duration_seconds"] = result.duration_seconds

        if hasattr(result, "final_standings"):
            # Log top 3 finishers if available
            # Support both dataclass (PlayerStanding) and dict standings
            standings = result.final_standings
            for idx, label in enumerate(
                ["first_place_rating", "second_place_rating", "third_place_rating"]
            ):
                if idx < len(standings):
                    entry = standings[idx]
                    rating = (
                        getattr(entry, "elo_rating", None)
                        or (entry.get("rating", 0.0) if isinstance(entry, dict) else 0.0)
                    )
                    data[f"tournament/summary/{label}"] = rating

        if data:
            self.wandb_logger.log(data, step=step)
            logger.info("Logged tournament summary")
        else:
            logger.warning("Tournament result has no loggable attributes")

    def log_elo_leaderboard(
        self,
        leaderboard: RatingLeaderboard,
        *,
        step: int | None = None,
    ) -> None:
        """Log ELO leaderboard as a W&B table.

        Creates a table visualization showing player rankings, ratings, and
        relative positions. Also logs summary statistics about rating spread.

        Args:
            leaderboard: RatingLeaderboard instance.
            step: Step counter.
        """
        if not self.wandb_logger.is_initialized:
            return

        # Get top players
        top_players = leaderboard.get_top(n=100)

        if not top_players:
            logger.warning("Empty leaderboard, skipping log")
            return

        # Prepare table data
        columns = ["Rank", "Player ID", "Rating"]
        rows = [[rank, player_id, rating] for player_id, rating, rank in top_players]

        self.wandb_logger.log_table(
            name="tournament/leaderboard",
            columns=columns,
            data=rows,
        )

        # Log summary statistics
        ratings = [rating for _, rating, _ in top_players]
        data: dict[str, Any] = {
            "tournament/elo_spread": max(ratings) - min(ratings),
            "tournament/elo_max": max(ratings),
            "tournament/elo_min": min(ratings),
            "tournament/elo_mean": sum(ratings) / len(ratings) if ratings else 0.0,
        }

        self.wandb_logger.log(data, step=step)
        logger.debug(f"Logged leaderboard with {len(top_players)} players")

    def log_head_to_head_matrix(
        self,
        matrix: Any,
        players: list[str],
        *,
        step: int | None = None,
    ) -> None:
        """Log head-to-head matchup matrix as a heatmap.

        Creates a table showing win rates between all pairs of players.
        The matrix should contain win rates (0-1) for player i vs player j.

        Args:
            matrix: Numpy array or list of win rates (NxN).
            players: List of player IDs corresponding to matrix rows/cols.
            step: Step counter.
        """
        if not self.wandb_logger.is_initialized:
            return

        # Convert to numpy array if needed
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)

        if matrix.shape[0] != len(players) or matrix.shape[1] != len(players):
            logger.error(
                f"Matrix shape {matrix.shape} does not match players count {len(players)}"
            )
            return

        # Prepare table with row/column headers
        columns = ["Player", *players]
        rows = []

        for i, player in enumerate(players):
            row = [player] + [float(matrix[i, j]) for j in range(len(players))]
            rows.append(row)

        self.wandb_logger.log_table(
            name="tournament/head_to_head_matrix",
            columns=columns,
            data=rows,
        )

        # Log average win rate (excluding diagonal)
        mask = np.ones_like(matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_win_rate = float(np.mean(matrix[mask]))

        data: dict[str, Any] = {
            "tournament/head_to_head/avg_win_rate": avg_win_rate,
            "tournament/head_to_head/num_players": len(players),
        }

        self.wandb_logger.log(data, step=step)
        logger.debug(f"Logged head-to-head matrix for {len(players)} players")

    def log_elo_history(
        self,
        trends: dict[str, Any],
        *,
        step: int | None = None,
    ) -> None:
        """Log per-player ELO rating history.

        Creates time-series visualizations of rating changes for each player.
        The `trends` dictionary should map player IDs to PlayerTrend objects
        with rating_history attribute.

        Args:
            trends: Dictionary mapping player_id to PlayerTrend objects.
            step: Step counter.
        """
        if not self.wandb_logger.is_initialized:
            return

        if not trends:
            logger.warning("Empty trends dictionary, skipping log")
            return

        # Prepare table with time series data
        columns = ["Player ID", "Timestamp", "Rating"]
        rows = []

        for player_id, trend in trends.items():
            # Handle different trend object formats
            if hasattr(trend, "rating_history"):
                history = trend.rating_history
            elif isinstance(trend, dict) and "rating_history" in trend:
                history = trend["rating_history"]
            else:
                logger.warning(f"Cannot extract rating_history from trend for {player_id}")
                continue

            # rating_history is expected to be list of (timestamp, rating) tuples
            for timestamp, rating in history:
                rows.append([player_id, timestamp, rating])

        if rows:
            self.wandb_logger.log_table(
                name="tournament/elo_history",
                columns=columns,
                data=rows,
            )
            logger.debug(f"Logged ELO history for {len(trends)} players")
        else:
            logger.warning("No ELO history data to log")
