"""Tournament-specific MLflow dashboards.

Wraps ``MLflowLogger`` with helpers that emit consistent metric names for
tournament events: per-match results, per-round summaries, leaderboards, and
head-to-head matrices. All methods degrade gracefully when the underlying
logger is disabled or uninitialized.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from alpha_zero.monitoring.metrics import compute_elo_spread

if TYPE_CHECKING:
    from alpha_zero.monitoring.mlflow_logger import MLflowLogger
    from alpha_zero.tournament.elo import RatingLeaderboard, RatingUpdate

logger = logging.getLogger(__name__)


class TournamentMlflowLogger:
    """Tournament-flavored facade over ``MLflowLogger``.

    Metrics are namespaced under ``tournament/`` and ``elo/`` so they show up
    in a coherent group in the MLflow UI. Methods are no-ops when the wrapped
    logger is not initialized.
    """

    def __init__(self, mlflow_logger: MLflowLogger) -> None:
        self.mlflow_logger = mlflow_logger

    # ------------------------------------------------------------------
    # Match-level
    # ------------------------------------------------------------------

    def log_match_result(
        self,
        round_index: int,
        match_index: int,
        player1_id: str,
        player2_id: str,
        score1: float,
        score2: float,
        game_length: int | None = None,
    ) -> None:
        """Log a single match result (win/loss/draw) for one round."""
        if not self.mlflow_logger.is_initialized:
            return
        prefix = f"tournament/round_{round_index:03d}/match_{match_index:03d}"
        metrics: dict[str, float] = {
            f"{prefix}/score_{player1_id}": float(score1),
            f"{prefix}/score_{player2_id}": float(score2),
        }
        if game_length is not None:
            metrics[f"{prefix}/game_length"] = float(game_length)
        self.mlflow_logger.log_metrics(metrics, step=round_index)

    # ------------------------------------------------------------------
    # Round-level aggregates
    # ------------------------------------------------------------------

    def log_round_results(
        self,
        round_index: int,
        match_count: int,
        avg_game_length: float,
        upset_rate: float = 0.0,
    ) -> None:
        """Log aggregate per-round statistics."""
        if not self.mlflow_logger.is_initialized:
            return
        metrics: dict[str, float] = {
            "tournament/round": float(round_index),
            "tournament/match_count": float(match_count),
            "tournament/avg_game_length": float(avg_game_length),
            "tournament/upset_rate": float(upset_rate),
        }
        self.mlflow_logger.log_metrics(metrics, step=round_index)

    # ------------------------------------------------------------------
    # Leaderboard & ELO
    # ------------------------------------------------------------------

    def log_elo_leaderboard(
        self,
        leaderboard: RatingLeaderboard,
        round_index: int,
        top_n: int = 10,
    ) -> None:
        """Log top-N leaderboard + aggregate spread."""
        if not self.mlflow_logger.is_initialized:
            return
        spread = compute_elo_spread(leaderboard)
        metrics: dict[str, float] = {f"elo/{k}": float(v) for k, v in spread.items()}
        for player_id, rating, rank in leaderboard.get_top(top_n):
            metrics[f"elo/leaderboard/{player_id}/rating"] = float(rating)
            metrics[f"elo/leaderboard/{player_id}/rank"] = float(rank)
        self.mlflow_logger.log_metrics(metrics, step=round_index)

    def log_elo_history(
        self,
        updates: list[RatingUpdate],
        round_index: int,
    ) -> None:
        """Log per-player rating deltas for a round."""
        if not self.mlflow_logger.is_initialized:
            return
        metrics: dict[str, float] = {}
        for update in updates:
            prefix = f"elo/history/{update.player_id}"
            metrics[f"{prefix}/rating_before"] = float(update.rating_before)
            metrics[f"{prefix}/rating_after"] = float(update.rating_after)
            metrics[f"{prefix}/rating_change"] = float(update.rating_change)
        self.mlflow_logger.log_metrics(metrics, step=round_index)

    def log_head_to_head_matrix(
        self,
        matrix: dict[str, dict[str, Any]],
        round_index: int,
        artifact_file: str = "head_to_head.json",
    ) -> None:
        """Log a head-to-head matrix as a JSON artifact."""
        if not self.mlflow_logger.is_initialized:
            return
        try:
            self.mlflow_logger.log_text(
                json.dumps({"round": round_index, "matrix": matrix}, indent=2),
                artifact_file,
            )
        except Exception as exc:
            logger.warning("Tournament log_head_to_head_matrix failed: %s", exc)
