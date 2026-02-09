"""Tournament analysis and statistics module.

This module provides advanced analytics for tournament data, including:
- Player trend analysis (ELO history, win rate progression)
- Head-to-head matrices for visualizing player matchups
- Meta-analysis of top player strategies and hyperparameters
- Upset frequency calculation
- Generation comparison for measuring training progress

Usage:
    from alpha_zero.tournament.analysis import TournamentAnalyzer
    from alpha_zero.tournament.history import MatchHistory
    from alpha_zero.tournament.elo import ELORatingSystem

    history = MatchHistory("tournament.db")
    elo_system = ELORatingSystem()
    analyzer = TournamentAnalyzer(match_history=history, elo_system=elo_system)

    # Get player trends
    trend = analyzer.get_player_trends("player1", window=20)
    print(f"Current ELO: {trend.elo_history[-1]:.1f}")
    print(f"Win rate: {trend.win_rate_history[-1]:.2%}")

    # Get head-to-head matrix
    players = ["player1", "player2", "player3"]
    matrix = analyzer.head_to_head_matrix(players)

    # Analyze top players
    meta = analyzer.get_meta_analysis(top_k=5, hyperparameters={"player1": {...}, ...})
    print(f"Top strategies: {meta.strongest_strategies}")

    # Compare generations
    gen_cmp = analyzer.compare_generations(
        gen_a_players=["gen_0_p1", "gen_0_p2"],
        gen_b_players=["gen_1_p1", "gen_1_p2"],
    )
    print(f"Gen 1 win rate vs Gen 0: {gen_cmp.gen_b_win_rate:.2%}")

    # Export report
    analyzer.export_report("analysis_report.json", format="json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from alpha_zero.tournament.elo import ELORatingSystem
from alpha_zero.tournament.history import MatchHistory

logger = logging.getLogger(__name__)


@dataclass
class PlayerTrend:
    """Player performance trend over time.

    Attributes:
        player_id: Player identifier
        elo_history: List of ELO ratings over time
        win_rate_history: List of win rates over time (rolling window)
        games_played: Total games played
        peak_elo: Highest ELO achieved
        current_streak: Current win/loss streak (positive=wins, negative=losses)
    """

    player_id: str
    elo_history: list[float] = field(default_factory=list)
    win_rate_history: list[float] = field(default_factory=list)
    games_played: int = 0
    peak_elo: float = 0.0
    current_streak: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "player_id": self.player_id,
            "elo_history": self.elo_history,
            "win_rate_history": self.win_rate_history,
            "games_played": self.games_played,
            "peak_elo": self.peak_elo,
            "current_streak": self.current_streak,
        }


@dataclass
class MetaAnalysis:
    """Meta-analysis of top player strategies.

    Attributes:
        top_hyperparameters: Map of hyperparameter name to {mean, std} statistics
        strongest_strategies: List of dominant strategy types
        diversity_index: How varied the top players are (0=identical, 1=max diversity)
    """

    top_hyperparameters: dict[str, dict[str, float]] = field(default_factory=dict)
    strongest_strategies: list[str] = field(default_factory=list)
    diversity_index: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "top_hyperparameters": self.top_hyperparameters,
            "strongest_strategies": self.strongest_strategies,
            "diversity_index": self.diversity_index,
        }


@dataclass
class GenerationComparison:
    """Comparison between two generations of players.

    Attributes:
        gen_a: Generation A identifier
        gen_b: Generation B identifier
        gen_a_win_rate: Generation A win rate in head-to-head matches
        gen_b_win_rate: Generation B win rate in head-to-head matches
        elo_difference: Average ELO difference (gen_b - gen_a)
        significant: Whether the difference is statistically significant
        sample_size: Number of matches used for comparison
    """

    gen_a: int
    gen_b: int
    gen_a_win_rate: float = 0.0
    gen_b_win_rate: float = 0.0
    elo_difference: float = 0.0
    significant: bool = False
    sample_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gen_a": self.gen_a,
            "gen_b": self.gen_b,
            "gen_a_win_rate": self.gen_a_win_rate,
            "gen_b_win_rate": self.gen_b_win_rate,
            "elo_difference": self.elo_difference,
            "significant": self.significant,
            "sample_size": self.sample_size,
        }


class TournamentAnalyzer:
    """Advanced analytics for tournament data.

    This class provides statistical analysis and visualization support
    for tournament results, including player trends, head-to-head
    comparisons, meta-analysis, and generation comparisons.

    Example:
        >>> analyzer = TournamentAnalyzer(match_history=history)
        >>> trend = analyzer.get_player_trends("player1")
        >>> matrix = analyzer.head_to_head_matrix(["p1", "p2", "p3"])
        >>> meta = analyzer.get_meta_analysis(top_k=10)
    """

    def __init__(
        self,
        *,
        match_history: MatchHistory | None = None,
        elo_system: ELORatingSystem | None = None,
    ) -> None:
        """Initialize tournament analyzer.

        Args:
            match_history: Match history database (optional)
            elo_system: ELO rating system (optional)
        """
        self.match_history = match_history
        self.elo_system = elo_system or ELORatingSystem()

        logger.info("TournamentAnalyzer initialized")

    def head_to_head_matrix(self, players: list[str]) -> np.ndarray:
        """Generate N x N head-to-head win rate matrix.

        Args:
            players: List of player IDs to include

        Returns:
            N x N matrix where entry [i, j] is player i's win rate vs player j.
            Diagonal entries are 0.5.
        """
        n = len(players)
        matrix = np.full((n, n), 0.5)

        if self.match_history is None:
            logger.warning("No match history available for head-to-head matrix")
            return matrix

        for i, player_i in enumerate(players):
            for j, player_j in enumerate(players):
                if i == j:
                    continue

                # Get head-to-head stats
                h2h = self.match_history.get_head_to_head(player_i, player_j)
                if h2h["total_games"] > 0:
                    matrix[i, j] = h2h["player1_win_rate"]

        logger.debug(f"Generated head-to-head matrix for {n} players")
        return matrix

    def get_player_trends(
        self, player_id: str, *, window: int = 20
    ) -> PlayerTrend:
        """Get player performance trends over time.

        Args:
            player_id: Player to analyze
            window: Rolling window size for win rate calculation

        Returns:
            PlayerTrend with ELO and win rate history.
        """
        trend = PlayerTrend(player_id=player_id)

        if self.match_history is None:
            logger.warning(f"No match history available for player {player_id}")
            return trend

        # Get match history
        matches = self.match_history.get_player_matches(player_id, limit=10000)

        if not matches:
            logger.debug(f"No matches found for player {player_id}")
            return trend

        # Build ELO and win rate history
        elo_values = []
        win_history = []

        for match in reversed(matches):  # Chronological order
            # Determine which player position
            if match.player1_id == player_id:
                elo = match.player1_rating_after
                won = match.outcome == "win"
            else:
                elo = match.player2_rating_after
                # Flip outcome for player 2
                won = match.outcome == "loss"

            elo_values.append(elo)
            win_history.append(1.0 if won else 0.0)

        trend.elo_history = elo_values
        trend.games_played = len(matches)
        trend.peak_elo = max(elo_values) if elo_values else 0.0

        # Calculate rolling win rate
        if len(win_history) >= window:
            for i in range(window, len(win_history) + 1):
                window_wins = win_history[i - window : i]
                trend.win_rate_history.append(sum(window_wins) / window)
        else:
            # Not enough data for full window, use all available
            if win_history:
                trend.win_rate_history.append(sum(win_history) / len(win_history))

        # Get current streak from stats
        stats = self.match_history.get_player_statistics(player_id)
        if stats:
            trend.current_streak = stats.current_streak

        logger.debug(
            f"Player {player_id}: {trend.games_played} games, "
            f"peak ELO {trend.peak_elo:.1f}"
        )
        return trend

    def get_meta_analysis(
        self,
        *,
        top_k: int = 5,
        hyperparameters: dict[str, dict[str, Any]] | None = None,
    ) -> MetaAnalysis:
        """Analyze dominant strategies among top players.

        Args:
            top_k: Number of top players to analyze
            hyperparameters: Map of player_id to hyperparameter dict (optional)

        Returns:
            MetaAnalysis with dominant hyperparameters and strategies.
        """
        meta = MetaAnalysis()

        if self.match_history is None:
            logger.warning("No match history available for meta-analysis")
            return meta

        # Get top players by rating
        leaderboard = self.match_history.get_leaderboard(limit=top_k, min_games=10)
        if not leaderboard:
            logger.warning("No players found on leaderboard for meta-analysis")
            return meta

        top_player_ids = [player_id for player_id, _, _, _ in leaderboard]

        # Analyze hyperparameters if provided
        if hyperparameters:
            top_hyperparams = {
                pid: hyperparameters[pid]
                for pid in top_player_ids
                if pid in hyperparameters
            }

            if top_hyperparams:
                # Calculate mean and std for each hyperparameter
                all_keys = set()
                for hparams in top_hyperparams.values():
                    all_keys.update(hparams.keys())

                for key in all_keys:
                    values = []
                    for hparams in top_hyperparams.values():
                        if key in hparams:
                            val = hparams[key]
                            # Only aggregate numeric values
                            if isinstance(val, (int, float)):
                                values.append(float(val))
                            elif isinstance(val, str):
                                # Track string strategy types
                                if key == "strategy" or key == "style":
                                    meta.strongest_strategies.append(val)

                    if values:
                        meta.top_hyperparameters[key] = {
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values)),
                        }

                # Remove duplicates from strategies
                meta.strongest_strategies = list(set(meta.strongest_strategies))

                # Calculate diversity index (coefficient of variation average)
                if meta.top_hyperparameters:
                    cvs = []
                    for stats in meta.top_hyperparameters.values():
                        if stats["mean"] != 0:
                            cv = abs(stats["std"] / stats["mean"])
                            cvs.append(min(cv, 1.0))  # Cap at 1.0
                    meta.diversity_index = float(np.mean(cvs)) if cvs else 0.0

        logger.debug(
            f"Meta-analysis of top {len(top_player_ids)} players: "
            f"diversity={meta.diversity_index:.3f}"
        )
        return meta

    def get_upset_frequency(self, *, elo_diff_threshold: float = 100.0) -> float:
        """Calculate how often lower-rated player wins.

        Args:
            elo_diff_threshold: Minimum ELO difference to count as upset

        Returns:
            Fraction of games where underdog won (0.0-1.0).
        """
        if self.match_history is None:
            logger.warning("No match history available for upset frequency")
            return 0.0

        # Get recent matches
        matches = self.match_history.get_recent_matches(limit=1000)

        if not matches:
            logger.debug("No matches found for upset frequency calculation")
            return 0.0

        upset_count = 0
        qualifying_games = 0

        for match in matches:
            elo_diff = abs(
                match.player1_rating_before - match.player2_rating_before
            )

            # Only count games with significant ELO difference
            if elo_diff < elo_diff_threshold:
                continue

            qualifying_games += 1

            # Determine underdog
            if match.player1_rating_before < match.player2_rating_before:
                # Player 1 is underdog
                if match.outcome == "win":
                    upset_count += 1
            else:
                # Player 2 is underdog
                if match.outcome == "loss":
                    upset_count += 1

        if qualifying_games == 0:
            return 0.0

        upset_rate = upset_count / qualifying_games
        logger.debug(
            f"Upset frequency: {upset_rate:.2%} "
            f"({upset_count}/{qualifying_games} games with ELO diff >= {elo_diff_threshold})"
        )
        return upset_rate

    def compare_generations(
        self,
        gen_a_players: list[str],
        gen_b_players: list[str],
    ) -> GenerationComparison:
        """Compare strength between two generations.

        Args:
            gen_a_players: List of player IDs from generation A
            gen_b_players: List of player IDs from generation B

        Returns:
            GenerationComparison with win rates and ELO difference.
        """
        # Extract generation numbers from player IDs if possible
        gen_a = 0
        gen_b = 1
        try:
            if gen_a_players:
                gen_a = int(gen_a_players[0].split("_")[1])
            if gen_b_players:
                gen_b = int(gen_b_players[0].split("_")[1])
        except (IndexError, ValueError):
            pass

        comparison = GenerationComparison(gen_a=gen_a, gen_b=gen_b)

        if self.match_history is None:
            logger.warning("No match history available for generation comparison")
            return comparison

        # Count wins for each generation in cross-generation matches
        gen_a_wins = 0
        gen_b_wins = 0
        total_matches = 0

        for player_a in gen_a_players:
            for player_b in gen_b_players:
                h2h = self.match_history.get_head_to_head(player_a, player_b)
                if h2h["total_games"] > 0:
                    gen_a_wins += h2h["player1_wins"]
                    gen_b_wins += h2h["player2_wins"]
                    total_matches += h2h["total_games"]

        comparison.sample_size = total_matches

        if total_matches > 0:
            comparison.gen_a_win_rate = gen_a_wins / total_matches
            comparison.gen_b_win_rate = gen_b_wins / total_matches
        else:
            logger.warning(
                f"No matches found between generations {gen_a} and {gen_b}"
            )

        # Calculate average ELO difference
        gen_a_elos = []
        gen_b_elos = []

        for player_id in gen_a_players:
            stats = self.match_history.get_player_statistics(player_id)
            if stats:
                gen_a_elos.append(stats.current_rating)

        for player_id in gen_b_players:
            stats = self.match_history.get_player_statistics(player_id)
            if stats:
                gen_b_elos.append(stats.current_rating)

        if gen_a_elos and gen_b_elos:
            avg_a = np.mean(gen_a_elos)
            avg_b = np.mean(gen_b_elos)
            comparison.elo_difference = float(avg_b - avg_a)

        # Statistical significance (simple z-test for proportions)
        # Significant if |win_rate - 0.5| > 1.96 * SE
        if total_matches >= 30:
            p = comparison.gen_b_win_rate
            se = np.sqrt(p * (1 - p) / total_matches)
            z_score = abs(p - 0.5) / se if se > 0 else 0
            comparison.significant = z_score > 1.96

        logger.debug(
            f"Generation comparison {gen_a} vs {gen_b}: "
            f"gen_b win rate={comparison.gen_b_win_rate:.2%}, "
            f"ELO diff={comparison.elo_difference:+.1f}, "
            f"significant={comparison.significant}"
        )
        return comparison

    def export_report(
        self, path: str | Path, *, format: str = "json"
    ) -> None:
        """Export analysis report to file.

        Args:
            path: Output file path
            format: Export format ("json" or "markdown")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            self._export_json(path)
        elif format == "markdown":
            self._export_markdown(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported analysis report to {path}")

    def _export_json(self, path: Path) -> None:
        """Export report as JSON."""
        report = {
            "tournament_statistics": {},
            "top_players": [],
            "upset_frequency": 0.0,
        }

        if self.match_history:
            # Global statistics
            report["tournament_statistics"] = (
                self.match_history.get_global_statistics()
            )

            # Top players
            leaderboard = self.match_history.get_leaderboard(limit=10, min_games=5)
            for player_id, rating, games, win_rate in leaderboard:
                trend = self.get_player_trends(player_id, window=20)
                report["top_players"].append(
                    {
                        "player_id": player_id,
                        "rating": rating,
                        "games": games,
                        "win_rate": win_rate,
                        "peak_elo": trend.peak_elo,
                        "current_streak": trend.current_streak,
                    }
                )

            # Upset frequency
            report["upset_frequency"] = self.get_upset_frequency()

        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    def _export_markdown(self, path: Path) -> None:
        """Export report as Markdown."""
        lines = [
            "# Tournament Analysis Report",
            "",
            "## Global Statistics",
            "",
        ]

        if self.match_history:
            stats = self.match_history.get_global_statistics()
            lines.extend(
                [
                    f"- Total Matches: {stats['total_matches']}",
                    f"- Total Players: {stats['total_players']}",
                    f"- Average Rating: {stats['avg_rating']:.1f}",
                    f"- Highest Rating Ever: {stats['highest_rating_ever']:.1f}",
                    f"- Average Game Length: {stats['avg_game_length']:.1f}",
                    "",
                    "## Top Players",
                    "",
                    "| Rank | Player ID | Rating | Games | Win Rate |",
                    "|------|-----------|--------|-------|----------|",
                ]
            )

            leaderboard = self.match_history.get_leaderboard(limit=10, min_games=5)
            for rank, (player_id, rating, games, win_rate) in enumerate(
                leaderboard, 1
            ):
                lines.append(
                    f"| {rank} | {player_id} | {rating:.1f} | {games} | "
                    f"{win_rate:.2%} |"
                )

            # Upset frequency
            upset_rate = self.get_upset_frequency()
            lines.extend(
                [
                    "",
                    "## Upset Frequency",
                    "",
                    f"Lower-rated player wins: {upset_rate:.2%}",
                    "",
                ]
            )

        with open(path, "w") as f:
            f.write("\n".join(lines))
