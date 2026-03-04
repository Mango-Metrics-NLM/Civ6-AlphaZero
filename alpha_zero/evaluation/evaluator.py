"""Model evaluation orchestrator.

This module provides the main ModelEvaluator class that coordinates evaluation
against multiple baselines, tracks ELO ratings, and aggregates results.

Usage:
    from alpha_zero.evaluation import ModelEvaluator, EvaluationConfig

    config = EvaluationConfig.from_yaml("configs/evaluation.yaml")
    evaluator = ModelEvaluator(config)
    results = evaluator.evaluate_model("checkpoints/model_1000.pt")
    print(results["random"].win_rate)
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alpha_zero.evaluation.baselines import BaselineFactory, BaselinePlayer
from alpha_zero.evaluation.config import BaselineConfig, EvaluationConfig
from alpha_zero.evaluation.diversity import DiversityMetrics, DiversityTracker

if TYPE_CHECKING:
    from alpha_zero.envs.microciv import MicroCivEnv
    from alpha_zero.tournament.elo import RatingUpdate

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single evaluation game.

    Attributes:
        winner: Winner player index (0=model, 1=opponent, None=draw)
        num_turns: Total number of turns played
        model_score: Score from model's perspective (1=win, 0.5=draw, 0=loss)
        opponent_score: Score from opponent's perspective
        game_history: List of (action, info) tuples for diversity analysis
        metadata: Additional game information
    """

    winner: int | None
    num_turns: int
    model_score: float
    opponent_score: float
    game_history: list[tuple[int, dict[str, Any]]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "winner": self.winner,
            "num_turns": self.num_turns,
            "model_score": self.model_score,
            "opponent_score": self.opponent_score,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Complete evaluation results against a baseline.

    Attributes:
        baseline_name: Name of baseline opponent
        games_played: Total games played
        wins: Number of wins for model
        losses: Number of losses for model
        draws: Number of draws
        avg_game_length: Average game length
        elo_update: ELO rating update if enabled
        game_results: Individual game results
    """

    baseline_name: str
    games_played: int
    wins: int
    losses: int
    draws: int
    avg_game_length: float
    elo_update: RatingUpdate | None = None
    game_results: list[GameResult] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Win rate (wins / games_played)."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    @property
    def loss_rate(self) -> float:
        """Loss rate (losses / games_played)."""
        if self.games_played == 0:
            return 0.0
        return self.losses / self.games_played

    @property
    def draw_rate(self) -> float:
        """Draw rate (draws / games_played)."""
        if self.games_played == 0:
            return 0.0
        return self.draws / self.games_played

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "baseline_name": self.baseline_name,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "draw_rate": self.draw_rate,
            "avg_game_length": self.avg_game_length,
            "elo_update": (asdict(self.elo_update) if self.elo_update else None),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Results vs {self.baseline_name}:",
            f"  Games: {self.games_played}",
            f"  Record: {self.wins}W-{self.losses}L-{self.draws}D",
            f"  Win Rate: {self.win_rate:.1%}",
            f"  Avg Game Length: {self.avg_game_length:.1f}",
        ]
        if self.elo_update:
            lines.append(f"  ELO Change: {self.elo_update.rating_change:+.1f}")
        return "\n".join(lines)


class ModelEvaluator:
    """Main model evaluation orchestrator.

    Coordinates evaluation against multiple baselines, tracks ELO ratings,
    and aggregates results with diversity metrics.

    Example:
        >>> config = EvaluationConfig.from_yaml("configs/evaluation.yaml")
        >>> evaluator = ModelEvaluator(config)
        >>> results = evaluator.evaluate_model("checkpoints/model_1000.pt")
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.win_rate:.1%}")
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Initialize model evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self._setup_logging()
        self._setup_elo_system()
        self._setup_diversity_tracker()

        # Lazy baseline initialization
        self._baselines: dict[str, BaselinePlayer] = {}
        self._model_player: BaselinePlayer | None = None

        logger.info(f"ModelEvaluator initialized: {config.experiment_name}")

    def _setup_logging(self) -> None:
        """Set up logging infrastructure."""
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.training_logger = None
        if self.config.tensorboard_enabled:
            try:
                from alpha_zero.utils.logging import TrainingLogger

                self.training_logger = TrainingLogger(
                    log_dir=str(self.output_dir / "tensorboard"),
                    name="evaluation",
                    tensorboard=True,
                )
            except ImportError:
                logger.warning("Could not import TrainingLogger, TensorBoard disabled")

    def _setup_elo_system(self) -> None:
        """Set up ELO rating system if enabled."""
        self.elo_system = None
        self.model_stats = None
        self._baseline_stats: dict[str, Any] = {}

        if self.config.elo_enabled:
            try:
                from alpha_zero.tournament.config import ELOConfig, PlayerStats
                from alpha_zero.tournament.elo import ELORatingSystem

                elo_config = ELOConfig(
                    initial_rating=self.config.elo_initial_rating,
                )
                self.elo_system = ELORatingSystem(elo_config)
                self.model_stats = PlayerStats(
                    player_id="evaluated_model",
                    display_name="Evaluated Model",
                    rating=self.config.elo_initial_rating,
                )
                logger.info("ELO rating system initialized")
            except ImportError:
                logger.warning("Could not import ELO system, ratings disabled")
                self.config.elo_enabled = False

    def _setup_diversity_tracker(self) -> None:
        """Set up diversity tracking if enabled."""
        if self.config.diversity.enabled:
            self.diversity_tracker = DiversityTracker(self.config.diversity)
        else:
            self.diversity_tracker = None  # type: ignore[assignment]

    def _create_environment(self) -> MicroCivEnv:
        """Create evaluation environment."""
        from alpha_zero.envs.microciv import MicroCivEnv

        return MicroCivEnv(
            board_size=self.config.env_board_size,
            max_turns=self.config.env_max_turns,
            seed=self.config.random_seed,
        )

    def get_baseline(self, config: BaselineConfig) -> BaselinePlayer:
        """Get or create baseline player.

        Args:
            config: Baseline configuration

        Returns:
            Baseline player instance
        """
        if config.name not in self._baselines:
            self._baselines[config.name] = BaselineFactory.create(
                config,
                device=self.config.device,
                seed=self.config.random_seed,
            )
        return self._baselines[config.name]

    def _get_baseline_stats(self, baseline_name: str) -> Any:
        """Get or create stats for a baseline."""
        if baseline_name not in self._baseline_stats:
            from alpha_zero.tournament.config import PlayerStats

            # Assign fixed ratings to baselines based on type
            baseline_ratings = {
                "random": 800.0,
                "random_baseline": 800.0,
                "heuristic": 1200.0,
                "heuristic_basic": 1200.0,
                "heuristic_defensive": 1250.0,
                "heuristic_aggressive": 1150.0,
                "heuristic_expansionist": 1200.0,
            }
            rating = baseline_ratings.get(baseline_name, 1000.0)
            self._baseline_stats[baseline_name] = PlayerStats(
                player_id=baseline_name,
                display_name=baseline_name,
                rating=rating,
            )
        return self._baseline_stats[baseline_name]

    def play_game(
        self,
        model_player: BaselinePlayer,
        opponent: BaselinePlayer,
        model_plays_first: bool = True,
    ) -> GameResult:
        """Play a single evaluation game.

        Args:
            model_player: The model being evaluated
            opponent: The baseline opponent
            model_plays_first: Whether model plays as player 0

        Returns:
            Game result
        """
        env = self._create_environment()
        obs, _info = env.reset()

        model_player.reset()
        opponent.reset()

        players = [model_player, opponent] if model_plays_first else [opponent, model_player]
        model_idx = 0 if model_plays_first else 1

        game_history: list[tuple[int, dict[str, Any]]] = []
        turn = 0

        while not getattr(env, "done", False):
            current_player_idx = getattr(env, "current_player", turn % 2)
            current_player = players[current_player_idx]

            action = current_player.select_action(env, obs)

            # Record for diversity analysis
            turn_info = {
                "turn": turn,
                "player": current_player_idx,
                "is_model": current_player_idx == model_idx,
            }
            game_history.append((action, turn_info))

            obs, _reward, terminated, truncated, _info = env.step(action)
            turn += 1

            if terminated or truncated:
                break

        # Determine winner from model's perspective
        env_winner = getattr(env, "winner", None)

        if env_winner is None:
            winner = None
            model_score = 0.5
            opponent_score = 0.5
        elif (env_winner == 0 and model_plays_first) or (env_winner == 1 and not model_plays_first):
            winner = 0  # Model won
            model_score = 1.0
            opponent_score = 0.0
        else:
            winner = 1  # Opponent won
            model_score = 0.0
            opponent_score = 1.0

        return GameResult(
            winner=winner,
            num_turns=turn,
            model_score=model_score,
            opponent_score=opponent_score,
            game_history=game_history,
            metadata={"model_played_first": model_plays_first},
        )

    def evaluate_against_baseline(
        self,
        model_player: BaselinePlayer,
        baseline_config: BaselineConfig,
    ) -> EvaluationResult:
        """Evaluate model against a specific baseline.

        Args:
            model_player: The model player
            baseline_config: Baseline configuration

        Returns:
            Evaluation results for this baseline
        """
        baseline = self.get_baseline(baseline_config)

        wins = 0
        losses = 0
        draws = 0
        total_turns = 0
        game_results: list[GameResult] = []

        logger.info(
            f"Starting evaluation vs {baseline_config.name} ({baseline_config.num_games} games)"
        )

        for game_idx in range(baseline_config.num_games):
            # Alternate who plays first for fairness
            model_plays_first = game_idx % 2 == 0

            result = self.play_game(model_player, baseline, model_plays_first)
            game_results.append(result)

            if result.winner == 0:  # Model won
                wins += 1
            elif result.winner == 1:  # Opponent won
                losses += 1
            else:
                draws += 1

            total_turns += result.num_turns

            # Track diversity
            if self.diversity_tracker:
                victory_type = (
                    "win" if result.winner == 0 else ("loss" if result.winner == 1 else "draw")
                )
                self.diversity_tracker.record_game(
                    result.game_history,
                    baseline_name=baseline_config.name,
                    victory_type=victory_type,
                    game_length=result.num_turns,
                )

            # Log progress
            if (game_idx + 1) % 10 == 0:
                logger.info(
                    f"vs {baseline_config.name}: {game_idx + 1}/{baseline_config.num_games} "
                    f"(W:{wins} L:{losses} D:{draws})"
                )

        # Update ELO if enabled
        elo_update = None
        if self.elo_system and self.model_stats:
            try:
                from alpha_zero.tournament.config import MatchOutcome

                baseline_stats = self._get_baseline_stats(baseline_config.name)

                # Determine aggregate outcome
                if wins > losses:
                    outcome = MatchOutcome.WIN
                elif losses > wins:
                    outcome = MatchOutcome.LOSS
                else:
                    outcome = MatchOutcome.DRAW

                elo_update, _ = self.elo_system.process_game_result(
                    self.model_stats,
                    baseline_stats,
                    outcome,
                )
            except Exception as e:
                logger.warning(f"ELO update failed: {e}")

        avg_game_length = (
            total_turns / baseline_config.num_games if baseline_config.num_games > 0 else 0
        )

        return EvaluationResult(
            baseline_name=baseline_config.name,
            games_played=baseline_config.num_games,
            wins=wins,
            losses=losses,
            draws=draws,
            avg_game_length=avg_game_length,
            elo_update=elo_update,
            game_results=game_results if self.config.save_game_logs else [],
        )

    def evaluate_model(
        self,
        checkpoint_path: str | None = None,
        model_player: BaselinePlayer | None = None,
    ) -> dict[str, EvaluationResult]:
        """Run full evaluation against all configured baselines.

        Args:
            checkpoint_path: Path to model checkpoint (uses config default if None)
            model_player: Pre-created model player (alternative to checkpoint)

        Returns:
            Dictionary mapping baseline names to results
        """
        # Create model player
        if model_player is None:
            checkpoint_path = checkpoint_path or self.config.model_checkpoint_path
            if Path(checkpoint_path).exists():
                from alpha_zero.evaluation.baselines import NeuralNetworkPlayer

                model_player = NeuralNetworkPlayer.from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    device=self.config.device,
                    use_mcts=True,
                    mcts_simulations=50,
                    name="evaluated_model",
                )
            else:
                logger.warning(f"Checkpoint not found: {checkpoint_path}, using random player")
                from alpha_zero.evaluation.baselines import RandomPlayer

                model_player = RandomPlayer(name="model_fallback")

        results: dict[str, EvaluationResult] = {}

        for baseline_config in self.config.baselines:
            if not baseline_config.enabled:
                continue

            logger.info(f"Evaluating against baseline: {baseline_config.name}")
            result = self.evaluate_against_baseline(model_player, baseline_config)
            results[baseline_config.name] = result

            # Log to training logger
            if self.training_logger:
                try:
                    self.training_logger.log_evaluation(
                        iteration=0,
                        win_rate=result.win_rate,
                        games_played=result.games_played,
                        opponent=baseline_config.name,
                    )
                except Exception as e:
                    logger.warning(f"Failed to log to TensorBoard: {e}")

            logger.info(result.summary())

        return results

    def get_diversity_metrics(self) -> DiversityMetrics | None:
        """Get computed diversity metrics.

        Returns:
            DiversityMetrics or None if diversity tracking disabled
        """
        if self.diversity_tracker:
            return self.diversity_tracker.compute_metrics()
        return None

    def get_model_elo(self) -> float | None:
        """Get current model ELO rating.

        Returns:
            ELO rating or None if ELO disabled
        """
        if self.model_stats:
            return self.model_stats.rating
        return None

    def summary(self, results: dict[str, EvaluationResult]) -> str:
        """Generate overall evaluation summary.

        Args:
            results: Evaluation results by baseline

        Returns:
            Formatted summary string
        """
        lines = [
            f"Evaluation Summary: {self.config.experiment_name}",
            "=" * 50,
        ]

        # Overall stats
        total_games = sum(r.games_played for r in results.values())
        total_wins = sum(r.wins for r in results.values())
        total_losses = sum(r.losses for r in results.values())
        total_draws = sum(r.draws for r in results.values())

        lines.extend(
            [
                f"Total Games: {total_games}",
                f"Overall Record: {total_wins}W-{total_losses}L-{total_draws}D",
                f"Overall Win Rate: {total_wins / total_games:.1%}" if total_games > 0 else "N/A",
                "",
                "Results by Baseline:",
                "-" * 30,
            ]
        )

        for name, result in results.items():
            lines.append(f"  {name}: {result.win_rate:.1%} ({result.wins}/{result.games_played})")

        if self.model_stats:
            lines.extend(["", f"Model ELO: {self.model_stats.rating:.0f}"])

        diversity = self.get_diversity_metrics()
        if diversity:
            lines.extend(
                [
                    "",
                    "Diversity Metrics:",
                    f"  Unique Openings: {diversity.unique_openings}",
                    f"  Opening Entropy: {diversity.opening_entropy:.2f}",
                    f"  Avg Game Length: {diversity.avg_game_length:.1f}",
                ]
            )

        return "\n".join(lines)
