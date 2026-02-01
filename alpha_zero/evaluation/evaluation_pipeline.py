"""Evaluation pipeline for measuring agent performance.

This module provides infrastructure for running games between agents
and tracking win rates, game statistics, and other metrics.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from alpha_zero.envs.microciv import MicroCivEnv
from alpha_zero.evaluation.baseline_agents import BaseAgent, MCTSAgentWrapper

if TYPE_CHECKING:
    from alpha_zero.core.civ6_mcts import Civ6MCTS

logger = logging.getLogger(__name__)


@dataclass
class GameResult:
    """Result of a single game."""

    winner: int | None  # 0, 1, or None for draw
    num_turns: int
    player0_cities: int
    player1_cities: int
    player0_units: int
    player1_units: int
    game_duration_seconds: float


@dataclass
class WinRateResult:
    """Win rate statistics from evaluation."""

    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_games: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate win rate as percentage."""
        if self.total_games == 0:
            return 0.0
        return self.wins / self.total_games * 100

    @property
    def loss_rate(self) -> float:
        """Calculate loss rate as percentage."""
        if self.total_games == 0:
            return 0.0
        return self.losses / self.total_games * 100

    @property
    def draw_rate(self) -> float:
        """Calculate draw rate as percentage."""
        if self.total_games == 0:
            return 0.0
        return self.draws / self.total_games * 100

    def __str__(self) -> str:
        return (
            f"WinRate: {self.win_rate:.1f}% "
            f"({self.wins}W/{self.losses}L/{self.draws}D of {self.total_games})"
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result with detailed statistics."""

    win_rate: WinRateResult
    game_results: list[GameResult] = field(default_factory=list)
    avg_game_length: float = 0.0
    avg_cities_won: float = 0.0
    avg_cities_lost: float = 0.0
    avg_units_won: float = 0.0
    avg_units_lost: float = 0.0
    total_evaluation_time: float = 0.0
    value_predictions: list[tuple[float, float]] = field(default_factory=list)
    value_correlation: float = 0.0

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"\n{'='*50}\n"
            f"EVALUATION SUMMARY\n"
            f"{'='*50}\n"
            f"Win Rate: {self.win_rate.win_rate:.1f}%\n"
            f"  Wins:   {self.win_rate.wins}\n"
            f"  Losses: {self.win_rate.losses}\n"
            f"  Draws:  {self.win_rate.draws}\n"
            f"  Total:  {self.win_rate.total_games}\n"
            f"\nGame Statistics:\n"
            f"  Avg Game Length: {self.avg_game_length:.1f} turns\n"
            f"  Avg Cities (won games): {self.avg_cities_won:.1f}\n"
            f"  Avg Cities (lost games): {self.avg_cities_lost:.1f}\n"
            f"\nValue Head Diagnostics:\n"
            f"  Prediction Correlation: {self.value_correlation:.3f}\n"
            f"\nEvaluation Time: {self.total_evaluation_time:.1f}s\n"
            f"{'='*50}"
        )


class EvaluationPipeline:
    """Pipeline for evaluating agent performance.

    Runs games between a target agent (typically trained model) and
    baseline opponents, tracking win rates and detailed statistics.
    """

    def __init__(
        self,
        board_size: int = 16,
        max_turns: int = 200,
        seed: int | None = None,
    ):
        """Initialize evaluation pipeline.

        Args:
            board_size: Size of the game board.
            max_turns: Maximum turns per game.
            seed: Random seed for reproducibility.
        """
        self.board_size = board_size
        self.max_turns = max_turns
        self.seed = seed
        self._game_seed = seed or 0

    def evaluate_vs_random(
        self,
        agent: BaseAgent | Civ6MCTS,
        n_games: int = 100,
        log_interval: int = 10,
    ) -> EvaluationResult:
        """Evaluate agent against random opponent.

        Args:
            agent: Agent to evaluate (BaseAgent or MCTS instance).
            n_games: Number of games to play.
            log_interval: How often to log progress.

        Returns:
            EvaluationResult with detailed statistics.
        """
        from alpha_zero.evaluation.baseline_agents import RandomAgent

        random_agent = RandomAgent(seed=self.seed)
        return self._run_evaluation(
            agent,
            random_agent,
            n_games,
            log_interval,
            opponent_name="Random",
        )

    def evaluate_vs_heuristic(
        self,
        agent: BaseAgent | Civ6MCTS,
        n_games: int = 50,
        log_interval: int = 10,
    ) -> EvaluationResult:
        """Evaluate agent against heuristic opponent.

        Args:
            agent: Agent to evaluate.
            n_games: Number of games to play.
            log_interval: How often to log progress.

        Returns:
            EvaluationResult with detailed statistics.
        """
        from alpha_zero.evaluation.baseline_agents import GreedyHeuristicAgent

        heuristic_agent = GreedyHeuristicAgent(seed=self.seed)
        return self._run_evaluation(
            agent,
            heuristic_agent,
            n_games,
            log_interval,
            opponent_name="Heuristic",
        )

    def evaluate_vs_checkpoint(
        self,
        agent: BaseAgent | Civ6MCTS,
        checkpoint_agent: BaseAgent | Civ6MCTS,
        n_games: int = 50,
        log_interval: int = 10,
    ) -> EvaluationResult:
        """Evaluate agent against a previous checkpoint.

        Args:
            agent: Current agent to evaluate.
            checkpoint_agent: Previous checkpoint agent.
            n_games: Number of games to play.
            log_interval: How often to log progress.

        Returns:
            EvaluationResult with detailed statistics.
        """
        return self._run_evaluation(
            agent,
            checkpoint_agent,
            n_games,
            log_interval,
            opponent_name="Checkpoint",
        )

    def _run_evaluation(
        self,
        agent: BaseAgent | Civ6MCTS,
        opponent: BaseAgent,
        n_games: int,
        log_interval: int,
        opponent_name: str,
    ) -> EvaluationResult:
        """Run evaluation games between two agents.

        Args:
            agent: Primary agent being evaluated.
            opponent: Opponent agent.
            n_games: Number of games.
            log_interval: Logging frequency.
            opponent_name: Name for logging.

        Returns:
            EvaluationResult with all statistics.
        """
        # Wrap MCTS if needed
        if not isinstance(agent, BaseAgent):
            agent = MCTSAgentWrapper(agent, temperature=0.0)

        start_time = time.time()
        win_rate = WinRateResult()
        game_results: list[GameResult] = []
        value_predictions: list[tuple[float, float]] = []

        logger.info(f"Starting evaluation: {n_games} games vs {opponent_name}")

        for game_idx in range(n_games):
            # Alternate who plays as player 0
            agent_is_player0 = game_idx % 2 == 0

            result, game_values = self._play_game(
                agent,
                opponent,
                agent_is_player0,
            )
            game_results.append(result)
            value_predictions.extend(game_values)

            # Update win rate
            if result.winner is None:
                win_rate.draws += 1
            elif (result.winner == 0 and agent_is_player0) or (
                result.winner == 1 and not agent_is_player0
            ):
                win_rate.wins += 1
            else:
                win_rate.losses += 1
            win_rate.total_games += 1

            # Log progress
            if (game_idx + 1) % log_interval == 0:
                logger.info(
                    f"  Game {game_idx + 1}/{n_games}: {win_rate}"
                )

        total_time = time.time() - start_time

        # Compute statistics
        eval_result = self._compute_statistics(
            win_rate,
            game_results,
            value_predictions,
            total_time,
        )

        logger.info(eval_result.summary())
        return eval_result

    def _play_game(
        self,
        agent: BaseAgent,
        opponent: BaseAgent,
        agent_is_player0: bool,
    ) -> tuple[GameResult, list[tuple[float, float]]]:
        """Play a single evaluation game.

        Args:
            agent: Agent being evaluated.
            opponent: Opponent agent.
            agent_is_player0: Whether agent plays as player 0.

        Returns:
            Tuple of (GameResult, list of (value_prediction, actual_outcome)).
        """
        env = MicroCivEnv(
            board_size=self.board_size,
            max_turns=self.max_turns,
            seed=self._game_seed,
        )
        self._game_seed += 1

        agent.reset()
        opponent.reset()

        obs, _ = env.reset()
        start_time = time.time()
        value_predictions: list[tuple[float, float]] = []

        while True:
            current_player = env.current_player
            is_agent_turn = (current_player == 0) == agent_is_player0

            if is_agent_turn:
                action = agent.select_action(env, obs)

                # Track value predictions if available
                if hasattr(agent, "mcts") and hasattr(agent.mcts, "network"):
                    try:
                        value = self._get_value_prediction(agent.mcts, env, obs)
                        value_predictions.append((value, 0.0))  # outcome filled later
                    except Exception:
                        pass
            else:
                action = opponent.select_action(env, obs)

            obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        game_duration = time.time() - start_time
        winner = info.get("winner")

        # Update value predictions with actual outcome
        if winner is not None:
            agent_won = (winner == 0) == agent_is_player0
            actual_outcome = 1.0 if agent_won else -1.0
        else:
            actual_outcome = 0.0

        value_predictions = [(v, actual_outcome) for v, _ in value_predictions]

        # Count final state
        player0_cities = len([c for c in env.cities if c.player == 0])
        player1_cities = len([c for c in env.cities if c.player == 1])
        player0_units = len([u for u in env.units if u.player == 0])
        player1_units = len([u for u in env.units if u.player == 1])

        return (
            GameResult(
                winner=winner,
                num_turns=env.turn,
                player0_cities=player0_cities,
                player1_cities=player1_cities,
                player0_units=player0_units,
                player1_units=player1_units,
                game_duration_seconds=game_duration,
            ),
            value_predictions,
        )

    def _get_value_prediction(
        self,
        mcts,
        env: MicroCivEnv,
        obs: dict[str, np.ndarray],
    ) -> float:
        """Get value prediction from network.

        Args:
            mcts: MCTS instance with network.
            env: Game environment.
            obs: Current observation.

        Returns:
            Value prediction from network.
        """
        import torch

        network = mcts.network
        device = mcts.device if hasattr(mcts, "device") else "cpu"

        spatial = torch.tensor(obs["spatial"], dtype=torch.float32).unsqueeze(0)
        vector = torch.tensor(obs["vector"], dtype=torch.float32).unsqueeze(0)

        spatial = spatial.to(device)
        vector = vector.to(device)

        network.eval()
        with torch.no_grad():
            _, value = network(spatial, vector)

        return float(value.item())

    def _compute_statistics(
        self,
        win_rate: WinRateResult,
        game_results: list[GameResult],
        value_predictions: list[tuple[float, float]],
        total_time: float,
    ) -> EvaluationResult:
        """Compute detailed statistics from game results.

        Args:
            win_rate: Win/loss/draw counts.
            game_results: List of individual game results.
            value_predictions: List of (predicted, actual) value pairs.
            total_time: Total evaluation time.

        Returns:
            EvaluationResult with all statistics.
        """
        # Average game length
        avg_game_length = np.mean([r.num_turns for r in game_results])

        # Average cities for won/lost games
        won_games = [r for r in game_results if r.winner == 0]
        lost_games = [r for r in game_results if r.winner == 1]

        avg_cities_won = (
            np.mean([r.player0_cities for r in won_games]) if won_games else 0.0
        )
        avg_cities_lost = (
            np.mean([r.player0_cities for r in lost_games]) if lost_games else 0.0
        )
        avg_units_won = (
            np.mean([r.player0_units for r in won_games]) if won_games else 0.0
        )
        avg_units_lost = (
            np.mean([r.player0_units for r in lost_games]) if lost_games else 0.0
        )

        # Value prediction correlation
        value_correlation = 0.0
        if len(value_predictions) > 10:
            predicted = np.array([p[0] for p in value_predictions])
            actual = np.array([p[1] for p in value_predictions])

            # Compute correlation if there's variance
            if np.std(predicted) > 1e-6 and np.std(actual) > 1e-6:
                value_correlation = float(np.corrcoef(predicted, actual)[0, 1])

        return EvaluationResult(
            win_rate=win_rate,
            game_results=game_results,
            avg_game_length=float(avg_game_length),
            avg_cities_won=float(avg_cities_won),
            avg_cities_lost=float(avg_cities_lost),
            avg_units_won=float(avg_units_won),
            avg_units_lost=float(avg_units_lost),
            total_evaluation_time=total_time,
            value_predictions=value_predictions,
            value_correlation=value_correlation,
        )


def run_baseline_evaluation(
    mcts,
    n_games_random: int = 100,
    n_games_heuristic: int = 50,
    board_size: int = 16,
    max_turns: int = 200,
    seed: int = 42,
) -> dict[str, EvaluationResult]:
    """Convenience function to run full baseline evaluation.

    Args:
        mcts: MCTS instance to evaluate.
        n_games_random: Games vs random agent.
        n_games_heuristic: Games vs heuristic agent.
        board_size: Board size for evaluation.
        max_turns: Max turns per game.
        seed: Random seed.

    Returns:
        Dictionary with 'random' and 'heuristic' evaluation results.
    """
    pipeline = EvaluationPipeline(
        board_size=board_size,
        max_turns=max_turns,
        seed=seed,
    )

    results = {}

    logger.info("=" * 60)
    logger.info("BASELINE EVALUATION")
    logger.info("=" * 60)

    results["random"] = pipeline.evaluate_vs_random(mcts, n_games=n_games_random)
    results["heuristic"] = pipeline.evaluate_vs_heuristic(
        mcts, n_games=n_games_heuristic
    )

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    logger.info(f"vs Random:    {results['random'].win_rate}")
    logger.info(f"vs Heuristic: {results['heuristic'].win_rate}")
    logger.info(f"Value Correlation (Random): {results['random'].value_correlation:.3f}")
    logger.info("=" * 60)

    return results
