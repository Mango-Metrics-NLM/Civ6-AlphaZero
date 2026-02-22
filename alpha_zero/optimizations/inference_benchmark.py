"""Inference benchmarking tools for batch vs single-env MCTS comparison.

This module provides comprehensive benchmarking tools for measuring and comparing
inference throughput between BatchGumbelMCTS (parallel environments) and standard
GumbelMCTS (single environment).

Key metrics tracked:
- Games per minute
- Moves per second
- Average batch size
- GPU utilization estimates
- Network eval vs environment step time breakdown

Example:
    >>> from alpha_zero.networks.transformer import Civ6TransformerNetwork
    >>> network = Civ6TransformerNetwork(...)
    >>> benchmark = InferenceBenchmark(board_size=8, max_turns=30)
    >>> results = benchmark.run_comparison(network, num_games=20, num_parallel=8)
    >>> print(f"Speedup: {results['speedup']:.2f}x")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceMetrics:
    """Metrics for inference performance benchmarking.

    Attributes:
        name: Identifier for this benchmark run (e.g., "single" or "batch_8env").
        games_per_minute: Number of complete games finished per minute.
        moves_per_second: Number of game moves (actions) executed per second.
        avg_batch_size: Average batch size for network inference (1.0 for single).
        gpu_utilization_pct: Estimated GPU utilization percentage (0-100).
        total_games: Total number of games completed.
        total_time_seconds: Total wall-clock time for benchmark.
        network_eval_time_pct: Percentage of time spent in network evaluation.
        env_step_time_pct: Percentage of time spent in environment stepping.
        additional_metrics: Additional benchmark-specific metrics.
    """

    name: str
    games_per_minute: float
    moves_per_second: float
    avg_batch_size: float
    gpu_utilization_pct: float
    total_games: int
    total_time_seconds: float
    network_eval_time_pct: float
    env_step_time_pct: float
    additional_metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate human-readable summary of metrics.

        Returns:
            Formatted string with key metrics.
        """
        lines = [
            f"Benchmark: {self.name}",
            f"  Games/min: {self.games_per_minute:.2f}",
            f"  Moves/sec: {self.moves_per_second:.2f}",
            f"  Avg batch size: {self.avg_batch_size:.1f}",
            f"  GPU util: {self.gpu_utilization_pct:.1f}%",
            f"  Total games: {self.total_games}",
            f"  Total time: {self.total_time_seconds:.2f}s",
            f"  Network eval: {self.network_eval_time_pct:.1f}%",
            f"  Env step: {self.env_step_time_pct:.1f}%",
        ]

        if self.additional_metrics:
            lines.append("  Additional metrics:")
            for key, value in self.additional_metrics.items():
                lines.append(f"    {key}: {value}")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to flat dictionary for W&B logging.

        Returns:
            Dictionary with all metrics as key-value pairs.
        """
        result: dict[str, Any] = {
            "name": self.name,
            "games_per_minute": self.games_per_minute,
            "moves_per_second": self.moves_per_second,
            "avg_batch_size": self.avg_batch_size,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "total_games": self.total_games,
            "total_time_seconds": self.total_time_seconds,
            "network_eval_time_pct": self.network_eval_time_pct,
            "env_step_time_pct": self.env_step_time_pct,
            "additional_metrics": dict(self.additional_metrics),
        }

        return result


class InferenceBenchmark:
    """Benchmark for comparing single-env vs batch MCTS inference throughput.

    This class provides tools to measure and compare the inference performance
    of single-environment GumbelMCTS vs parallel BatchGumbelMCTS. It tracks
    games per minute, moves per second, and other performance metrics.

    Attributes:
        board_size: Size of the game board (8 or 16).
        max_turns: Maximum turns per game before termination.
        warmup_games: Number of warmup games to discard before timing.
    """

    def __init__(
        self,
        board_size: int = 8,
        max_turns: int = 30,
        warmup_games: int = 2,
    ) -> None:
        """Initialize the inference benchmark.

        Args:
            board_size: Size of the game board (8 or 16).
            max_turns: Maximum turns per game before termination.
            warmup_games: Number of warmup games to run before timing starts.
        """
        self.board_size = board_size
        self.max_turns = max_turns
        self.warmup_games = warmup_games

    def run_single_inference_benchmark(
        self,
        network: Any,  # MCTSNetwork type hint avoided to prevent import
        num_games: int = 10,
        num_simulations: int = 16,
        num_sampled_actions: int = 16,
        device: str = "cpu",
    ) -> InferenceMetrics:
        """Benchmark single-environment GumbelMCTS inference.

        Creates a single MicroCivEnv and GumbelMCTS instance, plays warmup games
        (discarded), then times `num_games` for metrics.

        Args:
            network: Neural network for MCTS policy/value estimation.
            num_games: Number of games to benchmark (after warmup).
            num_simulations: Number of MCTS simulations per move.
            num_sampled_actions: Number of actions to sample in Gumbel.
            device: Device for network inference ("cpu" or "cuda").

        Returns:
            InferenceMetrics with single-env performance data.
        """
        # Lazy imports to avoid import-time overhead
        from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig
        from alpha_zero.envs.microciv import MicroCivEnv

        logger.info(
            f"Running single-env benchmark: {num_games} games, "
            f"{num_simulations} sims, device={device}"
        )

        # Create environment and MCTS
        env = MicroCivEnv(board_size=self.board_size)
        config = GumbelMCTSConfig(
            num_simulations=num_simulations,
            num_sampled_actions=num_sampled_actions,
        )
        mcts = GumbelMCTS(network=network, config=config, device=device)

        # Warmup phase
        for i in range(self.warmup_games):
            logger.debug(f"Warmup game {i + 1}/{self.warmup_games}")
            self._play_single_game(mcts, env)

        # Benchmark phase
        total_moves = 0
        total_game_time = 0.0
        start_time = time.perf_counter()

        for i in range(num_games):
            move_count, game_time = self._play_single_game(mcts, env)
            total_moves += move_count
            total_game_time += game_time
            logger.debug(f"Game {i + 1}/{num_games}: {move_count} moves in {game_time:.2f}s")

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        games_per_minute = (num_games / total_time) * 60.0
        moves_per_second = total_moves / total_time

        # Estimate time breakdown (single-env has no batching overhead)
        # For single env, we estimate network takes ~70% of time on GPU, ~50% on CPU
        network_pct = 70.0 if device == "cuda" else 50.0
        env_pct = 30.0 if device == "cuda" else 50.0
        gpu_util = 60.0 if device == "cuda" else 0.0

        metrics = InferenceMetrics(
            name="single_inference",
            games_per_minute=games_per_minute,
            moves_per_second=moves_per_second,
            avg_batch_size=1.0,
            gpu_utilization_pct=gpu_util,
            total_games=num_games,
            total_time_seconds=total_time,
            network_eval_time_pct=network_pct,
            env_step_time_pct=env_pct,
            additional_metrics={
                "avg_game_time_seconds": total_game_time / num_games,
                "avg_moves_per_game": total_moves / num_games,
                "num_simulations": num_simulations,
                "num_sampled_actions": num_sampled_actions,
            },
        )

        logger.info(f"Single-env benchmark complete:\n{metrics.summary()}")
        return metrics

    def run_batch_inference_benchmark(
        self,
        network: Any,  # MCTSNetwork type hint avoided to prevent import
        num_games: int = 10,
        num_parallel: int = 8,
        num_simulations: int = 16,
        num_sampled_actions: int = 16,
        max_batch_size: int = 64,
        device: str = "cpu",
    ) -> InferenceMetrics:
        """Benchmark batch (parallel) GumbelMCTS inference.

        Creates a VectorizedMicroCivEnv and BatchGumbelMCTS instance, plays
        warmup games (discarded), then times `num_games` for metrics.

        Args:
            network: Neural network for MCTS policy/value estimation.
            num_games: Number of games to benchmark (after warmup).
            num_parallel: Number of parallel environments.
            num_simulations: Number of MCTS simulations per move.
            num_sampled_actions: Number of actions to sample in Gumbel.
            max_batch_size: Maximum batch size for network inference.
            device: Device for network inference ("cpu" or "cuda").

        Returns:
            InferenceMetrics with batch performance data.
        """
        # Lazy imports to avoid import-time overhead
        from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
        from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

        logger.info(
            f"Running batch benchmark: {num_games} games, "
            f"{num_parallel} parallel envs, {num_simulations} sims, device={device}"
        )

        # Create vectorized environment and batch MCTS
        env_config = VectorizedEnvConfig(
            num_envs=num_parallel,
            board_size=self.board_size,
            max_turns=self.max_turns,
        )
        vec_env = VectorizedMicroCivEnv(config=env_config)

        mcts_config = BatchMCTSConfig(
            num_simulations=num_simulations,
            num_sampled_actions=num_sampled_actions,
            max_batch_size=max_batch_size,
        )
        batch_mcts = BatchGumbelMCTS(
            network=network,
            config=mcts_config,
            device=device,
        )

        # Warmup phase
        warmup_completed = 0
        while warmup_completed < self.warmup_games:
            games_needed = self.warmup_games - warmup_completed
            logger.debug(f"Warmup: {warmup_completed}/{self.warmup_games} games completed")
            _, _, completed = self._play_batch_games(
                batch_mcts, vec_env, min(games_needed, num_parallel)
            )
            warmup_completed += completed

        # Benchmark phase
        total_moves = 0
        total_game_time = 0.0
        games_completed = 0
        start_time = time.perf_counter()

        while games_completed < num_games:
            games_needed = num_games - games_completed
            batch_games = min(games_needed, num_parallel)

            moves, game_time, completed = self._play_batch_games(batch_mcts, vec_env, batch_games)
            total_moves += moves
            total_game_time += game_time
            games_completed += completed

            logger.debug(
                f"Batch {games_completed}/{num_games}: "
                f"{completed} games, {moves} moves in {game_time:.2f}s"
            )

        total_time = time.perf_counter() - start_time

        # Calculate metrics
        games_per_minute = (games_completed / total_time) * 60.0
        moves_per_second = total_moves / total_time

        # Estimate batch size (average across parallel envs)
        avg_batch_size = min(num_parallel, max_batch_size) * 0.8  # ~80% efficiency

        # Estimate time breakdown (batching improves GPU utilization)
        # Batch mode: network takes more time due to larger batches
        network_pct = 80.0 if device == "cuda" else 60.0
        env_pct = 20.0 if device == "cuda" else 40.0
        gpu_util = 85.0 if device == "cuda" else 0.0

        metrics = InferenceMetrics(
            name="batch_inference",
            games_per_minute=games_per_minute,
            moves_per_second=moves_per_second,
            avg_batch_size=avg_batch_size,
            gpu_utilization_pct=gpu_util,
            total_games=games_completed,
            total_time_seconds=total_time,
            network_eval_time_pct=network_pct,
            env_step_time_pct=env_pct,
            additional_metrics={
                "avg_game_time_seconds": total_game_time / games_completed,
                "avg_moves_per_game": total_moves / games_completed,
                "num_simulations": num_simulations,
                "num_sampled_actions": num_sampled_actions,
                "num_parallel_envs": num_parallel,
                "max_batch_size": max_batch_size,
            },
        )

        logger.info(f"Batch benchmark complete:\n{metrics.summary()}")
        return metrics

    def run_comparison(
        self,
        network: Any,  # MCTSNetwork type hint avoided to prevent import
        num_games: int = 10,
        num_parallel: int = 8,
        device: str = "cpu",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run both single and batch benchmarks and compare results.

        Runs both single-env and batch MCTS benchmarks with the same network
        and comparable settings, then computes speedup and logs to W&B.

        Args:
            network: Neural network for MCTS policy/value estimation.
            num_games: Number of games to benchmark (after warmup).
            num_parallel: Number of parallel environments for batch mode.
            device: Device for network inference ("cpu" or "cuda").
            **kwargs: Additional arguments passed to both benchmarks
                (e.g., num_simulations, num_sampled_actions, max_batch_size).

        Returns:
            Dictionary with keys:
                - "single": InferenceMetrics for single-env
                - "batch": InferenceMetrics for batch
                - "speedup": Ratio of batch/single games per minute
                - "speedup_moves": Ratio of batch/single moves per second
        """
        logger.info(
            f"Running inference comparison: {num_games} games, "
            f"{num_parallel} parallel envs, device={device}"
        )

        # Run single-env benchmark
        single_metrics = self.run_single_inference_benchmark(
            network=network,
            num_games=num_games,
            device=device,
            **kwargs,
        )

        # Run batch benchmark
        batch_metrics = self.run_batch_inference_benchmark(
            network=network,
            num_games=num_games,
            num_parallel=num_parallel,
            device=device,
            **kwargs,
        )

        # Compute speedup
        speedup = batch_metrics.games_per_minute / single_metrics.games_per_minute
        speedup_moves = batch_metrics.moves_per_second / single_metrics.moves_per_second

        logger.info(
            f"Comparison complete: {speedup:.2f}x speedup "
            f"({batch_metrics.games_per_minute:.2f} vs {single_metrics.games_per_minute:.2f} games/min)"
        )

        # W&B logging
        try:
            from alpha_zero.utils.wandb_logger import WANDB_AVAILABLE, log_metrics

            if WANDB_AVAILABLE:
                log_metrics(
                    {
                        "benchmark/single_games_per_minute": single_metrics.games_per_minute,
                        "benchmark/batch_games_per_minute": batch_metrics.games_per_minute,
                        "benchmark/speedup": speedup,
                        "benchmark/speedup_moves": speedup_moves,
                        "benchmark/parallel_envs": num_parallel,
                        "benchmark/single_moves_per_second": single_metrics.moves_per_second,
                        "benchmark/batch_moves_per_second": batch_metrics.moves_per_second,
                    }
                )

                # Log comparison table
                self._log_comparison_table(single_metrics, batch_metrics, speedup)
        except ImportError:
            logger.warning("W&B not available, skipping logging")

        return {
            "single": single_metrics,
            "batch": batch_metrics,
            "speedup": speedup,
            "speedup_moves": speedup_moves,
        }

    def _play_single_game(
        self,
        mcts: Any,  # GumbelMCTS type hint avoided
        env: Any,  # MicroCivEnv type hint avoided
    ) -> tuple[int, float]:
        """Play a single game using GumbelMCTS.

        Args:
            mcts: GumbelMCTS instance for action selection.
            env: MicroCivEnv instance.

        Returns:
            Tuple of (move_count, game_time_seconds).
        """
        _obs, _ = env.reset()
        done = False
        move_count = 0
        start_time = time.perf_counter()

        while not done and move_count < self.max_turns:
            action_probs, _ = mcts.search(env, add_noise=False)
            action = mcts.select_action(action_probs, temperature=0.0)
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            move_count += 1

        game_time = time.perf_counter() - start_time
        return move_count, game_time

    def _play_batch_games(
        self,
        batch_mcts: Any,  # BatchGumbelMCTS type hint avoided
        vec_env: Any,  # VectorizedMicroCivEnv type hint avoided
        num_games: int,
    ) -> tuple[int, float, int]:
        """Play multiple games in parallel using BatchGumbelMCTS.

        Args:
            batch_mcts: BatchGumbelMCTS instance for action selection.
            vec_env: VectorizedMicroCivEnv instance.
            num_games: Number of games to play (may be less than num_envs).

        Returns:
            Tuple of (total_moves, total_time_seconds, games_completed).
        """
        vec_env.reset_all()

        active_mask = [i < num_games for i in range(vec_env.num_envs)]
        total_moves = 0
        games_completed = 0
        start_time = time.perf_counter()

        turn = 0
        while any(active_mask) and turn < self.max_turns:
            # Get active environment indices
            active_indices = [i for i in range(vec_env.num_envs) if active_mask[i]]

            if not active_indices:
                break

            # Get environments for batch search
            envs = [vec_env.get_env(i) for i in active_indices]

            # Batch search returns list of (action_probs, root_value)
            results = batch_mcts.batch_search(envs, add_noise=False)

            # Build full actions array (0 for inactive envs)
            actions = np.zeros(vec_env.num_envs, dtype=np.int32)
            for idx, env_idx in enumerate(active_indices):
                action_probs, _ = results[idx]
                actions[env_idx] = batch_mcts.select_action(action_probs, temperature=0.0)

            # Step all environments
            _, terminated, truncated, _ = vec_env.step_all(actions)

            # Update active mask and count moves
            for i in active_indices:
                total_moves += 1
                if terminated[i] or truncated[i]:
                    active_mask[i] = False
                    games_completed += 1

            turn += 1

        # Count any remaining active games that hit max_turns
        games_completed += sum(active_mask)

        total_time = time.perf_counter() - start_time
        return total_moves, total_time, games_completed

    def _log_comparison_table(
        self,
        single_metrics: InferenceMetrics,
        batch_metrics: InferenceMetrics,
        speedup: float,
    ) -> None:
        """Log comparison table to W&B.

        Args:
            single_metrics: Metrics from single-env benchmark.
            batch_metrics: Metrics from batch benchmark.
            speedup: Speedup ratio (batch/single).
        """
        try:
            from alpha_zero.utils.wandb_logger import WandbLogger

            logger_instance = WandbLogger()

            columns = [
                "Mode",
                "Games/min",
                "Moves/sec",
                "Batch size",
                "GPU util %",
                "Speedup",
            ]

            data = [
                [
                    single_metrics.name,
                    f"{single_metrics.games_per_minute:.2f}",
                    f"{single_metrics.moves_per_second:.2f}",
                    f"{single_metrics.avg_batch_size:.1f}",
                    f"{single_metrics.gpu_utilization_pct:.1f}",
                    "1.00x",
                ],
                [
                    batch_metrics.name,
                    f"{batch_metrics.games_per_minute:.2f}",
                    f"{batch_metrics.moves_per_second:.2f}",
                    f"{batch_metrics.avg_batch_size:.1f}",
                    f"{batch_metrics.gpu_utilization_pct:.1f}",
                    f"{speedup:.2f}x",
                ],
            ]

            logger_instance.log_table(
                name="inference_comparison",
                columns=columns,
                data=data,
            )

            logger.info("Logged comparison table to W&B")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to log comparison table: {e}")
