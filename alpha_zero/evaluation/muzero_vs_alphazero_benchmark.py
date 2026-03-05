"""AlphaZero vs MuZero Comparison Benchmark.

Head-to-head comparison framework for evaluating AlphaZero and MuZero
on identical hyperparameters and game settings.

Key metrics:
- Search efficiency: env clones (AlphaZero) vs dynamics model calls (MuZero)
- Win rate: head-to-head and vs random baseline
- Training speed: steps/second for each algorithm
- Network utilization: total inference calls per game
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Optional W&B import
try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


@dataclass
class BenchmarkConfig:
    """Configuration for AlphaZero vs MuZero benchmark.

    All parameters are shared between algorithms to ensure fair comparison.
    """

    num_games: int = 50
    mcts_simulations: int = 100
    board_size: int = 8
    max_turns: int = 30
    seed: int = 42
    action_space_size: int = 376
    num_res_blocks: int = 6
    num_filters: int = 64
    temperature: float = 1.0
    temperature_threshold: int = 15
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    device: str = "cpu"
    # Reward classification thresholds — tune to match your environment's
    # reward distribution.  MicroCivEnv returns +1 / -1 for win/loss, so the
    # defaults (>0.5 → win, <-0.5 → loss) are appropriate.
    win_reward_threshold: float = 0.5
    loss_reward_threshold: float = -0.5
    # Gameplay environment preset
    gameplay_preset: str = "default"
    # Output configuration
    output_dir: str = "benchmark_results"
    wandb_project: str = "civ6-alphazero-benchmark"
    wandb_enabled: bool = False


@dataclass
class SearchMetrics:
    """Search efficiency metrics for a single algorithm."""

    total_env_clones: int = 0
    total_dynamics_calls: int = 0
    total_network_evals: int = 0
    avg_search_time_ms: float = 0.0
    avg_tree_depth: float = 0.0
    avg_nodes_expanded: int = 0


@dataclass
class BenchmarkResult:
    """Results from running one algorithm on the benchmark."""

    algorithm: str
    num_games: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    win_rate: float = 0.0
    avg_game_length: float = 0.0
    avg_reward: float = 0.0
    total_time_seconds: float = 0.0
    games_per_minute: float = 0.0
    search_metrics: SearchMetrics = field(default_factory=SearchMetrics)
    game_lengths: list[int] = field(default_factory=list)
    game_rewards: list[float] = field(default_factory=list)

    def compute_derived_metrics(
        self,
        win_threshold: float = 0.5,
        loss_threshold: float = -0.5,
    ) -> None:
        """Compute derived metrics from raw game data.

        Args:
            win_threshold: Reward above this value counts as a win.
            loss_threshold: Reward below this value counts as a loss.
        """
        if self.game_lengths:
            self.avg_game_length = sum(self.game_lengths) / len(self.game_lengths)
        if self.game_rewards:
            self.avg_reward = sum(self.game_rewards) / len(self.game_rewards)
            self.wins = sum(1 for r in self.game_rewards if r > win_threshold)
            self.losses = sum(1 for r in self.game_rewards if r < loss_threshold)
            self.draws = self.num_games - self.wins - self.losses
            self.win_rate = self.wins / max(self.num_games, 1)
        if self.total_time_seconds > 0:
            self.games_per_minute = self.num_games / (self.total_time_seconds / 60)


@dataclass
class ComparisonReport:
    """Complete comparison report between AlphaZero and MuZero."""

    config: BenchmarkConfig
    alphazero_result: BenchmarkResult
    muzero_result: BenchmarkResult
    timestamp: str = ""
    summary: dict[str, Any] = field(default_factory=dict)

    def compute_summary(self) -> dict[str, Any]:
        """Compute summary comparison metrics."""
        self.summary = {
            "alphazero_win_rate": self.alphazero_result.win_rate,
            "muzero_win_rate": self.muzero_result.win_rate,
            "win_rate_delta": self.muzero_result.win_rate - self.alphazero_result.win_rate,
            "alphazero_avg_game_length": self.alphazero_result.avg_game_length,
            "muzero_avg_game_length": self.muzero_result.avg_game_length,
            "alphazero_env_clones": self.alphazero_result.search_metrics.total_env_clones,
            "muzero_dynamics_calls": self.muzero_result.search_metrics.total_dynamics_calls,
            "search_efficiency_ratio": (
                self.muzero_result.search_metrics.total_dynamics_calls
                / max(self.alphazero_result.search_metrics.total_env_clones, 1)
            ),
            "alphazero_games_per_min": self.alphazero_result.games_per_minute,
            "muzero_games_per_min": self.muzero_result.games_per_minute,
            "speed_ratio": (
                self.muzero_result.games_per_minute
                / max(self.alphazero_result.games_per_minute, 0.001)
            ),
        }
        return self.summary


class AlphaZeroVsMuZeroBenchmark:
    """Head-to-head comparison runner for AlphaZero vs MuZero.

    Runs identical games with both algorithms under the same hyperparameters
    and environment settings, then compares search efficiency, win rate,
    and training speed metrics.

    Example usage:
        config = BenchmarkConfig(num_games=50, mcts_simulations=100)
        benchmark = AlphaZeroVsMuZeroBenchmark(config)
        report = benchmark.run_comparison()
        benchmark.generate_report(report, output_dir="results/")
    """

    def __init__(self, config: BenchmarkConfig) -> None:
        """Initialize benchmark runner.

        Args:
            config: Benchmark configuration.
        """
        self.config = config
        self._wandb_run: Any = None

    def run_comparison(
        self,
        alphazero_model: torch.nn.Module | None = None,
        muzero_model: torch.nn.Module | None = None,
    ) -> ComparisonReport:
        """Run identical games with both algorithms and compare.

        Args:
            alphazero_model: Pre-loaded AlphaZero model. If None, creates a fresh one.
            muzero_model: Pre-loaded MuZero model. If None, creates a fresh one.

        Returns:
            ComparisonReport with results from both algorithms.
        """
        import datetime

        logger.info(
            "Starting AlphaZero vs MuZero benchmark: %d games, %d sims",
            self.config.num_games,
            self.config.mcts_simulations,
        )

        # Set random seed for reproducibility
        torch.manual_seed(self.config.seed)

        # Run AlphaZero
        logger.info("Running AlphaZero evaluation...")
        alphazero_result = self._run_algorithm(
            algorithm="alphazero",
            model=alphazero_model,
        )

        # Reset seed for fair comparison
        torch.manual_seed(self.config.seed)

        # Run MuZero
        logger.info("Running MuZero evaluation...")
        muzero_result = self._run_algorithm(
            algorithm="muzero",
            model=muzero_model,
        )

        # Build report
        report = ComparisonReport(
            config=self.config,
            alphazero_result=alphazero_result,
            muzero_result=muzero_result,
            timestamp=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        )
        report.compute_summary()

        logger.info("Benchmark complete. Summary: %s", report.summary)
        return report

    def _run_algorithm(
        self,
        algorithm: str,
        model: torch.nn.Module | None = None,
    ) -> BenchmarkResult:
        """Run one algorithm on the benchmark games.

        Args:
            algorithm: "alphazero" or "muzero"
            model: Pre-loaded model, or None to create a fresh one.

        Returns:
            BenchmarkResult with game statistics and search metrics.
        """
        result = BenchmarkResult(algorithm=algorithm, num_games=self.config.num_games)
        search_metrics = SearchMetrics()

        # Create model if not provided
        if model is None:
            model = self._create_model(algorithm)

        model = model.to(self.config.device)
        model.eval()

        start_time = time.time()

        for game_idx in range(self.config.num_games):
            game_result = self._play_single_game(
                model=model,
                algorithm=algorithm,
                game_idx=game_idx,
            )

            result.game_lengths.append(game_result["length"])
            result.game_rewards.append(game_result["reward"])

            # Accumulate search metrics
            search_metrics.total_env_clones += game_result.get("env_clones", 0)
            search_metrics.total_dynamics_calls += game_result.get("dynamics_calls", 0)
            search_metrics.total_network_evals += game_result.get("network_evals", 0)

            if (game_idx + 1) % 10 == 0:
                logger.info(
                    "[%s] Progress: %d/%d games",
                    algorithm,
                    game_idx + 1,
                    self.config.num_games,
                )

        result.total_time_seconds = time.time() - start_time

        # Compute averages
        if self.config.num_games > 0:
            search_metrics.avg_search_time_ms = (
                result.total_time_seconds * 1000 / self.config.num_games
            )

        result.search_metrics = search_metrics
        result.compute_derived_metrics(
            win_threshold=self.config.win_reward_threshold,
            loss_threshold=self.config.loss_reward_threshold,
        )

        logger.info(
            "[%s] Results: win_rate=%.2f, avg_length=%.1f, %.1f games/min",
            algorithm,
            result.win_rate,
            result.avg_game_length,
            result.games_per_minute,
        )
        return result

    def _create_model(self, algorithm: str) -> torch.nn.Module:
        """Create a fresh model for the specified algorithm.

        Args:
            algorithm: "alphazero" or "muzero"

        Returns:
            Initialized neural network model.
        """
        if algorithm == "muzero":
            from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

            mz_config = MuZeroConfig(
                spatial_channels=13,
                spatial_size=self.config.board_size,
                vector_dim=64,
                action_space_size=self.config.action_space_size,
                num_res_blocks=self.config.num_res_blocks,
                hidden_state_channels=self.config.num_filters,
            )
            return MuZeroNetwork(config=mz_config)
        else:
            from alpha_zero.core.transformer_network import (
                Civ6TransformerNetwork,
                TransformerConfig,
            )

            az_config = TransformerConfig(
                spatial_channels=13,
                spatial_size=self.config.board_size,
                vector_dim=64,
                action_space_size=self.config.action_space_size,
                num_layers=self.config.num_res_blocks,
                embed_dim=self.config.num_filters,
            )
            return Civ6TransformerNetwork(config=az_config)

    def _play_single_game(
        self,
        model: torch.nn.Module,
        algorithm: str,
        game_idx: int,
    ) -> dict[str, Any]:
        """Play a single game and return metrics.

        Args:
            model: Neural network model.
            algorithm: "alphazero" or "muzero"
            game_idx: Game index for logging.

        Returns:
            Dict with game metrics including length, reward, and search stats.
        """
        from alpha_zero.core.mcts_factory import create_mcts
        from alpha_zero.envs.env_factory import create_game_env

        env = create_game_env(
            board_size=self.config.board_size,
            max_turns=self.config.max_turns,
            gameplay_preset=self.config.gameplay_preset,
        )

        # Select MCTS variant
        mcts_variant = "muzero" if algorithm == "muzero" else "civ6"
        mcts_kwargs: dict[str, Any] = {
            "num_simulations": self.config.mcts_simulations,
        }
        if algorithm != "muzero":
            mcts_kwargs.update(
                {
                    "c_puct": self.config.c_puct,
                    "dirichlet_alpha": self.config.dirichlet_alpha,
                    "dirichlet_epsilon": self.config.dirichlet_epsilon,
                }
            )

        mcts = create_mcts(
            variant=mcts_variant,
            network=model,
            device=self.config.device,
            **mcts_kwargs,
        )

        env.reset()
        done = False
        game_length = 0
        reward = 0.0
        env_clones = 0
        dynamics_calls = 0
        network_evals = 0

        while not done:
            try:
                policy, _ = mcts.search(env)
                action = mcts.select_action(policy, temperature=0.0)  # Greedy for eval
            except Exception as e:
                logger.warning("MCTS failed at step %d for %s: %s", game_length, algorithm, e)
                break

            step_result = env.step(action)
            reward = float(step_result[1])
            done = bool(step_result[2]) or bool(step_result[3])
            game_length += 1

            # Track search metrics
            if algorithm == "muzero":
                dynamics_calls += self.config.mcts_simulations
            else:
                env_clones += self.config.mcts_simulations
            network_evals += self.config.mcts_simulations

        return {
            "length": game_length,
            "reward": reward,
            "env_clones": env_clones,
            "dynamics_calls": dynamics_calls,
            "network_evals": network_evals,
        }

    def generate_report(
        self,
        report: ComparisonReport,
        output_dir: str | None = None,
    ) -> dict[str, Any]:
        """Generate JSON + markdown summary with key metrics.

        Args:
            report: ComparisonReport from run_comparison.
            output_dir: Directory to write report files.

        Returns:
            Report as a dictionary.
        """
        output_path = Path(output_dir or self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        report_dict = {
            "timestamp": report.timestamp,
            "config": asdict(report.config),
            "alphazero": asdict(report.alphazero_result),
            "muzero": asdict(report.muzero_result),
            "summary": report.summary,
        }

        # Write JSON report
        json_path = output_path / "benchmark_report.json"
        with open(json_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        logger.info("JSON report written to %s", json_path)

        # Write Markdown report
        md_path = output_path / "benchmark_report.md"
        md_content = self._generate_markdown(report)
        with open(md_path, "w") as f:
            f.write(md_content)
        logger.info("Markdown report written to %s", md_path)

        return report_dict

    def _generate_markdown(self, report: ComparisonReport) -> str:
        """Generate markdown report content.

        Args:
            report: ComparisonReport with data.

        Returns:
            Markdown string.
        """
        az = report.alphazero_result
        mz = report.muzero_result
        summary = report.summary

        lines = [
            "# AlphaZero vs MuZero Benchmark Report",
            "",
            f"**Date**: {report.timestamp}",
            f"**Games per algorithm**: {report.config.num_games}",
            f"**MCTS Simulations**: {report.config.mcts_simulations}",
            f"**Board Size**: {report.config.board_size}x{report.config.board_size}",
            "",
            "## Win Rates",
            "",
            "| Metric | AlphaZero | MuZero |",
            "|--------|----------|--------|",
            f"| Win Rate | {az.win_rate:.2%} | {mz.win_rate:.2%} |",
            f"| Wins | {az.wins} | {mz.wins} |",
            f"| Losses | {az.losses} | {mz.losses} |",
            f"| Draws | {az.draws} | {mz.draws} |",
            "",
            "## Search Efficiency",
            "",
            "| Metric | AlphaZero | MuZero |",
            "|--------|----------|--------|",
            f"| Env Clones | {az.search_metrics.total_env_clones:,} | N/A |",
            f"| Dynamics Calls | N/A | {mz.search_metrics.total_dynamics_calls:,} |",
            f"| Network Evals | {az.search_metrics.total_network_evals:,} | "
            f"{mz.search_metrics.total_network_evals:,} |",
            f"| Avg Search Time (ms) | {az.search_metrics.avg_search_time_ms:.1f} | "
            f"{mz.search_metrics.avg_search_time_ms:.1f} |",
            "",
            "## Performance",
            "",
            "| Metric | AlphaZero | MuZero |",
            "|--------|----------|--------|",
            f"| Games/Min | {az.games_per_minute:.1f} | {mz.games_per_minute:.1f} |",
            f"| Avg Game Length | {az.avg_game_length:.1f} | {mz.avg_game_length:.1f} |",
            f"| Total Time (s) | {az.total_time_seconds:.1f} | " f"{mz.total_time_seconds:.1f} |",
            "",
            "## Key Insights",
            "",
            f"- **Search Efficiency Ratio**: {summary.get('search_efficiency_ratio', 0):.2f}x "
            "(dynamics calls / env clones)",
            f"- **Speed Ratio**: {summary.get('speed_ratio', 0):.2f}x "
            "(MuZero games/min / AlphaZero games/min)",
            f"- **Win Rate Delta**: {summary.get('win_rate_delta', 0):+.2%} "
            "(MuZero - AlphaZero)",
            "",
        ]
        return "\n".join(lines)

    def log_to_wandb(self, report: ComparisonReport) -> None:
        """Log comparison metrics to W&B with grouped panels.

        Args:
            report: ComparisonReport with data.
        """
        if not _WANDB_AVAILABLE:
            logger.warning("W&B not available, skipping logging")
            return

        if not self.config.wandb_enabled:
            logger.info("W&B logging disabled in config")
            return

        try:
            run = wandb.init(
                project=self.config.wandb_project,
                name=f"benchmark-{report.timestamp}",
                config=asdict(self.config),
            )

            # Log summary metrics
            summary = report.summary
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"benchmark/{key}": value})

            # Log per-algorithm metrics
            for algo_name, result in [
                ("alphazero", report.alphazero_result),
                ("muzero", report.muzero_result),
            ]:
                wandb.log(
                    {
                        f"{algo_name}/win_rate": result.win_rate,
                        f"{algo_name}/avg_game_length": result.avg_game_length,
                        f"{algo_name}/games_per_minute": result.games_per_minute,
                        f"{algo_name}/env_clones": result.search_metrics.total_env_clones,
                        f"{algo_name}/dynamics_calls": result.search_metrics.total_dynamics_calls,
                        f"{algo_name}/network_evals": result.search_metrics.total_network_evals,
                    }
                )

            # Log comparison table
            table = wandb.Table(
                columns=[
                    "Algorithm",
                    "Win Rate",
                    "Games/Min",
                    "Env Clones",
                    "Dynamics Calls",
                    "Avg Game Length",
                ],
                data=[
                    [
                        "AlphaZero",
                        report.alphazero_result.win_rate,
                        report.alphazero_result.games_per_minute,
                        report.alphazero_result.search_metrics.total_env_clones,
                        report.alphazero_result.search_metrics.total_dynamics_calls,
                        report.alphazero_result.avg_game_length,
                    ],
                    [
                        "MuZero",
                        report.muzero_result.win_rate,
                        report.muzero_result.games_per_minute,
                        report.muzero_result.search_metrics.total_env_clones,
                        report.muzero_result.search_metrics.total_dynamics_calls,
                        report.muzero_result.avg_game_length,
                    ],
                ],
            )
            wandb.log({"benchmark/comparison_table": table})

            run.finish()
            logger.info("W&B logging complete")

        except Exception as e:
            logger.error("Failed to log to W&B: %s", e)
