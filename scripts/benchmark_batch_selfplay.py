#!/usr/bin/env python3
"""Benchmark sequential vs batched self-play on the local machine.

Builds a small Civ6TransformerNetwork, runs InferenceBenchmark.run_comparison
which logs to MLflow via experiment_tracker, and prints a comparison table.

Usage:
    python scripts/benchmark_batch_selfplay.py \\
      --games=20 --num_parallel_games=8 --max_batch_size=64 --device=cpu

Env vars honoured:
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME, MLFLOW_RUN_NAME, MLFLOW_TAGS.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from alpha_zero.cloud.gcp.training_worker import create_model
from alpha_zero.monitoring.config import MonitoringConfig
from alpha_zero.monitoring.mlflow_logger import MLflowLogger
from alpha_zero.optimizations.inference_benchmark import InferenceBenchmark
from alpha_zero.utils.game_budget import add_budget_arguments, enforce_cli

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark sequential vs batched self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--num_parallel_games", type=int, default=8)
    p.add_argument("--max_batch_size", type=int, default=64)
    p.add_argument("--mcts_simulations", type=int, default=8)
    p.add_argument("--board_size", type=int, default=8)
    p.add_argument("--max_turns", type=int, default=15)
    p.add_argument("--warmup_games", type=int, default=2)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--num_filters", type=int, default=64)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    add_budget_arguments(p)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    # The sequential and the batched variant each play `games` plus warmup.
    enforce_cli(
        "benchmark_batch_selfplay",
        {"games": args.games, "warmup_games": args.warmup_games},
        args,
    )

    device = torch.device(args.device)
    network = create_model(
        board_size=args.board_size,
        action_space_size=376,
        num_res_blocks=args.num_res_blocks,
        num_filters=args.num_filters,
    ).to(device)

    cfg = MonitoringConfig.from_env()
    if not cfg.experiment_name:
        cfg.experiment_name = "v1.5.0-bench"
    if not cfg.run_name:
        cfg.run_name = (
            f"bench-pgames{args.num_parallel_games}-batch{args.max_batch_size}-"
            f"sims{args.mcts_simulations}"
        )

    with MLflowLogger.from_config(cfg) as ml:
        ml.log_params(vars(args))

        bench = InferenceBenchmark(
            board_size=args.board_size,
            max_turns=args.max_turns,
            warmup_games=args.warmup_games,
        )
        # Call single/batch directly because InferenceBenchmark.run_comparison
        # forwards **kwargs to BOTH methods and run_single_inference_benchmark
        # does not accept max_batch_size.
        single = bench.run_single_inference_benchmark(
            network=network,
            num_games=args.games,
            num_simulations=args.mcts_simulations,
            device=args.device,
        )
        batch = bench.run_batch_inference_benchmark(
            network=network,
            num_games=args.games,
            num_parallel=args.num_parallel_games,
            num_simulations=args.mcts_simulations,
            max_batch_size=args.max_batch_size,
            device=args.device,
        )
        speedup = batch.games_per_minute / max(single.games_per_minute, 1e-8)
        speedup_moves = batch.moves_per_second / max(single.moves_per_second, 1e-8)
        ml.log_metrics(
            {
                "benchmark/single_games_per_minute": single.games_per_minute,
                "benchmark/batch_games_per_minute": batch.games_per_minute,
                "benchmark/speedup": speedup,
                "benchmark/speedup_moves": speedup_moves,
                "benchmark/parallel_envs": float(args.num_parallel_games),
                "benchmark/single_moves_per_second": single.moves_per_second,
                "benchmark/batch_moves_per_second": batch.moves_per_second,
            }
        )
        if single.peak_gpu_memory_mb is not None:
            ml.log_metric("benchmark/single_peak_gpu_memory_mb", single.peak_gpu_memory_mb)
        if batch.peak_gpu_memory_mb is not None:
            ml.log_metric("benchmark/batch_peak_gpu_memory_mb", batch.peak_gpu_memory_mb)

        # Print comparison table to stdout for human-readable summary.
        print("")
        print("=" * 72)
        print(
            f"{'Variant':<12} {'games/min':>12} {'moves/sec':>12} {'avg_batch':>12} {'peak_mb':>12}"
        )
        print("-" * 72)
        for label, m in (("sequential", single), ("batched", batch)):
            mem = f"{m.peak_gpu_memory_mb:.0f}" if m.peak_gpu_memory_mb is not None else "n/a"
            print(
                f"{label:<12} {m.games_per_minute:>12.2f} {m.moves_per_second:>12.2f} "
                f"{m.avg_batch_size:>12.2f} {mem:>12}"
            )
        print("-" * 72)
        print(f"speedup (batched / sequential games/min): {speedup:.2f}x")
        print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
