"""CLI entry point for the Strategic Planning SDK.

Provides commands for training, evaluation, and system health checks.
Uses argparse (stdlib) for zero-dependency CLI.

Usage::

    python -m src.cli train --config configs/config.yaml
    python -m src.cli train --env mini_resource --iterations 10
    python -m src.cli evaluate --env mini_resource --games 20
    python -m src.cli health-check
    python -m src.cli benchmark --env mini_resource
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from src.core.constants import DEFAULT_MAX_STEPS

logger = logging.getLogger(__name__)


def _get_default_registry() -> Any:
    """Create and populate the default environment registry."""
    from src.environments.mini_games import (
        MiniResourceEnv,
        TacticalCombatEnv,
        TerritoryControlEnv,
    )
    from src.environments.registry import EnvironmentRegistry

    registry = EnvironmentRegistry()
    registry.register("mini_resource", MiniResourceEnv)
    registry.register("territory_control", TerritoryControlEnv)
    registry.register("tactical_combat", TacticalCombatEnv)
    return registry


def cmd_train(args: argparse.Namespace) -> int:
    """Run training loop."""
    from src.core.config import SDKConfig, load_config
    from src.core.training import Learner, TrainingOrchestrator

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = SDKConfig()

    # Apply CLI overrides
    overrides: dict[str, Any] = {}
    if args.iterations is not None:
        overrides.setdefault("training", {})["max_iterations"] = args.iterations
    if args.games is not None:
        overrides.setdefault("training", {})["games_per_iteration"] = args.games
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size

    if overrides:
        config = load_config(overrides=overrides)

    config = config.apply_env_overrides()

    # Setup environment
    registry = _get_default_registry()
    env_name = args.env or "mini_resource"

    def env_factory() -> Any:
        return registry.create(env_name)

    # Setup learner (dry-run mode if no torch)
    learner = Learner(config=config.training)

    # Build orchestrator
    orchestrator = TrainingOrchestrator(
        config=config,
        env_factory=env_factory,
        learner=learner,
    )

    print(f"Starting training: env={env_name}, iterations={config.training.max_iterations}")
    if args.dry_run:
        print("  [DRY RUN MODE]")

    metrics_list = orchestrator.run(dry_run=args.dry_run)

    print(f"\nTraining complete: {len(metrics_list)} iterations")
    if metrics_list:
        last = metrics_list[-1]
        print(f"  Final Elo: {last.eval_elo:.1f}")
        print(f"  Final Loss: {last.training_loss:.4f}")
        print(f"  Best Elo: {orchestrator.best_elo:.1f}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run evaluation."""
    from src.core.config import MCTSConfig
    from src.core.training.evaluator import Evaluator

    registry = _get_default_registry()
    env_name = args.env or "mini_resource"

    def env_factory() -> Any:
        return registry.create(env_name)

    mcts_config = MCTSConfig()
    evaluator = Evaluator(
        mcts_config=mcts_config,
        num_eval_games=args.games,
    )

    # Use random baseline if no checkpoint specified
    print(f"Evaluating on {env_name}: {args.games} games vs random baseline")

    from src.core.training.evaluator import random_eval_fn

    result = evaluator.evaluate(
        candidate_fn=random_eval_fn,
        env_factory=env_factory,
        baseline_type="random",
    )

    print("\nEvaluation Results:")
    print(f"  Elo: {result.elo:.1f}")
    print(f"  Win Rate: {result.win_rate:.1%}")
    print(f"  W/L/D: {result.wins}/{result.losses}/{result.draws}")

    return 0


def cmd_health_check(args: argparse.Namespace) -> int:
    """Validate SDK installation and components."""
    checks: list[tuple[str, bool, str]] = []

    # Check core imports
    try:
        from src.core.types import GameOutcome  # noqa: F401
        checks.append(("Core types", True, "OK"))
    except Exception as e:
        checks.append(("Core types", False, str(e)))

    try:
        from src.core.config import SDKConfig  # noqa: F401
        checks.append(("Configuration", True, "OK"))
    except Exception as e:
        checks.append(("Configuration", False, str(e)))

    try:
        from src.core.protocols import StrategicEnvironment  # noqa: F401
        checks.append(("Protocols", True, "OK"))
    except Exception as e:
        checks.append(("Protocols", False, str(e)))

    try:
        from src.core.mcts import MCTSEngine  # noqa: F401
        checks.append(("MCTS Engine", True, "OK"))
    except Exception as e:
        checks.append(("MCTS Engine", False, str(e)))

    try:
        from src.core.networks import HAS_TORCH
        status = "Available" if HAS_TORCH else "Not installed (dry-run mode)"
        checks.append(("PyTorch", HAS_TORCH, status))
    except Exception as e:
        checks.append(("PyTorch", False, str(e)))

    try:
        from src.core.training import TrainingOrchestrator  # noqa: F401
        checks.append(("Training", True, "OK"))
    except Exception as e:
        checks.append(("Training", False, str(e)))

    try:
        from src.environments.mini_games import (
            MiniResourceEnv,
            TacticalCombatEnv,
            TerritoryControlEnv,
        )
        # Quick smoke test
        for cls in [MiniResourceEnv, TacticalCombatEnv, TerritoryControlEnv]:
            env = cls()
            env.reset(seed=42)
            assert not env.is_terminal()
        checks.append(("Mini-games", True, "3 environments OK"))
    except Exception as e:
        checks.append(("Mini-games", False, str(e)))

    try:
        reg = _get_default_registry()
        checks.append(("Registry", True, f"{len(reg)} environments"))
    except Exception as e:
        checks.append(("Registry", False, str(e)))

    # Check YAML configs
    config_dir = Path("configs")
    if config_dir.exists():
        yaml_files = list(config_dir.rglob("*.yaml"))
        checks.append(("YAML configs", True, f"{len(yaml_files)} files found"))
    else:
        checks.append(("YAML configs", False, "configs/ directory not found"))

    # Print results
    print("SDK Health Check")
    print("=" * 50)
    all_ok = True
    for name, ok, msg in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}: {msg}")
    print("=" * 50)
    print(f"Overall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")

    return 0 if all_ok else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run performance benchmark on an environment."""
    registry = _get_default_registry()
    env_name = args.env or "mini_resource"
    env = registry.create(env_name)

    import numpy as np

    num_episodes = args.episodes
    total_steps = 0
    start = time.monotonic()

    for ep in range(num_episodes):
        env.reset(seed=ep)
        steps = 0
        while not env.is_terminal() and steps < DEFAULT_MAX_STEPS:
            mask = env.get_valid_actions()
            valid_actions = np.where(mask)[0]
            if len(valid_actions) == 0:
                break
            action = int(valid_actions[ep % len(valid_actions)])
            env.step(action)
            steps += 1
        total_steps += steps

    elapsed = time.monotonic() - start
    print(f"Benchmark: {env_name}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total steps: {total_steps}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps/sec: {total_steps / elapsed:.0f}")
    print(f"  Episodes/sec: {num_episodes / elapsed:.1f}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="spsdk",
        description="Strategic Planning SDK - CLI",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable debug logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Run training loop")
    train_parser.add_argument("--config", type=str, help="Path to YAML config file")
    train_parser.add_argument("--env", type=str, help="Environment name")
    train_parser.add_argument("--iterations", type=int, help="Max training iterations")
    train_parser.add_argument("--games", type=int, help="Games per iteration")
    train_parser.add_argument("--batch-size", type=int, help="Training batch size")
    train_parser.add_argument(
        "--dry-run", action="store_true", help="Run one iteration for validation"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument("--env", type=str, help="Environment name")
    eval_parser.add_argument(
        "--games", type=int, default=20, help="Number of evaluation games"
    )
    eval_parser.add_argument("--checkpoint", type=str, help="Checkpoint path to evaluate")

    # Health check command
    subparsers.add_parser("health-check", help="Validate SDK installation")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run performance benchmark")
    bench_parser.add_argument("--env", type=str, help="Environment name")
    bench_parser.add_argument(
        "--episodes", type=int, default=100, help="Number of benchmark episodes"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "health-check": cmd_health_check,
        "benchmark": cmd_benchmark,
    }

    handler = commands.get(args.command)
    if handler is None:
        logger.error("Unknown command: %s", args.command)
        parser.print_help()
        return 1

    logger.debug("Executing command: %s", args.command)
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
