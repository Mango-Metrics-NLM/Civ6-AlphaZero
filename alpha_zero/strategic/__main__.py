"""CLI entry point for the Strategic Planning SDK.

Usage:
    python -m alpha_zero.strategic --help
    python -m alpha_zero.strategic plan --env resource_allocation --preset fast
    python -m alpha_zero.strategic train --env resource_allocation --preset fast
    python -m alpha_zero.strategic evaluate --env resource_allocation --scenarios scenarios.yaml
    python -m alpha_zero.strategic list-envs
    python -m alpha_zero.strategic --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import yaml

from alpha_zero.strategic.config import StrategicConfig
from alpha_zero.strategic.planner import StrategicPlanner
from alpha_zero.strategic.registry import EnvironmentRegistry
from alpha_zero.strategic.scenarios import Scenario

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="strategic-planning-sdk",
        description="Strategic Planning SDK - MCTS-based multi-objective planning",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Plan command
    plan_parser = subparsers.add_parser(
        "plan",
        help="Run strategic planning on an environment",
    )
    plan_parser.add_argument(
        "--env",
        type=str,
        default="resource_allocation",
        help="Environment name (default: resource_allocation)",
    )
    plan_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "thorough"],
        default="fast",
        help="Configuration preset (default: fast)",
    )
    plan_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    plan_parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of planning episodes (overrides preset)",
    )
    plan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save planning results (JSON)",
    )
    plan_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without running",
    )
    plan_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train a policy via iterative self-play planning",
    )
    train_parser.add_argument(
        "--env",
        type=str,
        default="resource_allocation",
        help="Environment name (default: resource_allocation)",
    )
    train_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "thorough"],
        default="fast",
        help="Configuration preset (default: fast)",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    train_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of training iterations (overrides preset)",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save training results (JSON)",
    )
    train_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without running",
    )
    train_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a strategy across multiple scenarios",
    )
    eval_parser.add_argument(
        "--env",
        type=str,
        default="resource_allocation",
        help="Environment name (default: resource_allocation)",
    )
    eval_parser.add_argument(
        "--preset",
        type=str,
        choices=["fast", "balanced", "thorough"],
        default="fast",
        help="Configuration preset (default: fast)",
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    eval_parser.add_argument(
        "--scenarios",
        type=str,
        required=True,
        help="Path to scenarios YAML file",
    )
    eval_parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=1,
        help="Episodes per scenario (default: 1)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results (JSON)",
    )
    eval_parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without running",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    # List environments command
    subparsers.add_parser(
        "list-envs",
        help="List available environments",
    )

    # Info command
    info_parser = subparsers.add_parser(
        "info",
        help="Show information about an environment",
    )
    info_parser.add_argument(
        "env_name",
        type=str,
        help="Environment name",
    )

    # Top-level flags
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate configuration without running",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def cmd_list_envs() -> None:
    """List all available environments."""
    envs = EnvironmentRegistry.get_info()
    print("Available Strategic Environments:")
    print("-" * 60)
    for name, desc in envs.items():
        print(f"  {name:30s} {desc}")
    print(f"\nTotal: {len(envs)} environments")


def cmd_info(env_name: str) -> None:
    """Show detailed information about an environment."""
    if not EnvironmentRegistry.is_registered(env_name):
        print(f"Error: Environment '{env_name}' not found.")
        print(f"Available: {', '.join(EnvironmentRegistry.list_environments())}")
        sys.exit(1)

    env = EnvironmentRegistry.create(env_name)
    print(f"Environment: {env_name}")
    print(f"  Description: {EnvironmentRegistry.get_description(env_name)}")
    print(f"  Objectives ({env.num_objectives}):")
    for name in env.objective_names:
        print(f"    - {name}")
    print(f"  Agents: {env.num_agents}")
    print("  State spec:")
    spec = env.state_spec
    if spec.has_spatial:
        print(f"    Spatial: {spec.spatial_shape}")
    print(f"    Vector dim: {spec.vector_dim}")
    print(f"  Action space: {env.action_spec.total_size} actions")


def cmd_train(args: argparse.Namespace) -> None:
    """Run iterative self-play training."""
    # Load config
    if args.config:
        config = StrategicConfig.from_yaml(args.config)
    else:
        config = StrategicConfig.from_preset(args.preset)

    if args.seed is not None:
        config.random_seed = args.seed

    # Validate
    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    if args.dry_run:
        print("Dry run - training configuration validated successfully")
        print(f"  Preset: {args.preset}")
        print(f"  Environment: {args.env}")
        print(f"  Iterations: {args.iterations or config.planning.num_planning_iterations}")
        print(f"  Strategies/iter: {config.planning.strategies_per_iteration}")
        return

    # Create environment
    if not EnvironmentRegistry.is_registered(args.env):
        print(f"Error: Environment '{args.env}' not found.")
        print(f"Available: {', '.join(EnvironmentRegistry.list_environments())}")
        sys.exit(1)

    env = EnvironmentRegistry.create(args.env)
    print(f"Training with environment: {args.env}")
    print(f"  Objectives: {', '.join(env.objective_names)}")

    planner = StrategicPlanner(config)
    result = planner.train(
        environment=env,
        num_iterations=args.iterations,
    )

    print("\n" + result.summary())

    if args.output:
        output_data: dict[str, Any] = {
            "summary": result.summary(),
            "num_strategies": result.num_strategies,
            "objective_names": result.objective_names,
            "elapsed_seconds": result.elapsed_seconds,
            "mode": "train",
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def _load_scenarios(scenarios_path: str) -> list[Scenario]:
    """Load scenarios from a YAML file.

    Args:
        scenarios_path: Path to YAML file containing scenario definitions.

    Returns:
        List of Scenario objects.
    """
    with open(scenarios_path) as f:
        data = yaml.safe_load(f)

    scenarios: list[Scenario] = []
    for entry in data.get("scenarios", []):
        scenarios.append(Scenario(
            name=str(entry["name"]),
            parameters=entry.get("parameters", {}),
            probability=float(entry.get("probability", 1.0)),
            description=str(entry.get("description", "")),
        ))
    return scenarios


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate a strategy across multiple scenarios."""
    # Load config
    if args.config:
        config = StrategicConfig.from_yaml(args.config)
    else:
        config = StrategicConfig.from_preset(args.preset)

    if args.seed is not None:
        config.random_seed = args.seed

    # Validate
    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    # Load scenarios
    scenarios = _load_scenarios(args.scenarios)
    if not scenarios:
        print(f"Error: No scenarios found in '{args.scenarios}'.")
        sys.exit(1)

    if args.dry_run:
        print("Dry run - evaluation configuration validated successfully")
        print(f"  Preset: {args.preset}")
        print(f"  Environment: {args.env}")
        print(f"  Scenarios: {len(scenarios)}")
        for s in scenarios:
            print(f"    - {s.name} (p={s.probability:.2f})")
        print(f"  Episodes/scenario: {args.episodes_per_scenario}")
        return

    # Create environment
    if not EnvironmentRegistry.is_registered(args.env):
        print(f"Error: Environment '{args.env}' not found.")
        print(f"Available: {', '.join(EnvironmentRegistry.list_environments())}")
        sys.exit(1)

    env = EnvironmentRegistry.create(args.env)
    print(f"Evaluating with environment: {args.env}")
    print(f"  Objectives: {', '.join(env.objective_names)}")
    print(f"  Scenarios: {len(scenarios)}")

    planner = StrategicPlanner(config)
    result = planner.evaluate(
        environment=env,
        scenarios=scenarios,
        num_episodes_per_scenario=args.episodes_per_scenario,
    )

    # Display results
    print(f"\nEvaluation Results:")
    print(f"  Robustness score: {result.robustness_score:.4f}")
    print(f"  Expected outcomes: {result.expected_outcomes.tolist()}")
    for scenario_name, outcomes in result.outcomes_by_scenario.items():
        print(f"  Scenario '{scenario_name}': {outcomes.tolist()}")

    if args.output:
        output_data: dict[str, Any] = {
            "robustness_score": result.robustness_score,
            "expected_outcomes": result.expected_outcomes.tolist(),
            "outcomes_by_scenario": {
                k: v.tolist() for k, v in result.outcomes_by_scenario.items()
            },
            "mode": "evaluate",
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def cmd_plan(args: argparse.Namespace) -> None:
    """Run strategic planning."""
    # Load config
    if args.config:
        config = StrategicConfig.from_yaml(args.config)
    else:
        config = StrategicConfig.from_preset(args.preset)

    if args.seed is not None:
        config.random_seed = args.seed

    # Validate
    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    if args.dry_run:
        print("Dry run - configuration validated successfully")
        print(f"  Preset: {args.preset}")
        print(f"  Environment: {args.env}")
        print(f"  Mode: {config.mode}")
        print(f"  Simulations: {config.planning.num_simulations}")
        print(f"  Iterations: {config.planning.num_planning_iterations}")
        return

    # Create environment
    if not EnvironmentRegistry.is_registered(args.env):
        print(f"Error: Environment '{args.env}' not found.")
        print(f"Available: {', '.join(EnvironmentRegistry.list_environments())}")
        sys.exit(1)

    env = EnvironmentRegistry.create(args.env)
    print(f"Planning with environment: {args.env}")
    print(f"  Objectives: {', '.join(env.objective_names)}")

    # Create and run planner
    planner = StrategicPlanner(config)
    result = planner.plan(
        environment=env,
        num_episodes=args.episodes,
    )

    # Display results
    print("\n" + result.summary())

    # Save if requested
    if args.output:
        output_data: dict[str, Any] = {
            "summary": result.summary(),
            "num_strategies": result.num_strategies,
            "objective_names": result.objective_names,
            "elapsed_seconds": result.elapsed_seconds,
            "pareto_frontier_size": (
                result.pareto_frontier.size if result.pareto_frontier else 0
            ),
        }
        if result.best_trace is not None:
            output_data["best_trace"] = result.best_trace.to_dict()
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


def main() -> None:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.command == "list-envs":
        cmd_list_envs()
    elif args.command == "info":
        cmd_info(args.env_name)
    elif args.command == "plan":
        cmd_plan(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif getattr(args, "dry_run", False):
        print("Strategic Planning SDK - dry run OK")
        print("Use 'python -m alpha_zero.strategic plan --help' for planning options")
        print("Use 'python -m alpha_zero.strategic list-envs' to see environments")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
