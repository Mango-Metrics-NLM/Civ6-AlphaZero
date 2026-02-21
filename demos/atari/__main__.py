"""CLI entry point for the Atari AlphaZero demo.

Usage:
    python -m demos.atari --game Pong --dry-run
    python -m demos.atari --config demos/atari/configs/breakout.yaml
    python -m demos.atari --game Breakout --iterations 50 --hierarchical
"""

from __future__ import annotations

import argparse
import logging

from demos.atari.config import AtariConfig
from demos.atari.trainer import AtariTrainer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AlphaZero-style training on Atari games",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source of configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=None,
        help="Atari game name (e.g. Pong, Breakout, SpaceInvaders)",
    )

    # Training overrides
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--episodes-per-iter", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-simulations", type=int, default=None)
    parser.add_argument("--num-res-blocks", type=int, default=None)
    parser.add_argument("--num-filters", type=int, default=None)
    parser.add_argument("--discount-factor", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    # Hierarchical reasoning
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=False,
        help="Enable 3-layer hierarchical reasoning",
    )

    # Novel features (patent techniques)
    parser.add_argument(
        "--reflection",
        action="store_true",
        default=False,
        help="Enable LATS-style reflection memory",
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        default=False,
        help="Enable RAG-enhanced MCTS with built-in game strategies",
    )

    # Execution mode
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Quick smoke test (1 iteration, 1 episode)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and launch training."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Build config
    if args.config:
        config = AtariConfig.from_yaml(args.config)
    elif args.game:
        config = AtariConfig.for_game(args.game)
    else:
        config = AtariConfig()

    # Apply env-var overrides
    config.apply_env_overrides()

    # Apply CLI overrides
    if args.game and not args.config:
        config.environment.game_name = args.game
    if args.iterations is not None:
        config.training.num_iterations = args.iterations
    if args.episodes_per_iter is not None:
        config.training.episodes_per_iteration = args.episodes_per_iter
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.num_simulations is not None:
        config.mcts.num_simulations = args.num_simulations
    if args.num_res_blocks is not None:
        config.network.num_res_blocks = args.num_res_blocks
    if args.num_filters is not None:
        config.network.num_filters = args.num_filters
    if args.discount_factor is not None:
        config.training.discount_factor = args.discount_factor
    if args.device is not None:
        config.training.device = args.device
    if args.checkpoint_dir is not None:
        config.training.checkpoint_dir = args.checkpoint_dir
    if args.hierarchical:
        config.hierarchical.enabled = True

    # Novel features (reflection memory + RAG advisor)
    if args.reflection or args.rag:
        from demos.atari.novel_features import NovelFeaturesConfig

        if config.novel_features is None:
            config.novel_features = NovelFeaturesConfig()
        if args.reflection:
            config.novel_features.reflection_enabled = True
        if args.rag:
            config.novel_features.rag_enabled = True

    # Launch
    trainer = AtariTrainer(config)
    result = trainer.train(dry_run=args.dry_run)

    logging.getLogger(__name__).info(
        "Done — %d iterations, final model: %s",
        result["iterations"],
        result["final_model"],
    )


if __name__ == "__main__":
    main()
