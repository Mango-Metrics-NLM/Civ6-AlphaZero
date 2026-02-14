"""
Standalone Tournament Worker for Civ6 AlphaZero.

Runs a round-robin tournament between model versions, computing ELO ratings
and head-to-head results. Can be run:
- As a standalone CLI: python -m alpha_zero.cloud.gcp.tournament_worker
- As a Vertex AI custom job via VertexTournamentJob

Usage:
    # Explicit model list
    python -m alpha_zero.cloud.gcp.tournament_worker \
        --model_versions=30,35,40,45,50 --tournament_games=10

    # Every 3rd model from GCS
    python -m alpha_zero.cloud.gcp.tournament_worker \
        --model_stride=3 --tournament_games=10

    # Subset range with stride
    python -m alpha_zero.cloud.gcp.tournament_worker \
        --model_stride=2 --model_range=30:60 --tournament_games=10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for tournament worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="[Tournament] %(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for tournament worker."""
    parser = argparse.ArgumentParser(description="Civ6 AlphaZero Tournament Worker")

    # Model selection (mutually prioritized: explicit > stride/range > all)
    parser.add_argument(
        "--model_versions",
        type=str,
        default=None,
        help="Comma-separated model versions (e.g., '30,35,40,45,50')",
    )
    parser.add_argument(
        "--model_stride",
        type=int,
        default=1,
        help="Use every Nth model (default: 1 = all models)",
    )
    parser.add_argument(
        "--model_range",
        type=str,
        default=None,
        help="Model version range as 'start:end' (e.g., '30:60')",
    )

    # Tournament parameters
    parser.add_argument(
        "--tournament_games",
        type=int,
        default=10,
        help="Games per matchup (default: 10)",
    )
    parser.add_argument(
        "--mcts_type",
        type=str,
        default="gumbel",
        choices=["civ6", "gumbel"],
        help="MCTS algorithm variant (default: gumbel)",
    )
    parser.add_argument(
        "--tournament_mcts_sims",
        type=int,
        default=25,
        help="MCTS simulations per move (default: 25)",
    )
    parser.add_argument(
        "--gumbel_num_actions",
        type=int,
        default=16,
        help="Gumbel sampled actions (default: 16)",
    )

    # Environment parameters
    parser.add_argument("--board_size", type=int, default=8, help="Board size (default: 8)")
    parser.add_argument(
        "--action_space_size", type=int, default=376, help="Action space size (default: 376)"
    )
    parser.add_argument("--max_turns", type=int, default=30, help="Max game turns (default: 30)")

    # Execution
    parser.add_argument(
        "--dry_run", action="store_true", help="Simulate tournament with dummy models"
    )

    return parser.parse_args()


def resolve_model_versions(
    args: argparse.Namespace,
    config: Any,
    dry_run: bool = False,
) -> list[int]:
    """Resolve model versions from CLI args.

    Supports three modes (in priority order):
    1. --model_versions: explicit comma-separated list
    2. --model_stride + optional --model_range: every Nth model in range
    3. Neither: use all available models from GCS

    Always ensures first and last model are included when subsampling.

    Args:
        args: Parsed CLI arguments.
        config: GCP configuration.
        dry_run: If True, generate synthetic version list.

    Returns:
        Sorted list of model version integers.
    """
    # Mode 1: Explicit list
    model_versions_str = getattr(args, "model_versions", None)
    if model_versions_str:
        versions = [int(v.strip()) for v in model_versions_str.split(",") if v.strip()]
        return sorted(set(versions))

    # Get all available versions
    if dry_run:
        all_versions = list(range(0, 10))
    else:
        from alpha_zero.cloud.gcp.storage import GCSModelStorage

        storage = GCSModelStorage(config)
        all_versions = sorted(storage.list_versions())

    if not all_versions:
        return []

    # Mode 2/3: Apply range filter
    model_range = getattr(args, "model_range", None)
    if model_range:
        parts = model_range.split(":")
        if len(parts) == 2:
            range_start = int(parts[0])
            range_end = int(parts[1])
            all_versions = [v for v in all_versions if range_start <= v <= range_end]

    if not all_versions:
        return []

    # Apply stride
    stride = getattr(args, "model_stride", 1) or 1
    if stride > 1:
        first, last = all_versions[0], all_versions[-1]
        sampled = [v for i, v in enumerate(all_versions) if i % stride == 0]
        if first not in sampled:
            sampled.insert(0, first)
        if last not in sampled:
            sampled.append(last)
        return sorted(set(sampled))

    return all_versions


def run_tournament(
    config: Any,
    model_versions: list[int],
    tournament_games: int = 10,
    mcts_type: str = "gumbel",
    mcts_sims: int = 25,
    gumbel_num_actions: int = 16,
    board_size: int = 8,
    max_turns: int = 30,
    device: torch.device | None = None,
    dry_run: bool = False,
    action_space_size: int = 376,
) -> dict[str, Any]:
    """Run round-robin tournament across model versions.

    Loads models from GCS, plays all pairings, computes ELO ratings.

    Args:
        config: GCP configuration.
        model_versions: List of model versions to include.
        tournament_games: Games per matchup.
        mcts_type: MCTS variant ("civ6" or "gumbel").
        mcts_sims: MCTS simulations per move.
        gumbel_num_actions: Gumbel sampled actions.
        board_size: Board size.
        max_turns: Max turns per game.
        device: PyTorch device (default: auto-detect).
        dry_run: If True, use dummy models.
        action_space_size: Action space size for dummy models.

    Returns:
        Dict with keys: model_versions, elo_ratings, matches,
        best_model, best_elo, metadata.
    """
    from alpha_zero.cloud.gcp.arena import run_arena

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if len(model_versions) < 2:
        logger.info("Tournament skipped: need >= 2 model versions (have %d)", len(model_versions))
        return {}

    num_matchups = len(model_versions) * (len(model_versions) - 1) // 2
    total_games = num_matchups * tournament_games

    logger.info("=" * 60)
    logger.info(
        "Tournament: %d models, %d matchups, %d total games",
        len(model_versions),
        num_matchups,
        total_games,
    )
    logger.info("Device: %s | MCTS: %s (%d sims) | Games/matchup: %d", device, mcts_type, mcts_sims, tournament_games)
    logger.info("=" * 60)

    tournament_start = time.time()

    # Load all models
    models: dict[str, torch.nn.Module] = {}
    for version in model_versions:
        player_id = f"v{version:03d}"
        if dry_run:
            from alpha_zero.core.transformer_network import (
                Civ6TransformerNetwork,
                TransformerConfig,
            )

            model = Civ6TransformerNetwork(
                config=TransformerConfig(
                    spatial_size=board_size,
                    action_space_size=action_space_size,
                )
            )
        else:
            from alpha_zero.cloud.gcp.evaluation_worker import load_model

            model = load_model(config, version)

        model = model.to(device)
        models[player_id] = model

    logger.info("Loaded %d models for tournament", len(models))

    # Round-robin: every pair plays tournament_games
    player_ids = sorted(models.keys())
    match_results: list[dict[str, Any]] = []
    elo_ratings: dict[str, float] = dict.fromkeys(player_ids, 1500.0)
    k_factor = 32.0
    completed_matchups = 0

    for i, player_a in enumerate(player_ids):
        for player_b in player_ids[i + 1:]:
            completed_matchups += 1
            logger.info(
                "Match %d/%d: %s vs %s (%d games)",
                completed_matchups,
                num_matchups,
                player_a,
                player_b,
                tournament_games,
            )

            result = run_arena(
                model_a=models[player_a],
                model_b=models[player_b],
                num_games=tournament_games,
                mcts_type=mcts_type,
                mcts_sims=mcts_sims,
                gumbel_num_actions=gumbel_num_actions,
                board_size=board_size,
                max_turns=max_turns,
                device=device,
            )

            logger.info(
                "Result: %s vs %s -> W:%d D:%d L:%d (win_rate=%.1f%%)",
                player_a,
                player_b,
                result.wins,
                result.draws,
                result.losses,
                result.win_rate * 100,
            )

            match_results.append(
                {
                    "player_a": player_a,
                    "player_b": player_b,
                    **result.to_dict(),
                }
            )

            # Update ELO ratings
            expected_a = 1.0 / (
                1.0 + 10.0 ** ((elo_ratings[player_b] - elo_ratings[player_a]) / 400.0)
            )
            score_a = (result.wins + 0.5 * result.draws) / max(result.total_games, 1)
            elo_ratings[player_a] += k_factor * (score_a - expected_a)
            elo_ratings[player_b] += k_factor * ((1.0 - score_a) - (1.0 - expected_a))

    tournament_duration = time.time() - tournament_start

    # Build results summary
    elo_ranking = sorted(elo_ratings.items(), key=lambda x: x[1], reverse=True)

    logger.info("=" * 60)
    logger.info("Tournament Results - ELO Rankings:")
    for rank, (player_id, elo) in enumerate(elo_ranking, 1):
        logger.info("  %d. %s: %.1f ELO", rank, player_id, elo)
    logger.info("Duration: %.1f min", tournament_duration / 60)
    logger.info("=" * 60)

    return {
        "model_versions": model_versions,
        "elo_ratings": dict(elo_ranking),
        "matches": match_results,
        "best_model": elo_ranking[0][0] if elo_ranking else None,
        "best_elo": elo_ranking[0][1] if elo_ranking else 0.0,
        "metadata": {
            "total_matchups": num_matchups,
            "total_games": total_games,
            "games_per_matchup": tournament_games,
            "mcts_type": mcts_type,
            "mcts_sims": mcts_sims,
            "board_size": board_size,
            "max_turns": max_turns,
            "duration_seconds": tournament_duration,
            "device": str(device),
        },
    }


def save_tournament_results(
    config: Any,
    results: dict[str, Any],
    dry_run: bool = False,
) -> str | None:
    """Save tournament results to GCS as JSON.

    Args:
        config: GCP configuration.
        results: Tournament results dict.
        dry_run: If True, log but don't save.

    Returns:
        GCS URI if saved, None if dry run.
    """
    if dry_run:
        logger.info("Dry run: skipping GCS save")
        return None

    try:
        from alpha_zero.cloud.gcp.storage import GCSBackend

        backend = GCSBackend(config.project_id)
        data = json.dumps(results, indent=2).encode("utf-8")
        uri = backend.upload_blob(config.bucket, "tournament/post_run_results.json", data)
        logger.info("Saved tournament results to %s", uri)
        return uri
    except Exception as e:
        logger.warning("Failed to save tournament results to GCS: %s", e)
        return None


def main() -> int:
    """Main entry point for tournament worker."""
    setup_logging()
    args = parse_args()

    logger.info("Civ6 AlphaZero Tournament Worker")
    logger.info("Dry run: %s", args.dry_run)

    # Load GCP config
    from alpha_zero.cloud.gcp.config import GCPConfig

    config = GCPConfig()

    # Resolve model versions
    model_versions = resolve_model_versions(args, config, dry_run=args.dry_run)
    if len(model_versions) < 2:
        logger.error("Need at least 2 model versions for tournament (found %d)", len(model_versions))
        return 1

    logger.info("Tournament models: %s", model_versions)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Run tournament
    results = run_tournament(
        config=config,
        model_versions=model_versions,
        tournament_games=args.tournament_games,
        mcts_type=args.mcts_type,
        mcts_sims=args.tournament_mcts_sims,
        gumbel_num_actions=args.gumbel_num_actions,
        board_size=args.board_size,
        max_turns=args.max_turns,
        device=device,
        dry_run=args.dry_run,
        action_space_size=args.action_space_size,
    )

    if not results:
        logger.error("Tournament produced no results")
        return 1

    # Save results
    save_tournament_results(config, results, dry_run=args.dry_run)

    # W&B logging (if enabled via env vars)
    try:
        import os

        if os.environ.get("WANDB_ENABLED", "").lower() == "true":
            from alpha_zero.utils.wandb_logger import WandbLogger

            wandb_logger = WandbLogger.from_env()
            if wandb_logger.is_enabled:
                wandb_logger.init(
                    config_dict={
                        "mode": "tournament",
                        "model_versions": model_versions,
                        "tournament_games": args.tournament_games,
                        "mcts_type": args.mcts_type,
                        "mcts_sims": args.tournament_mcts_sims,
                        "board_size": args.board_size,
                    },
                )
                elo_ratings = results.get("elo_ratings", {})
                for player_id, elo in elo_ratings.items():
                    wandb_logger.log({f"tournament/elo_{player_id}": elo})
                wandb_logger.log(
                    {
                        "tournament/best_model": results.get("best_model", ""),
                        "tournament/best_elo": results.get("best_elo", 0.0),
                        "tournament/num_models": len(model_versions),
                        "tournament/total_matches": len(results.get("matches", [])),
                    }
                )
    except Exception as e:
        logger.warning("W&B logging failed (non-fatal): %s", e)

    logger.info("Tournament worker completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
