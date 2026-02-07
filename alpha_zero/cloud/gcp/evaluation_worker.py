"""
Evaluation Worker Entry Point for Vertex AI.

Runs a model arena: plays new model vs. old model to measure improvement.
Results are saved as JSON to GCS for the orchestrator to read.

Usage:
    python -m alpha_zero.cloud.gcp.evaluation_worker \
        --model_version_new=2 \
        --model_version_old=1 \
        --iteration=10 \
        --games=20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

import torch

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage

logger = logging.getLogger(__name__)

RESULTS_PREFIX = "evaluations"


def setup_logging() -> None:
    """Configure logging for evaluation worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="[Evaluation] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Evaluation Worker")
    parser.add_argument("--model_version_new", type=int, required=True, help="New model version")
    parser.add_argument("--model_version_old", type=int, required=True, help="Old model version")
    parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    parser.add_argument("--games", type=int, default=20, help="Number of games to play")
    parser.add_argument(
        "--mcts_simulations", type=int, default=25, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--mcts_type",
        type=str,
        default="civ6",
        choices=["civ6", "gumbel"],
        help="MCTS algorithm variant",
    )
    parser.add_argument(
        "--gumbel_num_actions",
        type=int,
        default=16,
        help="Number of sampled actions for Gumbel MCTS",
    )
    parser.add_argument("--board_size", type=int, default=8, help="Board size")
    parser.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--num_filters", type=int, default=64, help="Conv filters in network")
    parser.add_argument(
        "--max_turns", type=int, default=30, help="Max game turns before truncation"
    )
    parser.add_argument("--mode", type=str, default="evaluation", help="Worker mode")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")
    return parser.parse_args()


def load_model(
    config: GCPConfig,
    version: int,
    board_size: int,
    action_space_size: int,
    num_res_blocks: int,
    num_filters: int,
    dry_run: bool = False,
) -> torch.nn.Module:
    """Load model from GCS or create fresh for dry run."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    model = MicroCivNetwork(
        board_size=board_size,
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )

    if dry_run:
        logger.info("Dry run: using fresh model v%d (no GCS load)", version)
        return model

    storage = GCSModelStorage(config)
    state_dict, metadata = storage.load_model(version)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Loaded model v%d from GCS", version)
    return model


def run_arena(
    model_new: torch.nn.Module,
    model_old: torch.nn.Module,
    num_games: int,
    mcts_type: str,
    mcts_sims: int,
    gumbel_num_actions: int,
    board_size: int,
    max_turns: int,
    device: torch.device,
) -> dict[str, float]:
    """Play new model vs. old model in an arena.

    Alternates starting player each game for fairness.

    Returns:
        Dict with win_rate, draw_rate, loss_rate (from new model's perspective).
    """
    from alpha_zero.core.mcts_factory import create_mcts
    from alpha_zero.envs.microciv import MicroCivEnv

    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(num_games):
        start_time = time.time()

        # Alternate which player the new model controls
        new_is_player_0 = game_idx % 2 == 0

        env = MicroCivEnv(board_size=board_size, max_turns=max_turns)
        mcts_new = create_mcts(
            mcts_type=mcts_type,
            network=model_new,
            device=str(device),
            num_simulations=mcts_sims,
            num_sampled_actions=gumbel_num_actions,
        )
        mcts_old = create_mcts(
            mcts_type=mcts_type,
            network=model_old,
            device=str(device),
            num_simulations=mcts_sims,
            num_sampled_actions=gumbel_num_actions,
        )

        obs, _ = env.reset()
        done = False
        move_count = 0

        while not done:
            current_is_new = (env.current_player == 0 and new_is_player_0) or (
                env.current_player == 1 and not new_is_player_0
            )
            mcts = mcts_new if current_is_new else mcts_old

            # Greedy play for evaluation (temperature=0)
            action_probs, _ = mcts.search(env, add_noise=False)
            action = mcts.select_action(action_probs, temperature=0.0)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            move_count += 1

        # Determine result from new model's perspective
        if env.winner is None:
            draws += 1
            result = "draw"
        elif (env.winner == 0 and new_is_player_0) or (env.winner == 1 and not new_is_player_0):
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"

        elapsed = time.time() - start_time
        player_str = "P0" if new_is_player_0 else "P1"
        logger.info(
            "Game %d/%d: new_model=%s, result=%s, moves=%d (%.1fs)",
            game_idx + 1,
            num_games,
            player_str,
            result,
            move_count,
            elapsed,
        )

    total = max(num_games, 1)
    return {
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "total_games": num_games,
    }


def save_results(
    config: GCPConfig,
    results: dict,
    iteration: int,
    model_version_new: int,
    model_version_old: int,
    dry_run: bool = False,
) -> str | None:
    """Save evaluation results as JSON to GCS."""
    results_with_metadata = {
        **results,
        "iteration": iteration,
        "model_version_new": model_version_new,
        "model_version_old": model_version_old,
    }

    if dry_run:
        logger.info("Dry run: would save results: %s", json.dumps(results_with_metadata, indent=2))
        return None

    from alpha_zero.cloud.gcp.storage import GCSBackend

    backend = GCSBackend(config.project_id)
    blob_path = f"{RESULTS_PREFIX}/iter_{iteration:03d}/results.json"
    data = json.dumps(results_with_metadata, indent=2).encode("utf-8")
    uri = backend.upload_blob(config.bucket, blob_path, data)
    logger.info("Saved evaluation results to %s", uri)
    return uri


def load_results(
    config: GCPConfig,
    iteration: int,
) -> dict | None:
    """Load evaluation results from GCS."""
    from alpha_zero.cloud.gcp.storage import GCSBackend, GCSStorageError

    backend = GCSBackend(config.project_id)
    blob_path = f"{RESULTS_PREFIX}/iter_{iteration:03d}/results.json"
    try:
        data = backend.download_blob(config.bucket, blob_path)
        return json.loads(data.decode("utf-8"))
    except GCSStorageError:
        return None


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    logger.info(
        "Starting evaluation: v%d vs v%d, %d games, %s MCTS (%d sims)",
        args.model_version_new,
        args.model_version_old,
        args.games,
        args.mcts_type,
        args.mcts_simulations,
    )

    try:
        config = GCPConfig()
        if not args.dry_run:
            config.validate()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Load both models
        model_kwargs = {
            "board_size": args.board_size,
            "action_space_size": args.action_space_size,
            "num_res_blocks": args.num_res_blocks,
            "num_filters": args.num_filters,
            "dry_run": args.dry_run,
        }
        model_new = load_model(config, args.model_version_new, **model_kwargs)
        model_old = load_model(config, args.model_version_old, **model_kwargs)
        model_new = model_new.to(device)
        model_old = model_old.to(device)

        # Run arena
        start_time = time.time()
        results = run_arena(
            model_new=model_new,
            model_old=model_old,
            num_games=args.games,
            mcts_type=args.mcts_type,
            mcts_sims=args.mcts_simulations,
            gumbel_num_actions=args.gumbel_num_actions,
            board_size=args.board_size,
            max_turns=args.max_turns,
            device=device,
        )
        elapsed = time.time() - start_time

        logger.info(
            "Arena complete in %.1fs: win_rate=%.2f, draw_rate=%.2f, loss_rate=%.2f",
            elapsed,
            results["win_rate"],
            results["draw_rate"],
            results["loss_rate"],
        )

        # Save results
        save_results(
            config=config,
            results=results,
            iteration=args.iteration,
            model_version_new=args.model_version_new,
            model_version_old=args.model_version_old,
            dry_run=args.dry_run,
        )

        return 0

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted")
        return 130
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
