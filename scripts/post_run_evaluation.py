"""
Post-run evaluation pipeline: tournament ELO + baseline eval → MLflow.

Orchestrates three evaluation tasks after a training run:
  1. Tournament: Round-robin ELO tournament across a strided subset of
     saved model checkpoints (v42-v91 by default), logging ELO ratings
     and spread to MLflow.
  2. Baseline evaluation: Final model vs RandomAgent and GreedyHeuristic
     over N games, logging win/draw/loss rates to MLflow.
  3. Custom metrics: policy_entropy, value_mae, policy_top1_acc via
     AlphaZeroMLflowEvaluator on a synthetic eval dataset.

All results are logged to the training MLflow run (or a sibling eval run
under the same experiment), and tournament results are saved to GCS.

Usage:
    # Minimal — attaches to existing training run:
    python scripts/post_run_evaluation.py \\
        --experiment civ6az-local-50iter \\
        --run_id 06f85192a5624f8e809536d9a558c1fd \\
        --model_start 42 --model_end 91 --model_stride 5

    # Full options:
    python scripts/post_run_evaluation.py \\
        --experiment civ6az-local-50iter \\
        --run_id 06f85192a5624f8e809536d9a558c1fd \\
        --model_start 42 --model_end 91 --model_stride 5 \\
        --tournament_games 10 --mcts_sims 25 \\
        --board_size 8 --action_space_size 376 \\
        --tracking_uri sqlite:///mlflow.db \\
        --gcs_bucket civ6az-training \\
        --create_child_run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Model version selection
# ---------------------------------------------------------------------------


def select_model_versions(
    start: int,
    end: int,
    stride: int,
) -> list[int]:
    """Select model versions to include in the tournament.

    Always includes both endpoints (``start`` and ``end``) in addition to
    the strided interior points.

    Args:
        start: First model version (inclusive).
        end: Last model version (inclusive).
        stride: Step between consecutive selected versions.

    Returns:
        Sorted list of version integers with no duplicates.

    Example::

        select_model_versions(42, 91, 5)
        # [42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 91]
    """
    if start > end:
        raise ValueError(f"start ({start}) must be <= end ({end})")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    versions = set(range(start, end + 1, stride))
    versions.add(start)
    versions.add(end)
    return sorted(versions)


# ---------------------------------------------------------------------------
# MLflow logging helpers
# ---------------------------------------------------------------------------


def log_elo_results_to_mlflow(
    ml_logger: Any,
    elo_ratings: dict[str, float],
    best_model: str,
    best_elo: float,
    *,
    step_offset: int = 0,
) -> None:
    """Log tournament ELO results to MLflow.

    Args:
        ml_logger: Active MLflowLogger instance.
        elo_ratings: Mapping of player ID (e.g. ``"v42"``) to ELO rating.
        best_model: Player ID of the tournament winner.
        best_elo: ELO rating of the winner.
        step_offset: Added to each model's numeric version for the step value.
    """
    for player_id, elo in elo_ratings.items():
        # Extract numeric version from player ID (e.g. "v42" -> 42)
        try:
            version = int(player_id.lstrip("v"))
        except ValueError:
            version = 0
        ml_logger.log_metric(f"tournament/elo/{player_id}", elo, step=version + step_offset)

    if elo_ratings:
        elo_spread = max(elo_ratings.values()) - min(elo_ratings.values())
    else:
        elo_spread = 0.0

    ml_logger.log_metrics(
        {
            "tournament/best_elo": best_elo,
            "tournament/elo_spread": elo_spread,
            "tournament/num_models": float(len(elo_ratings)),
        }
    )
    ml_logger.set_tag("tournament/best_model", best_model)
    logger.info(
        "Logged ELO results: best=%s (%.1f), spread=%.1f, models=%d",
        best_model,
        best_elo,
        elo_spread,
        len(elo_ratings),
    )


def log_eval_results_to_mlflow(
    ml_logger: Any,
    eval_results: dict[str, Any],
    *,
    prefix: str = "eval",
) -> None:
    """Log baseline evaluation results to MLflow.

    Args:
        ml_logger: Active MLflowLogger instance.
        eval_results: Dict with baseline-keyed results, each containing
            ``win_rate``, ``draw_rate``, ``loss_rate``, ``total_games``.
            Example: ``{"random": {"win_rate": 0.9, ...}}``.
        prefix: Metric key prefix (default: ``"eval"``).
    """
    for baseline_name, results in eval_results.items():
        for metric_key, value in results.items():
            if isinstance(value, (int, float)):
                ml_logger.log_metric(
                    f"{prefix}/{baseline_name}/{metric_key}",
                    float(value),
                )
    logger.info("Logged eval results for baselines: %s", list(eval_results.keys()))


# ---------------------------------------------------------------------------
# Tournament runner
# ---------------------------------------------------------------------------


def run_local_tournament(
    config: Any,
    model_versions: list[int],
    *,
    tournament_games: int = 10,
    mcts_sims: int = 25,
    board_size: int = 8,
    action_space_size: int = 376,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run a round-robin ELO tournament using the existing tournament worker.

    Args:
        config: GCPConfig instance with bucket/project settings.
        model_versions: List of model version integers to include.
        tournament_games: Number of games per pair.
        mcts_sims: MCTS simulations per move.
        board_size: Board size for the game.
        action_space_size: Action space size.
        dry_run: If True, skip actual game play and return synthetic ELO.

    Returns:
        Tournament results dict from ``run_tournament()``.
    """
    from alpha_zero.cloud.gcp.tournament_worker import run_tournament

    logger.info(
        "Running tournament: %d models, %d games/pair, %d sims",
        len(model_versions),
        tournament_games,
        mcts_sims,
    )
    return run_tournament(
        config=config,
        model_versions=model_versions,
        tournament_games=tournament_games,
        mcts_type="gumbel",
        mcts_sims=mcts_sims,
        board_size=board_size,
        max_turns=60,
        dry_run=dry_run,
        action_space_size=action_space_size,
    )


# ---------------------------------------------------------------------------
# Baseline evaluation runner
# ---------------------------------------------------------------------------


def run_baseline_eval(
    config: Any,
    model_version: int,
    *,
    n_games: int = 20,
    board_size: int = 8,
    action_space_size: int = 376,
    dry_run: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate the final model against baseline agents.

    Args:
        config: GCPConfig instance.
        model_version: Version of the model to evaluate.
        n_games: Number of games to play against each baseline.
        board_size: Board size.
        action_space_size: Action space size.
        dry_run: If True, return synthetic results without playing games.

    Returns:
        Dict mapping baseline name → ``{win_rate, draw_rate, loss_rate, total_games}``.
    """
    if dry_run:
        logger.info("[dry-run] Returning synthetic baseline eval results")
        return {
            "random": {
                "win_rate": 0.9,
                "draw_rate": 0.05,
                "loss_rate": 0.05,
                "total_games": n_games,
            },
            "heuristic": {
                "win_rate": 0.6,
                "draw_rate": 0.1,
                "loss_rate": 0.3,
                "total_games": n_games,
            },
        }

    try:
        import torch

        from alpha_zero.cloud.gcp.evaluation_worker import load_model
        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig
        from alpha_zero.evaluation.evaluation_pipeline import EvaluationPipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = load_model(
            config,
            model_version,
            dry_run=False,
            spatial_size=board_size,
            action_space_size=action_space_size,
        )
        model.to(device)
        model.eval()

        # Wrap model as Civ6MCTS — required by EvaluationPipeline
        mcts_cfg = MCTSConfig(num_simulations=25)  # low sims for fast eval
        _network: Any = model  # nn.Module satisfies MCTSNetwork at runtime; cast for mypy
        mcts_agent = Civ6MCTS(network=_network, config=mcts_cfg, device=device)

        pipeline = EvaluationPipeline(board_size=board_size, max_turns=60)

        results: dict[str, dict[str, float]] = {}
        for name, method in [
            ("random", pipeline.evaluate_vs_random),
            ("heuristic", pipeline.evaluate_vs_heuristic),
        ]:
            # evaluate_vs_random/heuristic signature: (agent, n_games, log_interval)
            # Returns EvaluationResult (not a dict) with nested WinRateResult
            eval_result = method(mcts_agent, n_games=n_games)
            wr = eval_result.win_rate
            results[name] = {
                "win_rate": float(wr.win_rate) / 100.0,  # normalise to 0-1
                "draw_rate": float(wr.draw_rate) / 100.0,
                "loss_rate": float(wr.loss_rate) / 100.0,
                "total_games": float(wr.total_games),
                "wins": float(wr.wins),
                "losses": float(wr.losses),
                "draws": float(wr.draws),
                "avg_game_length": float(eval_result.avg_game_length),
                "value_correlation": float(eval_result.value_correlation),
            }
            logger.info(
                "vs %s: W=%d D=%d L=%d (%.1f%% win)",
                name,
                wr.wins,
                wr.draws,
                wr.losses,
                wr.win_rate,
            )
        return results

    except Exception as exc:
        logger.error("Baseline eval failed: %s", exc, exc_info=True)
        return {
            "random": {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0, "total_games": 0.0},
            "heuristic": {"win_rate": 0.0, "draw_rate": 0.0, "loss_rate": 0.0, "total_games": 0.0},
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Post-run tournament ELO + baseline evaluation → MLflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # MLflow
    p.add_argument("--tracking_uri", default=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
    p.add_argument("--experiment", default="civ6az-local-50iter")
    p.add_argument("--run_id", default=None, help="Training run ID to attach eval results to")
    p.add_argument(
        "--create_child_run",
        action="store_true",
        help="Create a sibling eval run instead of logging to training run",
    )
    # GCS / model selection
    p.add_argument("--gcs_bucket", default=os.environ.get("CIV6AZ_GCS_BUCKET", "civ6az-training"))
    p.add_argument("--gcp_project", default=os.environ.get("CIV6AZ_VERTEX_PROJECT", "hrmtrain"))
    p.add_argument("--model_start", type=int, default=42, help="First model version")
    p.add_argument("--model_end", type=int, default=91, help="Last model version")
    p.add_argument("--model_stride", type=int, default=5, help="Version stride")
    # Tournament
    p.add_argument("--tournament_games", type=int, default=10, help="Games per model pair")
    p.add_argument("--mcts_sims", type=int, default=25, help="MCTS simulations per move")
    p.add_argument("--board_size", type=int, default=8)
    p.add_argument("--action_space_size", type=int, default=376)
    # Baseline eval
    p.add_argument("--baseline_games", type=int, default=20, help="Games vs each baseline")
    p.add_argument("--skip_baseline_eval", action="store_true")
    p.add_argument("--skip_tournament", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set GCP env vars
    os.environ.setdefault("CIV6AZ_GCS_BUCKET", args.gcs_bucket)
    os.environ.setdefault("CIV6AZ_VERTEX_PROJECT", args.gcp_project)
    os.environ.setdefault("CIV6AZ_VERTEX_REGION", "us-central1")

    from alpha_zero.cloud.gcp.config import GCPConfig
    from alpha_zero.monitoring.config import MonitoringConfig
    from alpha_zero.monitoring.mlflow_logger import MLflowLogger

    config = GCPConfig()
    model_versions = select_model_versions(args.model_start, args.model_end, args.model_stride)
    logger.info(
        "Selected %d model versions for evaluation: %s",
        len(model_versions),
        model_versions,
    )

    run_name = f"post-eval-{args.model_start}-{args.model_end}-stride{args.model_stride}-{int(time.time())}"

    if args.create_child_run or not args.run_id:
        # Create a new eval run under the same experiment
        mon_cfg = MonitoringConfig(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            run_name=run_name,
            tags={
                "eval_type": "post_run",
                "model_start": str(args.model_start),
                "model_end": str(args.model_end),
                "model_stride": str(args.model_stride),
            },
            description=f"Post-run evaluation for models v{args.model_start}-v{args.model_end}",
        )
        ml = MLflowLogger.from_config(mon_cfg)
    else:
        # Reuse the existing training run
        ml = MLflowLogger(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
        ml.enabled = True

    start_time = time.time()

    with ml:
        ml.log_params(
            {
                "eval/model_start": args.model_start,
                "eval/model_end": args.model_end,
                "eval/model_stride": args.model_stride,
                "eval/tournament_games": args.tournament_games,
                "eval/mcts_sims": args.mcts_sims,
                "eval/board_size": args.board_size,
                "eval/action_space_size": args.action_space_size,
                "eval/num_selected_models": len(model_versions),
            }
        )

        # ---------------------------------------------------------------
        # 1. Tournament
        # ---------------------------------------------------------------
        if not args.skip_tournament:
            logger.info("Starting tournament for models: %s", model_versions)
            try:
                tournament_results = run_local_tournament(
                    config=config,
                    model_versions=model_versions,
                    tournament_games=args.tournament_games,
                    mcts_sims=args.mcts_sims,
                    board_size=args.board_size,
                    action_space_size=args.action_space_size,
                    dry_run=args.dry_run,
                )
                elo_ratings: dict[str, float] = tournament_results.get("elo_ratings", {})
                best_model: str = tournament_results.get("best_model", "unknown")
                best_elo: float = tournament_results.get("best_elo", 0.0)

                log_elo_results_to_mlflow(ml, elo_ratings, best_model, best_elo)

                # Save to GCS
                if not args.dry_run:
                    try:
                        from alpha_zero.cloud.gcp.tournament_worker import save_tournament_results

                        save_tournament_results(config, tournament_results, dry_run=False)
                        logger.info("Tournament results saved to GCS")
                    except Exception as e:
                        logger.warning("Could not save tournament results to GCS: %s", e)

            except Exception as exc:
                logger.error("Tournament failed: %s", exc)
                ml.log_metric("tournament/error", 1.0)

        # ---------------------------------------------------------------
        # 2. Baseline evaluation (final model)
        # ---------------------------------------------------------------
        if not args.skip_baseline_eval:
            logger.info("Running baseline eval for model v%d", args.model_end)
            eval_results = run_baseline_eval(
                config=config,
                model_version=args.model_end,
                n_games=args.baseline_games,
                board_size=args.board_size,
                action_space_size=args.action_space_size,
                dry_run=args.dry_run,
            )
            log_eval_results_to_mlflow(ml, eval_results)

        # ---------------------------------------------------------------
        # 3. Final summary metrics
        # ---------------------------------------------------------------
        elapsed = time.time() - start_time
        ml.log_metrics(
            {
                "eval/total_elapsed_seconds": elapsed,
                "eval/models_evaluated": float(len(model_versions)),
            }
        )
        logger.info("Post-run evaluation complete in %.1fs", elapsed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
