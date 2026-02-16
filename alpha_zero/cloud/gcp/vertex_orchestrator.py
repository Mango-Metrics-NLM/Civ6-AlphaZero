"""
Vertex AI Orchestrator for Civ6 AlphaZero Training Loop.

Sequences the full AlphaZero pipeline on Vertex AI:
1. Launch N self-play workers to generate trajectories
2. Wait for all workers to complete
3. Launch training job on collected trajectories
4. Wait for training to complete
5. (Optional) Evaluate new model vs. old model in arena
6. Repeat for next iteration with updated model

Features:
- Configurable MCTS type (civ6/gumbel) via --mcts_type
- Optional curriculum learning via --curriculum
- Optional reward shaping via --reward_shaping
- Optional model evaluation via --evaluation_games
- W&B logging opt-in via WANDB_ENABLED env var

Usage:
    python -m alpha_zero.cloud.gcp.vertex_orchestrator \
        --iterations=10 \
        --workers=2 \
        --games_per_worker=50 \
        --mcts_type=gumbel \
        --evaluation_games=20 \
        --dry_run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.exceptions import GCSStorageError, PreemptionError, VertexJobError
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.cloud.gcp.vertex_jobs import (
    VertexAIClient,
    VertexEvaluationJob,
    VertexSelfPlayJob,
    VertexTrainingJob,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for orchestrator."""
    logging.basicConfig(
        level=logging.INFO,
        format="[Orchestrator] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Vertex AI Orchestrator")

    # Iteration control
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--start_iteration", type=int, default=0, help="Starting iteration number")
    parser.add_argument("--workers", type=int, default=2, help="Number of self-play workers")
    parser.add_argument(
        "--games_per_worker", type=int, default=50, help="Games per self-play worker"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument(
        "--trajectory_window",
        type=int,
        default=3,
        help="Number of recent iterations to use for training data",
    )
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries per failed job")

    # Network architecture
    parser.add_argument("--board_size", type=int, default=8, help="Board size")
    parser.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--num_filters", type=int, default=64, help="Conv filters in network")

    # MCTS configuration
    parser.add_argument(
        "--mcts_type",
        type=str,
        default="civ6",
        choices=["civ6", "gumbel"],
        help="MCTS algorithm variant",
    )
    parser.add_argument(
        "--mcts_simulations", type=int, default=400, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--gumbel_num_actions",
        type=int,
        default=16,
        help="Number of sampled actions for Gumbel MCTS",
    )
    parser.add_argument(
        "--max_turns", type=int, default=30, help="Max game turns before truncation"
    )

    # Evaluation (disabled by default)
    parser.add_argument(
        "--evaluation_games", type=int, default=0, help="Games for model evaluation (0=skip)"
    )
    parser.add_argument(
        "--evaluation_win_threshold",
        type=float,
        default=0.55,
        help="Win rate threshold to accept new model",
    )

    # Curriculum learning (disabled by default)
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument(
        "--curriculum_stages",
        type=int,
        default=3,
        choices=[3, 5],
        help="Number of curriculum stages",
    )

    # Reward shaping (disabled by default)
    parser.add_argument("--reward_shaping", action="store_true", help="Enable reward shaping")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Reward shaping scale")

    # Post-run tournament (disabled by default)
    parser.add_argument(
        "--post_run_tournament",
        action="store_true",
        help="Run round-robin tournament of all model versions after training",
    )
    parser.add_argument(
        "--tournament_games",
        type=int,
        default=10,
        help="Games per matchup in post-run tournament",
    )
    parser.add_argument(
        "--tournament_mcts_sims",
        type=int,
        default=25,
        help="MCTS simulations per move in tournament games",
    )
    parser.add_argument(
        "--tournament_cloud",
        action="store_true",
        help="Run tournament as a Vertex AI job instead of locally",
    )
    parser.add_argument(
        "--tournament_machine_type",
        type=str,
        default="n1-standard-8",
        help="Machine type for cloud tournament (default: n1-standard-8)",
    )
    parser.add_argument(
        "--tournament_accelerator_type",
        type=str,
        default="NVIDIA_TESLA_T4",
        help="GPU type for cloud tournament (default: NVIDIA_TESLA_T4)",
    )
    parser.add_argument(
        "--tournament_accelerator_count",
        type=int,
        default=1,
        help="Number of GPUs for cloud tournament (default: 1)",
    )
    parser.add_argument(
        "--tournament_model_stride",
        type=int,
        default=1,
        help="Use every Nth model in tournament (default: 1 = all models)",
    )
    parser.add_argument(
        "--tournament_model_range",
        type=str,
        default=None,
        help="Model version range for tournament as 'start:end' (e.g., '30:60')",
    )
    parser.add_argument(
        "--tournament_model_versions",
        type=str,
        default=None,
        help="Explicit comma-separated model versions for tournament (e.g., '79,88,95,110,129')",
    )
    parser.add_argument(
        "--tournament_only",
        action="store_true",
        help="Run tournament only (skip training loop). Use with --tournament_model_versions or auto-detect from GCS.",
    )

    # Learning rate schedule (passed through to training worker)
    parser.add_argument(
        "--lr_schedule_type",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "warmup_cosine", "exponential", "step"],
        help="LR schedule type for training worker",
    )
    parser.add_argument("--lr_initial", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_final", type=float, default=0.0001, help="Final learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="LR warmup steps")
    parser.add_argument("--lr_decay_steps", type=int, default=0, help="LR decay steps (0=auto)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "--value_lr_multiplier", type=float, default=1.0, help="LR multiplier for value head"
    )
    parser.add_argument(
        "--policy_lr_multiplier", type=float, default=1.0, help="LR multiplier for policy head"
    )

    # Stall detection
    parser.add_argument(
        "--stall_lookback",
        type=int,
        default=5,
        help="Iterations to check for improvement (stall detection)",
    )
    parser.add_argument(
        "--stall_threshold",
        type=float,
        default=0.02,
        help="Minimum win rate improvement to not be stalled",
    )

    parser.add_argument(
        "--dry_run", action="store_true", help="Validate without launching cloud jobs"
    )

    # Algorithm selection (MuZero support)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="alphazero",
        choices=["alphazero", "muzero"],
        help="Algorithm to use: alphazero (default) or muzero",
    )

    return parser.parse_args()


def get_initial_model_version(config: GCPConfig, dry_run: bool = False) -> int:
    """Get the latest model version from GCS, or 0 if none exists."""
    if dry_run:
        logger.info("Dry run: assuming model version 0")
        return 0

    try:
        storage = GCSModelStorage(config)
        version = storage.get_latest_version()
        logger.info("Found latest model version: %d", version)
        return version
    except GCSStorageError:
        logger.info("No existing models found, starting from version 0")
        return 0


def build_selfplay_extra_args(args: argparse.Namespace) -> list[str]:
    """Build extra args list for self-play container."""
    extra = [
        f"--mcts_type={args.mcts_type}",
        f"--mcts_simulations={args.mcts_simulations}",
        f"--gumbel_num_actions={args.gumbel_num_actions}",
        f"--max_turns={args.max_turns}",
        f"--board_size={args.board_size}",
        f"--action_space_size={args.action_space_size}",
        f"--num_res_blocks={args.num_res_blocks}",
        f"--num_filters={args.num_filters}",
    ]
    if args.reward_shaping:
        extra.append("--reward_shaping")
        extra.append(f"--reward_scale={args.reward_scale}")
    # MuZero: override mcts_type to muzero when algorithm=muzero
    algorithm = getattr(args, "algorithm", "alphazero")
    if algorithm == "muzero":
        # Replace mcts_type in extra args
        extra = [a for a in extra if not a.startswith("--mcts_type=")]
        extra.append("--mcts_type=muzero")
    return extra


def build_training_extra_args(args: argparse.Namespace) -> list[str]:
    """Build extra args list for training container."""
    extra = [
        f"--board_size={args.board_size}",
        f"--action_space_size={args.action_space_size}",
        f"--num_res_blocks={args.num_res_blocks}",
        f"--num_filters={args.num_filters}",
    ]
    # Pass through LR schedule args (only if non-default to keep backwards compat)
    if args.lr_schedule_type != "constant":
        extra.extend(
            [
                f"--lr_schedule_type={args.lr_schedule_type}",
                f"--lr_initial={args.lr_initial}",
                f"--lr_final={args.lr_final}",
                f"--lr_warmup_steps={args.lr_warmup_steps}",
                f"--lr_decay_steps={args.lr_decay_steps}",
            ]
        )
    if args.weight_decay != 0.01:
        extra.append(f"--weight_decay={args.weight_decay}")
    if getattr(args, "value_lr_multiplier", 1.0) != 1.0:
        extra.append(f"--value_lr_multiplier={args.value_lr_multiplier}")
    if getattr(args, "policy_lr_multiplier", 1.0) != 1.0:
        extra.append(f"--policy_lr_multiplier={args.policy_lr_multiplier}")
    # Pass algorithm flag for MuZero training
    algorithm = getattr(args, "algorithm", "alphazero")
    if algorithm == "muzero":
        extra.append("--algorithm=muzero")
    return extra


def build_evaluation_extra_args(args: argparse.Namespace) -> list[str]:
    """Build extra args list for evaluation container."""
    return [
        f"--mcts_type={args.mcts_type}",
        f"--mcts_simulations={args.mcts_simulations}",
        f"--gumbel_num_actions={args.gumbel_num_actions}",
        f"--max_turns={args.max_turns}",
        f"--board_size={args.board_size}",
        f"--action_space_size={args.action_space_size}",
        f"--num_res_blocks={args.num_res_blocks}",
        f"--num_filters={args.num_filters}",
    ]


def run_selfplay_phase(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    model_version: int,
    num_workers: int,
    games_per_worker: int,
    args: argparse.Namespace,
    max_retries: int = 2,
    dry_run: bool = False,
) -> bool:
    """Run the self-play phase of an iteration.

    Returns True if enough workers succeeded.
    """
    if dry_run:
        logger.info(
            "Dry run: would launch %d self-play workers (model v%d, %d games each, %s MCTS)",
            num_workers,
            model_version,
            games_per_worker,
            args.mcts_type,
        )
        return True

    # Override games_per_worker in config
    config.self_play.games_per_worker = games_per_worker

    extra_args = build_selfplay_extra_args(args)
    selfplay_manager = VertexSelfPlayJob(config, client)

    for attempt in range(max_retries + 1):
        try:
            job_names = selfplay_manager.launch(
                num_workers=num_workers,
                model_version=model_version,
                iteration=iteration,
                extra_args=extra_args,
            )
            logger.info("Launched %d self-play jobs (attempt %d)", len(job_names), attempt + 1)

            results = selfplay_manager.wait_for_completion(job_names)
            successful = sum(1 for r in results if r.success)

            if successful == 0:
                logger.error("All self-play workers failed")
                if attempt < max_retries:
                    logger.info("Retrying self-play phase...")
                    continue
                return False

            if successful < num_workers:
                logger.warning(
                    "Only %d/%d self-play workers succeeded (continuing anyway)",
                    successful,
                    num_workers,
                )

            return True

        except PreemptionError:
            logger.warning("Self-play preempted on attempt %d", attempt + 1)
            if attempt < max_retries:
                continue
            return False
        except VertexJobError as e:
            logger.error("Self-play failed: %s", e)
            if attempt < max_retries:
                time.sleep(30)
                continue
            return False

    return False


def run_training_phase(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    trajectory_iterations: list[int],
    args: argparse.Namespace,
    max_retries: int = 2,
    dry_run: bool = False,
) -> bool:
    """Run the training phase of an iteration.

    Returns True if training succeeded.
    """
    if dry_run:
        logger.info(
            "Dry run: would launch training job (iteration %d, trajectories from %s)",
            iteration,
            trajectory_iterations,
        )
        return True

    # Override epochs in config
    config.training.epochs_per_iteration = args.epochs

    extra_args = build_training_extra_args(args)
    training_manager = VertexTrainingJob(config, client)

    for attempt in range(max_retries + 1):
        try:
            job_name = training_manager.launch(
                iteration=iteration,
                trajectory_iterations=trajectory_iterations,
                extra_args=extra_args,
            )
            logger.info("Launched training job (attempt %d): %s", attempt + 1, job_name)

            result = training_manager.wait_for_completion(job_name)

            if result.success:
                logger.info("Training completed successfully")
                return True

            logger.error("Training failed: %s", result.error_message)
            if attempt < max_retries:
                logger.info("Retrying training phase...")
                time.sleep(30)
                continue
            return False

        except PreemptionError:
            logger.warning("Training preempted on attempt %d", attempt + 1)
            if attempt < max_retries:
                continue
            return False
        except VertexJobError as e:
            logger.error("Training job error: %s", e)
            if attempt < max_retries:
                time.sleep(30)
                continue
            return False

    return False


def run_evaluation_phase(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    model_version_new: int,
    model_version_old: int,
    args: argparse.Namespace,
    dry_run: bool = False,
) -> tuple[bool, float]:
    """Run model evaluation phase.

    Returns:
        Tuple of (success, win_rate). win_rate is 0.0 on failure.
    """
    if args.evaluation_games <= 0:
        return True, 0.0  # Skip evaluation

    if dry_run:
        logger.info(
            "Dry run: would evaluate v%d vs v%d (%d games)",
            model_version_new,
            model_version_old,
            args.evaluation_games,
        )
        return True, 0.55  # Simulated win rate

    extra_args = build_evaluation_extra_args(args)
    eval_manager = VertexEvaluationJob(config, client)

    try:
        job_name = eval_manager.launch(
            model_version_new=model_version_new,
            model_version_old=model_version_old,
            iteration=iteration,
            num_games=args.evaluation_games,
            extra_args=extra_args,
        )
        logger.info("Launched evaluation job: v%d vs v%d", model_version_new, model_version_old)

        result = eval_manager.wait_for_completion(job_name)

        if not result.success:
            logger.error("Evaluation job failed: %s", result.error_message)
            return False, 0.0

        # Load results from GCS
        from alpha_zero.cloud.gcp.evaluation_worker import load_results

        eval_results = load_results(config, iteration)
        if eval_results:
            win_rate = eval_results.get("win_rate", 0.0)
            logger.info(
                "Evaluation: v%d win_rate=%.2f%% vs v%d",
                model_version_new,
                win_rate * 100,
                model_version_old,
            )
            return True, win_rate

        logger.warning("Could not load evaluation results from GCS")
        return True, 0.0

    except (VertexJobError, PreemptionError) as e:
        logger.error("Evaluation failed: %s", e)
        return False, 0.0


def _apply_model_subsampling(
    model_versions: list[int],
    args: argparse.Namespace,
) -> list[int]:
    """Apply stride and range filters to model versions.

    Delegates to :func:`~alpha_zero.cloud.gcp.model_selection.subsample_model_versions`
    after extracting stride/range from CLI ``args`` (backwards-compatible via ``getattr``).

    Args:
        model_versions: Full list of model versions.
        args: Parsed CLI arguments (tournament_model_stride, tournament_model_range).

    Returns:
        Filtered and sorted list of model versions.
    """
    from alpha_zero.cloud.gcp.model_selection import subsample_model_versions

    stride = getattr(args, "tournament_model_stride", 1)
    version_range = getattr(args, "tournament_model_range", None)
    return subsample_model_versions(model_versions, stride=stride, version_range=version_range)


def _log_tournament_to_wandb(
    wandb_logger: object | None,
    tournament_results: dict,
) -> None:
    """Log tournament results to W&B.

    Args:
        wandb_logger: Optional WandbLogger instance.
        tournament_results: Dict with elo_ratings, matches, best_model, best_elo.
    """
    if wandb_logger is None or not getattr(wandb_logger, "is_enabled", False):
        return

    elo_ratings = tournament_results.get("elo_ratings", {})
    for player_id, elo in elo_ratings.items():
        wandb_logger.log({f"tournament/elo_{player_id}": elo})  # type: ignore[attr-defined]
    wandb_logger.log(
        {  # type: ignore[attr-defined]
            "tournament/best_model": tournament_results.get("best_model", ""),
            "tournament/best_elo": tournament_results.get("best_elo", 0.0),
            "tournament/num_models": len(tournament_results.get("model_versions", [])),
            "tournament/total_matches": len(tournament_results.get("matches", [])),
        }
    )


def _run_tournament_cloud(
    config: GCPConfig,
    model_versions: list[int],
    args: argparse.Namespace,
) -> dict:
    """Submit tournament as a Vertex AI job and wait for results.

    Args:
        config: GCP configuration.
        model_versions: Model versions to include.
        args: Parsed CLI arguments.

    Returns:
        Tournament results dict loaded from GCS, or empty dict on failure.
    """
    import json

    from alpha_zero.cloud.gcp.vertex_jobs import VertexAIClient, VertexTournamentJob

    client = VertexAIClient(project=config.project_id, region=config.region)
    tournament_job = VertexTournamentJob(config, client)

    job_name = tournament_job.launch(
        model_versions=model_versions,
        tournament_games=getattr(args, "tournament_games", 10),
        mcts_type=getattr(args, "mcts_type", "gumbel"),
        mcts_sims=getattr(args, "tournament_mcts_sims", 25),
        gumbel_num_actions=getattr(args, "gumbel_num_actions", 16),
        board_size=getattr(args, "board_size", 8),
        action_space_size=getattr(args, "action_space_size", config.network.action_space_size),
        max_turns=getattr(args, "max_turns", 30),
        machine_type=getattr(args, "tournament_machine_type", "n1-standard-8"),
        accelerator_type=getattr(args, "tournament_accelerator_type", "NVIDIA_TESLA_T4"),
        accelerator_count=getattr(args, "tournament_accelerator_count", 1),
    )

    result = tournament_job.wait_for_completion(job_name)

    if not result.success:
        logger.error("Cloud tournament job failed: %s", result.error_message)
        return {}

    # Load results from GCS
    try:
        from alpha_zero.cloud.gcp.storage import GCSBackend

        backend = GCSBackend(config.project_id)
        data = backend.download_blob(config.bucket, "tournament/post_run_results.json")
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        logger.warning("Failed to load cloud tournament results from GCS: %s", e)
        return {}


def run_post_run_evaluation(
    config: GCPConfig,
    model_versions: list[int],
    args: argparse.Namespace,
    wandb_logger: object | None = None,
    dry_run: bool = False,
) -> dict:
    """Run post-run tournament across model versions.

    Supports three execution modes:
    1. Local execution (default): runs tournament on the orchestrator machine
    2. Cloud execution (--tournament_cloud): submits a Vertex AI job
    3. Model subsampling (--tournament_model_stride, --tournament_model_range)

    Args:
        config: GCP configuration.
        model_versions: List of model versions to include in tournament.
        args: Parsed CLI arguments (tournament_games, mcts_type, etc.).
        wandb_logger: Optional WandbLogger for metrics logging.
        dry_run: If True, simulate tournament without loading real models.

    Returns:
        Dict with tournament results including ELO ratings and match results.
    """
    from alpha_zero.cloud.gcp.tournament_worker import run_tournament, save_tournament_results

    # Explicit model versions override subsampling
    explicit = getattr(args, "tournament_model_versions", None)
    if explicit:
        raw_versions = [int(v.strip()) for v in explicit.split(",") if v.strip()]
        versions = sorted(set(raw_versions))
        if len(versions) < len(raw_versions):
            logger.warning(
                "Duplicate entries in --tournament_model_versions removed; "
                "using %d unique versions: %s",
                len(versions),
                versions,
            )
        else:
            logger.info("Using explicit tournament model versions: %s", versions)
    else:
        # Apply subsampling
        versions = _apply_model_subsampling(model_versions, args)

    if len(versions) < 2:
        logger.info(
            "Post-run tournament skipped: need >= 2 model versions (have %d)", len(versions)
        )
        return {}

    if len(versions) < len(model_versions):
        logger.info(
            "Subsampled %d -> %d models (stride=%s, range=%s)",
            len(model_versions),
            len(versions),
            getattr(args, "tournament_model_stride", 1),
            getattr(args, "tournament_model_range", None),
        )

    # Cloud execution
    use_cloud = getattr(args, "tournament_cloud", False)
    if use_cloud and not dry_run:
        logger.info("Submitting tournament to Vertex AI (%d models)", len(versions))
        tournament_results = _run_tournament_cloud(config, versions, args)
        _log_tournament_to_wandb(wandb_logger, tournament_results)
        return tournament_results

    # Local execution — delegate to tournament_worker.run_tournament()
    tournament_results = run_tournament(
        config=config,
        model_versions=versions,
        tournament_games=getattr(args, "tournament_games", 10),
        mcts_type=getattr(args, "mcts_type", "gumbel"),
        mcts_sims=getattr(args, "tournament_mcts_sims", 25),
        gumbel_num_actions=getattr(args, "gumbel_num_actions", 16),
        board_size=getattr(args, "board_size", 8),
        max_turns=getattr(args, "max_turns", 30),
        dry_run=dry_run,
        action_space_size=getattr(args, "action_space_size", config.network.action_space_size),
    )

    _log_tournament_to_wandb(wandb_logger, tournament_results)
    save_tournament_results(config, tournament_results, dry_run=dry_run)

    return tournament_results


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    # Initialize W&B for orchestrator (opt-in via WANDB_ENABLED env var)
    wandb_logger = None
    try:
        from alpha_zero.utils.wandb_logger import WandbLogger

        wandb_logger = WandbLogger.from_env()
        if wandb_logger.is_enabled:
            wandb_logger.init(
                config_dict={
                    "iterations": args.iterations,
                    "workers": args.workers,
                    "games_per_worker": args.games_per_worker,
                    "mcts_type": args.mcts_type,
                    "mcts_simulations": args.mcts_simulations,
                    "board_size": args.board_size,
                    "evaluation_games": args.evaluation_games,
                    "epochs": args.epochs,
                    "mode": "orchestrator",
                }
            )
            logger.info("W&B logging enabled for orchestrator")
    except ImportError:
        logger.debug("W&B not available")

    logger.info("Civ6 AlphaZero Vertex AI Orchestrator")
    logger.info(
        "Config: %d iterations, %d workers, %d games/worker, %dx%d board, %s MCTS",
        args.iterations,
        args.workers,
        args.games_per_worker,
        args.board_size,
        args.board_size,
        args.mcts_type,
    )

    if args.dry_run:
        logger.info("DRY RUN MODE - no cloud jobs will be launched")

    # Set up curriculum manager if enabled
    curriculum_manager = None
    if args.curriculum:
        try:
            from alpha_zero.utils.curriculum_learning import create_curriculum_manager

            curriculum_manager = create_curriculum_manager(num_stages=args.curriculum_stages)
            logger.info("Curriculum learning enabled (%d stages)", args.curriculum_stages)
        except ImportError:
            logger.warning("Curriculum learning not available, continuing without it")

    try:
        config = GCPConfig()
        if not args.dry_run:
            config.validate()

        client = VertexAIClient(
            project=config.project_id,
            region=config.region,
            service_account=config.service_account,
        )

        # Tournament-only mode: skip training, just run tournament
        if getattr(args, "tournament_only", False):
            explicit = getattr(args, "tournament_model_versions", None)
            if explicit:
                raw_versions = [int(v.strip()) for v in explicit.split(",") if v.strip()]
                versions = sorted(set(raw_versions))
                if len(versions) < len(raw_versions):
                    logger.warning(
                        "Duplicate model versions in --tournament_model_versions; "
                        "using %d unique versions: %s",
                        len(versions),
                        versions,
                    )
            else:
                from alpha_zero.cloud.gcp.storage import GCSModelStorage

                storage = GCSModelStorage(config)
                versions = sorted(storage.list_versions())
                # Note: subsampling is applied inside run_post_run_evaluation()
                # to avoid double-subsampling when tournament_model_versions is unset.

            logger.info("Tournament-only mode: %d models", len(versions))
            results = run_post_run_evaluation(
                config=config,
                model_versions=versions,
                args=args,
                wandb_logger=wandb_logger,
                dry_run=args.dry_run,
            )
            if wandb_logger is not None and wandb_logger.is_enabled:
                wandb_logger.finish()
            return 0 if results else 1

        # Get starting model version (save for post-run tournament)
        model_version = get_initial_model_version(config, args.dry_run)
        initial_model_version = model_version

        # Initialize cost estimator
        from alpha_zero.cloud.gcp.cost import CostEstimator

        cost_estimator = CostEstimator(config)
        total_estimated_cost = 0.0

        total_start = time.time()
        successful_iterations = 0
        failed_iterations = 0
        win_rate_history: list[float] = []

        for i in range(args.iterations):
            iteration_start = time.time()
            iteration = args.start_iteration + i

            logger.info("=" * 60)
            logger.info("Starting iteration %d (model v%d)", iteration, model_version)
            logger.info("=" * 60)

            # Apply curriculum overrides if enabled
            if curriculum_manager is not None:
                env_params = curriculum_manager.get_environment_params()
                args.board_size = env_params.get("board_size", args.board_size)
                args.max_turns = env_params.get("max_turns", args.max_turns)
                logger.info(
                    "Curriculum stage=%s, board=%dx%d, max_turns=%d",
                    curriculum_manager.get_current_stage(),
                    args.board_size,
                    args.board_size,
                    args.max_turns,
                )

            # Phase 1: Self-play
            sp_start = time.time()
            logger.info(
                "Phase 1: Self-play (%d workers, %d games each, %s MCTS)",
                args.workers,
                args.games_per_worker,
                args.mcts_type,
            )
            selfplay_ok = run_selfplay_phase(
                config=config,
                client=client,
                iteration=iteration,
                model_version=model_version,
                num_workers=args.workers,
                games_per_worker=args.games_per_worker,
                args=args,
                max_retries=args.max_retries,
                dry_run=args.dry_run,
            )
            sp_duration = (time.time() - sp_start) / 3600.0

            # Estimate self-play cost
            sp_cost = (
                cost_estimator.estimate_job_cost(
                    machine_type=config.self_play.machine_type,
                    accelerator_type=config.self_play.accelerator_type,
                    accelerator_count=config.self_play.accelerator_count,
                    duration_hours=sp_duration,
                    is_spot=config.self_play.spot,
                )
                * args.workers
            )
            total_estimated_cost += sp_cost
            logger.info("Self-play phase cost (est): $%.2f", sp_cost)

            if not selfplay_ok:
                logger.error("Self-play phase failed for iteration %d", iteration)
                failed_iterations += 1
                continue

            # Phase 2: Training
            train_start = time.time()
            earliest = max(0, iteration - args.trajectory_window + 1)
            trajectory_iterations = list(range(earliest, iteration + 1))

            logger.info(
                "Phase 2: Training on trajectories from iterations %s", trajectory_iterations
            )
            training_ok = run_training_phase(
                config=config,
                client=client,
                iteration=iteration,
                trajectory_iterations=trajectory_iterations,
                args=args,
                max_retries=args.max_retries,
                dry_run=args.dry_run,
            )
            train_duration = (time.time() - train_start) / 3600.0

            # Estimate training cost
            train_cost = cost_estimator.estimate_job_cost(
                machine_type=config.training.machine_type,
                accelerator_type=config.training.accelerator_type,
                accelerator_count=config.training.accelerator_count,
                duration_hours=train_duration,
                is_spot=config.training.spot,
            )
            total_estimated_cost += train_cost
            logger.info("Training phase cost (est): $%.2f", train_cost)

            if not training_ok:
                logger.error("Training phase failed for iteration %d", iteration)
                failed_iterations += 1
                continue

            # Get new model version
            if args.dry_run:
                new_version = model_version + 1
            else:
                try:
                    storage = GCSModelStorage(config)
                    new_version = storage.get_latest_version()
                except Exception:
                    new_version = model_version + 1

            # Initialize metrics for logging
            win_rate = 0.0
            eval_cost = 0.0

            # Phase 3: Evaluation (optional)
            if args.evaluation_games > 0 and model_version > 0:
                eval_start = time.time()
                logger.info(
                    "Phase 3: Evaluating v%d vs v%d (%d games)",
                    new_version,
                    model_version,
                    args.evaluation_games,
                )
                eval_ok, win_rate = run_evaluation_phase(
                    config=config,
                    client=client,
                    iteration=iteration,
                    model_version_new=new_version,
                    model_version_old=model_version,
                    args=args,
                    dry_run=args.dry_run,
                )
                eval_duration = (time.time() - eval_start) / 3600.0

                # Estimate eval cost using the actual VertexEvaluationJob spec:
                # CPU-only worker on n1-standard-4 with no accelerators.
                eval_cost = cost_estimator.estimate_job_cost(
                    machine_type="n1-standard-4",
                    accelerator_type="",
                    accelerator_count=0,
                    duration_hours=eval_duration,
                    is_spot=False,
                )
                total_estimated_cost += eval_cost
                logger.info("Evaluation phase cost (est): $%.2f", eval_cost)

                if eval_ok and win_rate < args.evaluation_win_threshold:
                    logger.warning(
                        "New model rejected: win_rate=%.2f%% < threshold=%.2f%%",
                        win_rate * 100,
                        args.evaluation_win_threshold * 100,
                    )
                    new_version = model_version  # Keep old model

                # Record results for curriculum progression
                if curriculum_manager is not None and eval_ok:
                    curriculum_manager.record_game(
                        won=(win_rate >= args.evaluation_win_threshold),
                        metrics={"win_rate": win_rate},
                    )

            iteration_duration = (time.time() - iteration_start) / 60.0  # minutes

            # Stall detection (only when evaluation is enabled)
            stall_detected = False
            win_rate_trend = 0.0
            if args.evaluation_games > 0 and model_version > 0:
                win_rate_history.append(win_rate)

                try:
                    from alpha_zero.training.diagnostics import TrainingDiagnostics

                    _stall_diag = TrainingDiagnostics(stall_threshold=args.stall_threshold)
                    stall_detected = _stall_diag.detect_learning_stall(
                        win_rate_history,
                        lookback=args.stall_lookback,
                        threshold=args.stall_threshold,
                    )
                    win_rate_trend = _stall_diag.compute_win_rate_trend(
                        win_rate_history,
                        lookback=args.stall_lookback,
                    )

                    if stall_detected:
                        logger.warning(
                            "Learning stall detected at iteration %d: "
                            "win rate trend=%.4f over last %d iterations",
                            iteration,
                            win_rate_trend,
                            args.stall_lookback,
                        )
                        if wandb_logger is not None and wandb_logger.is_enabled:
                            wandb_logger.alert(
                                title="Learning Stall Detected",
                                text=(
                                    f"Win rate hasn't improved >{args.stall_threshold:.0%} "
                                    f"in {args.stall_lookback} iterations. "
                                    f"Current: {win_rate:.1%}, "
                                    f"Trend: {win_rate_trend:+.4f}/iter"
                                ),
                                level="WARN",
                            )
                except ImportError:
                    logger.debug("TrainingDiagnostics not available for stall detection")

            # Log iteration metrics to W&B
            if wandb_logger is not None and wandb_logger.is_enabled:
                iteration_metrics = {
                    "orchestrator/iteration": iteration,
                    "orchestrator/model_version": new_version,
                    "orchestrator/iteration_duration_min": iteration_duration,
                    "orchestrator/selfplay_cost": sp_cost,
                    "orchestrator/training_cost": train_cost,
                    "orchestrator/cumulative_cost": total_estimated_cost,
                    "orchestrator/stall_detected": int(stall_detected),
                    "orchestrator/win_rate_trend": win_rate_trend,
                }
                if args.evaluation_games > 0 and model_version > 0:
                    iteration_metrics["orchestrator/eval_win_rate"] = win_rate
                    iteration_metrics["orchestrator/eval_cost"] = eval_cost
                wandb_logger.log(iteration_metrics, step=iteration)

            logger.info(
                "Iteration %d complete in %.1f minutes. Model: v%d",
                iteration,
                iteration_duration,
                new_version,
            )

            model_version = new_version
            successful_iterations += 1

        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Training loop complete")
        logger.info(
            "Results: %d/%d iterations succeeded in %.1f minutes",
            successful_iterations,
            args.iterations,
            total_elapsed / 60,
        )
        logger.info("Final model version: v%d", model_version)
        logger.info("Total Estimated Cost: $%.2f", total_estimated_cost)
        logger.info("=" * 60)

        # Phase 4: Post-run tournament (optional)
        if args.post_run_tournament and successful_iterations > 0:
            # Collect all model versions: initial + all produced during training
            latest_version = get_initial_model_version(config, args.dry_run)
            all_versions = list(range(initial_model_version, latest_version + 1))
            if len(all_versions) >= 2:
                run_post_run_evaluation(
                    config=config,
                    model_versions=all_versions,
                    args=args,
                    wandb_logger=wandb_logger,
                    dry_run=args.dry_run,
                )

        # Log final summary to W&B
        if wandb_logger is not None and wandb_logger.is_enabled:
            wandb_logger.log(
                {
                    "orchestrator/total_iterations": args.iterations,
                    "orchestrator/successful_iterations": successful_iterations,
                    "orchestrator/failed_iterations": failed_iterations,
                    "orchestrator/total_cost": total_estimated_cost,
                    "orchestrator/total_duration_min": total_elapsed / 60,
                    "orchestrator/final_model_version": model_version,
                }
            )
            wandb_logger.finish()

        return 0 if failed_iterations == 0 else 1

    except KeyboardInterrupt:
        logger.warning("Orchestrator interrupted by user")
        if wandb_logger is not None and wandb_logger.is_enabled:
            wandb_logger.finish(exit_code=130)
        return 130
    except Exception as e:
        logger.exception("Orchestrator failed: %s", e)
        if wandb_logger is not None and wandb_logger.is_enabled:
            wandb_logger.finish(exit_code=1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
