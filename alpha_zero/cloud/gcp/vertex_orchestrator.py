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

    # Dry run
    parser.add_argument(
        "--dry_run", action="store_true", help="Validate without launching cloud jobs"
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
    return extra


def build_training_extra_args(args: argparse.Namespace) -> list[str]:
    """Build extra args list for training container."""
    return [
        f"--board_size={args.board_size}",
        f"--action_space_size={args.action_space_size}",
        f"--num_res_blocks={args.num_res_blocks}",
        f"--num_filters={args.num_filters}",
    ]


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





def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

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

        # Get starting model version
        model_version = get_initial_model_version(config, args.dry_run)

        # Initialize cost estimator
        from alpha_zero.cloud.gcp.cost import CostEstimator
        cost_estimator = CostEstimator(config)
        total_estimated_cost = 0.0

        total_start = time.time()
        successful_iterations = 0
        failed_iterations = 0

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
            sp_cost = cost_estimator.estimate_job_cost(
                machine_type=config.self_play.machine_type,
                accelerator_type=config.self_play.accelerator_type,
                accelerator_count=config.self_play.accelerator_count,
                duration_hours=sp_duration,
                is_spot=config.self_play.spot
            ) * args.workers
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

            logger.info("Phase 2: Training on trajectories from iterations %s", trajectory_iterations)
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
                is_spot=config.training.spot
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
                
                # Estimate calc cost (using training/eval machine type - assuming same as training for now or similar)
                # EvaluationJob uses CustomJob logic which defaults to n1-standard-4 + T4 if not specified in config override
                # Note: EvaluationJob logic in vertex_jobs.py uses default worker spec if not configures.
                # For estimation simplicity, we assume budget tier (n1-standard-4 + T4).
                eval_cost = cost_estimator.estimate_job_cost(
                    machine_type="n1-standard-4", 
                    accelerator_type="NVIDIA_TESLA_T4",
                    accelerator_count=1,
                    duration_hours=eval_duration,
                    is_spot=False
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
            
            iteration_duration = (time.time() - iteration_start) / 60.0 # minutes
            logger.info("Iteration %d complete in %.1f minutes. Model: v%d", iteration, iteration_duration, new_version)

            if new_version is not None:
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

        return 0 if failed_iterations == 0 else 1

    except KeyboardInterrupt:
        logger.warning("Orchestrator interrupted by user")
        return 130
    except Exception as e:
        logger.exception("Orchestrator failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
