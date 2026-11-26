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
- MLflow tracking opt-in via CIV6AZ_MLFLOW_ENABLED / MLFLOW_ENABLED (legacy WANDB_ENABLED honored)

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
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from alpha_zero.cloud.gcp.orchestrator_state import OrchestratorState

from alpha_zero.cloud.gcp.config import (
    BATCH_MCTS_TYPE,
    DEFAULT_MCTS_TYPE,
    MUZERO_MCTS_TYPE,
    VALID_MCTS_TYPES,
    GCPConfig,
    NetworkArchitectureConfig,
)
from alpha_zero.cloud.gcp.container_validator import validate_container_image
from alpha_zero.cloud.gcp.exceptions import GCSStorageError, PreemptionError, VertexJobError
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.cloud.gcp.vertex_jobs import (
    VertexAIClient,
    VertexEvaluationJob,
    VertexSelfPlayJob,
    VertexTrainingJob,
)
from alpha_zero.evaluation.promotion import (
    PromotionConfig,
    PromotionVerdict,
    log_promotion_decision,
    verdict_from_results,
)
from alpha_zero.utils.constants import DEFAULT_ACTION_SPACE_SIZE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry / backoff constants (overridable via CLI)
# ---------------------------------------------------------------------------
_INITIAL_RETRY_DELAY_S: Final[float] = 30.0
_MAX_RETRY_DELAY_S: Final[float] = 300.0
_BACKOFF_MULTIPLIER: Final[float] = 2.0
_JITTER_FRACTION: Final[float] = 0.25

# Machine type used by VertexEvaluationJob — kept in sync with VertexEvaluationJob.launch()
# to ensure cost estimates match actual spending.
_EVAL_MACHINE_TYPE: Final[str] = "n1-standard-4"


def _phase_run_name(iteration: int, phase: str) -> str:
    """Compose an MLflow run name for a per-phase worker run.

    Encodes the orchestrator's run group as the prefix so all per-phase
    runs sort together in the MLflow UI even when no explicit group/tag
    is set (the prefix carries the grouping information).

    Args:
        iteration: Current orchestrator iteration (0-based).
        phase: ``"selfplay"`` | ``"training"`` | ``"eval"``.

    Returns:
        e.g. ``"civ6az-selfplay-iter003"``.
    """
    base = os.environ.get("MLFLOW_RUN_GROUP", "civ6az")
    return f"{base}-{phase}-iter{iteration:03d}"


def compute_retry_delay(
    attempt: int,
    *,
    base: float = _INITIAL_RETRY_DELAY_S,
    maximum: float = _MAX_RETRY_DELAY_S,
    multiplier: float = _BACKOFF_MULTIPLIER,
    jitter: float = _JITTER_FRACTION,
) -> float:
    """Compute retry delay with exponential backoff and jitter.

    Formula: min(base * multiplier^attempt, maximum) * (1 + uniform(-jitter, jitter))

    Args:
        attempt: Zero-based retry attempt number.
        base: Initial delay in seconds.
        maximum: Cap on delay in seconds.
        multiplier: Exponential growth factor.
        jitter: Fraction of delay to add/subtract randomly (0 = deterministic).

    Returns:
        Delay in seconds (always >= 0).
    """
    raw = min(base * (multiplier**attempt), maximum)
    if jitter > 0:
        raw *= 1.0 + random.uniform(-jitter, jitter)  # noqa: S311
    return max(raw, 0.0)


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
    parser.add_argument(
        "--model_version_start",
        type=int,
        default=None,
        help="Override initial model version (use 0 for fresh random init, e.g. MuZero)",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Token forwarded to the training worker's --resume_from. "
            "'latest' loads the latest pointer; an integer loads that training "
            "step. When omitted, the orchestrator auto-detects and emits 'latest' "
            "if a checkpoint exists at the conventional GCS path."
        ),
    )
    parser.add_argument("--workers", type=int, default=2, help="Number of self-play workers")
    parser.add_argument(
        "--games_per_worker", type=int, default=100, help="Games per self-play worker"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per iteration")
    parser.add_argument(
        "--trajectory_window",
        type=int,
        default=5,
        help="Number of recent iterations to use for training data",
    )
    parser.add_argument("--max_retries", type=int, default=2, help="Max retries per failed job")

    # Network architecture
    parser.add_argument("--board_size", type=int, default=8, help="Board size")
    parser.add_argument(
        "--action_space_size",
        type=int,
        default=DEFAULT_ACTION_SPACE_SIZE,
        help="Action space size (defaults to the MicroCiv environment width)",
    )
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--num_filters", type=int, default=64, help="Conv filters in network")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument(
        "--policy_hidden_dim", type=int, default=128, help="Policy head hidden dimension"
    )
    parser.add_argument(
        "--value_head_dim", type=int, default=64, help="Value head hidden dimension"
    )
    parser.add_argument(
        "--architecture_preset",
        type=str,
        default=None,
        choices=["small", "medium", "large"],
        help="Architecture preset (overrides --num_res_blocks/--num_filters/--num_heads)",
    )

    # MCTS configuration
    parser.add_argument(
        "--mcts_type",
        type=str,
        default=DEFAULT_MCTS_TYPE,
        choices=VALID_MCTS_TYPES,
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
        "--max_turns", type=int, default=50, help="Max game turns before truncation"
    )

    # Vectorized self-play (opt-in, requires batch_gumbel MCTS)
    #
    # Default is 1 (sequential). It was 4, which combined with the default
    # --mcts_type=gumbel to fail the worker's batch dispatch gate: every
    # unflagged run advertised vectorized self-play and silently ran
    # sequential. 1 is what those runs actually did.
    parser.add_argument(
        "--num_parallel_games",
        type=int,
        default=1,
        help=(
            f"Number of parallel games per worker (1=sequential; >1 requires "
            f"--mcts_type={BATCH_MCTS_TYPE} and is otherwise ignored)"
        ),
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=64,
        help="Max batch size for vectorized inference",
    )

    # Evaluation (disabled by default)
    parser.add_argument(
        "--evaluation_games", type=int, default=20, help="Games for model evaluation (0=skip)"
    )
    parser.add_argument(
        "--evaluation_win_threshold",
        type=float,
        default=None,
        help=(
            "Promotion threshold for the gate's statistic (win rate under the "
            "default legacy policy, score rate under the draw-aware ones). "
            "When omitted, CIV6AZ_PROMOTION_THRESHOLD decides, then 0.55."
        ),
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
        "--learning_rate",
        type=float,
        default=0.001,
        help="Base learning rate (forwarded to training worker; applies to constant schedule)",
    )
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

    # Circuit breaker: halt after N consecutive failures to prevent cost burn
    parser.add_argument(
        "--max_consecutive_failures",
        type=int,
        default=3,
        help="Halt training after this many consecutive iteration failures (0=disabled)",
    )

    # Retry / backoff configuration
    parser.add_argument(
        "--initial_retry_delay",
        type=float,
        default=_INITIAL_RETRY_DELAY_S,
        help="Initial retry delay in seconds (exponential backoff base)",
    )
    parser.add_argument(
        "--max_retry_delay",
        type=float,
        default=_MAX_RETRY_DELAY_S,
        help="Maximum retry delay in seconds (backoff cap)",
    )

    parser.add_argument(
        "--dry_run", action="store_true", help="Validate without launching cloud jobs"
    )
    parser.add_argument(
        "--skip_image_validation",
        action="store_true",
        help="Skip container image pre-flight validation check",
    )

    # Algorithm selection (MuZero support)
    parser.add_argument(
        "--algorithm",
        type=str,
        default="alphazero",
        choices=["alphazero", "muzero"],
        help="Algorithm to use: alphazero (default) or muzero",
    )

    # Gameplay environment preset
    parser.add_argument(
        "--gameplay_preset",
        type=str,
        default="default",
        choices=["default", "enhanced"],
        help="Gameplay preset: 'default' (MicroCivEnv) or 'enhanced' (all gameplay systems)",
    )

    # Spatial encoder type
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="patch_embed",
        choices=["patch_embed", "hex_crystalline"],
        help="Spatial encoder type: 'patch_embed' (default Conv2d) or 'hex_crystalline' (hex-aware)",
    )

    # Agent ID for GCS path namespacing (A/B experiment isolation)
    parser.add_argument(
        "--agent_id",
        type=str,
        default=None,
        help="Agent ID for GCS path namespacing (isolates models/trajectories between runs)",
    )

    return parser.parse_args()


def _resolve_resume_token(
    args: argparse.Namespace,
    iteration: int,
    checkpoint_storage: Any | None = None,
) -> str | None:
    """Resolve the value to pass as ``--resume_from`` to the training worker.

    Returns:
        - The explicit ``args.resume_from`` value when set (operator override).
        - ``"latest"`` when a checkpoint exists at the conventional GCS path
          (``gs://{bucket}/checkpoints/civ6az/latest.pt``).
        - ``None`` when no checkpoint exists (cold start).

    The training worker (``training_worker.py``) parses the returned token and
    calls ``GCSCheckpointStorage.load_checkpoint(step=...)`` to resume optimizer
    state and the global step counter.  Without a token the worker starts from
    random init — which is what caused the v1.2.0 50-iter run to produce zero
    promotions.
    """
    explicit = getattr(args, "resume_from", None)
    if explicit:
        return str(explicit)

    if checkpoint_storage is None:
        from alpha_zero.cloud.gcp.storage import GCSCheckpointStorage

        # Honour CLI overrides (--gcs_bucket, --project, --region) when set;
        # fall back to env vars (CIV6AZ_*) otherwise. Only pass kwargs that
        # are explicitly set so GCPConfig defaults / env-var loading apply
        # for the rest.
        gcp_kwargs: dict[str, Any] = {}
        if getattr(args, "gcs_bucket", None):
            gcp_kwargs["bucket"] = args.gcs_bucket
        if getattr(args, "project", None):
            gcp_kwargs["project_id"] = args.project
        if getattr(args, "region", None):
            gcp_kwargs["region"] = args.region
        gcp_cfg = GCPConfig(**gcp_kwargs)
        checkpoint_storage = GCSCheckpointStorage(config=gcp_cfg)

    try:
        latest = checkpoint_storage.load_checkpoint(step=None)
    except (GCSStorageError, ImportError) as exc:
        # Expected failure modes (missing checkpoint, lazy-loaded backend
        # unavailable in tests) — cold-start is the right behaviour.
        logger.warning("resume_from resolution failed (%s); cold-starting", exc)
        latest = None
    except Exception:
        # Unexpected: misconfigured GCPConfig, torch.load deserialisation
        # crash, etc.  Silently cold-starting here reproduces the v1.2.0
        # zero-promotion bug — surface the traceback loudly so Cloud Logging
        # captures it and re-raise.
        logger.error(
            "Unexpected error resolving resume token; re-raising",
            exc_info=True,
        )
        raise

    if latest is None:
        if iteration > 0:
            logger.warning(
                "iteration=%d but no incumbent checkpoint at latest pointer - "
                "training will start from random init. Promotions are extremely "
                "unlikely.",
                iteration,
            )
        return None

    return "latest"


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
    # Architecture fields (new, backwards-compatible with getattr)
    num_heads = getattr(args, "num_heads", 8)
    if num_heads != 8:
        extra.append(f"--num_heads={num_heads}")
    policy_hidden_dim = getattr(args, "policy_hidden_dim", 128)
    if policy_hidden_dim != 128:
        extra.append(f"--policy_hidden_dim={policy_hidden_dim}")
    value_head_dim = getattr(args, "value_head_dim", 64)
    if value_head_dim != 64:
        extra.append(f"--value_head_dim={value_head_dim}")
    # Vectorized self-play args
    num_parallel = getattr(args, "num_parallel_games", 1)
    if num_parallel > 1:
        extra.append(f"--num_parallel_games={num_parallel}")
        extra.append(f"--max_batch_size={getattr(args, 'max_batch_size', 64)}")
    if args.reward_shaping:
        extra.append("--reward_shaping")
        extra.append(f"--reward_scale={args.reward_scale}")
    # MuZero: override mcts_type to muzero when algorithm=muzero
    algorithm = getattr(args, "algorithm", "alphazero")
    if algorithm == "muzero":
        # Replace mcts_type in extra args
        extra = [a for a in extra if not a.startswith("--mcts_type=")]
        extra.append("--mcts_type=muzero")
    # Gameplay preset (only propagate non-default to keep backward compat)
    gameplay_preset = getattr(args, "gameplay_preset", "default")
    if gameplay_preset != "default":
        extra.append(f"--gameplay_preset={gameplay_preset}")
    # Encoder type (only propagate non-default to keep backward compat)
    encoder_type = getattr(args, "encoder_type", "patch_embed")
    if encoder_type != "patch_embed":
        extra.append(f"--encoder_type={encoder_type}")
    return extra


def build_training_extra_args(
    args: argparse.Namespace, *, model_version: int | None = None
) -> list[str]:
    """Build extra args list for training container."""
    extra = [
        f"--board_size={args.board_size}",
        f"--action_space_size={args.action_space_size}",
        f"--num_res_blocks={args.num_res_blocks}",
        f"--num_filters={args.num_filters}",
    ]
    # Architecture fields (new, backwards-compatible with getattr)
    num_heads = getattr(args, "num_heads", 8)
    if num_heads != 8:
        extra.append(f"--num_heads={num_heads}")
    policy_hidden_dim = getattr(args, "policy_hidden_dim", 128)
    if policy_hidden_dim != 128:
        extra.append(f"--policy_hidden_dim={policy_hidden_dim}")
    value_head_dim = getattr(args, "value_head_dim", 64)
    if value_head_dim != 64:
        extra.append(f"--value_head_dim={value_head_dim}")
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
    # Encoder type (only propagate non-default to keep backward compat)
    encoder_type = getattr(args, "encoder_type", "patch_embed")
    if encoder_type != "patch_embed":
        extra.append(f"--encoder_type={encoder_type}")
    # Always forward learning rate (training worker default is 0.001)
    learning_rate = getattr(args, "learning_rate", 0.001)
    extra.append(f"--learning_rate={learning_rate}")
    # Forward --resume_from when set (training worker will resolve "latest"
    # via GCSCheckpointStorage and rehydrate optimizer state + global step).
    resume = getattr(args, "resume_from", None)
    if resume:
        extra.append(f"--resume_from={resume}")
    # NOTE: --model_version is intentionally NOT emitted. The training worker
    # no longer accepts it; checkpoint-resume happens via --resume_from=latest
    # (or an explicit token). The `model_version` kwarg is kept on the
    # signature for backwards-compat with callers that still pass it.
    _ = model_version  # silence unused; preserved for caller compat
    return extra


def build_evaluation_extra_args(args: argparse.Namespace) -> list[str]:
    """Build extra args list for evaluation container."""
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
    # Architecture fields (new, backwards-compatible with getattr)
    num_heads = getattr(args, "num_heads", 8)
    if num_heads != 8:
        extra.append(f"--num_heads={num_heads}")
    policy_hidden_dim = getattr(args, "policy_hidden_dim", 128)
    if policy_hidden_dim != 128:
        extra.append(f"--policy_hidden_dim={policy_hidden_dim}")
    value_head_dim = getattr(args, "value_head_dim", 64)
    if value_head_dim != 64:
        extra.append(f"--value_head_dim={value_head_dim}")
    # Gameplay preset (only propagate non-default to keep backward compat)
    gameplay_preset = getattr(args, "gameplay_preset", "default")
    if gameplay_preset != "default":
        extra.append(f"--gameplay_preset={gameplay_preset}")
    # Encoder type (only propagate non-default to keep backward compat)
    encoder_type = getattr(args, "encoder_type", "patch_embed")
    if encoder_type != "patch_embed":
        extra.append(f"--encoder_type={encoder_type}")
    return extra


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
            os.environ["MLFLOW_RUN_NAME"] = _phase_run_name(iteration, "selfplay")
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
                delay = compute_retry_delay(
                    attempt,
                    base=getattr(args, "initial_retry_delay", _INITIAL_RETRY_DELAY_S),
                    maximum=getattr(args, "max_retry_delay", _MAX_RETRY_DELAY_S),
                )
                logger.info(
                    "Retrying self-play in %.1fs (attempt %d/%d, backoff)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            return False
        finally:
            os.environ.pop("MLFLOW_RUN_NAME", None)

    return False


def run_training_phase(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    trajectory_iterations: list[int],
    args: argparse.Namespace,
    max_retries: int = 2,
    dry_run: bool = False,
    model_version: int | None = None,
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

    extra_args = build_training_extra_args(args, model_version=model_version)
    training_manager = VertexTrainingJob(config, client)

    for attempt in range(max_retries + 1):
        try:
            os.environ["MLFLOW_RUN_NAME"] = _phase_run_name(iteration, "training")
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
                delay = compute_retry_delay(
                    attempt,
                    base=getattr(args, "initial_retry_delay", _INITIAL_RETRY_DELAY_S),
                    maximum=getattr(args, "max_retry_delay", _MAX_RETRY_DELAY_S),
                )
                logger.info(
                    "Retrying training in %.1fs (attempt %d/%d, backoff)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
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
                delay = compute_retry_delay(
                    attempt,
                    base=getattr(args, "initial_retry_delay", _INITIAL_RETRY_DELAY_S),
                    maximum=getattr(args, "max_retry_delay", _MAX_RETRY_DELAY_S),
                )
                logger.info(
                    "Retrying training in %.1fs (attempt %d/%d, backoff)",
                    delay,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(delay)
                continue
            return False
        finally:
            os.environ.pop("MLFLOW_RUN_NAME", None)

    return False


def resolve_promotion(verdict: PromotionVerdict | None, win_rate: float, threshold: float) -> bool:
    """Decide promotion, preferring a policy verdict over the raw scalar.

    Both orchestration loops gate on this, so the production loop and the CLI
    loop cannot drift apart, and neither can the complementary comparison that
    feeds curriculum progression.

    Args:
        verdict: Policy verdict, or ``None`` when the evaluation artifact
            predates per-outcome counts (or the phase was skipped/dry-run).
        win_rate: Legacy scalar from the artifact.
        threshold: Configured win-rate threshold.

    Returns:
        Whether to promote. Falls back to the historical
        ``win_rate >= threshold`` comparison whenever no verdict is available,
        so a stale artifact behaves exactly as it always has.
    """
    if verdict is not None:
        return verdict.promoted
    return win_rate >= threshold


def resolve_evaluation_threshold(args: argparse.Namespace) -> float:
    """Resolve the promotion threshold with explicit precedence.

    An explicitly passed ``--evaluation_win_threshold`` wins; otherwise the
    env-aware :class:`PromotionConfig` default applies, so setting
    ``CIV6AZ_PROMOTION_THRESHOLD`` (the documented opt-in) is honoured
    whenever the CLI flag is absent. Despite the flag's historical name, the
    value gates whichever statistic the selected promotion policy uses.
    """
    cli_value = getattr(args, "evaluation_win_threshold", None)
    if cli_value is not None:
        return float(cli_value)
    return PromotionConfig().threshold


def resolve_effective_mcts_type(args: argparse.Namespace) -> str:
    """The MCTS variant the self-play worker will actually receive.

    ``--algorithm=muzero`` overrides ``--mcts_type`` when the worker arguments
    are built, so reading ``args.mcts_type`` directly gives the user's input
    rather than what runs -- and any advice derived from it can be wrong.

    Args:
        args: Parsed orchestrator arguments.

    Returns:
        The effective MCTS variant name.
    """
    if getattr(args, "algorithm", None) == MUZERO_MCTS_TYPE:
        return MUZERO_MCTS_TYPE
    return str(getattr(args, "mcts_type", DEFAULT_MCTS_TYPE))


def warn_on_ignored_parallelism(args: argparse.Namespace) -> bool:
    """Warn when vectorized self-play is requested but will not run.

    The self-play worker dispatches to the batch path only when
    ``num_parallel_games > 1`` **and** ``mcts_type`` is
    :data:`~alpha_zero.cloud.gcp.config.BATCH_MCTS_TYPE`. Any other
    combination silently runs sequential self-play while still forwarding
    ``--num_parallel_games``, so the run looks batch-configured and is not.

    This only warns; it deliberately does not rewrite ``args.mcts_type``.
    That namespace is shared with the evaluation, arena and tournament
    phases, so a rewrite here would change the search algorithm those phases
    use as a side effect of a self-play setting.

    Args:
        args: Parsed orchestrator arguments.

    Returns:
        ``True`` when a warning was emitted (the request will be ignored).
    """
    num_parallel = int(getattr(args, "num_parallel_games", 1) or 1)
    mcts_type = resolve_effective_mcts_type(args)

    if mcts_type == MUZERO_MCTS_TYPE:
        # MuZero's self-play path takes no num_parallel_games at all, so a
        # parallelism request is dropped just as completely here as anywhere
        # else. Previously exempted, which left the one case the fix was for.
        if num_parallel > 1:
            logger.warning(
                "num_parallel_games=%d ignored: MuZero self-play is sequential "
                "and has no vectorized path.",
                num_parallel,
            )
            return True
        return False

    if num_parallel > 1 and mcts_type != BATCH_MCTS_TYPE:
        logger.warning(
            "num_parallel_games=%d requested but mcts_type=%s: vectorized "
            "self-play runs only with --mcts_type=%s, so this run will use "
            "sequential self-play. Pass --mcts_type=%s to enable batching.",
            num_parallel,
            mcts_type,
            BATCH_MCTS_TYPE,
            BATCH_MCTS_TYPE,
        )
        return True

    if mcts_type == BATCH_MCTS_TYPE and num_parallel <= 1:
        # The inverse mistake, and the likelier one now that the default is 1:
        # batch MCTS driving a single environment pays the batching overhead
        # with none of the benefit.
        logger.warning(
            "mcts_type=%s with num_parallel_games=%d runs batched search over a "
            "single environment -- all of the overhead, none of the throughput. "
            "Pass --num_parallel_games>1 to actually batch.",
            BATCH_MCTS_TYPE,
            num_parallel,
        )
        return True

    return False


def run_evaluation_phase(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    model_version_new: int,
    model_version_old: int,
    args: argparse.Namespace,
    dry_run: bool = False,
    max_retries: int = 2,
) -> tuple[bool, float, PromotionVerdict | None]:
    """Run model evaluation phase.

    Args:
        config: GCP configuration.
        client: Vertex AI client.
        iteration: Current training iteration.
        model_version_new: New model version to evaluate.
        model_version_old: Baseline model version.
        args: Parsed CLI arguments.
        dry_run: If True, simulate without launching cloud jobs.
        max_retries: Maximum number of retry attempts on failure.

    Returns:
        Tuple of (success, win_rate, promotion_verdict).
        - success: whether evaluation completed (not dry-run/skipped)
        - win_rate: scalar win rate (candidate_wins / num_games), 0.0 on failure
        - promotion_verdict: PromotionVerdict or None if not computed
    """
    if args.evaluation_games <= 0:
        return True, 0.0, None  # Skip evaluation

    if dry_run:
        logger.info(
            "Dry run: would evaluate v%d vs v%d (%d games)",
            model_version_new,
            model_version_old,
            args.evaluation_games,
        )
        return True, 0.55, None  # Simulated win rate

    extra_args = build_evaluation_extra_args(args)
    eval_manager = VertexEvaluationJob(config, client)

    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                delay = compute_retry_delay(
                    attempt - 1,
                    base=getattr(args, "initial_retry_delay", _INITIAL_RETRY_DELAY_S),
                    maximum=getattr(args, "max_retry_delay", _MAX_RETRY_DELAY_S),
                )
                logger.info("Evaluation retry %d/%d in %.0fs...", attempt, max_retries, delay)
                time.sleep(delay)
            os.environ["MLFLOW_RUN_NAME"] = _phase_run_name(iteration, "eval")
            job_name = eval_manager.launch(
                model_version_new=model_version_new,
                model_version_old=model_version_old,
                iteration=iteration,
                num_games=args.evaluation_games,
                extra_args=extra_args,
            )
            logger.info(
                "Launched evaluation job: v%d vs v%d (attempt %d/%d)",
                model_version_new,
                model_version_old,
                attempt + 1,
                max_retries + 1,
            )

            result = eval_manager.wait_for_completion(job_name)

            if not result.success:
                logger.error(
                    "Evaluation job failed (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    result.error_message,
                )
                if attempt == max_retries:
                    return False, 0.0, None
                continue

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
                # Decide here, where the per-outcome counts still exist. By the
                # time the gate runs, only the scalar survives.
                # Thread the resolved threshold through the verdict so the verdict
                # and the scalar fallback gate on the same number. The threshold
                # bounds whichever statistic the selected policy uses -- win rate
                # under the legacy default, score rate under the draw-aware ones.
                eval_config = PromotionConfig(threshold=resolve_evaluation_threshold(args))
                verdict = verdict_from_results(eval_results, config=eval_config)
                if verdict is not None:
                    log_promotion_decision(
                        verdict,
                        target=logger,
                        context={
                            "iteration": iteration,
                            "model_version_new": model_version_new,
                            "model_version_old": model_version_old,
                        },
                    )
                return True, win_rate, verdict

            logger.warning("Could not load evaluation results from GCS")
            return True, 0.0, None

        except (VertexJobError, PreemptionError) as e:
            logger.error(
                "Evaluation failed (attempt %d/%d): %s",
                attempt + 1,
                max_retries + 1,
                e,
            )
            if attempt == max_retries:
                return False, 0.0, None
        finally:
            os.environ.pop("MLFLOW_RUN_NAME", None)

    return False, 0.0, None


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
    wandb_logger: Any | None,
    tournament_results: dict[str, Any],
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
        wandb_logger.log({f"tournament/elo_{player_id}": elo})
    wandb_logger.log(
        {
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
) -> dict[str, Any]:
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
        parsed = json.loads(data.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        logger.warning("Failed to load cloud tournament results from GCS: %s", e)
        return {}


def run_post_run_evaluation(
    config: GCPConfig,
    model_versions: list[int],
    args: argparse.Namespace,
    wandb_logger: object | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
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
        gameplay_preset=getattr(args, "gameplay_preset", "default"),
    )

    _log_tournament_to_wandb(wandb_logger, tournament_results)
    save_tournament_results(config, tournament_results, dry_run=dry_run)

    return tournament_results


def run_iterations(
    config: GCPConfig,
    client: VertexAIClient,
    args: argparse.Namespace,
    *,
    start_iteration: int = 0,
    end_iteration: int = 10,
    initial_model_version: int = 0,
    eval_win_threshold: float = 0.55,
    max_consecutive_failures: int = 3,
    wandb_logger: object | None = None,
    dry_run: bool = False,
    resume: bool = True,
) -> OrchestratorState:
    """Run a sequence of training iterations with state recovery.

    This is the core multi-iteration loop extracted from ``main()``.
    It persists state to GCS after each iteration so the run can be
    resumed after preemption.

    Args:
        config: GCP configuration.
        client: Vertex AI client.
        args: Parsed CLI arguments (must contain all orchestrator fields).
        start_iteration: First iteration number.
        end_iteration: Last iteration number (exclusive).
        initial_model_version: Starting model version.
        eval_win_threshold: Minimum win rate to accept a new model.
        max_consecutive_failures: Circuit breaker threshold.
        wandb_logger: Optional W&B logger instance.
        dry_run: If True, simulate without cloud jobs.
        resume: If True, attempt to load previous state from GCS.

    Returns:
        Final OrchestratorState after all iterations (or early stop).
    """
    from alpha_zero.cloud.gcp.cost import CostEstimator
    from alpha_zero.cloud.gcp.orchestrator_state import OrchestratorState

    agent_id = getattr(args, "agent_id", None)
    run_id = getattr(args, "run_id", "")

    # Attempt to resume from GCS state
    state: OrchestratorState | None = None
    if resume and not dry_run:
        state = OrchestratorState.load(config, agent_id=agent_id, run_id=run_id)

    if state is None:
        state = OrchestratorState(
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            current_iteration=start_iteration,
            current_model_version=initial_model_version,
            initial_model_version=initial_model_version,
            agent_id=agent_id,
            run_id=run_id,
        )

    cost_estimator = CostEstimator(config)
    max_consec = max_consecutive_failures

    for iteration in state.remaining_iterations():
        iteration_start = time.time()
        model_version = state.current_model_version

        logger.info("=" * 60)
        logger.info("Starting iteration %d (model v%d)", iteration, model_version)
        logger.info("=" * 60)

        # Phase 1: Self-play
        sp_start = time.time()
        selfplay_ok = run_selfplay_phase(
            config=config,
            client=client,
            iteration=iteration,
            model_version=model_version,
            num_workers=args.workers,
            games_per_worker=args.games_per_worker,
            args=args,
            max_retries=args.max_retries,
            dry_run=dry_run,
        )
        sp_duration = (time.time() - sp_start) / 3600.0
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

        if not selfplay_ok:
            logger.error("Self-play failed for iteration %d", iteration)
            state.record_failure(iteration, cost_usd=sp_cost)
            if max_consec > 0 and state.consecutive_failures >= max_consec:
                logger.critical(
                    "CIRCUIT BREAKER: %d consecutive failures (threshold=%d).",
                    state.consecutive_failures,
                    max_consec,
                )
                break
            continue

        # Phase 2: Training
        train_start = time.time()
        earliest = max(0, iteration - args.trajectory_window + 1)
        trajectory_iterations = list(range(earliest, iteration + 1))
        training_ok = run_training_phase(
            config=config,
            client=client,
            iteration=iteration,
            trajectory_iterations=trajectory_iterations,
            args=args,
            max_retries=args.max_retries,
            dry_run=dry_run,
        )
        train_duration = (time.time() - train_start) / 3600.0
        train_cost = cost_estimator.estimate_job_cost(
            machine_type=config.training.machine_type,
            accelerator_type=config.training.accelerator_type,
            accelerator_count=config.training.accelerator_count,
            duration_hours=train_duration,
            is_spot=config.training.spot,
        )

        if not training_ok:
            logger.error("Training failed for iteration %d", iteration)
            state.record_failure(iteration, cost_usd=sp_cost + train_cost)
            if max_consec > 0 and state.consecutive_failures >= max_consec:
                logger.critical(
                    "CIRCUIT BREAKER: %d consecutive failures (threshold=%d).",
                    state.consecutive_failures,
                    max_consec,
                )
                break
            continue

        # Get new model version
        if dry_run:
            new_version = model_version + 1
        else:
            try:
                storage = GCSModelStorage(config)
                new_version = storage.get_latest_version()
            except Exception:
                new_version = model_version + 1

        # Phase 3: Evaluation (optional)
        win_rate = 0.0
        eval_cost = 0.0
        rejected = False
        # None means "no policy verdict available" -- skipped phase, dry run, or
        # an artifact without per-outcome counts. resolve_promotion() falls back
        # to the legacy scalar comparison in that case.
        eval_verdict: PromotionVerdict | None = None

        if args.evaluation_games > 0 and model_version > 0:
            eval_start = time.time()
            eval_ok, win_rate, eval_verdict = run_evaluation_phase(
                config=config,
                client=client,
                iteration=iteration,
                model_version_new=new_version,
                model_version_old=model_version,
                args=args,
                dry_run=dry_run,
                max_retries=args.max_retries,
            )
            eval_duration = (time.time() - eval_start) / 3600.0
            eval_cost = cost_estimator.estimate_job_cost(
                machine_type=_EVAL_MACHINE_TYPE,
                accelerator_type="",
                accelerator_count=0,
                duration_hours=eval_duration,
                is_spot=False,
            )

            if not eval_ok:
                state.record_failure(
                    iteration,
                    cost_usd=sp_cost + train_cost + eval_cost,
                )
                if max_consec > 0 and state.consecutive_failures >= max_consec:
                    break
                continue

            # Evaluation gate
            threshold = eval_win_threshold
            if not resolve_promotion(eval_verdict, win_rate, threshold):
                if eval_verdict is not None:
                    logger.warning(
                        "Model rejected by %s policy: %s (win_rate=%.2f%%)",
                        eval_verdict.policy or "unknown",
                        eval_verdict.decision.name,
                        win_rate * 100,
                    )
                else:
                    logger.warning(
                        "Model rejected: win_rate=%.2f%% < threshold=%.2f%%",
                        win_rate * 100,
                        threshold * 100,
                    )
                rejected = True

        total_cost = sp_cost + train_cost + eval_cost
        duration = time.time() - iteration_start

        state.record_success(
            iteration=iteration,
            model_version=new_version,
            win_rate=win_rate,
            cost_usd=total_cost,
            duration_seconds=duration,
            rejected=rejected,
        )

        # Log to W&B
        if wandb_logger is not None and getattr(wandb_logger, "is_enabled", False):
            wandb_logger.log(  # type: ignore[attr-defined]
                {
                    "orchestrator/iteration": iteration,
                    "orchestrator/model_version": state.current_model_version,
                    "orchestrator/iteration_duration_min": duration / 60,
                    "orchestrator/cumulative_cost": state.total_cost_usd,
                    "orchestrator/eval_win_rate": win_rate,
                },
                step=iteration,
            )

        logger.info(
            "Iteration %d complete in %.1f min. Model: v%d (cost: $%.2f)",
            iteration,
            duration / 60,
            state.current_model_version,
            total_cost,
        )

        # Persist state for preemption recovery
        if not dry_run:
            state.save(config)

    # Final summary
    summary = state.to_summary()
    logger.info("=" * 60)
    logger.info("Training run complete: %s", summary)
    logger.info("=" * 60)

    return state


@dataclass
class _OrchestratorLoopState:
    """Mutable loop counters carried across orchestrator iterations."""

    model_version: int
    initial_model_version: int
    total_estimated_cost: float = 0.0
    successful_iterations: int = 0
    failed_iterations: int = 0
    consecutive_failures: int = 0
    win_rate_history: list[float] = field(default_factory=list)


def _apply_architecture_preset_to_args(args: argparse.Namespace) -> None:
    """Sync CLI args with the named architecture preset when specified."""
    if not getattr(args, "architecture_preset", None):
        return
    preset_config = NetworkArchitectureConfig.from_preset(args.architecture_preset)
    args.num_res_blocks = preset_config.num_layers
    args.num_filters = preset_config.embed_dim
    args.num_heads = preset_config.num_heads
    args.policy_hidden_dim = preset_config.policy_hidden_dim
    args.value_head_dim = preset_config.value_head_dim


def _init_orchestrator_tracker(args: argparse.Namespace) -> Any:
    """Initialize opt-in MLflow/WandbCompatAdapter tracking and propagate group tags."""
    try:
        from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger
    except ImportError:
        logger.debug("Experiment-tracking backend not available")
        return None

    wandb_logger = WandbLogger.from_env()
    if not wandb_logger.is_enabled:
        return wandb_logger

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
            "max_turns": args.max_turns,
            "trajectory_window": args.trajectory_window,
            "architecture_preset": getattr(args, "architecture_preset", None),
            "embed_dim": args.num_filters,
            "num_layers": args.num_res_blocks,
            "num_heads": getattr(args, "num_heads", 8),
            "policy_hidden_dim": getattr(args, "policy_hidden_dim", 128),
            "value_head_dim": getattr(args, "value_head_dim", 64),
            "num_parallel_games": getattr(args, "num_parallel_games", 1),
            "gameplay_preset": getattr(args, "gameplay_preset", "default"),
            "encoder_type": getattr(args, "encoder_type", "patch_embed"),
            "agent_id": getattr(args, "agent_id", None),
            "mode": "orchestrator",
        }
    )
    try:
        group_name = getattr(wandb_logger, "run_name", None)
        if group_name:
            os.environ["MLFLOW_RUN_GROUP"] = group_name
            existing_tags = os.environ.get("MLFLOW_TAGS", "").strip()
            group_tag = f"group={group_name}"
            os.environ["MLFLOW_TAGS"] = (
                f"{existing_tags},{group_tag}" if existing_tags else group_tag
            )
            logger.info("MLflow run group set to: %s", group_name)
    except Exception as exc:
        logger.debug("Could not set MLFLOW_RUN_GROUP: %s", exc)
    logger.info("Experiment tracking enabled for orchestrator (MLflow)")
    return wandb_logger


def _init_curriculum_manager(args: argparse.Namespace) -> Any:
    """Create the curriculum manager if ``--curriculum`` is enabled."""
    if not args.curriculum:
        return None
    try:
        from alpha_zero.utils.curriculum_learning import create_curriculum_manager

        manager = create_curriculum_manager(num_stages=args.curriculum_stages)
        logger.info("Curriculum learning enabled (%d stages)", args.curriculum_stages)
        return manager
    except ImportError:
        logger.warning("Curriculum learning not available, continuing without it")
        return None


def _build_and_validate_config(args: argparse.Namespace) -> GCPConfig | None:
    """Build ``GCPConfig`` from CLI args and run config + container image preflight."""
    config = GCPConfig()
    agent_id = getattr(args, "agent_id", None)
    if agent_id:
        config.paths.agent_id = agent_id
        logger.info("GCS path namespace: agent_id=%s", agent_id)

    encoder_type = getattr(args, "encoder_type", "patch_embed")
    if encoder_type != "patch_embed":
        config.network.encoder_type = encoder_type
        logger.info("Spatial encoder type: %s", encoder_type)

    if getattr(args, "architecture_preset", None):
        config.network = NetworkArchitectureConfig.from_preset(args.architecture_preset)
        logger.info(
            "Architecture preset '%s': embed=%d, layers=%d, heads=%d",
            args.architecture_preset,
            config.network.embed_dim,
            config.network.num_layers,
            config.network.num_heads,
        )

    if args.dry_run:
        return config

    config.validate()
    if getattr(args, "skip_image_validation", False):
        return config

    image_uri = config.container_image_uri
    if not image_uri:
        logger.error("Container image URI is not set in config.")
        return None
    logger.info("Validating container image: %s", image_uri)
    val_result = validate_container_image(image_uri)
    if not val_result.exists:
        logger.error(
            "Container image not found: %s (error: %s). "
            "Build and push the image first, or use --skip_image_validation.",
            image_uri,
            val_result.error,
        )
        return None
    if val_result.digest:
        logger.info("Container digest: %s", val_result.digest)
    return config


def _finish_tracker(wandb_logger: Any, exit_code: int | None = None) -> None:
    """Finalize the experiment tracker when active."""
    if wandb_logger is None or not wandb_logger.is_enabled:
        return
    if exit_code is None:
        wandb_logger.finish()
    else:
        wandb_logger.finish(exit_code=exit_code)


def _run_tournament_only_mode(
    config: GCPConfig, args: argparse.Namespace, wandb_logger: Any
) -> int:
    """Execute standalone post-run tournament evaluation and exit cleanly."""
    explicit = getattr(args, "tournament_model_versions", None)
    if explicit:
        versions: list[int] = []
    else:
        storage = GCSModelStorage(config)
        versions = sorted(storage.list_versions())

    logger.info(
        "Tournament-only mode%s",
        f": {explicit}" if explicit else f": {len(versions)} models from GCS",
    )
    results = run_post_run_evaluation(
        config=config,
        model_versions=versions,
        args=args,
        wandb_logger=wandb_logger,
        dry_run=args.dry_run,
    )
    _finish_tracker(wandb_logger)
    return 0 if results else 1


def _check_circuit_breaker(
    consecutive_failures: int,
    max_consec: int,
    iteration: int,
    wandb_logger: Any,
    *,
    phase_label: str = "",
) -> bool:
    """Return True and emit alerts if the consecutive-failure threshold is reached."""
    if max_consec <= 0 or consecutive_failures < max_consec:
        return False
    suffix = f" ({phase_label})" if phase_label else ""
    logger.critical(
        "CIRCUIT BREAKER: %d consecutive failures%s reached "
        "(threshold=%d). Halting to prevent cost burn.",
        consecutive_failures,
        suffix,
        max_consec,
    )
    if wandb_logger is not None and wandb_logger.is_enabled:
        wandb_logger.alert(
            title="Circuit Breaker Triggered",
            text=(
                f"{consecutive_failures} consecutive failure(s){suffix}. "
                f"Training halted at iteration {iteration}."
            ),
            level="ERROR",
        )
    return True


def _run_eval_and_promotion(
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    model_version: int,
    new_version: int,
    args: argparse.Namespace,
    cost_estimator: Any,
    curriculum_manager: Any,
) -> tuple[bool, int, float, float]:
    """Run Phase 3 evaluation and resolve promotion and curriculum recording."""
    if args.evaluation_games <= 0 or model_version <= 0:
        return True, new_version, 0.0, 0.0

    eval_start = time.time()
    logger.info(
        "Phase 3: Evaluating v%d vs v%d (%d games)",
        new_version,
        model_version,
        args.evaluation_games,
    )
    eval_ok, win_rate, eval_verdict = run_evaluation_phase(
        config=config,
        client=client,
        iteration=iteration,
        model_version_new=new_version,
        model_version_old=model_version,
        args=args,
        dry_run=args.dry_run,
        max_retries=args.max_retries,
    )
    eval_duration = (time.time() - eval_start) / 3600.0
    eval_cost = float(
        cost_estimator.estimate_job_cost(
            machine_type=_EVAL_MACHINE_TYPE,
            accelerator_type="",
            accelerator_count=0,
            duration_hours=eval_duration,
            is_spot=False,
        )
    )
    logger.info("Evaluation phase cost (est): $%.2f", eval_cost)
    if not eval_ok:
        logger.error("Evaluation phase failed for iteration %d after retries", iteration)
        return False, new_version, win_rate, eval_cost

    promoted = resolve_promotion(eval_verdict, win_rate, args.evaluation_win_threshold)
    if not promoted:
        if eval_verdict is not None:
            logger.warning(
                "New model rejected by %s policy: %s (win_rate=%.2f%%)",
                eval_verdict.policy or "unknown",
                eval_verdict.decision.name,
                win_rate * 100,
            )
        else:
            logger.warning(
                "New model rejected: win_rate=%.2f%% < threshold=%.2f%%",
                win_rate * 100,
                args.evaluation_win_threshold * 100,
            )
        resolved_version = model_version
    else:
        logger.info(
            "New model promoted: v%d -> v%d (win_rate=%.2f%%)",
            model_version,
            new_version,
            win_rate * 100,
        )
        resolved_version = new_version

    if curriculum_manager is not None:
        curriculum_manager.record_game(won=promoted, metrics={"win_rate": win_rate})
    return True, resolved_version, win_rate, eval_cost


def _detect_learning_stall(
    args: argparse.Namespace,
    iteration: int,
    model_version: int,
    win_rate: float,
    win_rate_history: list[float],
    wandb_logger: Any,
) -> tuple[bool, float]:
    """Evaluate win-rate progression for learning stalls when evaluation ran."""
    if args.evaluation_games <= 0 or model_version <= 0:
        return False, 0.0
    win_rate_history.append(win_rate)
    try:
        from alpha_zero.training.diagnostics import TrainingDiagnostics
    except ImportError:
        logger.debug("TrainingDiagnostics not available for stall detection")
        return False, 0.0

    diag = TrainingDiagnostics(stall_threshold=args.stall_threshold)
    stall_detected = bool(
        diag.detect_learning_stall(
            win_rate_history,
            lookback=args.stall_lookback,
            threshold=args.stall_threshold,
        )
    )
    win_rate_trend = float(
        diag.compute_win_rate_trend(win_rate_history, lookback=args.stall_lookback)
    )
    if stall_detected:
        logger.warning(
            "Learning stall detected at iteration %d: win rate trend=%.4f over last %d iterations",
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
                    f"Current: {win_rate:.1%}, Trend: {win_rate_trend:+.4f}/iter"
                ),
                level="WARN",
            )
    return stall_detected, win_rate_trend


def _execute_selfplay_step(
    state: _OrchestratorLoopState,
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    args: argparse.Namespace,
    cost_estimator: Any,
    curriculum_manager: Any,
) -> tuple[bool, float]:
    """Run curriculum override resolution and Phase 1 self-play for one iteration."""
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
        model_version=state.model_version,
        num_workers=args.workers,
        games_per_worker=args.games_per_worker,
        args=args,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
    )
    sp_duration = (time.time() - sp_start) / 3600.0
    sp_cost = float(
        cost_estimator.estimate_job_cost(
            machine_type=config.self_play.machine_type,
            accelerator_type=config.self_play.accelerator_type,
            accelerator_count=config.self_play.accelerator_count,
            duration_hours=sp_duration,
            is_spot=config.self_play.spot,
        )
        * args.workers
    )
    state.total_estimated_cost += sp_cost
    logger.info("Self-play phase cost (est): $%.2f", sp_cost)
    return selfplay_ok, sp_cost


def _execute_training_step(
    state: _OrchestratorLoopState,
    config: GCPConfig,
    client: VertexAIClient,
    iteration: int,
    args: argparse.Namespace,
    cost_estimator: Any,
) -> tuple[bool, float]:
    """Resolve resume token and run Phase 2 training for one iteration."""
    train_start = time.time()
    earliest = max(0, iteration - args.trajectory_window + 1)
    trajectory_iterations = list(range(earliest, iteration + 1))
    logger.info("Phase 2: Training on trajectories from iterations %s", trajectory_iterations)

    resume_token = (
        getattr(args, "resume_from", None)
        if args.dry_run
        else _resolve_resume_token(args, iteration=iteration)
    )
    if resume_token:
        args.resume_from = resume_token
        logger.info("iteration %d: resuming training from %s", iteration, resume_token)
    else:
        args.resume_from = None
        logger.info("iteration %d: cold-start training (no incumbent checkpoint)", iteration)

    training_ok = run_training_phase(
        config=config,
        client=client,
        iteration=iteration,
        trajectory_iterations=trajectory_iterations,
        args=args,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        model_version=state.model_version,
    )
    train_duration = (time.time() - train_start) / 3600.0
    train_cost = float(
        cost_estimator.estimate_job_cost(
            machine_type=config.training.machine_type,
            accelerator_type=config.training.accelerator_type,
            accelerator_count=config.training.accelerator_count,
            duration_hours=train_duration,
            is_spot=config.training.spot,
        )
    )
    state.total_estimated_cost += train_cost
    logger.info("Training phase cost (est): $%.2f", train_cost)
    return training_ok, train_cost


def _resolve_trained_version(config: GCPConfig, model_version: int, dry_run: bool) -> int:
    """Resolve the newly trained model version from GCS or dry-run increment."""
    if dry_run:
        return model_version + 1
    try:
        return GCSModelStorage(config).get_latest_version()
    except Exception:
        return model_version + 1


def _run_orchestrator_loop(
    config: GCPConfig,
    client: VertexAIClient,
    args: argparse.Namespace,
    wandb_logger: Any,
    curriculum_manager: Any,
) -> int:
    """Run all orchestrator training/evaluation iterations and post-run tournament."""
    from alpha_zero.cloud.gcp.cost import CostEstimator

    start_version = (
        args.model_version_start
        if args.model_version_start is not None
        else get_initial_model_version(config, args.dry_run)
    )
    if args.model_version_start is not None:
        logger.info("Using explicit model version start: %d", start_version)

    state = _OrchestratorLoopState(model_version=start_version, initial_model_version=start_version)
    cost_estimator = CostEstimator(config)
    total_start = time.time()
    max_consec = getattr(args, "max_consecutive_failures", 3)

    for i in range(args.iterations):
        iteration_start = time.time()
        iteration = args.start_iteration + i
        logger.info("=" * 60)
        logger.info("Starting iteration %d (model v%d)", iteration, state.model_version)
        logger.info("=" * 60)

        selfplay_ok, sp_cost = _execute_selfplay_step(
            state, config, client, iteration, args, cost_estimator, curriculum_manager
        )
        if not selfplay_ok:
            logger.error("Self-play phase failed for iteration %d", iteration)
            state.failed_iterations += 1
            state.consecutive_failures += 1
            if _check_circuit_breaker(
                state.consecutive_failures, max_consec, iteration, wandb_logger
            ):
                break
            continue

        training_ok, train_cost = _execute_training_step(
            state, config, client, iteration, args, cost_estimator
        )
        if not training_ok:
            logger.error("Training phase failed for iteration %d", iteration)
            state.failed_iterations += 1
            state.consecutive_failures += 1
            if _check_circuit_breaker(
                state.consecutive_failures, max_consec, iteration, wandb_logger
            ):
                break
            continue

        candidate_version = _resolve_trained_version(config, state.model_version, args.dry_run)
        eval_ok, resolved_version, win_rate, eval_cost = _run_eval_and_promotion(
            config,
            client,
            iteration,
            state.model_version,
            candidate_version,
            args,
            cost_estimator,
            curriculum_manager,
        )
        state.total_estimated_cost += eval_cost
        if not eval_ok:
            state.failed_iterations += 1
            state.consecutive_failures += 1
            if _check_circuit_breaker(
                state.consecutive_failures, max_consec, iteration, wandb_logger, phase_label="eval"
            ):
                break
            continue

        iteration_duration = (time.time() - iteration_start) / 60.0
        stall_detected, win_rate_trend = _detect_learning_stall(
            args, iteration, state.model_version, win_rate, state.win_rate_history, wandb_logger
        )
        if wandb_logger is not None and wandb_logger.is_enabled:
            metrics = {
                "orchestrator/iteration": iteration,
                "orchestrator/model_version": resolved_version,
                "orchestrator/iteration_duration_min": iteration_duration,
                "orchestrator/selfplay_cost": sp_cost,
                "orchestrator/training_cost": train_cost,
                "orchestrator/cumulative_cost": state.total_estimated_cost,
                "orchestrator/stall_detected": int(stall_detected),
                "orchestrator/win_rate_trend": win_rate_trend,
            }
            if args.evaluation_games > 0 and state.model_version > 0:
                metrics["orchestrator/eval_win_rate"] = win_rate
                metrics["orchestrator/eval_cost"] = eval_cost
            wandb_logger.log(metrics, step=iteration)

        logger.info(
            "Iteration %d complete in %.1f minutes. Model: v%d",
            iteration,
            iteration_duration,
            resolved_version,
        )
        state.model_version = resolved_version
        state.successful_iterations += 1
        state.consecutive_failures = 0

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info("Training loop complete")
    logger.info(
        "Results: %d/%d iterations succeeded in %.1f minutes",
        state.successful_iterations,
        args.iterations,
        total_elapsed / 60,
    )
    logger.info("Final model version: v%d", state.model_version)
    logger.info("Total Estimated Cost: $%.2f", state.total_estimated_cost)
    logger.info("=" * 60)

    if args.post_run_tournament and state.successful_iterations > 0:
        latest_version = get_initial_model_version(config, args.dry_run)
        all_versions = list(range(state.initial_model_version, latest_version + 1))
        if len(all_versions) >= 2:
            run_post_run_evaluation(
                config=config,
                model_versions=all_versions,
                args=args,
                wandb_logger=wandb_logger,
                dry_run=args.dry_run,
            )

    if wandb_logger is not None and wandb_logger.is_enabled:
        wandb_logger.log(
            {
                "orchestrator/total_iterations": args.iterations,
                "orchestrator/successful_iterations": state.successful_iterations,
                "orchestrator/failed_iterations": state.failed_iterations,
                "orchestrator/total_cost": state.total_estimated_cost,
                "orchestrator/total_duration_min": total_elapsed / 60,
                "orchestrator/final_model_version": state.model_version,
            }
        )
        wandb_logger.finish()

    return 0 if state.failed_iterations == 0 else 1


def main() -> int:
    """Main entry point.

    .. note:: This orchestrator is designed to run **locally** (or on a small
       GCE VM with appropriate IAM permissions).  It dispatches self-play,
       training, and evaluation jobs to Vertex AI Custom Jobs.  Running it
       inside a Vertex AI container would create a recursive loop.
    """
    args = parse_args()
    setup_logging()
    args.evaluation_win_threshold = resolve_evaluation_threshold(args)
    warn_on_ignored_parallelism(args)

    cloud_job_id = os.environ.get("CLOUD_ML_JOB_ID")
    if cloud_job_id and not getattr(args, "dry_run", False):
        logger.error(
            "ARCHITECTURE ERROR: The orchestrator is running inside Vertex AI "
            "container (job=%s). The orchestrator must run LOCALLY and dispatch "
            "worker jobs to Vertex AI.  Running it in a container creates a "
            "recursive loop. Use selfplay_worker.py or training_worker.py as "
            "container entrypoints instead.",
            cloud_job_id,
        )
        return 1

    _apply_architecture_preset_to_args(args)
    wandb_logger = _init_orchestrator_tracker(args)

    logger.info("Civ6 AlphaZero Vertex AI Orchestrator")
    logger.info(
        "Config: %d iterations, %d workers, %d games/worker, %dx%d board, %s MCTS, encoder=%s",
        args.iterations,
        args.workers,
        args.games_per_worker,
        args.board_size,
        args.board_size,
        args.mcts_type,
        getattr(args, "encoder_type", "patch_embed"),
    )
    if args.dry_run:
        logger.info("DRY RUN MODE - no cloud jobs will be launched")

    curriculum_manager = _init_curriculum_manager(args)
    try:
        config = _build_and_validate_config(args)
        if config is None:
            return 1
        client = VertexAIClient(
            project=config.project_id,
            region=config.region,
            service_account=config.service_account,
        )
        if getattr(args, "tournament_only", False):
            return _run_tournament_only_mode(config, args, wandb_logger)
        return _run_orchestrator_loop(config, client, args, wandb_logger, curriculum_manager)
    except KeyboardInterrupt:
        logger.warning("Orchestrator interrupted by user")
        _finish_tracker(wandb_logger, exit_code=130)
        return 130
    except Exception as e:
        logger.exception("Orchestrator failed: %s", e)
        _finish_tracker(wandb_logger, exit_code=1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
