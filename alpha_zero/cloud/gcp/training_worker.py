"""
Training Worker Entry Point for Vertex AI.

This is the containerized entry point for training jobs.
Each training job:
1. Resumes from GCS checkpoint if available
2. Loads trajectories from specified iterations
3. Trains for specified epochs
4. Checkpoints every N steps to GCS
5. Saves final model with version increment

Usage:
    python -m alpha_zero.cloud.gcp.training_worker \
        --iteration=10 \
        --trajectory_iterations=8,9,10 \
        --epochs=10 \
        --resume_from=gs://bucket/checkpoints/latest.pt
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: N817
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import (
    GCSCheckpointStorage,
    GCSModelStorage,
    GCSTrajectoryStorage,
)
from alpha_zero.core.losses import LossConfig, compute_reward_loss
from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

if TYPE_CHECKING:
    from alpha_zero.orchestrator.scheduler import HyperparameterScheduler
    from alpha_zero.training.diagnostics import TrainingDiagnostics
    from alpha_zero.training.training_profiler import TrainingProfiler
    from alpha_zero.training.training_report import TrainingReport

if TYPE_CHECKING:
    from alpha_zero.orchestrator.scheduler import HyperparameterScheduler
    from alpha_zero.training.diagnostics import TrainingDiagnostics
    from alpha_zero.training.training_profiler import TrainingProfiler
    from alpha_zero.training.training_report import TrainingReport

logger = logging.getLogger(__name__)

CHECKPOINT_RETENTION: int = 3


def setup_logging() -> None:
    """Configure logging for training worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="[Training] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def setup_ddp() -> tuple[int, int, int]:
    """Initialize distributed training.

    Returns:
        Tuple of (rank, local_rank, world_size).
    """
    if "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Select backend based on CUDA availability to support CPU-only envs.
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        logger.info(
            "Initialized DDP: rank=%d, local_rank=%d, world_size=%d, backend=%s",
            rank,
            local_rank,
            world_size,
            backend,
        )
        return rank, local_rank, world_size
    return 0, 0, 1


def cleanup_ddp() -> None:
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def report_tuning_metric(metric: dict[str, float], step: int) -> None:
    """Report hyperparameter tuning metric to Vertex AI Vizier.

    Writes to the file specified by CLOUD_ML_HP_METRIC_FILE environment variable.
    """
    metric_file = os.environ.get("CLOUD_ML_HP_METRIC_FILE")
    if not metric_file:
        return

    try:
        import json

        # Format required by Vertex AI Vizier
        data = {
            "timestamp": time.time(),
            "step": step,
            **metric,
        }

        # Write generic metric to file (append or overwrite?)
        # Vizier expects one JSON per line or just one JSON?
        # Documentation says: "Write the metric to the file... Each line in the file is a JSON object"

        with open(metric_file, "a") as f:
            f.write(json.dumps(data) + "\n")

        logger.debug("Reported tuning metric: %s", data)

    except Exception as e:
        logger.warning("Failed to report tuning metric: %s", e)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Training Worker")
    parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    parser.add_argument(
        "--trajectory_iterations",
        type=str,
        required=True,
        help="Comma-separated iterations to load trajectories from",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--agent_id", type=str, default=None, help="PBT Agent ID")
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1000, help="Steps between checkpoints"
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    # Network architecture args passed by the Vertex orchestrator
    parser.add_argument("--board_size", type=int, default=8, help="Board size (spatial_size)")
    parser.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    parser.add_argument(
        "--num_res_blocks", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--num_filters", type=int, default=64, help="Model embedding dimension")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument(
        "--policy_hidden_dim", type=int, default=128, help="Policy head hidden dimension"
    )
    parser.add_argument(
        "--value_head_dim", type=int, default=64, help="Value head hidden dimension"
    )
    parser.add_argument(
        "--wandb_log_interval", type=int, default=10, help="W&B logging frequency (batches)"
    )
    # Learning rate schedule args (backwards-compatible: constant=no change)
    parser.add_argument(
        "--lr_schedule_type",
        type=str,
        default="constant",
        choices=["constant", "linear", "cosine", "warmup_cosine", "exponential", "step"],
        help="LR schedule type (default: constant=fixed LR)",
    )
    parser.add_argument("--lr_initial", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--lr_final", type=float, default=0.0001, help="Final learning rate")
    parser.add_argument("--lr_warmup_steps", type=int, default=0, help="LR warmup steps")
    parser.add_argument(
        "--lr_decay_steps",
        type=int,
        default=0,
        help="LR decay steps (0=auto-compute from epochs*batches)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument(
        "--value_lr_multiplier", type=float, default=1.0, help="LR multiplier for value head"
    )
    parser.add_argument(
        "--policy_lr_multiplier", type=float, default=1.0, help="LR multiplier for policy head"
    )
    parser.add_argument(
        "--diagnostic_window", type=int, default=100, help="Diagnostics rolling window size"
    )
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")
    parser.add_argument(
        "--algorithm",
        type=str,
        default="alphazero",
        choices=["alphazero", "muzero"],
        help="Training algorithm (default: alphazero)",
    )
    # MuZero-specific training args
    parser.add_argument(
        "--muzero_unroll_steps",
        type=int,
        default=5,
        help="MuZero dynamics unroll steps for training",
    )
    parser.add_argument(
        "--muzero_td_steps",
        type=int,
        default=10,
        help="MuZero TD-lambda steps for value targets",
    )
    parser.add_argument(
        "--muzero_discount",
        type=float,
        default=0.997,
        help="MuZero discount factor",
    )
    # Spatial encoder type
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="patch_embed",
        choices=["patch_embed", "hex_crystalline"],
        help="Spatial encoder type (default: patch_embed)",
    )
    return parser.parse_args()


def load_trajectories(
    config: GCPConfig,
    iterations: list[int],
    max_per_iteration: int = 50000,
    board_size: int = 8,
    action_space_size: int = 376,
    dry_run: bool = False,
) -> list[dict]:
    """Load trajectories from specified iterations."""
    if dry_run:
        logger.info("Dry run: creating dummy trajectories")
        spatial_dim = ENCODER_SPATIAL_CHANNELS * board_size * board_size
        vector_dim = 64
        policy_val = 1.0 / action_space_size
        return [
            {
                "spatial": [[0.0] * spatial_dim],
                "vector": [[0.0] * vector_dim],
                "policies": [[policy_val] * action_space_size],
                "rewards": [0.5],
            }
            for _ in range(1000)
        ]

    storage = GCSTrajectoryStorage(config)
    all_trajectories = []

    for iteration in iterations:
        trajectories = storage.load_trajectories(iteration, max_examples=max_per_iteration)
        all_trajectories.extend(trajectories)
        logger.info("Loaded %d trajectories from iteration %d", len(trajectories), iteration)

    logger.info("Total trajectories loaded: %d", len(all_trajectories))
    return all_trajectories


# MicroCivEncoder.TOTAL_CHANNELS — input spatial feature planes from the encoder
ENCODER_SPATIAL_CHANNELS: int = 13
# MicroCivEnv default vector observation dimension
VECTOR_DIM: int = 64


def create_model(
    board_size: int = 8,
    action_space_size: int = 376,
    num_res_blocks: int = 6,
    num_filters: int = 64,
    spatial_channels: int = ENCODER_SPATIAL_CHANNELS,
    num_heads: int = 8,
    policy_hidden_dim: int = 128,
    value_head_dim: int = 64,
    encoder_type: str = "patch_embed",
) -> torch.nn.Module:
    """Create model architecture.

    Args:
        board_size: Board size = spatial_size input dimension (H=W).
        action_space_size: Total number of actions.
        num_res_blocks: Maps to TransformerConfig.num_layers.
        num_filters: Maps to TransformerConfig.embed_dim.
        spatial_channels: Input spatial channels (from encoder, default 13).
        num_heads: Number of attention heads.
        policy_hidden_dim: Hidden dimension for policy head.
        value_head_dim: Hidden dimension for value head.
        encoder_type: Spatial encoder type ('patch_embed' or 'hex_crystalline').
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    model_config = TransformerConfig(
        spatial_channels=spatial_channels,
        spatial_size=board_size,
        vector_dim=VECTOR_DIM,
        action_space_size=action_space_size,
        num_layers=num_res_blocks,
        embed_dim=num_filters,
        num_heads=num_heads,
        policy_hidden_dim=policy_hidden_dim,
        value_head_dim=value_head_dim,
        encoder_type=encoder_type,
    )
    logger.info(
        "Creating model: spatial=(%d, %d, %d), actions=%d, layers=%d, embed=%d, heads=%d, "
        "encoder=%s",
        spatial_channels,
        board_size,
        board_size,
        action_space_size,
        num_res_blocks,
        num_filters,
        num_heads,
        encoder_type,
    )
    return Civ6TransformerNetwork(config=model_config)


def create_muzero_model(
    board_size: int = 8,
    action_space_size: int = 376,
    num_res_blocks: int = 6,
    num_filters: int = 64,
    spatial_channels: int = ENCODER_SPATIAL_CHANNELS,
) -> torch.nn.Module:
    """Create MuZero model architecture.

    Args:
        board_size: Board size = spatial_size input dimension (H=W).
        action_space_size: Total number of actions.
        num_res_blocks: Number of residual blocks in representation/dynamics.
        num_filters: Hidden channel dimension.
        spatial_channels: Input spatial channels (from encoder, default 13).
    """
    from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

    muzero_config = MuZeroConfig(
        spatial_channels=spatial_channels,
        spatial_size=board_size,
        vector_dim=VECTOR_DIM,
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        hidden_state_channels=num_filters,
    )
    logger.info(
        "Creating MuZero model: spatial=(%d, %d, %d), actions=%d, res_blocks=%d, "
        "hidden_channels=%d",
        spatial_channels,
        board_size,
        board_size,
        action_space_size,
        num_res_blocks,
        num_filters,
    )
    return MuZeroNetwork(config=muzero_config)


def train_muzero_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    unroll_steps: int = 5,
    discount: float = 0.997,
    global_step: int = 0,
    wandb_logger: WandbLogger | None = None,
    wandb_log_interval: int = 10,
    lr_scheduler: HyperparameterScheduler | None = None,
) -> tuple[float, float, float, float, int]:
    """Train MuZero model for one epoch using trajectory-based multi-step learning.

    Args:
        model: MuZero neural network.
        dataloader: Training data loader with MuZero trajectories.
        optimizer: Optimizer instance.
        device: Training device.
        unroll_steps: Number of dynamics unroll steps.
        discount: Discount factor for value targets.
        global_step: Global step counter for logging.
        wandb_logger: Optional W&B logger.
        wandb_log_interval: W&B logging frequency in batches.
        lr_scheduler: Optional LR scheduler.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, reward_loss, num_batches).
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_reward_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        spatial = batch[0].to(device)
        vector = batch[1].to(device)
        # batch[2] is action_mask (added in prepare_dataset); policies and values shift by 1
        target_policies = batch[3].to(device)
        target_values = batch[4].to(device)

        # MuZero-specific targets (if available in the dataset)
        target_rewards = batch[5].to(device) if len(batch) > 5 else torch.zeros_like(target_values)
        target_actions = (
            batch[6].to(device).long()
            if len(batch) > 6
            else torch.zeros(spatial.shape[0], dtype=torch.long, device=device)
        )

        # Update learning rate from scheduler
        current_step = global_step + num_batches
        if lr_scheduler is not None:
            scheduled_lr = lr_scheduler.get_learning_rate(current_step)
            for param_group in optimizer.param_groups:
                ratio = param_group.get("base_lr_ratio", 1.0)
                param_group["lr"] = scheduled_lr * ratio

        # Forward pass: initial inference via representation + prediction
        batch_size = spatial.shape[0]
        spatial_channels = ENCODER_SPATIAL_CHANNELS
        # Reshape to (B, C, H, W)
        expected_flat = spatial.shape[1]
        hw = int((expected_flat / spatial_channels) ** 0.5)
        spatial_4d = spatial.view(batch_size, spatial_channels, hw, hw)

        # MuZero initial inference
        if hasattr(model, "representation"):
            hidden_state = model.representation(spatial_4d, vector)  # type: ignore[operator]
            policy_logits, value = model.prediction_function(hidden_state)  # type: ignore[operator]
        else:
            policy_logits, value = model(spatial_4d, vector)

        # Policy loss (cross-entropy with MCTS targets)
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(target_policies * policy_log_probs, dim=-1).mean()

        # Value loss (MSE)
        value_loss = torch.nn.functional.mse_loss(value.squeeze(), target_values)

        # Reward loss (accumulated from dynamics predictions in unroll loop)
        reward_loss = torch.tensor(0.0, device=device)

        # Detect categorical reward config from model for proper loss computation
        use_categorical_reward = getattr(
            getattr(model, "config", None), "use_categorical_reward", False
        )
        reward_support_size = getattr(getattr(model, "config", None), "reward_support_size", 601)
        reward_loss_config = LossConfig(
            use_categorical=use_categorical_reward,
            support_size=reward_support_size,
        )

        # Dynamics unrolling for multi-step losses (if model supports it)
        if hasattr(model, "dynamics_function") and unroll_steps > 0:
            unroll_policy_loss = torch.tensor(0.0, device=device)
            unroll_value_loss = torch.tensor(0.0, device=device)
            current_hidden = hidden_state.detach() if hidden_state is not None else None

            # Only unroll if we have valid hidden states
            if current_hidden is not None:
                for k in range(unroll_steps):
                    dynamics_out = model.dynamics_function(current_hidden, target_actions)  # type: ignore[operator]
                    current_hidden = dynamics_out.next_hidden_state

                    # Prediction from unrolled state
                    step_policy_logits, step_value = model.prediction_function(current_hidden)  # type: ignore[operator]

                    # Accumulate losses with discount
                    discount_factor = discount ** (k + 1)
                    step_policy_log_probs = torch.log_softmax(step_policy_logits, dim=-1)
                    unroll_policy_loss = unroll_policy_loss + discount_factor * (
                        -torch.sum(target_policies * step_policy_log_probs, dim=-1).mean()
                    )
                    unroll_value_loss = unroll_value_loss + discount_factor * (
                        torch.nn.functional.mse_loss(step_value.squeeze(), target_values)
                    )

                    # Reward prediction loss (handles both categorical and scalar)
                    reward_pred = dynamics_out.reward
                    if reward_pred is not None:
                        reward_loss = reward_loss + compute_reward_loss(
                            pred=reward_pred,
                            target=target_rewards,
                            config=reward_loss_config,
                        )

                # Add unrolled losses (weighted down)
                policy_loss = policy_loss + 0.5 * unroll_policy_loss
                value_loss = value_loss + 0.5 * unroll_value_loss

        # Combined loss
        loss = policy_loss + value_loss + 0.1 * reward_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_reward_loss += reward_loss.item()
        num_batches += 1

        # Log to W&B periodically
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
            and num_batches % wandb_log_interval == 0
        ):
            wandb_logger.log_training_step(
                step=global_step + num_batches,
                loss=loss.item(),
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
                learning_rate=optimizer.param_groups[0]["lr"],
                reward_loss=reward_loss.item(),
                algorithm="muzero",
            )

    return (
        total_loss / max(num_batches, 1),
        total_policy_loss / max(num_batches, 1),
        total_value_loss / max(num_batches, 1),
        total_reward_loss / max(num_batches, 1),
        num_batches,
    )


def prepare_dataset(
    trajectories: list[dict],
    vector_dim: int | None = None,
    action_space_size: int | None = None,
) -> TensorDataset:
    """Convert trajectories to PyTorch dataset.

    Args:
        trajectories: List of trajectory dictionaries.
        vector_dim: Vector dimension for legacy format handling.
            If None, uses default from TransformerConfig.
        action_space_size: Action space size for legacy mask fallback.
            If None, uses default from TransformerConfig.

    Returns:
        TensorDataset with spatial, vector, action_mask, policy, and value tensors.
        Order: (spatial, vector, action_mask, policies, values)
    """
    from torch.utils.data import TensorDataset

    # Get vector_dim from config if not specified
    if vector_dim is None:
        from alpha_zero.core.transformer_network import TransformerConfig

        vector_dim = TransformerConfig().vector_dim

    # Get action_space_size from config if not specified
    if action_space_size is None:
        from alpha_zero.core.transformer_network import TransformerConfig

        action_space_size = TransformerConfig().action_space_size

    spatial_list = []
    vector_list = []
    action_masks_list = []
    policies = []
    values = []

    legacy_format_warned = False

    for traj in trajectories:
        # Handle both old format (states) and new format (spatial + vector)
        if "spatial" in traj:
            for i, spatial in enumerate(traj["spatial"]):
                spatial_list.append(spatial)
                vector_list.append(traj["vector"][i])
                # Use stored mask if available; fall back to all-ones (legacy trajectories)
                if "action_masks" in traj:
                    action_masks_list.append(traj["action_masks"][i])
                else:
                    action_masks_list.append([1.0] * action_space_size)
                policies.append(traj["policies"][i])
                values.append(traj["rewards"][i])
        else:
            # Legacy format: split states into spatial and dummy vector
            if not legacy_format_warned:
                logger.warning(
                    "Legacy trajectory format detected (missing 'spatial' key). "
                    "Using zero-filled vector of dimension %d. "
                    "Consider re-generating trajectories with updated selfplay worker.",
                    vector_dim,
                )
                legacy_format_warned = True
            for i, state in enumerate(traj["states"]):
                spatial_list.append(state)
                vector_list.append([0.0] * vector_dim)
                action_masks_list.append([1.0] * action_space_size)
                policies.append(traj["policies"][i])
                values.append(traj["rewards"][i])

    # Convert to tensors
    spatial_tensor = torch.tensor(spatial_list, dtype=torch.float32)
    vector_tensor = torch.tensor(vector_list, dtype=torch.float32)
    action_masks_tensor = torch.tensor(action_masks_list, dtype=torch.float32)
    policies_tensor = torch.tensor(policies, dtype=torch.float32)
    values_tensor = torch.tensor(values, dtype=torch.float32)

    return TensorDataset(
        spatial_tensor, vector_tensor, action_masks_tensor, policies_tensor, values_tensor
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int = 0,
    wandb_logger: WandbLogger | None = None,
    spatial_channels: int = 13,
    spatial_size: int = 8,
    wandb_log_interval: int = 10,
    lr_scheduler: HyperparameterScheduler | None = None,
    diagnostics: TrainingDiagnostics | None = None,
    report: TrainingReport | None = None,
    profiler: TrainingProfiler | None = None,
) -> tuple[float, float, float, int]:
    """Train for one epoch.

    Args:
        model: Neural network model.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        device: Training device.
        global_step: Global step counter for W&B logging.
        wandb_logger: Optional W&B logger.
        spatial_channels: Number of spatial channels.
        spatial_size: Board size.
        wandb_log_interval: W&B logging frequency in batches.
        lr_scheduler: Optional HyperparameterScheduler for LR scheduling.
        diagnostics: Optional TrainingDiagnostics for metric tracking.
        report: Optional TrainingReport for periodic summary generation.
        profiler: Optional PerformanceProfiler for section timing.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, num_batches).
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for spatial, vector, action_mask, target_policies, target_values in dataloader:
        spatial = spatial.to(device)
        vector = vector.to(device)
        action_mask = action_mask.to(device)
        target_policies = target_policies.to(device)
        target_values = target_values.to(device)

        # Update learning rate from scheduler if available (preserving per-group ratios)
        current_step = global_step + num_batches
        if lr_scheduler is not None:
            scheduled_lr = lr_scheduler.get_learning_rate(current_step)
            for param_group in optimizer.param_groups:
                ratio = param_group.get("base_lr_ratio", 1.0)
                param_group["lr"] = scheduled_lr * ratio

        # Forward pass
        # Reshape spatial for transformer input and validate
        batch_size = spatial.shape[0]
        expected_flat = spatial_channels * spatial_size * spatial_size
        if spatial.shape[1] != expected_flat:
            raise ValueError(
                f"Spatial tensor flat dim {spatial.shape[1]} != "
                f"expected {spatial_channels}x{spatial_size}x{spatial_size}={expected_flat}. "
                f"Ensure encoder channels and board_size match between selfplay and training."
            )
        spatial = spatial.view(
            batch_size, spatial_channels, spatial_size, spatial_size
        )  # B, C, H, W

        if profiler is not None:
            profiler.start("forward_pass")

        policy_logits, value = model(spatial, vector, action_mask)

        # Policy loss: cross-entropy over legal actions only.
        # Mask illegal actions from target policy, then renormalize.
        masked_target = target_policies * action_mask
        masked_target = masked_target / masked_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(masked_target * policy_log_probs, dim=-1).mean()

        # Value loss (MSE)
        value_loss = torch.nn.functional.mse_loss(value, target_values)

        # Combined loss
        loss = policy_loss + value_loss

        if profiler is not None:
            profiler.stop("forward_pass")

        # Backward pass
        if profiler is not None:
            profiler.start("backward_pass")

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient stats BEFORE clipping (for diagnostics)
        grad_norm_pre_clip = 0.0
        if diagnostics is not None:
            grad_stats = diagnostics.compute_gradient_stats(model)
            grad_norm_pre_clip = grad_stats["total_norm"]

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if profiler is not None:
            profiler.stop("backward_pass")

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

        # Record step in training report
        if report is not None:
            report.record_step(
                step=current_step,
                loss=loss.item(),
                grad_norm=grad_norm_pre_clip,
                lr=optimizer.param_groups[0]["lr"],
                policy_loss=policy_loss.item(),
                value_loss=value_loss.item(),
            )

        # Compute diagnostic metrics
        diagnostic_kwargs: dict[str, float] = {}
        if diagnostics is not None:
            policy_entropy = diagnostics.compute_policy_entropy(policy_logits)
            value_accuracy = diagnostics.compute_value_accuracy(value, target_values)
            diagnostic_kwargs = {
                "policy_entropy": policy_entropy,
                "value_sign_accuracy": value_accuracy["sign_accuracy"],
                "value_correlation": value_accuracy["correlation"],
                "value_mae": value_accuracy["mae"],
                "grad_norm_pre_clip": grad_norm_pre_clip,
            }
            logger.debug(
                "Step %d: entropy=%.3f, sign_acc=%.3f, corr=%.3f",
                current_step,
                policy_entropy,
                value_accuracy["sign_accuracy"],
                value_accuracy["correlation"],
            )

        # Log to W&B periodically
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            if num_batches % wandb_log_interval == 0:
                lr = optimizer.param_groups[0]["lr"]
                # Log per-head learning rates
                for pg in optimizer.param_groups:
                    name = pg.get("name", "")
                    if name:
                        diagnostic_kwargs[f"learning_rate_{name}"] = pg["lr"]
                wandb_logger.log_training_step(
                    step=global_step + num_batches,
                    loss=loss.item(),
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    learning_rate=lr,
                    **diagnostic_kwargs,
                )

    # Generate training report if interval reached
    end_step = global_step + num_batches
    if report is not None and report.should_report(end_step):
        report_data = report.generate_report(end_step)
        if (
            report_data is not None
            and wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.log(report_data.to_wandb_dict())

    return (
        total_loss / max(num_batches, 1),
        total_policy_loss / max(num_batches, 1),
        total_value_loss / max(num_batches, 1),
        num_batches,
    )


def save_checkpoint(
    config: GCPConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    dry_run: bool = False,
) -> str | None:
    """Save training checkpoint to GCS."""
    if dry_run:
        logger.info("Dry run: would save checkpoint at step %d", step)
        return None

    storage = GCSCheckpointStorage(config)
    uri = storage.save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=step,
        extra={"epoch": epoch},
    )
    logger.info("Saved checkpoint at step %d to %s", step, uri)
    return uri


def save_final_model(
    config: GCPConfig,
    model: torch.nn.Module,
    iteration: int,
    dry_run: bool = False,
) -> tuple[str | None, int]:
    """Save final trained model to GCS.

    Returns:
        Tuple of (uri, new_version).
    """
    if dry_run:
        new_version = iteration
        logger.info("Dry run: would save model v%d", new_version)
        return None, new_version

    storage = GCSModelStorage(config)

    try:
        latest = storage.get_latest_version()
        new_version = latest + 1
    except Exception:
        new_version = 1

    # Extract model config for architecture reconstruction
    model_config_dict = None
    if hasattr(model, "config"):
        import dataclasses
        from enum import Enum

        model_config_dict = dataclasses.asdict(model.config)  # type: ignore[call-overload]
        # Convert enums to string values for JSON serialization
        for key, value in model_config_dict.items():
            if isinstance(value, Enum):
                model_config_dict[key] = value.value

    # Detect algorithm type for checkpoint metadata
    algorithm = "alphazero"
    config_module = type(model).__module__ if model is not None else ""
    if "muzero" in config_module.lower() or (
        model_config_dict and "hidden_channels" in model_config_dict
    ):
        algorithm = "muzero"

    uri = storage.save_model(
        model=model,
        version=new_version,
        metadata={"iteration": iteration, "algorithm": algorithm},
        model_config=model_config_dict,
    )
    logger.info(
        "Saved model v%d (%s) to %s (with config: %s)",
        new_version,
        algorithm,
        uri,
        model_config_dict is not None,
    )
    return uri, new_version


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    trajectory_iterations = [int(x) for x in args.trajectory_iterations.split(",")]
    logger.info("Starting training for iteration %d", args.iteration)
    logger.info(
        "Config: trajectories from %s, %d epochs, board=%dx%d",
        trajectory_iterations,
        args.epochs,
        args.board_size,
        args.board_size,
    )

    # Initialize W&B (disabled by default, opt-in via WANDB_ENABLED env var)
    wandb_logger = None
    try:
        from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

        wandb_logger = WandbLogger.from_env()
        if wandb_logger.is_enabled:
            wandb_logger.init(
                config_dict={
                    "iteration": args.iteration,
                    "trajectory_iterations": trajectory_iterations,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "board_size": args.board_size,
                    "embed_dim": args.num_filters,
                    "num_layers": args.num_res_blocks,
                    "num_heads": getattr(args, "num_heads", 8),
                    "policy_hidden_dim": getattr(args, "policy_hidden_dim", 128),
                    "value_head_dim": getattr(args, "value_head_dim", 64),
                    "mode": "training",
                }
            )
            logger.info("W&B logging enabled")
    except ImportError:
        logger.debug("W&B not available")

    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    is_main_process = rank == 0

    try:
        config = GCPConfig()
        if args.agent_id:
            config.paths.agent_id = args.agent_id
            if is_main_process:
                logger.info("Using Agent ID: %s", args.agent_id)

        if not args.dry_run and is_main_process:
            config.validate()

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

        if is_main_process:
            logger.info("Using device: %s (World size: %d)", device, world_size)

        # Load trajectories (only on main process or cached?)
        # For simplicity, all processes load data. In data-parallel, this is fine for small datasets.
        # For huge datasets, we'd want to shard reading.
        # GCS storage might be slow if 8 GPUs hit it at once.
        # But for now assuming each node caches it independently or it's fast enough.
        trajectories = load_trajectories(
            config,
            trajectory_iterations,
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            dry_run=args.dry_run,
        )

        # Validate we have data to train on
        if not trajectories:
            logger.error(
                "No trajectories found for iterations %s. "
                "You must run self-play first to generate training data:\n"
                "  python -m alpha_zero.cloud.gcp.cli selfplay --iteration %d --wait",
                trajectory_iterations,
                args.iteration,
            )
            return 1

        # Prepare dataset
        dataset = prepare_dataset(trajectories)

        if len(dataset) == 0:
            logger.error(
                "Dataset is empty after processing trajectories. "
                "This may indicate corrupted trajectory data."
            )
            return 1

        # Distributed sampler
        sampler: DistributedSampler | None = DistributedSampler(dataset) if world_size > 1 else None
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            # num_workers=0 avoids multiprocessing issues inside containers
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        if is_main_process:
            logger.info("Dataset size: %d examples, %d batches", len(dataset), len(dataloader))

        # Create model (AlphaZero or MuZero based on --algorithm)
        is_muzero = getattr(args, "algorithm", "alphazero") == "muzero"

        if is_muzero:
            model = create_muzero_model(
                board_size=args.board_size,
                action_space_size=args.action_space_size,
                num_res_blocks=args.num_res_blocks,
                num_filters=args.num_filters,
            )
            if is_main_process:
                logger.info("Using MuZero algorithm for training")
        else:
            model = create_model(
                board_size=args.board_size,
                action_space_size=args.action_space_size,
                num_res_blocks=args.num_res_blocks,
                num_filters=args.num_filters,
                num_heads=getattr(args, "num_heads", 8),
                policy_hidden_dim=getattr(args, "policy_hidden_dim", 128),
                value_head_dim=getattr(args, "value_head_dim", 64),
                encoder_type=getattr(args, "encoder_type", "patch_embed"),
            )
        model = model.to(device)
        raw_model = model  # Keep reference before DDP for param group construction

        # Wrap DDP
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank])

        # Enable W&B model watching for automatic gradient histograms
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            try:
                wandb_logger.watch_model(raw_model, log="gradients", log_freq=100)
                logger.info("W&B model gradient watching enabled")
            except Exception as e:
                logger.debug("Failed to enable W&B model watching: %s", e)

        # Determine effective base LR: use lr_initial when a schedule is active,
        # otherwise fall back to --learning_rate. This prevents an abrupt LR jump
        # on the first scheduler step when lr_initial != learning_rate.
        lr_schedule_type = getattr(args, "lr_schedule_type", "constant")
        base_lr = (
            getattr(args, "lr_initial", args.learning_rate)
            if lr_schedule_type != "constant"
            else args.learning_rate
        )

        # Create optimizer with per-head learning rate multipliers
        if is_muzero:
            # MuZero uses simpler param groups (representation, dynamics, prediction)
            param_groups = [
                {
                    "params": list(raw_model.parameters()),
                    "lr": base_lr,
                    "base_lr_ratio": 1.0,
                    "name": "muzero_all",
                },
            ]
        else:
            param_groups = [
                {
                    "params": list(raw_model.encoder.parameters()),  # type: ignore[union-attr]
                    "lr": base_lr,
                    "base_lr_ratio": 1.0,
                    "name": "encoder",
                },
                {
                    "params": list(raw_model.policy_head.parameters()),  # type: ignore[union-attr]
                    "lr": base_lr * args.policy_lr_multiplier,
                    "base_lr_ratio": args.policy_lr_multiplier,
                    "name": "policy_head",
                },
                {
                    "params": list(raw_model.value_head.parameters()),  # type: ignore[union-attr]
                    "lr": base_lr * args.value_lr_multiplier,
                    "base_lr_ratio": args.value_lr_multiplier,
                    "name": "value_head",
                },
            ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

        # Create LR scheduler (backwards-compatible: constant=no change)
        lr_scheduler = None
        try:
            from alpha_zero.orchestrator.config import ScheduleConfig
            from alpha_zero.orchestrator.scheduler import HyperparameterScheduler

            if args.lr_schedule_type != "constant":
                lr_schedule_config = ScheduleConfig(
                    schedule_type=args.lr_schedule_type,
                    initial_value=args.lr_initial,
                    final_value=args.lr_final,
                    warmup_steps=args.lr_warmup_steps,
                    decay_steps=args.lr_decay_steps
                    if args.lr_decay_steps > 0
                    else (args.epochs * len(dataloader)),
                )
                lr_scheduler = HyperparameterScheduler(lr_schedule=lr_schedule_config)
                if is_main_process:
                    logger.info(
                        "LR schedule: %s (%.6f -> %.6f, warmup=%d, decay=%d steps)",
                        args.lr_schedule_type,
                        args.lr_initial,
                        args.lr_final,
                        args.lr_warmup_steps,
                        lr_schedule_config.decay_steps,
                    )
        except ImportError:
            logger.debug("HyperparameterScheduler not available, using constant LR")

        # Initialize training diagnostics
        diagnostics = None
        try:
            from alpha_zero.training.diagnostics import TrainingDiagnostics

            diagnostics = TrainingDiagnostics(window_size=args.diagnostic_window)
            if is_main_process:
                logger.info("Training diagnostics enabled (window=%d)", args.diagnostic_window)
        except ImportError:
            logger.debug("TrainingDiagnostics not available")

        # Initialize training report
        report = None
        try:
            from alpha_zero.training.training_report import TrainingReport, TrainingReportConfig

            report = TrainingReport(TrainingReportConfig(report_interval=max(len(dataloader), 1)))
            if is_main_process:
                logger.info("Training report enabled (interval=%d steps)", max(len(dataloader), 1))
        except ImportError:
            logger.debug("TrainingReport not available")

        # Initialize performance profiler
        profiler = None
        try:
            from alpha_zero.utils.logging_config import PerformanceProfiler

            profiler = PerformanceProfiler()
            if is_main_process:
                logger.info("Performance profiler enabled")
        except ImportError:
            logger.debug("PerformanceProfiler not available")

        # Log trajectory quality before training
        if diagnostics is not None and is_main_process:
            traj_quality = diagnostics.analyze_trajectory_quality(trajectories)
            if (
                wandb_logger is not None
                and hasattr(wandb_logger, "is_enabled")
                and wandb_logger.is_enabled
            ):
                wandb_logger.log(
                    {f"train/{k}": v for k, v in traj_quality.items()},
                    commit=False,
                )

        # Resume from checkpoint if specified
        start_epoch = 0
        global_step = 0
        if args.resume_from and not args.dry_run:
            checkpoint_storage = GCSCheckpointStorage(config)
            step_to_load: int | None = None
            if args.resume_from.isdigit():
                step_to_load = int(args.resume_from)
            elif "training_step_" in args.resume_from:
                try:
                    step_str = args.resume_from.split("training_step_")[1].replace(".pt", "")
                    step_to_load = int(step_str)
                except (IndexError, ValueError):
                    logger.warning(
                        "Failed to parse resume_from value '%s'; loading latest checkpoint",
                        args.resume_from,
                    )

            checkpoint = checkpoint_storage.load_checkpoint(step=step_to_load)
            if checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                global_step = checkpoint.get("step", 0)
                start_epoch = checkpoint.get("epoch", 0)
                if is_main_process:
                    logger.info(
                        "Resumed from checkpoint at step %d, epoch %d", global_step, start_epoch
                    )

        # Training loop
        start_time = time.time()

        for epoch in range(start_epoch, args.epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)

            epoch_start = time.time()

            # Spatial dims must match encoder output, NOT model embed_dim
            spatial_channels = ENCODER_SPATIAL_CHANNELS
            spatial_size = args.board_size

            if is_muzero:
                loss, policy_loss, value_loss, _reward_loss, num_batches = train_muzero_epoch(
                    model,
                    dataloader,
                    optimizer,
                    device,
                    unroll_steps=getattr(args, "muzero_unroll_steps", 5),
                    discount=getattr(args, "muzero_discount", 0.997),
                    global_step=global_step,
                    wandb_logger=wandb_logger,
                    wandb_log_interval=args.wandb_log_interval,
                    lr_scheduler=lr_scheduler,
                )
            else:
                loss, policy_loss, value_loss, num_batches = train_epoch(
                    model,
                    dataloader,
                    optimizer,
                    device,
                    global_step=global_step,
                    wandb_logger=wandb_logger,
                    spatial_channels=spatial_channels,
                    spatial_size=spatial_size,
                    wandb_log_interval=args.wandb_log_interval,
                    lr_scheduler=lr_scheduler,
                    diagnostics=diagnostics,
                    report=report,
                    profiler=profiler,
                )

            epoch_time = time.time() - epoch_start

            # Log profiler timing summary
            if (
                profiler is not None
                and is_main_process
                and wandb_logger is not None
                and hasattr(wandb_logger, "is_enabled")
                and wandb_logger.is_enabled
            ):
                for section_name in ["forward_pass", "backward_pass"]:
                    summary = profiler.get_summary(section_name)
                    if summary:
                        wandb_logger.log(
                            {f"profiler/{section_name}_mean_ms": summary["mean"] * 1000},
                            commit=False,
                        )

            prev_step = global_step
            global_step += num_batches

            if is_main_process:
                logger.info(
                    "Epoch %d/%d completed: loss=%.4f (policy=%.4f, value=%.4f), "
                    "time=%.2fs, steps %d-%d",
                    epoch + 1,
                    args.epochs,
                    loss,
                    policy_loss,
                    value_loss,
                    epoch_time,
                    prev_step + 1,
                    global_step,
                )

                # Report to Vizier
                report_tuning_metric(
                    {
                        "loss": loss,
                        "policy_loss": policy_loss,
                        "value_loss": value_loss,
                    },
                    step=global_step,
                )

            # Checkpoint if we crossed an interval boundary
            if is_main_process and (global_step // args.checkpoint_interval) > (
                prev_step // args.checkpoint_interval
            ):
                save_checkpoint(config, model, optimizer, global_step, epoch + 1, args.dry_run)

        # Save final model
        total_time = time.time() - start_time
        if is_main_process:
            logger.info("Training complete in %.1f seconds", total_time)

            # Unwrap model if DDP
            model_to_save: torch.nn.Module = model.module if isinstance(model, DDP) else model
            uri, version = save_final_model(config, model_to_save, args.iteration, args.dry_run)
            logger.info("Saved final model version %d to %s", version, uri)

            # Cleanup old checkpoints
            if not args.dry_run:
                checkpoint_storage = GCSCheckpointStorage(config)
                deleted = checkpoint_storage.cleanup_old_checkpoints(keep_last=CHECKPOINT_RETENTION)
                if deleted:
                    logger.info("Cleaned up %d old checkpoints", deleted)

        return 0

    except KeyboardInterrupt:
        logger.warning("Training interrupted")
        return 130
    except Exception as e:
        logger.exception("Training failed: %s", e)
        return 1
    finally:
        cleanup_ddp()
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.finish()


if __name__ == "__main__":
    sys.exit(main())
