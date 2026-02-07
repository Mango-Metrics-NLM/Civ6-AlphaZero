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
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from alpha_zero.cloud.gcp.config import GCPConfig
from alpha_zero.cloud.gcp.storage import (
    GCSCheckpointStorage,
    GCSModelStorage,
    GCSTrajectoryStorage,
)

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    """Configure logging for training worker."""
    logging.basicConfig(
        level=logging.INFO,
        format="[Training] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


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
    parser.add_argument(
        "--checkpoint_interval", type=int, default=1000, help="Steps between checkpoints"
    )
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--board_size", type=int, default=8, help="Board size")
    parser.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    parser.add_argument("--num_res_blocks", type=int, default=6, help="Residual blocks in network")
    parser.add_argument("--num_filters", type=int, default=64, help="Conv filters in network")
    parser.add_argument("--mode", type=str, default="training", help="Worker mode (training)")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")
    return parser.parse_args()


class TrajectoryDataset(Dataset):
    """Dataset that holds pre-extracted training examples from trajectories."""

    def __init__(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
        policy: torch.Tensor,
        value: torch.Tensor,
    ):
        self.spatial = spatial
        self.vector = vector
        self.action_mask = action_mask
        self.policy = policy
        self.value = value

    def __len__(self) -> int:
        return self.spatial.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.spatial[idx],
            self.vector[idx],
            self.action_mask[idx],
            self.policy[idx],
            self.value[idx],
        )


def load_trajectories(
    config: GCPConfig,
    iterations: list[int],
    max_per_iteration: int = 50000,
    board_size: int = 8,
    action_space_size: int = 381,
    dry_run: bool = False,
) -> list[dict]:
    """Load trajectories from specified iterations."""
    if dry_run:
        logger.info("Dry run: creating dummy trajectories")
        spatial_shape = (13, board_size, board_size)
        trajectories = []
        for _ in range(100):
            n_states = 20
            trajectories.append(
                {
                    "states": [
                        {
                            "spatial": np.random.randn(*spatial_shape).astype(np.float32),
                            "vector": np.random.randn(64).astype(np.float32),
                            "action_mask": np.ones(action_space_size, dtype=np.float32),
                        }
                        for _ in range(n_states)
                    ],
                    "policies": [
                        np.full(action_space_size, 1.0 / action_space_size, dtype=np.float32)
                        for _ in range(n_states)
                    ],
                    "values": [0.5] * n_states,
                }
            )
        return trajectories

    storage = GCSTrajectoryStorage(config)
    all_trajectories = []

    for iteration in iterations:
        trajectories = storage.load_trajectories(iteration, max_examples=max_per_iteration)
        all_trajectories.extend(trajectories)
        logger.info("Loaded %d trajectories from iteration %d", len(trajectories), iteration)

    logger.info("Total trajectories loaded: %d", len(all_trajectories))
    return all_trajectories


def create_model(
    board_size: int = 8,
    action_space_size: int = 381,
    num_res_blocks: int = 6,
    num_filters: int = 64,
) -> torch.nn.Module:
    """Create model architecture."""
    from alpha_zero.core.civ6_network import MicroCivNetwork

    return MicroCivNetwork(
        board_size=board_size,
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
    )


def prepare_dataset(trajectories: list[dict], action_space_size: int) -> TrajectoryDataset:
    """Convert trajectories to PyTorch dataset."""
    all_spatial = []
    all_vector = []
    all_mask = []
    all_policy = []
    all_value = []

    for traj in trajectories:
        states = traj["states"]
        policies = traj["policies"]
        values = traj.get("values", traj.get("rewards", [0.0] * len(states)))

        for i, state in enumerate(states):
            all_spatial.append(state["spatial"])
            all_vector.append(state["vector"])
            all_mask.append(state["action_mask"])

            # Ensure policy has correct size
            policy = policies[i]
            if len(policy) < action_space_size:
                padded = np.zeros(action_space_size, dtype=np.float32)
                padded[: len(policy)] = policy
                policy = padded
            all_policy.append(policy[:action_space_size])

            all_value.append(float(values[i]))

    return TrajectoryDataset(
        spatial=torch.tensor(np.array(all_spatial), dtype=torch.float32),
        vector=torch.tensor(np.array(all_vector), dtype=torch.float32),
        action_mask=torch.tensor(np.array(all_mask), dtype=torch.float32),
        policy=torch.tensor(np.array(all_policy), dtype=torch.float32),
        value=torch.tensor(all_value, dtype=torch.float32),
    )


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_step: int = 0,
    wandb_logger: object | None = None,
) -> tuple[float, float, float, int]:
    """Train for one epoch.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss, num_batches).
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0

    for spatial, vector, action_mask, target_policy, target_value in dataloader:
        spatial = spatial.to(device)
        vector = vector.to(device)
        action_mask = action_mask.to(device)
        target_policy = target_policy.to(device)
        target_value = target_value.to(device)

        # Forward pass with proper 3-input API
        policy_logits, value = model(spatial, vector, action_mask)

        # Policy loss (cross-entropy with target distribution)
        policy_log_probs = torch.log_softmax(policy_logits, dim=-1)
        policy_loss = -torch.sum(target_policy * policy_log_probs, dim=-1).mean()

        # Value loss (MSE)
        value_loss = torch.nn.functional.mse_loss(value, target_value)

        # Combined loss
        loss = policy_loss + value_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1

        # Log to W&B periodically
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            if num_batches % 10 == 0:
                lr = optimizer.param_groups[0]["lr"]
                wandb_logger.log_training_step(
                    step=global_step + num_batches,
                    loss=loss.item(),
                    policy_loss=policy_loss.item(),
                    value_loss=value_loss.item(),
                    learning_rate=lr,
                )

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

    uri = storage.save_model(
        model=model,
        version=new_version,
        metadata={"iteration": iteration},
    )
    logger.info("Saved model v%d to %s", new_version, uri)
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
        from alpha_zero.utils.wandb_logger import WandbLogger

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
                    "mode": "training",
                }
            )
            logger.info("W&B logging enabled")
    except ImportError:
        logger.debug("W&B not available")

    try:
        config = GCPConfig()
        if not args.dry_run:
            config.validate()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Load trajectories
        trajectories = load_trajectories(
            config,
            trajectory_iterations,
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            dry_run=args.dry_run,
        )

        # Prepare dataset
        dataset = prepare_dataset(trajectories, args.action_space_size)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        logger.info("Dataset size: %d examples, %d batches", len(dataset), len(dataloader))

        # Create model
        model = create_model(
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            num_res_blocks=args.num_res_blocks,
            num_filters=args.num_filters,
        )
        model = model.to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

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
                    pass

            checkpoint = checkpoint_storage.load_checkpoint(step=step_to_load)
            if checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                global_step = checkpoint.get("step", 0)
                start_epoch = checkpoint.get("epoch", 0)
                logger.info(
                    "Resumed from checkpoint at step %d, epoch %d", global_step, start_epoch
                )

        # Training loop
        start_time = time.time()

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()

            loss, policy_loss, value_loss, num_batches = train_epoch(
                model,
                dataloader,
                optimizer,
                device,
                global_step=global_step,
                wandb_logger=wandb_logger,
            )

            epoch_time = time.time() - epoch_start
            prev_step = global_step
            global_step += num_batches

            logger.info(
                "Epoch %d/%d - Loss: %.4f (P: %.4f, V: %.4f) - %.1fs - Steps: %d->%d",
                epoch + 1,
                args.epochs,
                loss,
                policy_loss,
                value_loss,
                epoch_time,
                prev_step,
                global_step,
            )

            # Checkpoint if we crossed an interval boundary
            if (global_step // args.checkpoint_interval) > (prev_step // args.checkpoint_interval):
                save_checkpoint(config, model, optimizer, global_step, epoch + 1, args.dry_run)

        # Save final model
        total_time = time.time() - start_time
        logger.info("Training complete in %.1f seconds", total_time)

        uri, version = save_final_model(config, model, args.iteration, args.dry_run)
        logger.info("Saved final model version %d", version)

        # Cleanup old checkpoints
        if not args.dry_run:
            checkpoint_storage = GCSCheckpointStorage(config)
            deleted = checkpoint_storage.cleanup_old_checkpoints(keep_last=3)
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
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.finish()


if __name__ == "__main__":
    sys.exit(main())
