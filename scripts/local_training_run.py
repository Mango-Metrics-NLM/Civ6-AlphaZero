"""
Local 50-Iteration AlphaZero Training Run with MLflow.

Pulls real trajectory data from GCS, trains locally on CPU/GPU,
saves model checkpoints back to GCS, and tracks everything in MLflow.

Usage:
    python scripts/local_training_run.py \
        --iterations 50 \
        --start_iteration 1 \
        --epochs_per_iter 10 \
        --batch_size 128 \
        --learning_rate 0.001 \
        --mlflow_experiment civ6az-local-50iter \
        --mlflow_run_name "local-50iter-$(date +%Y%m%d-%H%M)" \
        --trajectories_per_iter 3

Set env vars before running:
    $env:CIV6AZ_GCS_BUCKET = "civ6az-training"
    $env:CIV6AZ_VERTEX_PROJECT = "hrmtrain"
    $env:CIV6AZ_VERTEX_REGION = "us-central1"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Ensure the project root is importable. The ``sys.path`` mutation MUST
# happen before any ``alpha_zero.*`` import below, hence the ``noqa: E402``
# on the deferred imports.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from alpha_zero.cloud.gcp.config import GCPConfig  # noqa: E402
from alpha_zero.cloud.gcp.training_worker import (  # noqa: E402
    ENCODER_SPATIAL_CHANNELS,
    create_model,
    load_trajectories,
    prepare_dataset,
    save_final_model,
    train_epoch,
)
from alpha_zero.monitoring.config import MonitoringConfig  # noqa: E402
from alpha_zero.monitoring.mlflow_logger import MLflowLogger  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="[LocalTrain] %(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local 50-iteration AlphaZero training with MLflow")
    p.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    p.add_argument(
        "--start_iteration",
        type=int,
        default=1,
        help="First iteration index (GCS trajectory iter_XXX)",
    )
    p.add_argument("--epochs_per_iter", type=int, default=10, help="Epochs per iteration")
    p.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    p.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay")
    p.add_argument("--board_size", type=int, default=8, help="Board size")
    p.add_argument("--action_space_size", type=int, default=376, help="Action space size")
    p.add_argument("--num_res_blocks", type=int, default=6, help="Transformer layers")
    p.add_argument("--num_filters", type=int, default=64, help="Embed dim")
    p.add_argument("--num_heads", type=int, default=8, help="Attention heads")
    p.add_argument(
        "--encoder_type",
        default="patch_embed",
        choices=["patch_embed", "hex_crystalline"],
        help="Spatial encoder (default: patch_embed). Recorded in the checkpoint's "
        "model_config, so the choice survives save/load.",
    )
    p.add_argument(
        "--trajectories_per_iter",
        type=int,
        default=3,
        help="Number of consecutive trajectory iterations to load per training step",
    )
    p.add_argument(
        "--max_traj_per_iter",
        type=int,
        default=50000,
        help="Max trajectory examples per GCS iteration",
    )
    p.add_argument(
        "--fresh_start",
        action="store_true",
        help="Train from random weights (skip champion loading)",
    )
    p.add_argument(
        "--start_model_version",
        type=int,
        default=None,
        help="GCS model version to fine-tune from (default: latest=v268)",
    )
    # MLflow
    p.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="mlruns",
        help="MLflow tracking URI (local dir or http://...)",
    )
    p.add_argument(
        "--mlflow_experiment",
        type=str,
        default="civ6az-local-50iter",
        help="MLflow experiment name",
    )
    p.add_argument(
        "--mlflow_run_name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if omitted)",
    )
    p.add_argument(
        "--save_to_gcs",
        action="store_true",
        help="Save final model back to GCS after each iteration",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # -----------------------------------------------------------------------
    # GCP config — reads CIV6AZ_* env vars
    # -----------------------------------------------------------------------
    os.environ.setdefault("CIV6AZ_GCS_BUCKET", "civ6az-training")
    os.environ.setdefault("CIV6AZ_VERTEX_PROJECT", "hrmtrain")
    os.environ.setdefault("CIV6AZ_VERTEX_REGION", "us-central1")
    config = GCPConfig()

    # -----------------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # -----------------------------------------------------------------------
    # MLflow
    # -----------------------------------------------------------------------
    run_name = args.mlflow_run_name or f"local-{args.iterations}iter-{int(time.time())}"
    mon_cfg = MonitoringConfig(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment,
        run_name=run_name,
        tags={
            "iterations": str(args.iterations),
            "device": str(device),
            "board_size": str(args.board_size),
            "algorithm": "alphazero",
        },
        description=f"Local {args.iterations}-iteration AlphaZero training run",
    )

    with MLflowLogger.from_config(mon_cfg) as ml:
        # Log hyperparameters once
        ml.log_params(
            {
                "iterations": args.iterations,
                "start_iteration": args.start_iteration,
                "epochs_per_iter": args.epochs_per_iter,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "board_size": args.board_size,
                "action_space_size": args.action_space_size,
                "num_res_blocks": args.num_res_blocks,
                "num_filters": args.num_filters,
                "num_heads": args.num_heads,
                "encoder_type": args.encoder_type,
                "trajectories_per_iter": args.trajectories_per_iter,
                "device": str(device),
                "fresh_start": args.fresh_start,
                "start_model_version": args.start_model_version,
            }
        )

        logger.info(
            "MLflow run: %s  experiment: %s  run_id: %s",
            run_name,
            args.mlflow_experiment,
            ml.run_id,
        )

        # -------------------------------------------------------------------
        # Create model
        # -------------------------------------------------------------------
        model = create_model(
            board_size=args.board_size,
            action_space_size=args.action_space_size,
            num_res_blocks=args.num_res_blocks,
            num_filters=args.num_filters,
            num_heads=args.num_heads,
            encoder_type=args.encoder_type,
        )

        # Fine-tune from GCS champion unless --fresh_start
        if not args.fresh_start:
            try:
                # Try worktree version first (has _load_champion_weights)
                try:
                    from alpha_zero.cloud.gcp.training_worker import _load_champion_weights

                    _load_champion_weights(
                        config=config,
                        model=model,
                        start_model_version=args.start_model_version,
                        is_main_process=True,
                    )
                except ImportError:
                    # Inline fallback: load latest champion from GCS directly
                    from alpha_zero.cloud.gcp.storage import GCSModelStorage

                    storage = GCSModelStorage(config)
                    latest = storage.get_latest_version()
                    ver = (
                        args.start_model_version if args.start_model_version is not None else latest
                    )
                    state_dict, metadata, _ = storage.load_model(version=ver)
                    model.load_state_dict(state_dict, strict=True)
                    logger.info(
                        "Loaded champion model v%d (ELO=%s, iter=%s)",
                        ver,
                        metadata.elo,
                        metadata.iteration,
                    )
            except Exception as e:
                logger.warning("Could not load champion weights (%s); starting fresh", e)

        model = model.to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        global_step = 0
        total_start = time.time()

        # -------------------------------------------------------------------
        # 50-iteration loop
        # -------------------------------------------------------------------
        for iteration in range(args.start_iteration, args.start_iteration + args.iterations):
            iter_start = time.time()
            logger.info("=" * 60)
            logger.info("ITERATION %d / %d", iteration, args.start_iteration + args.iterations - 1)

            # Load real GCS trajectories for this window
            traj_iters = list(
                range(
                    max(0, iteration - args.trajectories_per_iter),
                    iteration,
                )
            )
            if not traj_iters:
                traj_iters = [0]

            logger.info("Loading trajectories from GCS iterations: %s", traj_iters)
            try:
                trajectories = load_trajectories(
                    config=config,
                    iterations=traj_iters,
                    max_per_iteration=args.max_traj_per_iter,
                    board_size=args.board_size,
                    action_space_size=args.action_space_size,
                    dry_run=False,
                )
            except Exception as e:
                logger.error("Failed to load trajectories for iter %d: %s", iteration, e)
                ml.log_metric("error/trajectory_load", 1.0, step=iteration)
                continue

            if not trajectories:
                logger.warning("No trajectories found for iter %d, skipping", iteration)
                continue

            dataset = prepare_dataset(
                trajectories,
                action_space_size=args.action_space_size,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(device.type == "cuda"),
            )
            logger.info("Dataset: %d examples, %d batches", len(dataset), len(dataloader))

            ml.log_metrics(
                {
                    "data/trajectory_count": len(trajectories),
                    "data/example_count": len(dataset),
                    "data/batch_count": len(dataloader),
                },
                step=iteration,
            )

            # ---------------------------------------------------------------
            # Train for epochs_per_iter epochs
            # ---------------------------------------------------------------
            epoch_losses: list[float] = []
            epoch_policy_losses: list[float] = []
            epoch_value_losses: list[float] = []

            for epoch in range(args.epochs_per_iter):
                spatial_channels = ENCODER_SPATIAL_CHANNELS
                # Account for Sprint 6 N-player channels (28 if present, else 13)
                if trajectories and "spatial" in trajectories[0]:
                    sample_flat = len(trajectories[0]["spatial"][0])
                    inferred_channels = sample_flat // (args.board_size * args.board_size)
                    if inferred_channels in (13, 28):
                        spatial_channels = inferred_channels

                total_loss, policy_loss, value_loss, n_batches = train_epoch(
                    model=model,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    device=device,
                    global_step=global_step,
                    spatial_channels=spatial_channels,
                    spatial_size=args.board_size,
                )
                global_step += n_batches

                epoch_losses.append(total_loss)
                epoch_policy_losses.append(policy_loss)
                epoch_value_losses.append(value_loss)

                logger.info(
                    "  Iter %d Epoch %d/%d — loss=%.4f  policy=%.4f  value=%.4f",
                    iteration,
                    epoch + 1,
                    args.epochs_per_iter,
                    total_loss,
                    policy_loss,
                    value_loss,
                )

                ml.log_metrics(
                    {
                        "train/loss": total_loss,
                        "train/policy_loss": policy_loss,
                        "train/value_loss": value_loss,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            # ---------------------------------------------------------------
            # Per-iteration summary
            # ---------------------------------------------------------------
            iter_elapsed = time.time() - iter_start
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_policy = sum(epoch_policy_losses) / len(epoch_policy_losses)
            avg_value = sum(epoch_value_losses) / len(epoch_value_losses)

            ml.log_metrics(
                {
                    "iter/avg_loss": avg_loss,
                    "iter/avg_policy_loss": avg_policy,
                    "iter/avg_value_loss": avg_value,
                    "iter/elapsed_seconds": iter_elapsed,
                    "iter/global_step": global_step,
                },
                step=iteration,
            )

            logger.info(
                "Iteration %d done in %.1fs — avg_loss=%.4f",
                iteration,
                iter_elapsed,
                avg_loss,
            )

            # ---------------------------------------------------------------
            # Optionally save model back to GCS
            # ---------------------------------------------------------------
            if args.save_to_gcs:
                try:
                    uri, new_ver = save_final_model(
                        config=config,
                        model=model,
                        iteration=iteration,
                        dry_run=False,
                    )
                    logger.info("Saved model v%d to %s", new_ver, uri)
                    ml.log_metrics({"model/saved_version": new_ver}, step=iteration)
                    ml.log_param("latest_model_version", new_ver)
                except Exception as e:
                    logger.warning("GCS model save failed: %s", e)

        # -------------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------------
        total_elapsed = time.time() - total_start
        ml.log_metrics(
            {
                "run/total_elapsed_seconds": total_elapsed,
                "run/total_global_steps": global_step,
                "run/iterations_completed": args.iterations,
            }
        )
        logger.info(
            "Training complete: %d iterations in %.1fs (%.1f min)",
            args.iterations,
            total_elapsed,
            total_elapsed / 60,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
