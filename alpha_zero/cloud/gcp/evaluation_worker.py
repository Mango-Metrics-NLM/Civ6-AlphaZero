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
from typing import Any

import torch

from alpha_zero.cloud.gcp.arena import run_arena
from alpha_zero.cloud.gcp.config import DEFAULT_MCTS_TYPE, VALID_MCTS_TYPES, GCPConfig
from alpha_zero.cloud.gcp.storage import GCSModelStorage
from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

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
        default=DEFAULT_MCTS_TYPE,
        choices=VALID_MCTS_TYPES,
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

    # Gameplay environment preset
    parser.add_argument(
        "--gameplay_preset",
        type=str,
        default="default",
        choices=["default", "enhanced"],
        help="Gameplay preset: 'default' (MicroCivEnv) or 'enhanced' (all gameplay systems)",
    )

    return parser.parse_args()


def load_model(
    config: GCPConfig,
    version: int,
    dry_run: bool = False,
    *,
    spatial_size: int | None = None,
    action_space_size: int | None = None,
) -> torch.nn.Module:
    """Load model from GCS or create fresh for dry run.

    Args:
        config: GCP configuration.
        version: Model version to load.
        dry_run: If True, create dummy model without GCS access.
        spatial_size: Override spatial_size for dry-run model (keyword-only).
        action_space_size: Override action_space_size for dry-run model (keyword-only).

    Returns:
        Loaded neural network model.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    if dry_run:
        logger.info("Dry run: using fresh model v%d (no GCS load)", version)
        cfg_kwargs: dict[str, Any] = {}
        if spatial_size is not None:
            cfg_kwargs["spatial_size"] = spatial_size
        if action_space_size is not None:
            cfg_kwargs["action_space_size"] = action_space_size
        return Civ6TransformerNetwork(config=TransformerConfig(**cfg_kwargs))

    storage = GCSModelStorage(config)
    state_dict, _metadata, model_config_dict = storage.load_model(version)

    # Build TransformerConfig from saved config or use defaults
    if model_config_dict is not None:
        logger.info("Using model config from checkpoint for v%d", version)
        config_copy = model_config_dict.copy()

        # Convert string enums to actual enum values
        if "norm_type" in config_copy and isinstance(config_copy["norm_type"], str):
            from alpha_zero.core.transformer_network import NormType

            config_copy["norm_type"] = NormType(config_copy["norm_type"])
        if "attention_type" in config_copy and isinstance(config_copy["attention_type"], str):
            from alpha_zero.core.transformer_network import AttentionType

            config_copy["attention_type"] = AttentionType(config_copy["attention_type"])

        # Remove computed fields that shouldn't be passed to constructor
        config_copy.pop("_num_patches", None)
        config_copy.pop("_head_dim", None)

        # Infer head dimensions from state_dict if not in saved config
        # (older models saved config without policy_hidden_dim/value_head_dim)
        if "policy_hidden_dim" not in config_copy and "policy_head.mlp.0.weight" in state_dict:
            inferred = state_dict["policy_head.mlp.0.weight"].shape[0]
            config_copy["policy_hidden_dim"] = inferred
            logger.info("Inferred policy_hidden_dim=%d from state_dict", inferred)
        if "value_head_dim" not in config_copy and "value_head.mlp.0.weight" in state_dict:
            inferred = state_dict["value_head.mlp.0.weight"].shape[0]
            config_copy["value_head_dim"] = inferred
            logger.info("Inferred value_head_dim=%d from state_dict", inferred)

        try:
            transformer_config = TransformerConfig(**config_copy)
        except TypeError as e:
            net_cfg = config.network
            logger.warning("Failed to parse model config (%s), using pipeline defaults", e)
            transformer_config = TransformerConfig(
                spatial_channels=net_cfg.spatial_channels,
                spatial_size=net_cfg.board_size,
                vector_dim=net_cfg.vector_dim,
                action_space_size=net_cfg.action_space_size,
                num_layers=net_cfg.num_layers,
                embed_dim=net_cfg.embed_dim,
            )
    else:
        # Legacy model without config - use NetworkArchitectureConfig as fallback
        net_cfg = config.network
        logger.warning(
            "Model v%d has no saved config. Using pipeline architecture defaults: "
            "spatial_size=%d, action_space=%d, embed_dim=%d, num_layers=%d",
            version,
            net_cfg.board_size,
            net_cfg.action_space_size,
            net_cfg.embed_dim,
            net_cfg.num_layers,
        )
        transformer_config = TransformerConfig(
            spatial_channels=net_cfg.spatial_channels,
            spatial_size=net_cfg.board_size,
            vector_dim=net_cfg.vector_dim,
            action_space_size=net_cfg.action_space_size,
            num_layers=net_cfg.num_layers,
            embed_dim=net_cfg.embed_dim,
        )

    model = Civ6TransformerNetwork(config=transformer_config)
    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        logger.warning("Missing keys in state_dict: %s", incompatible.missing_keys[:5])
    if incompatible.unexpected_keys:
        logger.warning("Unexpected keys in state_dict: %s", incompatible.unexpected_keys[:5])

    model.eval()
    logger.info("Loaded model v%d from GCS", version)
    return model


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
        parsed = json.loads(data.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None
    except GCSStorageError:
        return None


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging()

    # Initialize tracking for evaluation worker (opt-in via MLFLOW_ENABLED env var)
    wandb_logger: WandbLogger | None = None
    try:
        wandb_logger = WandbLogger.from_env()
        if wandb_logger.is_enabled:
            wandb_logger.init(
                config_dict={
                    "mode": "evaluation",
                    "model_version_new": args.model_version_new,
                    "model_version_old": args.model_version_old,
                    "iteration": args.iteration,
                    "games": args.games,
                    "mcts_type": args.mcts_type,
                    "mcts_simulations": args.mcts_simulations,
                    "board_size": args.board_size,
                    "action_space_size": args.action_space_size,
                    "max_turns": args.max_turns,
                }
            )
            logger.info("W&B logging enabled for evaluation worker")
    except Exception as _e:
        logger.debug("W&B initialization failed (non-fatal): %s", _e)
        wandb_logger = None

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
        model_new = load_model(
            config,
            args.model_version_new,
            dry_run=args.dry_run,
            spatial_size=args.board_size,
            action_space_size=args.action_space_size,
        )
        model_old = load_model(
            config,
            args.model_version_old,
            dry_run=args.dry_run,
            spatial_size=args.board_size,
            action_space_size=args.action_space_size,
        )
        model_new = model_new.to(device)
        model_old = model_old.to(device)

        # Run arena
        start_time = time.time()
        results = run_arena(
            model_a=model_new,
            model_b=model_old,
            num_games=args.games,
            mcts_type=args.mcts_type,
            mcts_sims=args.mcts_simulations,
            gumbel_num_actions=args.gumbel_num_actions,
            board_size=args.board_size,
            max_turns=args.max_turns,
            device=device,
            gameplay_preset=getattr(args, "gameplay_preset", "default"),
        )
        elapsed = time.time() - start_time

        logger.info(
            "Arena complete in %.1fs: win_rate=%.2f, draw_rate=%.2f, loss_rate=%.2f",
            elapsed,
            results.win_rate,
            results.draw_rate,
            results.loss_rate,
        )

        # Log evaluation metrics to W&B
        if wandb_logger is not None and wandb_logger.is_enabled:
            games_per_min = results.total_games / (elapsed / 60.0) if elapsed > 0 else 0.0
            wandb_logger.log(
                {
                    "eval/win_rate": results.win_rate,
                    "eval/draw_rate": results.draw_rate,
                    "eval/loss_rate": results.loss_rate,
                    "eval/total_games": results.total_games,
                    "eval/games_per_minute": games_per_min,
                    "eval/elapsed_seconds": elapsed,
                },
                step=args.iteration,
            )

        # Save results
        save_results(
            config=config,
            results=results.to_dict(),
            iteration=args.iteration,
            model_version_new=args.model_version_new,
            model_version_old=args.model_version_old,
            dry_run=args.dry_run,
        )

        # Log final summary to W&B
        if wandb_logger is not None and wandb_logger.is_enabled:
            wandb_logger.log(
                {
                    "eval/model_version_new": args.model_version_new,
                    "eval/model_version_old": args.model_version_old,
                }
            )

        return 0

    except KeyboardInterrupt:
        logger.warning("Evaluation interrupted")
        return 130
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return 1
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()


if __name__ == "__main__":
    sys.exit(main())
