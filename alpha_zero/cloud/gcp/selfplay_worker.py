"""
Self-Play Worker Entry Point for Vertex AI.

This is the containerized entry point for self-play workers.
Each worker:
1. Loads the latest model from GCS
2. Runs self-play games using MCTS
3. Writes trajectories to GCS immediately
4. Exits cleanly (stateless design for Spot VM resilience)

Usage:
    python -m alpha_zero.cloud.gcp.selfplay_worker \
        --worker_id=0 \
        --model_version=1 \
        --iteration=10 \
        --games=100
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import sys
import time
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    import numpy as np

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _PSUTIL_AVAILABLE = False

from alpha_zero.cloud.gcp.config import (
    BATCH_MCTS_TYPE,
    DEFAULT_MCTS_TYPE,
    VALID_MCTS_TYPES,
    GCPConfig,
    NetworkArchitectureConfig,
)
from alpha_zero.cloud.gcp.storage import GCSModelStorage, GCSTrajectoryStorage

logger = logging.getLogger(__name__)

# Memory threshold for triggering garbage collection (in MB)
MEMORY_THRESHOLD_MB: int = 6000

# Default network constants (from NetworkArchitectureConfig defaults)
_DEFAULT_NET_CONFIG = NetworkArchitectureConfig()
_SPATIAL_CHANNELS: int = _DEFAULT_NET_CONFIG.spatial_channels
_VECTOR_DIM: int = _DEFAULT_NET_CONFIG.vector_dim
_ACTION_SPACE_SIZE: int = _DEFAULT_NET_CONFIG.action_space_size


def setup_logging(worker_id: int) -> None:
    """Configure logging for worker."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{worker_id:02d}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Civ6AZ Self-Play Worker")
    parser.add_argument("--worker_id", type=int, required=True, help="Worker ID")
    parser.add_argument("--model_version", type=int, required=True, help="Model version to use")
    parser.add_argument("--iteration", type=int, required=True, help="Training iteration")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--agent_id", type=str, default=None, help="PBT Agent ID")
    parser.add_argument("--mode", type=str, default="selfplay", help="Worker mode (selfplay)")

    # MCTS configuration arguments
    parser.add_argument(
        "--mcts_simulations", type=int, default=400, help="MCTS simulations per move"
    )
    parser.add_argument(
        "--mcts_type",
        type=str,
        default=DEFAULT_MCTS_TYPE,
        choices=VALID_MCTS_TYPES,
        help="MCTS variant to use",
    )
    parser.add_argument("--board_size", type=int, default=8, help="Size of the hex board")
    parser.add_argument("--max_turns", type=int, default=30, help="Max turns per game")
    parser.add_argument("--c_puct", type=float, default=1.5, help="MCTS exploration constant")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.3, help="Dirichlet noise alpha")
    parser.add_argument(
        "--dirichlet_epsilon", type=float, default=0.25, help="Dirichlet noise weight"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Action selection temperature"
    )
    parser.add_argument(
        "--temperature_threshold", type=int, default=30, help="Turn for temperature=0"
    )
    parser.add_argument(
        "--temperature_schedule",
        type=str,
        default="threshold",
        choices=["threshold", "linear", "cosine"],
        help="Temperature decay schedule (threshold=step at threshold, linear/cosine=smooth decay)",
    )

    # Additional hyperparameters
    parser.add_argument(
        "--gumbel_num_actions", type=int, default=16, help="Gumbel MCTS sampled actions"
    )
    parser.add_argument(
        "--action_space_size", type=int, default=376, help="Size of the action space"
    )
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
    parser.add_argument("--reward_shaping", action="store_true", help="Enable reward shaping")
    parser.add_argument("--reward_scale", type=float, default=1.0, help="Reward scaling factor")
    parser.add_argument("--dry_run", action="store_true", help="Run without GCS access")

    # Gameplay environment preset
    parser.add_argument(
        "--gameplay_preset",
        type=str,
        default="default",
        choices=["default", "enhanced"],
        help="Gameplay preset: 'default' (MicroCivEnv) or 'enhanced' (all gameplay systems)",
    )

    # Batch inference configuration
    parser.add_argument(
        "--num_parallel_games",
        type=int,
        default=1,
        help="Number of parallel games (>1 enables batch MCTS)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=64,
        help="Maximum batch size for batched network inference",
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


def load_model(
    config: GCPConfig,
    version: int,
    dry_run: bool = False,
    *,
    spatial_size: int = 8,
    action_space_size: int = 376,
) -> torch.nn.Module:
    """Load model from GCS or create dummy for dry run.

    Args:
        config: GCP configuration.
        version: Model version to load.
        dry_run: If True, create dummy model without GCS access.
        spatial_size: Board spatial size for the dry-run dummy model (keyword-only).
        action_space_size: Action space size for the dry-run dummy model (keyword-only).

    Returns:
        Loaded neural network model.

    Raises:
        RuntimeError: If model loading fails.
    """
    from alpha_zero.core.transformer_network import Civ6TransformerNetwork, TransformerConfig

    if dry_run:
        logger.info("Dry run: creating dummy model")
        # Create a minimal dummy model for testing using CLI-specified dimensions
        dummy_config = TransformerConfig(
            spatial_channels=13,
            spatial_size=spatial_size,
            vector_dim=64,
            action_space_size=action_space_size,
        )
        return Civ6TransformerNetwork(config=dummy_config)

    # Load from GCS
    storage = GCSModelStorage(config)
    try:
        state_dict, metadata, model_config_dict = storage.load_model(version)
    except Exception as e:
        logger.error("Failed to download model v%d from GCS: %s", version, e)
        raise RuntimeError(f"Failed to load model v{version} from GCS") from e

    # Build TransformerConfig from saved config or use defaults
    if model_config_dict is not None:
        logger.info("Using model config from checkpoint")
        # Handle enum conversions if needed
        config_copy = model_config_dict.copy()

        # Enum re-hydration and dropping the init=False derived fields is shared
        # with the other checkpoint readers; see TransformerConfig.from_saved_dict.
        config_copy = TransformerConfig.normalize_saved_dict(config_copy)

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
            "spatial_size=%d, action_space=%d, embed_dim=%d, num_layers=%d, encoder=%s",
            version,
            net_cfg.board_size,
            net_cfg.action_space_size,
            net_cfg.embed_dim,
            net_cfg.num_layers,
            net_cfg.encoder_type,
        )
        transformer_config = TransformerConfig(
            spatial_channels=net_cfg.spatial_channels,
            spatial_size=net_cfg.board_size,
            vector_dim=net_cfg.vector_dim,
            action_space_size=net_cfg.action_space_size,
            num_layers=net_cfg.num_layers,
            embed_dim=net_cfg.embed_dim,
            encoder_type=net_cfg.encoder_type,
        )

    # Create model and load state dict
    try:
        model = Civ6TransformerNetwork(config=transformer_config)
        incompatible = model.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning("Missing keys in state_dict: %s", incompatible.missing_keys[:5])
        if incompatible.unexpected_keys:
            logger.warning("Unexpected keys in state_dict: %s", incompatible.unexpected_keys[:5])

        # Detect silently cross-loaded encoder weights (PR #107 follow-up).
        from alpha_zero.utils.checkpoint import validate_encoder_state

        expected_encoder = getattr(transformer_config, "encoder_type", "patch_embed")
        validate_encoder_state(
            model,
            expected_encoder,
            checkpoint_keys=state_dict.keys(),
            strict=False,  # warn, don't crash — operator decides
        )

    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(
                "Model architecture mismatch for v%d. The saved model has different "
                "dimensions than expected. Ensure model config is saved with checkpoints.",
                version,
            )
        raise RuntimeError(f"Failed to load state_dict for v{version}: {e}") from e

    model.eval()
    logger.info("Loaded model v%d (ELO: %s)", version, metadata.elo)
    return model


def load_muzero_model(
    config: GCPConfig,
    version: int,
    board_size: int = 8,
    action_space_size: int = 376,
    num_res_blocks: int = 6,
    num_filters: int = 64,
    dry_run: bool = False,
) -> torch.nn.Module:
    """Load MuZero model from GCS or create dummy for dry run.

    Args:
        config: GCP configuration.
        version: Model version to load.
        board_size: Board spatial size (H=W).
        action_space_size: Total number of actions.
        num_res_blocks: Number of residual blocks.
        num_filters: Hidden channel dimension.
        dry_run: If True, create dummy model without GCS access.

    Returns:
        Loaded MuZero neural network model.

    Raises:
        RuntimeError: If model loading fails.
    """
    from alpha_zero.core.muzero_network import MuZeroConfig, MuZeroNetwork

    muzero_config = MuZeroConfig(
        spatial_channels=_SPATIAL_CHANNELS,
        spatial_size=board_size,
        vector_dim=_VECTOR_DIM,
        action_space_size=action_space_size,
        num_res_blocks=num_res_blocks,
        hidden_state_channels=num_filters,
    )

    if dry_run:
        logger.info("Dry run: creating dummy MuZero model")
        return MuZeroNetwork(config=muzero_config)

    # Load from GCS
    storage = GCSModelStorage(config)
    try:
        state_dict, metadata, model_config_dict = storage.load_model(version)
    except Exception as e:
        logger.error("Failed to download MuZero model v%d from GCS: %s", version, e)
        raise RuntimeError(f"Failed to load MuZero model v{version} from GCS") from e

    # Reconstruct config from saved config if available
    if model_config_dict is not None:
        logger.info("Using saved MuZero model config from checkpoint")
        config_copy = model_config_dict.copy()
        # Filter to only known MuZeroConfig fields (tolerates ViT fields like patch_size)
        import dataclasses as _dc

        valid_fields = {f.name for f in _dc.fields(MuZeroConfig)}
        unknown_keys = [k for k in config_copy if k not in valid_fields]
        if unknown_keys:
            logger.info("Filtering %d unknown config keys: %s", len(unknown_keys), unknown_keys[:5])
        config_copy = {k: v for k, v in config_copy.items() if k in valid_fields}
        try:
            muzero_config = MuZeroConfig(**config_copy)
        except TypeError as e:
            logger.warning("Failed to parse saved MuZero config (%s), using CLI defaults", e)

    # Create model and load state dict
    try:
        model = MuZeroNetwork(config=muzero_config)
        incompatible = model.load_state_dict(state_dict, strict=False)

        # Check for severe architecture mismatch (e.g. ViT checkpoint vs CNN MuZero)
        total_keys = len(state_dict)
        mismatch_keys = len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
        mismatch_ratio = mismatch_keys / max(total_keys, 1)

        if mismatch_ratio > 0.2:
            logger.warning(
                "Architecture mismatch: %d/%d keys incompatible (%.0f%%). "
                "Checkpoint appears to be a different architecture (e.g. ViT vs CNN). "
                "Using fresh random MuZero weights instead.",
                mismatch_keys,
                total_keys,
                mismatch_ratio * 100,
            )
            model = MuZeroNetwork(config=muzero_config)
            model.eval()
            logger.info("Created fresh MuZero model (random init) for self-play")
            return model

        if incompatible.missing_keys:
            logger.warning("Missing keys in MuZero state_dict: %s", incompatible.missing_keys[:5])
        if incompatible.unexpected_keys:
            logger.warning(
                "Unexpected keys in MuZero state_dict: %s",
                incompatible.unexpected_keys[:5],
            )

        # MuZero uses CNN, not hex encoder — but still validate that encoder
        # weights were actually loaded, not left randomly initialised.
        from alpha_zero.utils.checkpoint import classify_encoder_keys, validate_encoder_state

        ckpt_encoder = classify_encoder_keys(state_dict.keys())
        validate_encoder_state(
            model,
            ckpt_encoder,
            checkpoint_keys=state_dict.keys(),
            strict=False,  # warn, don't crash
        )

    except RuntimeError as e:
        if "size mismatch" in str(e):
            logger.error(
                "MuZero architecture mismatch for v%d. Ensure config is saved with checkpoints.",
                version,
            )
        raise RuntimeError(f"Failed to load MuZero state_dict v{version}: {e}") from e

    model.eval()
    logger.info("Loaded MuZero model v%d (ELO: %s)", version, metadata.elo)
    return model


def compute_temperature(
    turn: int,
    max_turns: int,
    temperature: float,
    temperature_threshold: int,
    schedule: str = "threshold",
) -> float:
    """Compute action selection temperature for a given turn.

    Args:
        turn: Current turn number (0-indexed).
        max_turns: Maximum turns in the game.
        temperature: Base temperature value.
        temperature_threshold: Turn at which threshold schedule drops to 0.
        schedule: Schedule type ("threshold", "linear", "cosine").

    Returns:
        Temperature value for this turn.
    """
    if schedule == "threshold":
        return temperature if turn < temperature_threshold else 0.0
    elif schedule == "linear":
        # Linear decay from temperature to 0 over max_turns
        progress = min(turn / max(max_turns, 1), 1.0)
        return temperature * (1.0 - progress)
    elif schedule == "cosine":
        # Cosine annealing from temperature to 0
        progress = min(turn / max(max_turns, 1), 1.0)
        return temperature * 0.5 * (1.0 + math.cos(math.pi * progress))
    else:
        return temperature


def outcome_value_for_player(winner: int | None, player: int) -> float:
    """Classic AlphaZero value target for one position, from ``player``'s view.

    ``+1`` if ``player`` won, ``-1`` if another player won, ``0`` for a draw.

    This exists because the game outcome must be signed **per position**. The
    terminal reward returned by ``env.step`` is relative to whoever was
    ``current_player`` at the terminal action, so broadcasting it to every
    position gives roughly half of them the wrong sign in a two-player zero-sum
    game. ``env.winner`` is absolute and therefore frame-independent.

    Matches the ``winner_take_all`` convention the local pipeline already uses
    (``alpha_zero/training_civ6.py``).

    Args:
        winner: Winning player index, or ``None`` for a draw.
        player: The player to move at the position being labelled.

    Returns:
        The value target in ``[-1.0, 1.0]``.
    """
    if winner is None:
        return 0.0
    return 1.0 if int(winner) == int(player) else -1.0


# --- Trajectory construction -------------------------------------------------
#
# All three self-play paths (sequential, MuZero, batch) persist the same base
# trajectory shape. They previously built it inline, three times over, and the
# batch path silently omitted ``action_masks`` -- which the trainer then
# substitutes with an all-ones mask, disabling masked policy loss for every
# batch-generated sample. These helpers give the shape one definition so a path
# cannot forget a field.
#
# The stored object stays a plain ``dict``: shards are pickled
# (``gcp/storage.py``) and read back across a rolling window of iterations, so a
# dataclass here would be a cross-version unpickling hazard.

_TRAJECTORY_FIELDS: tuple[str, ...] = (
    "spatial",
    "vector",
    "action_masks",
    "policies",
    "actions",
    "players",
)


def should_use_batch(args: argparse.Namespace) -> bool:
    """Whether this invocation runs vectorized (batched) self-play.

    Batching requires **both** more than one parallel game and the batch MCTS
    variant; it is additionally disabled under ``--dry_run``, which returns
    synthetic trajectories without building an engine. Any other combination
    runs sequential self-play, so a parallelism request paired with a
    non-batch ``mcts_type`` is silently ignored -- ``main`` warns when that
    happens.

    Args:
        args: Parsed worker arguments.

    Returns:
        ``True`` when :func:`run_batch_selfplay` should drive this run.
    """
    return (
        int(getattr(args, "num_parallel_games", 1) or 1) > 1
        and getattr(args, "mcts_type", DEFAULT_MCTS_TYPE) == BATCH_MCTS_TYPE
        and not getattr(args, "dry_run", False)
    )


def new_trajectory(algorithm: str | None = None) -> dict[str, Any]:
    """Create an empty trajectory with every per-step field present.

    Args:
        algorithm: Optional algorithm tag (e.g. ``"muzero"``) recorded on the
            trajectory and used by the trainer to select its target builder.
            MuZero trajectories also carry ``reward_targets``/``value_targets``.

    Returns:
        A trajectory dict whose per-step lists are all empty.
    """
    trajectory: dict[str, Any] = {field: [] for field in _TRAJECTORY_FIELDS}
    if algorithm == "muzero":
        trajectory["reward_targets"] = []
        trajectory["value_targets"] = []
        trajectory["algorithm"] = algorithm
    elif algorithm is not None:
        trajectory["algorithm"] = algorithm
    return trajectory


def append_step(
    trajectory: dict[str, Any],
    *,
    observation: dict[str, np.ndarray],
    policy: Any,
    action: int,
    player: int,
) -> None:
    """Append one position to ``trajectory``, writing every per-step field.

    Callers must invoke this only after the search that produced ``policy``
    has succeeded, so a failed search never leaves the per-step lists ragged
    (unequal lengths make ``prepare_dataset`` index past the shorter list).

    Args:
        trajectory: Trajectory from :func:`new_trajectory`, mutated in place.
        observation: The ``env.get_observation()`` dict for this position,
            carrying ``spatial``, ``vector`` and ``action_mask``.
        policy: Search policy for this position (array-like or list).
        action: The action selected at this position.
        player: The player to move at this position (see
            :func:`outcome_value_for_player`).
    """
    # Build the whole row before touching the trajectory. Appending field by
    # field would leave the lists ragged if a later field raised (a malformed
    # observation missing "action_mask", say) -- which is the exact state this
    # helper exists to make unrepresentable.
    row = {
        "spatial": observation["spatial"].flatten().tolist(),
        "vector": observation["vector"].flatten().tolist(),
        "action_masks": observation["action_mask"].flatten().tolist(),
        "policies": policy.tolist() if hasattr(policy, "tolist") else policy,
        "actions": int(action),
        "players": int(player),
    }
    for field, value in row.items():
        trajectory[field].append(value)


def dummy_trajectory(
    board_size: int, num_steps: int = 5, algorithm: str | None = None
) -> dict[str, Any]:
    """Build one dry-run trajectory matching the real per-step shapes.

    Dimensions come from :class:`NetworkArchitectureConfig` rather than
    literals so a dry run cannot emit, say, a 100-wide policy against a
    376-wide action space.

    Args:
        board_size: Board edge length used for the spatial plane size.
        num_steps: Number of dummy positions to emit.
        algorithm: Optional algorithm tag; ``"muzero"`` adds MuZero targets.

    Returns:
        A fully populated trajectory dict.
    """
    trajectory = new_trajectory(algorithm)
    trajectory["spatial"] = [
        [0.0] * (_SPATIAL_CHANNELS * board_size * board_size) for _ in range(num_steps)
    ]
    trajectory["vector"] = [[0.0] * _VECTOR_DIM for _ in range(num_steps)]
    trajectory["action_masks"] = [[1.0] * _ACTION_SPACE_SIZE for _ in range(num_steps)]
    trajectory["policies"] = [
        [1.0 / _ACTION_SPACE_SIZE] * _ACTION_SPACE_SIZE for _ in range(num_steps)
    ]
    trajectory["actions"] = [0] * num_steps
    trajectory["players"] = [i % 2 for i in range(num_steps)]
    trajectory["rewards"] = [1.0 if i % 2 == 0 else -1.0 for i in range(num_steps)]
    if algorithm == "muzero":
        trajectory["reward_targets"] = [0.0] * num_steps
        trajectory["value_targets"] = [1.0 if i % 2 == 0 else -1.0 for i in range(num_steps)]
    return trajectory


def run_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    mcts_type: str = "civ6",
    board_size: int = 8,
    max_turns: int = 30,
    gumbel_num_actions: int = 16,
    mcts_config_override: dict[str, Any] | None = None,
    dry_run: bool = False,
    device: str = "cpu",
    temperature_schedule: str = "threshold",
    gameplay_preset: str = "default",
    seed: int | None = None,
) -> list[dict]:
    """Run self-play games and collect trajectories.

    Args:
        model: Neural network model for MCTS evaluation.
        num_games: Number of self-play games to run.
        mcts_sims: Number of MCTS simulations per move.
        mcts_type: MCTS variant to use ("civ6" or "gumbel").
        board_size: Size of the game board.
        max_turns: Maximum turns per game.
        gumbel_num_actions: Number of sampled actions for Gumbel MCTS.
        mcts_config_override: Optional dict of MCTS config overrides
            (c_puct, dirichlet_alpha, dirichlet_epsilon, temperature, temperature_threshold).
        dry_run: If True, create dummy trajectories without running games.
        device: Device for neural network inference (e.g., "cuda" or "cpu").
        seed: Base seed for environment generation. Each game ``i`` is seeded
            with ``seed + i``, so a run is reproducible while games still
            differ from one another. ``None`` (the default) leaves environments
            unseeded, preserving historical behavior.

            Note that seeding the global RNGs is not sufficient on its own:
            ``MicroCivEnv`` holds its own ``numpy`` generator, so without this
            argument the game stream varies between runs even under a fixed
            global seed.

    Returns:
        List of trajectory dictionaries, each containing:
        - spatial: list of spatial features (flattened)
        - vector: list of vector features (flattened)
        - policies: MCTS search policies (action probability distributions)
        - actions: actions taken at each step
        - rewards: game outcome signed per player-to-move (+1/-1/0)
    """
    # For dry run, return dummy trajectories without needing actual env/MCTS
    if dry_run:
        logger.info("Dry run: creating dummy trajectories")
        return [dummy_trajectory(board_size) for _ in range(num_games)]

    # Extract temperature params from override before the game loop
    # These are action selection params, not MCTS config fields
    # Shallow copy to avoid mutating the caller-provided dict
    sel_temperature = 1.0
    sel_threshold = max_turns
    if mcts_config_override:
        mcts_config_override = dict(mcts_config_override)
        sel_temperature = mcts_config_override.pop("temperature", 1.0)
        sel_threshold = mcts_config_override.pop("temperature_threshold", max_turns)

    logger.info(
        "Temperature schedule: %s (temp=%.2f, threshold=%d, max_turns=%d)",
        temperature_schedule,
        sel_temperature,
        sel_threshold,
        max_turns,
    )

    trajectories: list[dict[str, Any]] = []
    failed_games = 0

    for game_idx in range(num_games):
        start_time = time.time()
        game_failed = False

        # Create environment and MCTS
        from alpha_zero.core.mcts_factory import create_mcts
        from alpha_zero.envs.env_factory import create_game_env

        env = create_game_env(
            board_size=board_size,
            max_turns=max_turns,
            gameplay_preset=gameplay_preset,
            seed=None if seed is None else seed + game_idx,
        )

        # Build variant-aware MCTS kwargs to avoid passing invalid params
        puct_variants = {"puct", "civ6"}
        gumbel_variants = {"gumbel"}

        mcts_kwargs: dict[str, Any] = {
            "num_simulations": mcts_sims,
        }

        if mcts_type in puct_variants:
            mcts_kwargs.update(
                {
                    "c_puct": 1.5,
                    "dirichlet_alpha": 0.3,
                    "dirichlet_epsilon": 0.25,
                }
            )
        elif mcts_type in gumbel_variants:
            mcts_kwargs["num_sampled_actions"] = mcts_kwargs.get(
                "num_sampled_actions",
                gumbel_num_actions,
            )

        # Apply remaining MCTS config overrides (temperature already extracted)
        # Filter to only keys valid for the current MCTS variant to avoid
        # passing e.g. c_puct/dirichlet_* to GumbelMCTS
        civ6_only_keys = {"c_puct", "dirichlet_alpha", "dirichlet_epsilon"}
        gumbel_only_keys = {"num_sampled_actions", "c_visit", "c_scale"}
        if mcts_config_override:
            for k, v in mcts_config_override.items():
                if v is None:
                    continue
                # Skip keys that belong to a different MCTS variant
                if mcts_type in gumbel_variants and k in civ6_only_keys:
                    continue
                if mcts_type in puct_variants and k in gumbel_only_keys:
                    continue
                mcts_kwargs[k] = v

        mcts = create_mcts(
            variant=mcts_type,
            network=model,
            device=device,
            **mcts_kwargs,
        )

        trajectory: dict[str, Any] = new_trajectory()

        env.reset()
        done = False

        while not done:
            # Get observation with spatial, vector, action_mask
            obs = env.get_observation()

            # Run MCTS with error handling
            try:
                # Pass env directly since MicroCivEnv implements MCTSEnvironment protocol
                policy, _ = mcts.search(env)
                turn = len(trajectory["spatial"])
                temp = compute_temperature(
                    turn=turn,
                    max_turns=max_turns,
                    temperature=sel_temperature,
                    temperature_threshold=sel_threshold,
                    schedule=temperature_schedule,
                )
                action = mcts.select_action(policy, temperature=temp)
            except Exception as e:
                logger.error(
                    "MCTS search failed at step %d: %s",
                    len(trajectory["spatial"]),
                    e,
                )
                if device != "cpu":
                    torch.cuda.empty_cache()
                game_failed = True
                break  # Skip this game, move to next

            # Store experience. Player to move at this position: MicroCiv
            # advances current_player only inside _end_turn, so a player takes
            # many actions per turn and the acting player cannot be recovered
            # from the step index.
            append_step(
                trajectory,
                observation=obs,
                policy=policy,
                action=action,
                player=env.current_player,
            )

            # Take action. The step reward is deliberately not captured: the
            # value target comes from the absolute winner, not from a reward
            # whose sign depends on whose turn it was at termination.
            step_result = env.step(action)
            done = bool(step_result[2]) or bool(step_result[3])  # terminated or truncated
            final_info = step_result[4] if len(step_result) > 4 else {}

        # Track failed games
        if game_failed:
            failed_games += 1
            logger.warning("Game %d/%d failed, skipping", game_idx + 1, num_games)
        # Sign the game outcome per player-to-move (only if we collected data).
        elif trajectory["spatial"]:
            winner = final_info.get("winner", getattr(env, "winner", None))
            trajectory["rewards"] = [
                outcome_value_for_player(winner, player) for player in trajectory["players"]
            ]
            trajectories.append(trajectory)

        elapsed = time.time() - start_time
        if not game_failed:
            logger.debug("Game %d/%d completed in %.1fs", game_idx + 1, num_games, elapsed)

        # Cleanup after each game to prevent memory accumulation
        del trajectory
        del mcts
        del env
        if device != "cpu":
            torch.cuda.empty_cache()

        # Log progress and memory usage periodically
        if (game_idx + 1) % 10 == 0:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024 if _PSUTIL_AVAILABLE else 0.0
            logger.info(
                "Progress: %d/%d games, Memory: %.0f MB",
                game_idx + 1,
                num_games,
                mem_mb,
            )
            # Force garbage collection if memory is getting high
            if mem_mb > MEMORY_THRESHOLD_MB:
                logger.warning("High memory usage (%.0f MB), forcing GC", mem_mb)
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

    # Log final statistics including any failed games
    if failed_games > 0:
        logger.warning(
            "Self-play completed with %d/%d games failed",
            failed_games,
            num_games,
        )

    return trajectories


def run_muzero_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    board_size: int = 8,
    max_turns: int = 30,
    mcts_config_override: dict[str, Any] | None = None,
    dry_run: bool = False,
    device: str = "cpu",
    temperature_schedule: str = "threshold",
    gameplay_preset: str = "default",
) -> list[dict]:
    """Run MuZero self-play games using learned dynamics model.

    Unlike AlphaZero self-play, MuZero MCTS searches using the dynamics
    model (no environment cloning). Trajectories include additional
    reward_targets and value_targets for multi-step training.

    Args:
        model: MuZero neural network with representation, dynamics, prediction.
        num_games: Number of self-play games to run.
        mcts_sims: Number of MCTS simulations per move.
        board_size: Size of the game board.
        max_turns: Maximum turns per game.
        mcts_config_override: Optional dict of MCTS config overrides.
        dry_run: If True, create dummy trajectories.
        device: Device for neural network inference.
        temperature_schedule: Temperature decay schedule type.

    Returns:
        List of trajectory dictionaries with MuZero-specific fields:
        - spatial, vector, policies, actions, players, rewards (same as AlphaZero)
        - reward_targets: per-step reward predictions for training
        - value_targets: per-step value targets for training
    """
    if dry_run:
        logger.info("Dry run: creating dummy MuZero trajectories")
        return [dummy_trajectory(board_size, algorithm="muzero") for _ in range(num_games)]

    # Extract temperature params
    sel_temperature = 1.0
    sel_threshold = max_turns
    if mcts_config_override:
        mcts_config_override = dict(mcts_config_override)
        sel_temperature = mcts_config_override.pop("temperature", 1.0)
        sel_threshold = mcts_config_override.pop("temperature_threshold", max_turns)

    logger.info(
        "MuZero self-play: %d games, %d sims, device=%s, schedule=%s",
        num_games,
        mcts_sims,
        device,
        temperature_schedule,
    )

    trajectories: list[dict[str, Any]] = []
    failed_games = 0

    for game_idx in range(num_games):
        start_time = time.time()
        game_failed = False

        # Create environment and MuZero MCTS
        from alpha_zero.core.mcts_factory import create_mcts
        from alpha_zero.envs.env_factory import create_game_env

        env = create_game_env(
            board_size=board_size,
            max_turns=max_turns,
            gameplay_preset=gameplay_preset,
        )

        # MuZero MCTS uses the dynamics model for tree search
        mcts_kwargs: dict[str, Any] = {
            "num_simulations": mcts_sims,
        }
        # Apply overrides (skip variant-specific filtering since muzero has its own config)
        if mcts_config_override:
            for k, v in mcts_config_override.items():
                if v is not None:
                    mcts_kwargs[k] = v

        mcts = create_mcts(
            variant="muzero",
            network=model,
            device=device,
            **mcts_kwargs,
        )

        trajectory: dict[str, Any] = new_trajectory(algorithm="muzero")

        env.reset()
        done = False

        while not done:
            obs = env.get_observation()

            try:
                # MuZeroMCTS.search needs observation dict + action mask. The
                # mask is already a key in the observation produced by
                # env.get_observation() above, so reuse it instead of calling
                # env.get_valid_actions() a second time.
                action_mask = obs["action_mask"]
                policy, value_estimate = mcts.search(obs, action_mask)
                turn = len(trajectory["spatial"])
                temp = compute_temperature(
                    turn=turn,
                    max_turns=max_turns,
                    temperature=sel_temperature,
                    temperature_threshold=sel_threshold,
                    schedule=temperature_schedule,
                )
                action = mcts.select_action(policy, temperature=temp)
            except Exception as e:
                logger.error(
                    "MuZero MCTS search failed at step %d: %s",
                    len(trajectory["spatial"]),
                    e,
                )
                if device != "cpu":
                    torch.cuda.empty_cache()
                game_failed = True
                break

            # Store experience (player to move: see outcome_value_for_player).
            append_step(
                trajectory,
                observation=obs,
                policy=policy,
                action=action,
                player=env.current_player,
            )

            # Store value estimate from MCTS root for training targets
            trajectory["value_targets"].append(
                float(value_estimate) if value_estimate is not None else 0.0
            )

            # Take action
            step_result = env.step(action)
            reward = float(step_result[1])
            done = bool(step_result[2]) or bool(step_result[3])
            final_info = step_result[4] if len(step_result) > 4 else {}
            trajectory["reward_targets"].append(reward)

        # Handle completed game
        if game_failed:
            failed_games += 1
            logger.warning("MuZero game %d/%d failed", game_idx + 1, num_games)
        elif trajectory["spatial"]:
            winner = final_info.get("winner", getattr(env, "winner", None))
            trajectory["rewards"] = [
                outcome_value_for_player(winner, player) for player in trajectory["players"]
            ]
            trajectories.append(trajectory)

        elapsed = time.time() - start_time
        if not game_failed:
            logger.debug("MuZero game %d/%d completed in %.1fs", game_idx + 1, num_games, elapsed)

        # Cleanup
        del trajectory
        del mcts
        del env
        if device != "cpu":
            torch.cuda.empty_cache()

        # Log progress periodically
        if (game_idx + 1) % 10 == 0:
            mem_mb = psutil.Process().memory_info().rss / 1024 / 1024 if _PSUTIL_AVAILABLE else 0.0
            logger.info(
                "MuZero progress: %d/%d games, Memory: %.0f MB",
                game_idx + 1,
                num_games,
                mem_mb,
            )
            if mem_mb > MEMORY_THRESHOLD_MB:
                logger.warning("High memory usage (%.0f MB), forcing GC", mem_mb)
                gc.collect()
                if device != "cpu":
                    torch.cuda.empty_cache()

    if failed_games > 0:
        logger.warning("MuZero self-play: %d/%d games failed", failed_games, num_games)

    return trajectories


def run_batch_selfplay(
    model: torch.nn.Module,
    num_games: int,
    mcts_sims: int,
    board_size: int = 8,
    max_turns: int = 30,
    num_parallel_games: int = 8,
    max_batch_size: int = 64,
    gumbel_num_actions: int = 16,
    device: str = "cpu",
    temperature_schedule: str = "threshold",
    temperature: float = 1.0,
    temperature_threshold: int = 30,
    gameplay_preset: str = "default",
) -> list[dict]:
    """Run batch self-play games using BatchGumbelMCTS and vectorized environments.

    Uses BatchGumbelMCTS with VectorizedMicroCivEnv to play multiple games
    simultaneously, batching network evaluations for higher GPU utilization.

    Args:
        model: Neural network model for MCTS evaluation.
        num_games: Total number of self-play games to run.
        mcts_sims: Number of MCTS simulations per move.
        board_size: Size of the game board.
        max_turns: Maximum turns per game.
        num_parallel_games: Number of games to run in parallel.
        max_batch_size: Maximum batch size for network inference.
        gumbel_num_actions: Number of sampled actions for Gumbel MCTS.
        device: Device for neural network inference (e.g., "cuda" or "cpu").
        temperature_schedule: Temperature decay schedule type.
        temperature: Base temperature for action selection.
        temperature_threshold: Turn for temperature=0 (threshold schedule).

    Returns:
        List of trajectory dictionaries (same format as run_selfplay).
    """
    import numpy as np

    from alpha_zero.core.batch_mcts import BatchGumbelMCTS, BatchMCTSConfig
    from alpha_zero.core.civ6_mcts import MCTSNetwork
    from alpha_zero.envs.vectorized_env import VectorizedEnvConfig, VectorizedMicroCivEnv

    logger.info(
        "Starting batch self-play: %d games, %d parallel, %d sims, device=%s",
        num_games,
        num_parallel_games,
        mcts_sims,
        device,
    )

    # Create vectorized environment
    env_config = VectorizedEnvConfig(
        num_envs=num_parallel_games,
        board_size=board_size,
        max_turns=max_turns,
        gameplay_preset=gameplay_preset,
    )
    vec_env = VectorizedMicroCivEnv(config=env_config)

    # Create batch MCTS
    mcts_config = BatchMCTSConfig(
        num_simulations=mcts_sims,
        num_sampled_actions=gumbel_num_actions,
        max_batch_size=max_batch_size,
        num_parallel_envs=num_parallel_games,
    )
    batch_mcts = BatchGumbelMCTS(
        # ``model`` is typed as a generic nn.Module; the concrete networks
        # implement the MCTSNetwork protocol (predict/eval) at runtime.
        network=cast(MCTSNetwork, model),
        config=mcts_config,
        device=device,
    )

    trajectories: list[dict[str, Any]] = []
    games_completed = 0
    batch_num = 0
    start_time = time.time()

    while games_completed < num_games:
        batch_start = time.time()
        batch_num += 1

        # Determine how many games to run in this batch
        remaining = num_games - games_completed
        batch_size = min(remaining, num_parallel_games)

        # Reset environments
        vec_env.reset_all()

        # Initialize per-env trajectory storage
        batch_trajectories: list[dict[str, Any]] = [new_trajectory() for _ in range(batch_size)]

        # Track which envs are still active
        active = [True] * batch_size + [False] * (num_parallel_games - batch_size)
        turn = 0

        while any(active) and turn < max_turns:
            # Get active environment indices
            active_indices = [i for i in range(num_parallel_games) if active[i]]
            if not active_indices:
                break

            # Collect observations for active envs before search. These are
            # buffered rather than appended: a failed search below breaks out
            # of the loop, and appending here would leave spatial/vector longer
            # than policies/actions/players -- a ragged trajectory that makes
            # the trainer index past the shorter lists.
            pending_observations: dict[int, dict[str, np.ndarray]] = {
                i: vec_env.get_env(i).get_observation() for i in active_indices
            }

            # Get environments for batch search
            envs = [vec_env.get_env(i) for i in active_indices]

            # Batch MCTS search
            try:
                results = batch_mcts.batch_search(envs, add_noise=False)  # type: ignore[arg-type]
            except Exception as e:
                logger.error("Batch search failed at turn %d: %s", turn, e, exc_info=True)
                if device != "cpu":
                    torch.cuda.empty_cache()
                break

            # Build actions array and store policies
            actions = np.zeros(num_parallel_games, dtype=np.int32)
            for idx, env_idx in enumerate(active_indices):
                action_probs, _ = results[idx]

                # Compute temperature for this turn
                temp = compute_temperature(
                    turn=turn,
                    max_turns=max_turns,
                    temperature=temperature,
                    temperature_threshold=temperature_threshold,
                    schedule=temperature_schedule,
                )
                action = batch_mcts.select_action(action_probs, temperature=temp)
                actions[env_idx] = action

                # Commit the buffered observation together with the search
                # result, so every per-step field advances in lockstep.
                # Player to move at this position: see outcome_value_for_player.
                append_step(
                    batch_trajectories[env_idx],
                    observation=pending_observations[env_idx],
                    policy=action_probs,
                    action=action,
                    player=vec_env.get_env(env_idx).current_player,
                )

            # Step all environments
            _rewards, terminated, truncated, _ = vec_env.step_all(actions)

            # Update active mask
            for i in active_indices:
                if terminated[i] or truncated[i]:
                    active[i] = False

            turn += 1

        # Finalize trajectories for completed games
        batch_games = 0
        for i in range(batch_size):
            traj = batch_trajectories[i]
            if traj["spatial"]:
                # Sign the outcome per player-to-move, from the absolute winner.
                # This also removes a divergence from the sequential path, which
                # previously labelled unfinished games with their last observed
                # reward while this path labelled every position 0.0.
                winner = getattr(vec_env.get_env(i), "winner", None)
                traj["rewards"] = [
                    outcome_value_for_player(winner, player) for player in traj["players"]
                ]
                trajectories.append(traj)
                batch_games += 1

        games_completed += batch_games
        batch_elapsed = time.time() - batch_start

        logger.info(
            "Batch %d: %d games in %.1fs (%.1f games/min), total %d/%d",
            batch_num,
            batch_games,
            batch_elapsed,
            batch_games / max(batch_elapsed / 60, 0.001),
            games_completed,
            num_games,
        )

        # Log batch metrics to W&B
        try:
            from alpha_zero.utils.experiment_tracker import log_metrics

            log_metrics(
                {
                    "selfplay/batch_games_completed": batch_games,
                    "selfplay/batch_time_seconds": batch_elapsed,
                    "selfplay/parallel_envs": num_parallel_games,
                    "selfplay/total_games_completed": games_completed,
                }
            )
        except ImportError:
            logger.debug("experiment tracker not available, skipping batch metrics logging")

        # Memory cleanup between batches
        if device != "cpu":
            torch.cuda.empty_cache()
        gc.collect()

    total_elapsed = time.time() - start_time
    total_states = sum(len(t["spatial"]) for t in trajectories)
    logger.info(
        "Batch self-play complete: %d games, %d states in %.1fs (%.1f games/min)",
        len(trajectories),
        total_states,
        total_elapsed,
        len(trajectories) / max(total_elapsed / 60, 0.001),
    )

    return trajectories


def save_trajectories(
    config: GCPConfig,
    trajectories: list[dict],
    iteration: int,
    worker_id: int,
    model_version: int,
    dry_run: bool = False,
) -> str | None:
    """Save trajectories to GCS."""
    if dry_run:
        total_states = sum(len(t["spatial"]) for t in trajectories)
        logger.info(
            "Dry run: would save %d trajectories (%d states)", len(trajectories), total_states
        )
        return None

    storage = GCSTrajectoryStorage(config, compress=True)
    uri = storage.save_trajectories(
        trajectories=trajectories,
        iteration=iteration,
        worker_id=worker_id,
        model_version=model_version,
    )
    logger.info("Saved %d trajectories to %s", len(trajectories), uri)
    return uri


def main() -> int:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.worker_id)

    logger.info("Starting self-play worker %d for iteration %d", args.worker_id, args.iteration)
    logger.info(
        "Config: model_v%d, %d games, %d MCTS sims (%s), %dx%d board",
        args.model_version,
        args.games,
        args.mcts_simulations,
        args.mcts_type,
        args.board_size,
        args.board_size,
    )

    # Initialize tracking (disabled by default; opt-in via CIV6AZ_MLFLOW_ENABLED /
    # MLFLOW_ENABLED, legacy WANDB_ENABLED honored)
    wandb_logger = None
    try:
        from alpha_zero.monitoring.mlflow_logger import WandbCompatAdapter as WandbLogger

        wandb_logger = WandbLogger.from_env()
        if wandb_logger.is_enabled:
            wandb_logger.init(
                config_dict={
                    "worker_id": args.worker_id,
                    "iteration": args.iteration,
                    "model_version": args.model_version,
                    "mcts_type": args.mcts_type,
                    "mcts_simulations": args.mcts_simulations,
                    "board_size": args.board_size,
                    "max_turns": args.max_turns,
                    "games": args.games,
                    "num_parallel_games": args.num_parallel_games,
                    "max_batch_size": args.max_batch_size,
                    "embed_dim": args.num_filters,
                    "num_layers": args.num_res_blocks,
                    "num_heads": getattr(args, "num_heads", 8),
                    "policy_hidden_dim": getattr(args, "policy_hidden_dim", 128),
                    "value_head_dim": getattr(args, "value_head_dim", 64),
                    "gameplay_preset": getattr(args, "gameplay_preset", "default"),
                    "mode": "selfplay",
                }
            )
            logger.info("W&B logging enabled")
    except ImportError:
        logger.debug("W&B not available")

    try:
        config = GCPConfig()
        if args.agent_id:
            config.paths.agent_id = args.agent_id
            logger.info("Using Agent ID: %s", args.agent_id)

        if not args.dry_run:
            config.validate()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)

        # Build MCTS override dict from CLI args
        mcts_override = {
            "c_puct": args.c_puct,
            "dirichlet_alpha": args.dirichlet_alpha,
            "dirichlet_epsilon": args.dirichlet_epsilon,
            "temperature": args.temperature,
            "temperature_threshold": args.temperature_threshold,
        }

        # Run self-play (batch, MuZero, or sequential based on configuration)
        start_time = time.time()
        use_batch = should_use_batch(args)
        use_muzero = args.mcts_type == "muzero"

        # The worker can be launched standalone, so repeat the orchestrator's
        # warning here: a parallelism request paired with a non-batch MCTS type
        # is silently dropped, and the run looks batched while running
        # sequentially.
        if args.num_parallel_games > 1 and not use_batch and not use_muzero:
            logger.warning(
                "num_parallel_games=%d ignored: vectorized self-play requires "
                "--mcts_type=%s (got %s)%s. Running sequential self-play.",
                args.num_parallel_games,
                BATCH_MCTS_TYPE,
                args.mcts_type,
                " and is disabled under --dry_run" if args.dry_run else "",
            )

        if use_muzero:
            # MuZero self-play: load MuZero model and use dynamics-based MCTS
            logger.info("Using MuZero self-play (dynamics-based MCTS, no env cloning)")
            model = load_muzero_model(
                config,
                args.model_version,
                board_size=args.board_size,
                action_space_size=args.action_space_size,
                num_res_blocks=args.num_res_blocks,
                num_filters=args.num_filters,
                dry_run=args.dry_run,
            )
            model = model.to(device)
            model.eval()
            trajectories = run_muzero_selfplay(
                model=model,
                num_games=args.games,
                mcts_sims=args.mcts_simulations,
                board_size=args.board_size,
                max_turns=args.max_turns,
                mcts_config_override=mcts_override,
                dry_run=args.dry_run,
                device=str(device),
                temperature_schedule=args.temperature_schedule,
                gameplay_preset=getattr(args, "gameplay_preset", "default"),
            )
        else:
            # AlphaZero/Gumbel: load standard model
            model = load_model(
                config,
                args.model_version,
                dry_run=args.dry_run,
                spatial_size=args.board_size,
                action_space_size=args.action_space_size,
            )
            model = model.to(device)
            model.eval()

            if use_batch:
                logger.info(
                    "Using batch self-play: %d parallel games, max_batch=%d",
                    args.num_parallel_games,
                    args.max_batch_size,
                )
                trajectories = run_batch_selfplay(
                    model=model,
                    num_games=args.games,
                    mcts_sims=args.mcts_simulations,
                    board_size=args.board_size,
                    max_turns=args.max_turns,
                    num_parallel_games=args.num_parallel_games,
                    max_batch_size=args.max_batch_size,
                    gumbel_num_actions=args.gumbel_num_actions,
                    device=str(device),
                    temperature_schedule=args.temperature_schedule,
                    temperature=args.temperature,
                    temperature_threshold=args.temperature_threshold,
                    gameplay_preset=getattr(args, "gameplay_preset", "default"),
                )
            else:
                trajectories = run_selfplay(
                    model=model,
                    num_games=args.games,
                    mcts_sims=args.mcts_simulations,
                    mcts_type=args.mcts_type,
                    board_size=args.board_size,
                    max_turns=args.max_turns,
                    gumbel_num_actions=args.gumbel_num_actions,
                    mcts_config_override=mcts_override,
                    dry_run=args.dry_run,
                    device=str(device),
                    temperature_schedule=args.temperature_schedule,
                    gameplay_preset=getattr(args, "gameplay_preset", "default"),
                )
        elapsed = time.time() - start_time

        # Calculate total states generated (using spatial list length)
        total_states = sum(len(t["spatial"]) for t in trajectories)
        logger.info(
            "Self-play complete: %d games, %d states in %.1fs (%.1f games/min)",
            len(trajectories),
            total_states,
            elapsed,
            len(trajectories) / max(elapsed / 60, 0.001),
        )

        # Log aggregate selfplay metrics to W&B
        if (
            wandb_logger is not None
            and hasattr(wandb_logger, "is_enabled")
            and wandb_logger.is_enabled
        ):
            wandb_logger.log_selfplay(
                games_completed=len(trajectories),
                buffer_size=total_states,
                samples_generated=total_states,
            )

            # Log batch-specific metrics
            if use_batch:
                from alpha_zero.utils.experiment_tracker import log_metrics

                log_metrics(
                    {
                        "selfplay/games_per_minute": len(trajectories) / max(elapsed / 60, 0.001),
                        "selfplay/avg_batch_size": args.num_parallel_games,
                        "selfplay/parallel_envs": args.num_parallel_games,
                        "selfplay/batch_utilization_pct": (
                            args.num_parallel_games / args.max_batch_size * 100
                        ),
                    }
                )

        # Save trajectories
        save_trajectories(
            config=config,
            trajectories=trajectories,
            iteration=args.iteration,
            worker_id=args.worker_id,
            model_version=args.model_version,
            dry_run=args.dry_run,
        )

        logger.info("Worker %d completed successfully", args.worker_id)
        return 0

    except KeyboardInterrupt:
        logger.warning("Worker %d interrupted", args.worker_id)
        return 130
    except Exception as e:
        logger.exception("Worker %d failed: %s", args.worker_id, e)
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
