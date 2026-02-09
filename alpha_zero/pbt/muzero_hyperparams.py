"""MuZero-specific hyperparameter space definitions for Population-Based Training.

This module provides factory functions to create HyperparameterSpace instances
configured for MuZero training, which requires additional parameters beyond
standard AlphaZero (e.g., unroll steps, TD steps, reward prediction).

MuZero extends AlphaZero with learned dynamics models, requiring:
- Unroll steps: Number of steps to unroll the learned model
- TD steps: Temporal difference learning horizon
- Discount factor: Reward discounting gamma
- Consistency loss weight: Weight for dynamics consistency
- Reward loss weight: Weight for reward prediction loss

Usage:
    from alpha_zero.pbt.muzero_hyperparams import create_muzero_hyperparameter_space

    # Create MuZero-specific space
    space = create_muzero_hyperparameter_space()

    # Or merge with existing base space
    from alpha_zero.pbt.config import HyperparameterSpace
    base_space = HyperparameterSpace()
    muzero_space = merge_with_base_space(base_space)
"""

from __future__ import annotations

import logging

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace

logger = logging.getLogger(__name__)

# MuZero-specific parameter names
MUZERO_PARAM_NAMES: frozenset[str] = frozenset({
    "unroll_steps",
    "td_steps",
    "discount",
    "consistency_loss_weight",
    "reward_loss_weight",
})


def get_muzero_custom_params() -> dict[str, HyperparameterRange]:
    """Get MuZero-specific custom hyperparameters.

    Returns:
        Dictionary mapping MuZero parameter names to their ranges.
    """
    return {
        "unroll_steps": HyperparameterRange(
            name="unroll_steps",
            min_value=3,
            max_value=10,
            scale="linear",
            discrete=True,
        ),
        "td_steps": HyperparameterRange(
            name="td_steps",
            min_value=5,
            max_value=20,
            scale="linear",
            discrete=True,
        ),
        "discount": HyperparameterRange(
            name="discount",
            min_value=0.95,
            max_value=0.999,
            scale="linear",
        ),
        "consistency_loss_weight": HyperparameterRange(
            name="consistency_loss_weight",
            min_value=0.5,
            max_value=5.0,
            scale="linear",
        ),
        "reward_loss_weight": HyperparameterRange(
            name="reward_loss_weight",
            min_value=0.5,
            max_value=2.0,
            scale="linear",
        ),
    }


def create_muzero_hyperparameter_space() -> HyperparameterSpace:
    """Create a HyperparameterSpace pre-configured for MuZero training.

    MuZero requires different default ranges compared to base AlphaZero:
    - num_simulations: log scale (50-400) instead of linear (25-400)
    - batch_size: smaller max (256 vs 512) due to larger unroll memory
    - c_puct: wider range (0.5-5.0) for learned dynamics exploration

    Returns:
        HyperparameterSpace configured for MuZero with custom parameters.
    """
    logger.info("Creating MuZero hyperparameter space")

    # Create base space with MuZero-specific overrides
    space = HyperparameterSpace(
        # Keep base learning_rate (log scale 1e-5 to 1e-2)
        learning_rate=HyperparameterRange(
            name="learning_rate",
            min_value=1e-5,
            max_value=1e-2,
            scale="log",
        ),
        # Keep base weight_decay
        weight_decay=HyperparameterRange(
            name="weight_decay",
            min_value=1e-6,
            max_value=1e-3,
            scale="log",
        ),
        # Wider c_puct range for learned dynamics
        c_puct=HyperparameterRange(
            name="c_puct",
            min_value=0.5,
            max_value=5.0,
            scale="linear",
        ),
        # Keep base dirichlet_alpha
        dirichlet_alpha=HyperparameterRange(
            name="dirichlet_alpha",
            min_value=0.1,
            max_value=1.0,
            scale="linear",
        ),
        # Keep base temperature ranges
        temperature=HyperparameterRange(
            name="temperature",
            min_value=0.5,
            max_value=2.0,
            scale="linear",
        ),
        temperature_initial=HyperparameterRange(
            name="temperature_initial",
            min_value=1.0,
            max_value=2.5,
            scale="linear",
        ),
        temperature_final=HyperparameterRange(
            name="temperature_final",
            min_value=0.1,
            max_value=1.0,
            scale="linear",
        ),
        temperature_decay_steps=HyperparameterRange(
            name="temperature_decay_steps",
            min_value=10,
            max_value=50,
            scale="linear",
            discrete=True,
        ),
        # Smaller batch_size max due to unroll memory overhead
        batch_size=HyperparameterRange(
            name="batch_size",
            min_value=32,
            max_value=256,
            scale="log",
            discrete=True,
        ),
        # Log scale for num_simulations (MuZero benefits from log sampling)
        num_simulations=HyperparameterRange(
            name="num_simulations",
            min_value=50,
            max_value=400,
            scale="log",
            discrete=True,
        ),
        # Keep base replay_buffer_size
        replay_buffer_size=HyperparameterRange(
            name="replay_buffer_size",
            min_value=10000,
            max_value=200000,
            scale="linear",
            discrete=True,
        ),
        # Add MuZero-specific custom parameters
        custom_params=get_muzero_custom_params(),
    )

    logger.info(
        "Created MuZero space with %d total parameters (%d base + %d custom)",
        len(space.get_all_params()),
        11,  # Base parameters
        len(MUZERO_PARAM_NAMES),
    )

    return space


def merge_with_base_space(base: HyperparameterSpace) -> HyperparameterSpace:
    """Merge MuZero-specific parameters with an existing base space.

    This creates a new HyperparameterSpace that combines the base space with
    MuZero-specific overrides and custom parameters. The original base space
    is not modified.

    Args:
        base: Existing base HyperparameterSpace to extend

    Returns:
        New HyperparameterSpace with MuZero parameters merged in.
    """
    logger.info("Merging MuZero parameters with base space")

    # Get MuZero custom params
    muzero_params = get_muzero_custom_params()

    # Create new custom_params dict combining base and MuZero
    merged_custom = {**base.custom_params, **muzero_params}

    # Create new space with MuZero overrides
    merged_space = HyperparameterSpace(
        learning_rate=base.learning_rate,  # Keep base
        weight_decay=base.weight_decay,  # Keep base
        # Override c_puct with wider range
        c_puct=HyperparameterRange(
            name="c_puct",
            min_value=0.5,
            max_value=5.0,
            scale="linear",
        ),
        dirichlet_alpha=base.dirichlet_alpha,  # Keep base
        temperature=base.temperature,  # Keep base
        temperature_initial=base.temperature_initial,  # Keep base
        temperature_final=base.temperature_final,  # Keep base
        temperature_decay_steps=base.temperature_decay_steps,  # Keep base
        # Override batch_size with smaller max and log scale
        batch_size=HyperparameterRange(
            name="batch_size",
            min_value=32,
            max_value=256,
            scale="log",
            discrete=True,
        ),
        # Override num_simulations with log scale
        num_simulations=HyperparameterRange(
            name="num_simulations",
            min_value=50,
            max_value=400,
            scale="log",
            discrete=True,
        ),
        replay_buffer_size=base.replay_buffer_size,  # Keep base
        custom_params=merged_custom,
    )

    logger.info(
        "Merged space has %d total parameters (added %d MuZero-specific)",
        len(merged_space.get_all_params()),
        len(MUZERO_PARAM_NAMES),
    )

    return merged_space
