"""Factory for creating MCTS instances.

This module provides a unified interface for creating different MCTS variants,
making it easier to switch between implementations and ensuring consistent
configuration.

All MCTS variants share a common interface:
    search(env, add_noise) -> (action_probs, root_value)
    select_action(action_probs, temperature) -> int

Supported variants:
- "puct": Standard AlphaZero PUCT-based MCTS (Civ6MCTS)
- "gumbel": Gumbel MuZero MCTS for efficient inference
- "simple": Random rollout MCTS for bootstrapping/testing
- "muzero": MuZero MCTS with learned dynamics

Usage:
    from alpha_zero.core.mcts_factory import create_mcts, MCTSVariant

    # Create simple MCTS for testing
    mcts = create_mcts("simple", num_simulations=20)

    # Create network-guided MCTS
    mcts = create_mcts("puct", network=network, num_simulations=100)

    # Using enum
    mcts = create_mcts(MCTSVariant.GUMBEL, network=network, config=config)
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
from enum import Enum
from typing import Any, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MCTSVariant(Enum):
    """Available MCTS variants."""

    CIV6 = "civ6"
    PUCT = "puct"
    GUMBEL = "gumbel"
    SIMPLE = "simple"
    MUZERO = "muzero"


class MCTSProtocol(Protocol):
    """Protocol that all MCTS implementations should follow.

    This protocol defines the minimal interface that MCTS implementations
    must provide to be compatible with the training pipeline.
    """

    def search(
        self,
        env_or_obs: Any,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search and return policy and value.

        Args:
            env_or_obs: Environment instance or observation dict.
            add_noise: Whether to add exploration noise at root.

        Returns:
            Tuple of (action_probabilities, root_value).
        """
        ...

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Select action from probability distribution.

        Args:
            action_probs: Action probability distribution.
            temperature: Temperature for action selection.
            action_mask: Optional mask indicating which actions are legal/valid.
                If provided, implementations should respect this mask.

        Returns:
            Selected action index.
        """
        ...


# Registry of MCTS types -> (class_name, config_class_name, default_simulations)
_MCTS_REGISTRY: dict[str, tuple[str, str, int]] = {
    "civ6": ("Civ6MCTS", "MCTSConfig", 200),
    "puct": ("Civ6MCTS", "MCTSConfig", 200),
    "gumbel": ("GumbelMCTS", "GumbelMCTSConfig", 16),
    "muzero": ("MuZeroMCTS", "MuZeroMCTSConfig", 50),
    "simple": ("SimpleMCTS", "None", 20),
}


def create_mcts(
    variant: str | MCTSVariant = "puct",
    network: torch.nn.Module | None = None,
    config: Any | None = None,
    device: str = "cpu",
    **kwargs: Any,
) -> MCTSProtocol:
    """Factory function to create MCTS instances.

    This is the recommended way to create MCTS instances as it provides
    a consistent interface across all variants and handles configuration
    validation.

    Args:
        variant: MCTS variant to create. One of:
            - "puct": Standard AlphaZero PUCT-based MCTS
            - "gumbel": Gumbel MuZero for efficient inference
            - "simple": Random rollout MCTS (no network needed)
            - "muzero": MuZero with learned dynamics
        network: Neural network for evaluation (required for puct, gumbel, muzero).
        config: Optional configuration object. If None, defaults are used.
        device: Device for network inference ("cpu" or "cuda").
        **kwargs: Additional arguments passed to MCTS constructor.
            Common kwargs:
            - num_simulations: Number of MCTS simulations (default: variant default)
            - c_puct: Exploration constant
            - dirichlet_alpha: Dirichlet noise parameter
            - dirichlet_epsilon: Noise weight

    Returns:
        MCTS instance implementing MCTSProtocol.

    Raises:
        ValueError: If variant is unknown or required arguments are missing.
    """
    # Normalize variant to string
    if isinstance(variant, MCTSVariant):
        variant = variant.value

    variant = variant.lower()

    if variant not in _MCTS_REGISTRY:
        valid_variants = list(_MCTS_REGISTRY.keys())
        raise ValueError(f"Unknown MCTS variant: {variant!r}. Valid variants are: {valid_variants}")

    class_name, config_class_name, default_sims = _MCTS_REGISTRY[variant]

    # Normalize simulations
    num_simulations = kwargs.pop("num_simulations", default_sims)

    # Lazy import to avoid circular dependencies
    from alpha_zero.core.civ6_mcts import (
        Civ6MCTS,
        GumbelMCTS,
        GumbelMCTSConfig,
        MCTSConfig,
        MuZeroMCTS,
        MuZeroMCTSConfig,
        SimpleMCTS,
    )

    if variant == "simple":
        return SimpleMCTS(
            num_simulations=num_simulations,
            **kwargs,
        )

    if network is None:
        raise ValueError(f"Network is required for {variant} MCTS variant")

    # Resolve classes
    mcts_classes = {
        "Civ6MCTS": Civ6MCTS,
        "GumbelMCTS": GumbelMCTS,
        "MuZeroMCTS": MuZeroMCTS,
    }
    config_classes = {
        "MCTSConfig": MCTSConfig,
        "GumbelMCTSConfig": GumbelMCTSConfig,
        "MuZeroMCTSConfig": MuZeroMCTSConfig,
    }

    mcts_cls = mcts_classes[class_name]
    config_cls = config_classes[config_class_name]
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}

    if config is None:
        # Build config with overrides
        config_kwargs = {"num_simulations": num_simulations}

        # Filter kwargs to only include fields that the config class accepts
        for key, value in list(kwargs.items()):
            if key in valid_fields:
                config_kwargs[key] = value
                kwargs.pop(key)  # Remove from kwargs so it doesn't get passed to MCTS twice

        config = config_cls(**config_kwargs)

    # Filter remaining kwargs to only include params the MCTS class __init__ accepts
    valid_init_params = set(inspect.signature(mcts_cls.__init__).parameters.keys()) - {
        "self",
        "network",
        "config",
        "device",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_init_params}

    # Reject unrecognized kwargs — likely indicates a caller bug
    unrecognized = set(kwargs.keys()) - set(filtered_kwargs.keys())
    if unrecognized:
        raise TypeError(
            f"create_mcts() got unexpected keyword argument(s): {unrecognized}. "
            f"Valid config fields: {valid_fields}, valid init params: {valid_init_params}"
        )

    return mcts_cls(network=network, config=config, device=device, **filtered_kwargs)


def get_default_config(variant: str | MCTSVariant) -> Any:
    """Get default configuration for a variant.

    Args:
        variant: MCTS variant.

    Returns:
        Default configuration object for the variant.
    """
    if isinstance(variant, MCTSVariant):
        variant = variant.value

    variant = variant.lower()

    if variant == "simple":
        return None  # SimpleMCTS doesn't use a config object

    # Lazy import
    from alpha_zero.core.civ6_mcts import (
        GumbelMCTSConfig,
        MCTSConfig,
        MuZeroMCTSConfig,
    )

    config_classes = {
        "civ6": MCTSConfig,
        "puct": MCTSConfig,
        "gumbel": GumbelMCTSConfig,
        "muzero": MuZeroMCTSConfig,
    }

    if variant in config_classes:
        return config_classes[variant]()

    valid_variants = list(_MCTS_REGISTRY.keys())
    raise ValueError(f"Unknown MCTS variant: {variant!r}. Valid variants are: {valid_variants}")


def list_variants() -> list[str]:
    """List all available MCTS variants.

    Returns:
        List of variant names.
    """
    return list(_MCTS_REGISTRY.keys())


def get_available_mcts_types() -> list[str]:
    """List all available MCTS types (alias for list_variants).

    Returns:
        List of variant names.
    """
    return list_variants()
