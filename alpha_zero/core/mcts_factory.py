"""Factory for creating MCTS instances.

This module provides a unified interface for creating different MCTS variants,
making it easier to switch between implementations and ensuring consistent
configuration.

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

from enum import Enum
from typing import Any, Protocol

import numpy as np

from alpha_zero.utils.constants import MCTS as MCTS_CONSTANTS


class MCTSVariant(Enum):
    """Available MCTS variants."""

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


def create_mcts(
    variant: str | MCTSVariant = "puct",
    network: Any | None = None,
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
            - num_simulations: Number of MCTS simulations (default: 50)
            - c_puct: Exploration constant (default: 1.5)
            - dirichlet_alpha: Dirichlet noise parameter (default: 0.3)
            - dirichlet_epsilon: Noise weight (default: 0.25)

    Returns:
        MCTS instance implementing MCTSProtocol.

    Raises:
        ValueError: If variant is unknown or required arguments are missing.

    Examples:
        >>> # Simple MCTS for testing (no network needed)
        >>> mcts = create_mcts("simple", num_simulations=20)

        >>> # Network-guided MCTS for training
        >>> mcts = create_mcts("puct", network=my_network, num_simulations=100)

        >>> # Efficient inference with Gumbel MCTS
        >>> mcts = create_mcts("gumbel", network=my_network, num_simulations=16)
    """
    # Normalize variant to string
    if isinstance(variant, MCTSVariant):
        variant = variant.value

    variant = variant.lower()

    # Extract common kwargs with defaults from constants
    num_simulations = kwargs.pop("num_simulations", MCTS_CONSTANTS.DEFAULT_NUM_SIMULATIONS)
    c_puct = kwargs.pop("c_puct", MCTS_CONSTANTS.DEFAULT_C_PUCT)
    dirichlet_alpha = kwargs.pop("dirichlet_alpha", MCTS_CONSTANTS.DEFAULT_DIRICHLET_ALPHA)
    dirichlet_epsilon = kwargs.pop("dirichlet_epsilon", MCTS_CONSTANTS.DEFAULT_DIRICHLET_EPSILON)

    if variant == "simple":
        from alpha_zero.core.civ6_mcts import SimpleMCTS

        return SimpleMCTS(
            num_simulations=num_simulations,
            c_puct=c_puct,
            **kwargs,
        )

    elif variant == "puct":
        if network is None:
            raise ValueError("Network is required for PUCT MCTS variant")

        from alpha_zero.core.civ6_mcts import Civ6MCTS, MCTSConfig

        if config is None:
            config = MCTSConfig(
                num_simulations=num_simulations,
                c_puct=c_puct,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_epsilon=dirichlet_epsilon,
            )

        return Civ6MCTS(network, config, device=device, **kwargs)

    elif variant == "gumbel":
        if network is None:
            raise ValueError("Network is required for Gumbel MCTS variant")

        from alpha_zero.core.civ6_mcts import GumbelMCTS, GumbelMCTSConfig

        if config is None:
            config = GumbelMCTSConfig(
                num_simulations=num_simulations,
                c_puct=c_puct,
            )

        return GumbelMCTS(network, config, device=device, **kwargs)

    elif variant == "muzero":
        if network is None:
            raise ValueError("Network is required for MuZero MCTS variant")

        from alpha_zero.core.civ6_mcts import MuZeroMCTS, MuZeroMCTSConfig

        if config is None:
            config = MuZeroMCTSConfig(
                num_simulations=num_simulations,
                c_puct=c_puct,
            )

        return MuZeroMCTS(network, config, device=device, **kwargs)

    else:
        valid_variants = [v.value for v in MCTSVariant]
        raise ValueError(f"Unknown MCTS variant: {variant!r}. Valid variants are: {valid_variants}")


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

    elif variant == "puct":
        from alpha_zero.core.civ6_mcts import MCTSConfig

        return MCTSConfig()

    elif variant == "gumbel":
        from alpha_zero.core.civ6_mcts import GumbelMCTSConfig

        return GumbelMCTSConfig()

    elif variant == "muzero":
        from alpha_zero.core.civ6_mcts import MuZeroMCTSConfig

        return MuZeroMCTSConfig()

    else:
        valid_variants = [v.value for v in MCTSVariant]
        raise ValueError(f"Unknown MCTS variant: {variant!r}. Valid variants are: {valid_variants}")


def list_variants() -> list[str]:
    """List all available MCTS variants.

    Returns:
        List of variant names.
    """
    return [v.value for v in MCTSVariant]
