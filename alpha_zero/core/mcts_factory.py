"""Factory for creating MCTS instances by type name.

All MCTS variants share a common interface:
    search(env, add_noise) -> (action_probs, root_value)
    select_action(action_probs, temperature) -> int

This factory allows runtime selection of the MCTS algorithm via a string
type name, enabling configuration-driven switching without code changes.

Usage:
    mcts = create_mcts("gumbel", network, device="cuda", num_simulations=16)
    action_probs, value = mcts.search(env, add_noise=True)
    action = mcts.select_action(action_probs, temperature=1.0)
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from alpha_zero.core.civ6_mcts import Civ6MCTS, GumbelMCTS, MuZeroMCTS

# Registry of MCTS types -> (class_name, config_class_name, default_simulations)
_MCTS_REGISTRY: dict[str, tuple[str, str, int]] = {
    "civ6": ("Civ6MCTS", "MCTSConfig", 200),
    "gumbel": ("GumbelMCTS", "GumbelMCTSConfig", 16),
    "muzero": ("MuZeroMCTS", "MuZeroMCTSConfig", 50),
}


def get_available_mcts_types() -> list[str]:
    """Return list of supported MCTS type names."""
    return list(_MCTS_REGISTRY.keys())


def create_mcts(
    mcts_type: str = "civ6",
    network: torch.nn.Module | None = None,
    device: str = "cpu",
    num_simulations: int | None = None,
    **kwargs,
) -> Civ6MCTS | GumbelMCTS | MuZeroMCTS:
    """Create an MCTS instance by type name.

    Args:
        mcts_type: MCTS variant to create ("civ6", "gumbel", or "muzero").
        network: Neural network for evaluation.
        device: Device for neural network inference.
        num_simulations: Override default simulation count.
            Defaults: civ6=200, gumbel=16, muzero=50.
        **kwargs: Additional config parameters passed to the config dataclass.
            For gumbel: num_sampled_actions, max_considered_actions, c_visit, etc.
            For muzero: use_gumbel, discount, etc.
            For civ6: c_puct, dirichlet_alpha, dirichlet_epsilon, etc.

    Returns:
        An MCTS instance with search() and select_action() methods.

    Raises:
        ValueError: If mcts_type is not recognized.
    """
    if mcts_type not in _MCTS_REGISTRY:
        available = ", ".join(f"'{t}'" for t in _MCTS_REGISTRY)
        raise ValueError(f"Unknown MCTS type: '{mcts_type}'. Available types: {available}")

    # Lazy import to avoid circular dependencies
    from alpha_zero.core.civ6_mcts import (
        Civ6MCTS,
        GumbelMCTS,
        GumbelMCTSConfig,
        MCTSConfig,
        MuZeroMCTS,
        MuZeroMCTSConfig,
    )

    class_name, config_class_name, default_sims = _MCTS_REGISTRY[mcts_type]

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

    # Build config with overrides
    config_kwargs = {"num_simulations": num_simulations or default_sims}

    # Filter kwargs to only include fields that the config class accepts
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    for key, value in kwargs.items():
        if key in valid_fields:
            config_kwargs[key] = value

    config = config_cls(**config_kwargs)

    return mcts_cls(network=network, config=config, device=device)
