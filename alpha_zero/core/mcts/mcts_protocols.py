"""Structural typing contracts for the MCTS family.

Protocols only -- no runtime behaviour. Extracted from ``civ6_mcts`` so the
search variants can depend on the contracts without importing each other.
"""

from __future__ import annotations

from typing import Any, Protocol, SupportsFloat

import numpy as np
import torch


class MCTSNetwork(Protocol):
    """Protocol for neural networks used in MCTS."""

    def __call__(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the network directly."""
        ...

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value from state.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, value).
        """
        ...

    def eval(self) -> Any:
        """Set network to evaluation mode."""
        ...


class MuZeroNetworkProtocol(Protocol):
    """Protocol for MuZero-compatible networks with learned dynamics."""

    def initial_inference(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> Any:
        """Initial inference from observation.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Output with hidden_state, policy_logits, value attributes.
        """
        ...

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recurrent inference during MCTS search.

        Args:
            hidden_state: Current hidden state (B, C, H, W).
            action: Action indices (B,).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (next_hidden_state, reward, policy_logits, value).
        """
        ...

    def prediction_function(
        self,
        hidden_state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get policy and value from hidden state.

        Args:
            hidden_state: Hidden state (B, C, H, W).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_logits, value).
        """
        ...

    def eval(self) -> Any:
        """Set network to evaluation mode."""
        ...


class ActionSpace(Protocol):
    """Protocol for action space with size attribute."""

    n: int


class MCTSEnvironment(Protocol):
    """Protocol for environments compatible with MCTS.

    MCTS requires environments that can be cloned for simulation,
    stepped through, and queried for observations and valid actions.

    Attributes:
        action_space: Action space with `n` attribute for size.
        current_player: Index of the current player (for two-player games).

    Note:
        Implementations must provide the public `get_observation()` and
        `get_info()` methods. The private `_get_observation()` and
        `_get_info()` methods are deprecated and should not be used
        in new code.
    """

    @property
    def action_space(self) -> ActionSpace:
        """Action space with an ``n`` attribute for size (read-only)."""
        ...

    @property
    def current_player(self) -> int:
        """Index of the current player (read-only)."""
        ...

    def clone(self) -> MCTSEnvironment:
        """Create a deep copy of the environment for simulation."""
        ...

    def step(self, action: int) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info)."""
        ...

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation dict with 'spatial', 'vector', 'action_mask'."""
        ...

    def get_info(self) -> dict[str, Any]:
        """Get environment info dict (may contain 'winner' key)."""
        ...

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions (1 = valid, 0 = invalid)."""
        ...
