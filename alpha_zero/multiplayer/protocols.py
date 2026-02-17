"""Protocol definitions for multiplayer components.

Extends the core protocols with N-player support using structural subtyping.
Any class implementing these methods is considered compatible without
explicit inheritance.

Example:
    >>> class MyMultiEnv:
    ...     @property
    ...     def num_players(self) -> int: return 4
    ...     @property
    ...     def current_player(self) -> int: return 0
    ...     # ... other required methods
    >>> isinstance(MyMultiEnv(), MultiplayerEnvironment)
    True
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, SupportsFloat, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import torch


class ActionSpace(Protocol):
    """Protocol for action space with size attribute."""

    n: int


@runtime_checkable
class MultiplayerEnvironment(Protocol):
    """Protocol for N-player game environments.

    Extends the standard MCTSEnvironment with multi-player awareness.
    Backward-compatible: 2-player environments satisfy this protocol
    if they provide num_players.

    Attributes:
        action_space: Action space with `n` attribute for size.
        current_player: Index of the currently active player (0 to num_players-1).
        num_players: Total number of players in the game.
    """

    action_space: ActionSpace
    current_player: int
    num_players: int

    def clone(self) -> MultiplayerEnvironment:
        """Create a deep copy of the environment for simulation."""
        ...

    def step(
        self, action: int
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute action and return (obs, reward, terminated, truncated, info).

        The info dict should contain:
            - 'winner': int | None - ID of winning player, or None for draw/ongoing
            - 'rankings': list[int] | None - Final player rankings (optional)
            - 'player_scores': dict[int, float] | None - Per-player scores (optional)
            - 'eliminated': set[int] | None - Set of eliminated player IDs (optional)
        """
        ...

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation dict with 'spatial', 'vector', 'action_mask'."""
        ...

    def get_info(self) -> dict[str, Any]:
        """Get environment info dict."""
        ...

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions (1 = valid, 0 = invalid)."""
        ...


@runtime_checkable
class MultiplayerNetwork(Protocol):
    """Protocol for neural networks with N-player value output.

    The key difference from the standard MCTSNetwork is that the value
    output is a vector of per-player values instead of a single scalar.

    For backward compatibility, implementations may return:
        - (policy, value) where value shape is (B,) for 2-player
        - (policy, values) where values shape is (B, N) for N-player
    """

    @property
    def num_players(self) -> int:
        """Number of players this network supports."""
        ...

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and per-player values from state.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, player_values).
            policy_probs shape: (B, A)
            player_values shape: (B, N) where N = num_players, or (B,)
                for legacy 2-player networks (interpreted as player 0's value;
                player 1's value is the negation).
        """
        ...

    def eval(self) -> None:
        """Set network to evaluation mode."""
        ...


@runtime_checkable
class OpponentModelProtocol(Protocol):
    """Protocol for opponent behavior modeling.

    Tracks opponent actions and predicts their likely moves,
    which can be used to improve MCTS search quality.
    """

    def update(
        self,
        player_id: int,
        game_state: dict[str, Any],
        action: int,
    ) -> None:
        """Record an opponent's action for modeling.

        Args:
            player_id: ID of the opponent (0 to num_players-1).
            game_state: State in which the action was taken.
            action: Action index chosen by the opponent.
        """
        ...

    def predict_action_distribution(
        self,
        player_id: int,
        game_state: dict[str, Any],
        valid_actions: np.ndarray,
    ) -> np.ndarray:
        """Predict action probability distribution for an opponent.

        Args:
            player_id: ID of the opponent.
            game_state: Current game state.
            valid_actions: Mask of valid actions.

        Returns:
            Probability distribution over actions.
        """
        ...

    def reset(self) -> None:
        """Reset model state for a new game."""
        ...
