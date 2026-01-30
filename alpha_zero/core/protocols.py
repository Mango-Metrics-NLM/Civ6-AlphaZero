"""Protocol definitions for duck typing.

This module provides Protocol classes that define interfaces for
common patterns used throughout the codebase, enabling better
type checking and interchangeability of implementations.

Protocols are used for structural subtyping - any class that implements
the required methods will be considered compatible, without explicit
inheritance.

Example:
    >>> from alpha_zero.core.protocols import Serializable, NetworkProtocol
    >>>
    >>> # Check if an object implements Serializable
    >>> isinstance(my_object, Serializable)  # True if has to_dict/from_dict
    >>>
    >>> def train(network: NetworkProtocol):
    ...     policy, value = network.forward(spatial, vector)
    ...     # Works with any network implementing the protocol
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


@runtime_checkable
class Serializable(Protocol):
    """Protocol for objects that can be serialized to/from dictionaries.

    Classes implementing this protocol can be easily serialized to JSON,
    YAML, or other formats for persistence and configuration.

    Example:
        >>> @dataclass
        ... class MyConfig:
        ...     value: int = 10
        ...
        ...     def to_dict(self) -> dict[str, Any]:
        ...         return {"value": self.value}
        ...
        ...     @classmethod
        ...     def from_dict(cls, data: dict[str, Any]) -> "MyConfig":
        ...         return cls(**data)
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Serializable:
        """Create instance from dictionary."""
        ...


@runtime_checkable
class StateEncoderProtocol(Protocol):
    """Protocol for state encoders.

    State encoders transform game state dictionaries into tensor
    representations suitable for neural network input.

    Example:
        >>> encoder: StateEncoderProtocol = MyEncoder()
        >>> encoded = encoder.encode(game_state)
        >>> assert encoded.shape == (encoder.output_dim,)
    """

    def encode(self, game_state: dict[str, Any]) -> np.ndarray:
        """Encode game state into tensor representation.

        Args:
            game_state: Dictionary containing game state information.

        Returns:
            Encoded state as numpy array.
        """
        ...

    @property
    def output_dim(self) -> int:
        """Get output dimension of encoded state."""
        ...


@runtime_checkable
class ActionMaskable(Protocol):
    """Protocol for objects that support action masking.

    Action masking is used to filter out invalid actions from
    policy distributions during MCTS and training.

    Example:
        >>> maskable: ActionMaskable = MyPolicy()
        >>> mask = maskable.get_valid_actions()
        >>> masked_logits = maskable.apply_action_mask(logits, mask)
    """

    def get_valid_actions(self) -> np.ndarray:
        """Get mask of valid actions.

        Returns:
            Boolean or float array where 1 = valid, 0 = invalid.
        """
        ...

    def apply_action_mask(
        self,
        logits: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Apply action mask to logits.

        Args:
            logits: Policy logits of shape (B, A).
            mask: Valid action mask of shape (B, A).

        Returns:
            Masked logits with invalid actions set to very negative value.
        """
        ...


@runtime_checkable
class NetworkProtocol(Protocol):
    """Protocol for neural networks with policy/value heads.

    This protocol defines the interface that all AlphaZero-style
    networks must implement. It supports both training (forward)
    and inference (predict) modes.

    Example:
        >>> network: NetworkProtocol = MicroCivNetwork(...)
        >>> policy, value = network.forward(spatial, vector)
        >>> probs, val = network.predict(spatial, vector)
    """

    def forward(
        self,
        spatial: Tensor,
        vector: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass returning (policy_logits, value).

        Args:
            spatial: Spatial features of shape (B, C, H, W).
            vector: Vector features of shape (B, V).
            action_mask: Optional action mask of shape (B, A).

        Returns:
            Tuple of (policy_logits, value) tensors.
        """
        ...

    def predict(
        self,
        spatial: Tensor,
        vector: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Predict policy probabilities and value.

        Like forward() but returns probabilities instead of logits.

        Args:
            spatial: Spatial features of shape (B, C, H, W).
            vector: Vector features of shape (B, V).
            action_mask: Optional action mask of shape (B, A).

        Returns:
            Tuple of (policy_probs, value) tensors.
        """
        ...


@runtime_checkable
class CheckpointableProtocol(Protocol):
    """Protocol for objects that can be checkpointed.

    Objects implementing this protocol can save and load their state,
    enabling model persistence and training resumption.

    Example:
        >>> model: CheckpointableProtocol = MyModel()
        >>> model.save("checkpoint.pt")
        >>> loaded = MyModel.load("checkpoint.pt")
    """

    def save(self, path: str) -> None:
        """Save state to path.

        Args:
            path: Path to save checkpoint to.
        """
        ...

    def load(self, path: str, device: str = "cpu") -> None:
        """Load state from path.

        Args:
            path: Path to load checkpoint from.
            device: Device to load weights to.
        """
        ...


@runtime_checkable
class EnvironmentProtocol(Protocol):
    """Protocol for game environments.

    Defines the interface for environments compatible with MCTS
    and training pipelines.

    Example:
        >>> env: EnvironmentProtocol = MicroCivEnv()
        >>> obs = env.reset()
        >>> obs, reward, done, truncated, info = env.step(action)
    """

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray]:
        """Reset the environment to initial state.

        Args:
            seed: Optional random seed for reproducibility.

        Returns:
            Initial observation.
        """
        ...

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Take a step in the environment.

        Args:
            action: Action to take.

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        ...

    def get_observation(self) -> dict[str, np.ndarray]:
        """Get current observation without stepping.

        Returns:
            Current observation dictionary.
        """
        ...

    def clone(self) -> EnvironmentProtocol:
        """Clone the environment state.

        Returns:
            A copy of the environment with identical state.
        """
        ...


@runtime_checkable
class MCTSProtocol(Protocol):
    """Protocol for MCTS implementations.

    Defines the interface for all MCTS variants (SimpleMCTS, Civ6MCTS,
    GumbelMCTS, etc.).

    Example:
        >>> mcts: MCTSProtocol = Civ6MCTS(network, config)
        >>> action_probs, value = mcts.search(env)
        >>> action = mcts.select_action(action_probs, temperature=1.0)
    """

    def search(
        self,
        env: EnvironmentProtocol,
        num_simulations: int | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run MCTS search from current state.

        Args:
            env: Environment at current state.
            num_simulations: Optional override for simulation count.

        Returns:
            Tuple of (action_probabilities, value_estimate).
        """
        ...

    def select_action(
        self,
        action_probs: np.ndarray,
        temperature: float = 1.0,
    ) -> int:
        """Select action from probability distribution.

        Args:
            action_probs: Action probabilities from search.
            temperature: Sampling temperature (0 = argmax, 1 = sample).

        Returns:
            Selected action index.
        """
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for training loops.

    Defines the interface for components that train neural networks.

    Example:
        >>> trainer: TrainerProtocol = AlphaZeroTrainer(...)
        >>> for epoch in range(epochs):
        ...     loss = trainer.train_step(batch)
    """

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Training batch with observations, policies, values.

        Returns:
            Dictionary of loss values and metrics.
        """
        ...

    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint.

        Args:
            path: Path to save checkpoint to.
        """
        ...

    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint.

        Args:
            path: Path to load checkpoint from.
        """
        ...
