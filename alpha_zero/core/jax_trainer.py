"""JAX training utilities for AlphaZero self-play learning.

Provides JAX-native training components using optax for optimization
and eqx.filter_value_and_grad for gradient computation.

Usage:
    from alpha_zero.core.jax_trainer import (
        JaxTrainState,
        create_train_state,
        jax_compute_loss,
        jax_train_step,
        JaxDataLoader,
    )

    state = create_train_state(wrapper.model, config)
    for batch in JaxDataLoader(examples, batch_size=128):
        state, metrics = jax_train_step(state, batch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _require_jax() -> None:
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX trainer requires: pip install -e '.[jax]' "
            "(installs jax, jaxlib, equinox, optax)"
        )


# ---------------------------------------------------------------------------
# Training Configuration
# ---------------------------------------------------------------------------


@dataclass
class JaxTrainConfig:
    """Configuration for JAX training.

    All values are configurable — no hardcoded defaults in the training loop.
    """

    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    jit_enabled: bool = True
    seed: int = 42

    @classmethod
    def from_training_config(cls, config: Any) -> JaxTrainConfig:
        """Create from existing TrainingConfig or OrchestratorConfig."""
        return cls(
            learning_rate=getattr(config, "learning_rate", 0.001),
            weight_decay=getattr(config, "weight_decay", 1e-4),
            gradient_clip=getattr(config, "gradient_clip", 1.0),
        )


# ---------------------------------------------------------------------------
# Train State
# ---------------------------------------------------------------------------


@dataclass
class JaxTrainState:
    """Immutable training state for JAX training loops.

    Holds the model, optimizer state, step counter, and PRNG key.
    New states are created on each step (functional updates).
    """

    model: Any
    opt_state: Any
    optimizer: Any
    step: int = 0
    rng_key: Any = field(default=None)

    def increment_step(self) -> JaxTrainState:
        """Return a new state with step incremented."""
        return JaxTrainState(
            model=self.model,
            opt_state=self.opt_state,
            optimizer=self.optimizer,
            step=self.step + 1,
            rng_key=self.rng_key,
        )


def create_train_state(
    model: Any,
    config: JaxTrainConfig,
) -> JaxTrainState:
    """Create initial training state with optax optimizer.

    Args:
        model: Equinox model to train.
        config: Training configuration.

    Returns:
        Initialized JaxTrainState.
    """
    _require_jax()

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
    )

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng_key = jax.random.PRNGKey(config.seed)

    logger.info(
        "Created JaxTrainState: lr=%.6f, weight_decay=%.6f, grad_clip=%.1f",
        config.learning_rate,
        config.weight_decay,
        config.gradient_clip,
    )

    return JaxTrainState(
        model=model,
        opt_state=opt_state,
        optimizer=optimizer,
        step=0,
        rng_key=rng_key,
    )


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------


def jax_compute_loss(
    model: Any,
    spatial: Any,
    vector: Any,
    action_mask: Any,
    target_policy: Any,
    target_value: Any,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    *,
    key: Any = None,
    inference: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Compute AlphaZero loss (policy cross-entropy + value MSE).

    Pure function suitable for use with jax.grad / eqx.filter_value_and_grad.
    Matches the loss computation in civ6_network.py compute_loss().

    Args:
        model: Equinox model.
        spatial: Spatial input (C, H, W) — single sample.
        vector: Vector input (V,).
        action_mask: Valid action mask (A,).
        target_policy: Target policy distribution (A,).
        target_value: Target value scalar.
        policy_loss_weight: Weight for policy loss.
        value_loss_weight: Weight for value loss.
        key: Optional PRNG key for dropout.
        inference: Whether to run in inference mode.

    Returns:
        Tuple of (total_loss, metrics_dict).
    """
    policy_logits, value = model(
        spatial, vector, action_mask, key=key, inference=inference
    )

    # Policy loss: cross-entropy with soft targets and action masking
    masked_logits = jnp.where(action_mask > 0, policy_logits, -1e9)
    log_probs = jax.nn.log_softmax(masked_logits)
    policy_loss = -jnp.sum(target_policy * log_probs)

    # Value loss: MSE
    value_loss = jnp.square(value - target_value)

    total_loss = policy_loss_weight * policy_loss + value_loss_weight * value_loss

    metrics = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }

    return total_loss, metrics


def _batch_loss(
    model: Any,
    batch: dict[str, Any],
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    *,
    key: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Compute mean loss over a batch using vmap.

    Args:
        model: Equinox model.
        batch: Dict with keys 'spatial', 'vector', 'action_mask', 'policy', 'value'.
        policy_loss_weight: Weight for policy loss.
        value_loss_weight: Weight for value loss.
        key: PRNG key.

    Returns:
        Tuple of (mean_total_loss, mean_metrics_dict).
    """
    batch_size = batch["spatial"].shape[0]

    # Generate per-sample keys if needed
    if key is not None:
        keys = jax.random.split(key, batch_size)
    else:
        keys = [None] * batch_size

    def single_loss(spatial: Any, vector: Any, mask: Any, policy: Any, value: Any, k: Any) -> tuple[Any, dict[str, Any]]:
        return jax_compute_loss(
            model, spatial, vector, mask, policy, value,
            policy_loss_weight=policy_loss_weight,
            value_loss_weight=value_loss_weight,
            key=k, inference=True,
        )

    # vmap over the batch dimension
    losses, metrics = jax.vmap(single_loss)(
        batch["spatial"],
        batch["vector"],
        batch["action_mask"],
        batch["policy"],
        batch["value"],
        jnp.stack(keys) if key is not None else jnp.zeros((batch_size,)),
    )

    mean_loss = jnp.mean(losses)
    mean_metrics = {k: jnp.mean(v) for k, v in metrics.items()}

    return mean_loss, mean_metrics


# ---------------------------------------------------------------------------
# Training Step
# ---------------------------------------------------------------------------


def jax_train_step(
    state: JaxTrainState,
    batch: dict[str, Any],
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
) -> tuple[JaxTrainState, dict[str, float]]:
    """Execute a single training step.

    Computes gradients, applies optimizer updates, and returns new state.

    Args:
        state: Current training state.
        batch: Training batch dict with JAX arrays.
        policy_loss_weight: Weight for policy loss term.
        value_loss_weight: Weight for value loss term.

    Returns:
        Tuple of (new_state, metrics_dict).
    """
    _require_jax()

    # Split key for this step
    rng_key, step_key = jax.random.split(state.rng_key)

    def loss_fn(model: Any) -> tuple[Any, dict[str, Any]]:
        return _batch_loss(
            model, batch,
            policy_loss_weight=policy_loss_weight,
            value_loss_weight=value_loss_weight,
            key=step_key,
        )

    # Compute loss and gradients (only differentiate array leaves)
    (_loss, metrics), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(state.model)

    # Apply optimizer updates
    updates, new_opt_state = state.optimizer.update(
        eqx.filter(grads, eqx.is_array),
        state.opt_state,
        eqx.filter(state.model, eqx.is_array),
    )
    new_model = eqx.apply_updates(state.model, updates)

    new_state = JaxTrainState(
        model=new_model,
        opt_state=new_opt_state,
        optimizer=state.optimizer,
        step=state.step + 1,
        rng_key=rng_key,
    )

    metrics_float = {k: float(v) for k, v in metrics.items()}
    metrics_float["step"] = float(new_state.step)

    return new_state, metrics_float


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


class JaxDataLoader:
    """Iterator that yields batches of JAX arrays from NumPy training data.

    Replaces PyTorch's DataLoader for JAX training loops.
    """

    def __init__(
        self,
        examples: list[Any],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        """Initialize data loader.

        Args:
            examples: List of training examples with spatial/vector/action_mask/policy/value.
            batch_size: Batch size.
            shuffle: Whether to shuffle examples each epoch.
            seed: Random seed for shuffling.
        """
        self._examples = examples
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return max(1, len(self._examples) // self._batch_size)

    def __iter__(self) -> JaxDataLoader:
        """Start iteration (optionally shuffle)."""
        _require_jax()

        indices = np.arange(len(self._examples))
        if self._shuffle:
            self._rng.shuffle(indices)

        self._current_indices = indices
        self._pos = 0
        return self

    def __next__(self) -> dict[str, Any]:
        """Get next batch as dict of JAX arrays."""
        if self._pos >= len(self._current_indices):
            raise StopIteration

        end = min(self._pos + self._batch_size, len(self._current_indices))
        batch_indices = self._current_indices[self._pos:end]
        self._pos = end

        batch_examples = [self._examples[i] for i in batch_indices]

        # Stack into arrays
        spatial = jnp.array(np.stack([ex.spatial for ex in batch_examples]))
        vector = jnp.array(np.stack([ex.vector for ex in batch_examples]))
        action_mask = jnp.array(np.stack([ex.action_mask for ex in batch_examples]))
        policy = jnp.array(np.stack([ex.policy for ex in batch_examples]))
        value = jnp.array(np.array([ex.value for ex in batch_examples]))

        return {
            "spatial": spatial,
            "vector": vector,
            "action_mask": action_mask,
            "policy": policy,
            "value": value,
        }
