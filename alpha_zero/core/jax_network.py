"""JAX/Equinox neural network implementations for Civ6 AlphaZero.

Mirrors the PyTorch architecture in civ6_network.py using Equinox modules:
- JaxResidualBlock: Residual block with GroupNorm (stateless)
- JaxSpatialEncoder: Convolutional spatial feature encoder
- JaxVectorEncoder: MLP for global features
- JaxPolicyHead: Action logit prediction
- JaxValueHead: Position evaluation
- JaxCiv6Network: Combined network
- JaxMicroCivNetwork: Simplified variant for MicroCiv environment
- JaxNetworkWrapper: Stateful wrapper implementing predict()/save()/load()

Requires: pip install -e '.[jax]' (jax, equinox, optax)

Usage:
    from alpha_zero.core.jax_network import JaxMicroCivNetwork, JaxNetworkWrapper

    key = jax.random.PRNGKey(42)
    model = JaxMicroCivNetwork(board_size=16, action_space_size=381, key=key)
    wrapper = JaxNetworkWrapper(model, seed=42)

    policy, value = wrapper.predict(spatial_np, vector_np, mask_np)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


def _require_jax() -> None:
    """Raise ImportError if JAX is not installed."""
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX backend requires: pip install -e '.[jax]' "
            "(installs jax, jaxlib, equinox, optax)"
        )


# ---------------------------------------------------------------------------
# Network Configuration (reusable with both frameworks)
# ---------------------------------------------------------------------------


@dataclass
class JaxNetworkConfig:
    """Configuration for JAX network architecture.

    Mirrors NetworkConfig from civ6_network.py with JAX-specific additions.
    """

    spatial_channels: int = 13
    spatial_size: int = 16
    vector_dim: int = 64
    num_res_blocks: int = 6
    num_filters: int = 64
    fc_hidden_dim: int = 256
    action_space_size: int = 381
    value_head_hidden: int = 128
    dropout: float = 0.0
    num_groups: int = 8  # GroupNorm groups (JAX-specific, replaces BatchNorm)

    @classmethod
    def from_network_config(cls, config: Any) -> JaxNetworkConfig:
        """Create from an existing NetworkConfig (from civ6_network.py or orchestrator).

        Args:
            config: A NetworkConfig-like object with standard fields.

        Returns:
            JaxNetworkConfig with matching parameters.
        """
        return cls(
            spatial_channels=getattr(config, "spatial_channels", 13),
            spatial_size=getattr(config, "spatial_size", 16),
            vector_dim=getattr(config, "vector_dim", 64),
            num_res_blocks=getattr(config, "num_res_blocks", 6),
            num_filters=getattr(config, "num_filters", 64),
            fc_hidden_dim=getattr(config, "fc_hidden_dim", 256),
            action_space_size=getattr(config, "action_space_size", 381),
            value_head_hidden=getattr(config, "value_head_hidden", 128),
            dropout=getattr(config, "dropout", 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "spatial_channels": self.spatial_channels,
            "spatial_size": self.spatial_size,
            "vector_dim": self.vector_dim,
            "num_res_blocks": self.num_res_blocks,
            "num_filters": self.num_filters,
            "fc_hidden_dim": self.fc_hidden_dim,
            "action_space_size": self.action_space_size,
            "value_head_hidden": self.value_head_hidden,
            "dropout": self.dropout,
            "num_groups": self.num_groups,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JaxNetworkConfig:
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Equinox Network Modules
# ---------------------------------------------------------------------------

if _JAX_AVAILABLE:

    class JaxGroupNorm(eqx.Module):
        """GroupNorm implementation for JAX.

        Stateless normalization (no running statistics), works identically
        in training and inference mode.
        """

        weight: jax.Array
        bias: jax.Array
        num_groups: int
        num_channels: int
        eps: float

        def __init__(
            self,
            num_channels: int,
            num_groups: int = 8,
            eps: float = 1e-5,
        ) -> None:
            self.num_channels = num_channels
            self.num_groups = num_groups
            self.eps = eps
            self.weight = jnp.ones(num_channels)
            self.bias = jnp.zeros(num_channels)

        def __call__(self, x: jax.Array) -> jax.Array:
            """Apply GroupNorm. Input shape: (C, H, W) — no batch dim."""
            c, h, w = x.shape
            g = self.num_groups
            x_grouped = x.reshape(g, c // g, h, w)
            mean = x_grouped.mean(axis=(1, 2, 3), keepdims=True)
            var = x_grouped.var(axis=(1, 2, 3), keepdims=True)
            x_norm = (x_grouped - mean) / jnp.sqrt(var + self.eps)
            x_norm = x_norm.reshape(c, h, w)
            return x_norm * self.weight[:, None, None] + self.bias[:, None, None]

    class JaxResidualBlock(eqx.Module):
        """Residual block with two convolutional layers and GroupNorm."""

        conv1: eqx.nn.Conv2d
        gn1: JaxGroupNorm
        conv2: eqx.nn.Conv2d
        gn2: JaxGroupNorm

        def __init__(self, num_filters: int, num_groups: int = 8, *, key: jax.Array) -> None:
            key1, key2 = jax.random.split(key)
            self.conv1 = eqx.nn.Conv2d(
                num_filters, num_filters, kernel_size=3, padding=1, use_bias=False, key=key1
            )
            self.gn1 = JaxGroupNorm(num_filters, num_groups)
            self.conv2 = eqx.nn.Conv2d(
                num_filters, num_filters, kernel_size=3, padding=1, use_bias=False, key=key2
            )
            self.gn2 = JaxGroupNorm(num_filters, num_groups)

        def __call__(self, x: jax.Array) -> jax.Array:
            """Forward pass. Input: (C, H, W)."""
            residual = x
            out = jax.nn.relu(self.gn1(self.conv1(x)))
            out = self.gn2(self.conv2(out))
            out = out + residual
            return jax.nn.relu(out)

    class JaxSpatialEncoder(eqx.Module):
        """Encodes spatial (map) features using convolutional layers."""

        conv_in: eqx.nn.Conv2d
        gn_in: JaxGroupNorm
        res_blocks: list[JaxResidualBlock]

        def __init__(
            self,
            in_channels: int,
            num_filters: int,
            num_res_blocks: int,
            num_groups: int = 8,
            *,
            key: jax.Array,
        ) -> None:
            keys = jax.random.split(key, num_res_blocks + 1)
            self.conv_in = eqx.nn.Conv2d(
                in_channels, num_filters, kernel_size=3, padding=1, use_bias=False, key=keys[0]
            )
            self.gn_in = JaxGroupNorm(num_filters, num_groups)
            self.res_blocks = [
                JaxResidualBlock(num_filters, num_groups, key=keys[i + 1])
                for i in range(num_res_blocks)
            ]

        def __call__(self, x: jax.Array) -> jax.Array:
            """Forward pass. Input: (C, H, W). Output: (num_filters, H, W)."""
            out = jax.nn.relu(self.gn_in(self.conv_in(x)))
            for block in self.res_blocks:
                out = block(out)
            return out

    class JaxVectorEncoder(eqx.Module):
        """Encodes non-spatial (vector) features using MLP."""

        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear
        dropout: eqx.nn.Dropout

        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            dropout: float = 0.0,
            *,
            key: jax.Array,
        ) -> None:
            key1, key2 = jax.random.split(key)
            self.fc1 = eqx.nn.Linear(input_dim, hidden_dim, key=key1)
            self.fc2 = eqx.nn.Linear(hidden_dim, output_dim, key=key2)
            self.dropout = eqx.nn.Dropout(p=dropout)

        def __call__(
            self, x: jax.Array, *, key: jax.Array | None = None, inference: bool = True
        ) -> jax.Array:
            """Forward pass. Input: (D,). Output: (output_dim,)."""
            out = jax.nn.relu(self.fc1(x))
            if key is not None and not inference:
                out = self.dropout(out, key=key, inference=False)
            return self.fc2(out)

    class JaxPolicyHead(eqx.Module):
        """Policy head for action prediction."""

        conv: eqx.nn.Conv2d
        gn: JaxGroupNorm
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(
            self,
            spatial_filters: int,
            spatial_size: int,
            vector_dim: int,
            action_space_size: int,
            hidden_dim: int = 256,
            *,
            key: jax.Array,
        ) -> None:
            key1, key2, key3 = jax.random.split(key, 3)
            self.conv = eqx.nn.Conv2d(
                spatial_filters, 2, kernel_size=1, use_bias=False, key=key1
            )
            self.gn = JaxGroupNorm(2, num_groups=1)
            flat_spatial_dim = 2 * spatial_size * spatial_size
            combined_dim = flat_spatial_dim + vector_dim
            self.fc1 = eqx.nn.Linear(combined_dim, hidden_dim, key=key2)
            self.fc2 = eqx.nn.Linear(hidden_dim, action_space_size, key=key3)

        def __call__(
            self, spatial_features: jax.Array, vector_features: jax.Array
        ) -> jax.Array:
            """Forward pass. Returns logits of shape (action_space_size,)."""
            s = jax.nn.relu(self.gn(self.conv(spatial_features)))
            s = s.ravel()
            combined = jnp.concatenate([s, vector_features])
            out = jax.nn.relu(self.fc1(combined))
            return self.fc2(out)

    class JaxValueHead(eqx.Module):
        """Value head for position evaluation."""

        conv: eqx.nn.Conv2d
        gn: JaxGroupNorm
        fc1: eqx.nn.Linear
        fc2: eqx.nn.Linear

        def __init__(
            self,
            spatial_filters: int,
            spatial_size: int,
            vector_dim: int,
            hidden_dim: int = 128,
            *,
            key: jax.Array,
        ) -> None:
            key1, key2, key3 = jax.random.split(key, 3)
            self.conv = eqx.nn.Conv2d(
                spatial_filters, 1, kernel_size=1, use_bias=False, key=key1
            )
            self.gn = JaxGroupNorm(1, num_groups=1)
            flat_spatial_dim = spatial_size * spatial_size
            combined_dim = flat_spatial_dim + vector_dim
            self.fc1 = eqx.nn.Linear(combined_dim, hidden_dim, key=key2)
            self.fc2 = eqx.nn.Linear(hidden_dim, 1, key=key3)

        def __call__(
            self, spatial_features: jax.Array, vector_features: jax.Array
        ) -> jax.Array:
            """Forward pass. Returns scalar value in [-1, 1]."""
            s = jax.nn.relu(self.gn(self.conv(spatial_features)))
            s = s.ravel()
            combined = jnp.concatenate([s, vector_features])
            out = jax.nn.relu(self.fc1(combined))
            return jax.nn.tanh(self.fc2(out)).squeeze()

    class JaxCiv6Network(eqx.Module):
        """Full AlphaZero network for Civilization VI (JAX/Equinox).

        Architecture mirrors Civ6Network from civ6_network.py:
        1. Spatial encoder (ResNet with GroupNorm) processes map features
        2. Vector encoder (MLP) processes global features
        3. Policy head outputs action logits
        4. Value head outputs scalar value estimate
        """

        spatial_encoder: JaxSpatialEncoder
        vector_encoder: JaxVectorEncoder
        policy_head: JaxPolicyHead
        value_head: JaxValueHead
        config: JaxNetworkConfig

        def __init__(self, config: JaxNetworkConfig, *, key: jax.Array) -> None:
            """Initialize network from configuration.

            Args:
                config: Network architecture configuration.
                key: JAX PRNG key for weight initialization.
            """
            key1, key2, key3, key4 = jax.random.split(key, 4)

            self.config = config
            self.spatial_encoder = JaxSpatialEncoder(
                in_channels=config.spatial_channels,
                num_filters=config.num_filters,
                num_res_blocks=config.num_res_blocks,
                num_groups=config.num_groups,
                key=key1,
            )
            self.vector_encoder = JaxVectorEncoder(
                input_dim=config.vector_dim,
                hidden_dim=config.fc_hidden_dim,
                output_dim=config.fc_hidden_dim,
                dropout=config.dropout,
                key=key2,
            )
            self.policy_head = JaxPolicyHead(
                spatial_filters=config.num_filters,
                spatial_size=config.spatial_size,
                vector_dim=config.fc_hidden_dim,
                action_space_size=config.action_space_size,
                hidden_dim=config.fc_hidden_dim,
                key=key3,
            )
            self.value_head = JaxValueHead(
                spatial_filters=config.num_filters,
                spatial_size=config.spatial_size,
                vector_dim=config.fc_hidden_dim,
                hidden_dim=config.value_head_hidden,
                key=key4,
            )

        def __call__(
            self,
            spatial: jax.Array,
            vector: jax.Array,
            action_mask: jax.Array | None = None,
            *,
            key: jax.Array | None = None,
            inference: bool = True,
        ) -> tuple[jax.Array, jax.Array]:
            """Forward pass for a single sample (no batch dimension).

            Args:
                spatial: Spatial input (C, H, W).
                vector: Vector input (V,).
                action_mask: Valid action mask (A,). If None, all actions valid.
                key: PRNG key for dropout (only used when inference=False).
                inference: If True, dropout is disabled.

            Returns:
                Tuple of (policy_logits, value).
            """
            dropout_key = key
            spatial_features = self.spatial_encoder(spatial)
            vector_features = self.vector_encoder(
                vector, key=dropout_key, inference=inference
            )

            policy_logits = self.policy_head(spatial_features, vector_features)
            value = self.value_head(spatial_features, vector_features)

            if action_mask is not None:
                policy_logits = jnp.where(action_mask > 0, policy_logits, -1e9)

            return policy_logits, value

    class JaxMicroCivNetwork(JaxCiv6Network):
        """Simplified network for MicroCiv environment.

        Uses smaller architecture for faster training (13 channels, 16x16 board).
        """

        def __init__(
            self,
            board_size: int = 16,
            action_space_size: int = 381,
            num_res_blocks: int = 6,
            num_filters: int = 64,
            num_groups: int = 8,
            config: JaxNetworkConfig | None = None,
            *,
            key: jax.Array,
        ) -> None:
            if config is None:
                config = JaxNetworkConfig(
                    spatial_channels=13,
                    spatial_size=board_size,
                    vector_dim=64,
                    num_res_blocks=num_res_blocks,
                    num_filters=num_filters,
                    fc_hidden_dim=128,
                    action_space_size=action_space_size,
                    value_head_hidden=64,
                    num_groups=num_groups,
                )
            super().__init__(config=config, key=key)


# ---------------------------------------------------------------------------
# Wrapper providing stateful interface (predict, save, load)
# ---------------------------------------------------------------------------


class JaxNetworkWrapper:
    """Wraps an Equinox model with a stateful predict/save/load interface.

    Manages PRNG key state, JIT compilation, and NumPy I/O conversion.
    This is the primary interface used by MCTS adapters and training code.
    """

    def __init__(
        self,
        model: Any,
        seed: int = 0,
        jit_enabled: bool = True,
    ) -> None:
        """Initialize wrapper.

        Args:
            model: An Equinox module (JaxCiv6Network or JaxMicroCivNetwork).
            seed: Random seed for PRNG key management.
            jit_enabled: Whether to JIT-compile the forward pass.
        """
        _require_jax()

        self._model = model
        self._key = jax.random.PRNGKey(seed)
        self._jit_enabled = jit_enabled

        # Pre-compile forward pass for single sample
        if jit_enabled:
            self._jit_forward = eqx.filter_jit(self._forward_single)
        else:
            self._jit_forward = self._forward_single

        # Count parameters
        leaves = jax.tree.leaves(eqx.filter(model, eqx.is_array))
        self._param_count = sum(leaf.size for leaf in leaves)

        logger.info(
            "JaxNetworkWrapper created: params=%d (%.2f MB), jit=%s",
            self._param_count,
            self._param_count * 4 / (1024 * 1024),
            jit_enabled,
        )

    @property
    def model(self) -> Any:
        """The underlying Equinox model."""
        return self._model

    @property
    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return self._param_count

    def _next_key(self) -> Any:
        """Split and return a new PRNG subkey."""
        self._key, subkey = jax.random.split(self._key)
        return subkey

    def _forward_single(
        self,
        model: Any,
        spatial: Any,
        vector: Any,
        action_mask: Any,
    ) -> tuple[Any, Any]:
        """Forward pass for a single sample. Pure function for JIT."""
        policy_logits, value = model(
            spatial, vector, action_mask, inference=True
        )
        policy_probs = jax.nn.softmax(policy_logits)
        return policy_probs, value

    def predict(
        self,
        spatial: np.ndarray,
        vector: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict policy probabilities and value from NumPy arrays.

        Handles batched input (B, ...) by iterating over the batch dimension,
        since Equinox modules operate on single samples by default.

        Args:
            spatial: Spatial features, shape (B, C, H, W) or (C, H, W).
            vector: Vector features, shape (B, V) or (V,).
            action_mask: Valid action mask, shape (B, A) or (A,).

        Returns:
            Tuple of (policy_probs, value) as NumPy arrays.
        """
        _require_jax()

        is_batched = spatial.ndim == 4

        if is_batched:
            # Process batch by vmapping
            sp = jnp.array(spatial)
            vec = jnp.array(vector)
            mask = jnp.array(action_mask)

            batched_fn = jax.vmap(
                lambda s, v, m: self._jit_forward(self._model, s, v, m)
            )
            policy_probs, values = batched_fn(sp, vec, mask)

            return np.asarray(policy_probs), np.asarray(values)
        else:
            # Single sample
            sp = jnp.array(spatial)
            vec = jnp.array(vector)
            mask = jnp.array(action_mask)

            policy_probs, value = self._jit_forward(self._model, sp, vec, mask)

            return np.asarray(policy_probs), np.asarray(value)

    def predict_single(
        self,
        spatial: np.ndarray,
        vector: np.ndarray,
        action_mask: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Predict for a single state (no batch dimension).

        Args:
            spatial: Spatial input (C, H, W).
            vector: Vector input (V,).
            action_mask: Valid action mask (A,).

        Returns:
            Tuple of (policy_probs, value_scalar).
        """
        policy, value = self.predict(spatial, vector, action_mask)
        return policy, float(value)

    def get_params(self) -> Any:
        """Get the model's parameter pytree."""
        return eqx.filter(self._model, eqx.is_array)

    def set_params(self, params: Any) -> None:
        """Update the model's parameters from a pytree.

        Args:
            params: Parameter pytree matching the model structure.
        """
        self._model = eqx.combine(params, eqx.filter(self._model, lambda x: not eqx.is_array(x)))

    def save(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """Save model to disk.

        Saves:
        - {path}.eqx: Equinox model parameters
        - {path}.json: Configuration and metadata

        Args:
            path: Base path (without extension).
            metadata: Optional metadata to include.
        """
        model_path = f"{path}.eqx"
        meta_path = f"{path}.json"

        eqx.tree_serialise_leaves(model_path, self._model)

        meta = {
            "config": self._model.config.to_dict(),
            "param_count": self._param_count,
            "framework": "jax",
        }
        if metadata:
            meta.update(metadata)

        Path(meta_path).write_text(json.dumps(meta, indent=2))

        logger.info("Saved JAX model to %s", model_path)

    @classmethod
    def load(
        cls,
        path: str,
        seed: int = 0,
        jit_enabled: bool = True,
    ) -> JaxNetworkWrapper:
        """Load model from disk.

        Args:
            path: Base path (without extension).
            seed: Random seed for the wrapper.
            jit_enabled: Whether to JIT-compile the forward pass.

        Returns:
            JaxNetworkWrapper with loaded parameters.
        """
        _require_jax()

        model_path = f"{path}.eqx"
        meta_path = f"{path}.json"

        # Load metadata to reconstruct config
        meta = json.loads(Path(meta_path).read_text())
        config = JaxNetworkConfig.from_dict(meta["config"])

        # Create template model with same architecture
        template_key = jax.random.PRNGKey(0)
        template = JaxCiv6Network(config=config, key=template_key)

        # Load saved parameters into template
        model = eqx.tree_deserialise_leaves(model_path, template)

        logger.info("Loaded JAX model from %s", model_path)

        return cls(model, seed=seed, jit_enabled=jit_enabled)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_jax_network(
    config: Any,
    board_size: int,
    action_space_size: int,
    seed: int = 0,
    jit_enabled: bool = True,
) -> JaxNetworkWrapper:
    """Create a JAX network from configuration.

    Factory function matching create_network() signature pattern.

    Args:
        config: NetworkConfig-like object with architecture parameters.
        board_size: Game board size.
        action_space_size: Size of action space.
        seed: Random seed.
        jit_enabled: Whether to JIT-compile inference.

    Returns:
        JaxNetworkWrapper ready for inference.
    """
    _require_jax()

    jax_config = JaxNetworkConfig.from_network_config(config)
    jax_config.spatial_size = board_size
    jax_config.action_space_size = action_space_size

    key = jax.random.PRNGKey(seed)

    if board_size <= 16 and jax_config.spatial_channels <= 13:
        logger.info(
            "Creating JaxMicroCivNetwork: board=%d, actions=%d, filters=%d",
            board_size,
            action_space_size,
            jax_config.num_filters,
        )
        model = JaxMicroCivNetwork(
            board_size=board_size,
            action_space_size=action_space_size,
            num_res_blocks=jax_config.num_res_blocks,
            num_filters=jax_config.num_filters,
            num_groups=jax_config.num_groups,
            key=key,
        )
    else:
        logger.info(
            "Creating JaxCiv6Network: board=%d, channels=%d, filters=%d",
            board_size,
            jax_config.spatial_channels,
            jax_config.num_filters,
        )
        model = JaxCiv6Network(config=jax_config, key=key)

    return JaxNetworkWrapper(model, seed=seed, jit_enabled=jit_enabled)
