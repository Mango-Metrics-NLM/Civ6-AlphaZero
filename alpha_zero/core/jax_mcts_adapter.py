"""Adapter making JAX networks compatible with the MCTSNetwork Protocol.

The existing MCTS implementations (Civ6MCTS, GumbelMCTS) require networks
that accept torch.Tensor inputs and return torch.Tensor outputs via the
MCTSNetwork Protocol defined in civ6_mcts.py.

This adapter bridges JAX networks (which work with NumPy/JAX arrays) to
satisfy that Protocol, allowing MCTS to use JAX networks with zero changes
to the MCTS code.

Usage:
    from alpha_zero.core.jax_network import JaxMicroCivNetwork, JaxNetworkWrapper
    from alpha_zero.core.jax_mcts_adapter import JaxMCTSAdapter
    from alpha_zero.core.civ6_mcts import Civ6MCTS

    wrapper = JaxNetworkWrapper(model, seed=42)
    adapter = JaxMCTSAdapter(wrapper)

    # adapter now satisfies MCTSNetwork Protocol
    mcts = Civ6MCTS(adapter, mcts_config, device="cpu")
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class JaxMCTSAdapter:
    """Adapts a JaxNetworkWrapper to the MCTSNetwork Protocol.

    The MCTSNetwork Protocol (civ6_mcts.py:40) requires:
    - predict(spatial: torch.Tensor, vector: torch.Tensor,
              action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]
    - eval() -> None

    This adapter converts torch.Tensor → NumPy → JAX inference → NumPy → torch.Tensor
    at the boundary, keeping the MCTS code completely unchanged.
    """

    def __init__(self, wrapper: Any, device: str = "cpu") -> None:
        """Initialize the adapter.

        Args:
            wrapper: A JaxNetworkWrapper instance with predict() method.
            device: PyTorch device for output tensors.
        """
        self._wrapper = wrapper
        self._device = device
        logger.info("JaxMCTSAdapter created: device=%s", device)

    @property
    def wrapper(self) -> Any:
        """The underlying JaxNetworkWrapper."""
        return self._wrapper

    def predict(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict policy and value, accepting and returning torch.Tensor.

        Satisfies the MCTSNetwork Protocol. Internally converts to NumPy,
        runs JAX inference, and converts back to torch.Tensor.

        Args:
            spatial: Spatial features (B, C, H, W).
            vector: Vector features (B, D).
            action_mask: Valid action mask (B, A).

        Returns:
            Tuple of (policy_probs, value) as torch.Tensor on self._device.
        """
        # Convert torch → numpy
        sp_np = spatial.detach().cpu().numpy()
        vec_np = vector.detach().cpu().numpy()
        mask_np = action_mask.detach().cpu().numpy()

        # Run JAX inference
        policy_np, value_np = self._wrapper.predict(sp_np, vec_np, mask_np)

        # Ensure writable arrays (JAX may return read-only views)
        if not policy_np.flags.writeable:
            policy_np = np.array(policy_np, copy=True)
        if not value_np.flags.writeable:
            value_np = np.array(value_np, copy=True)

        # Convert numpy → torch
        policy_tensor = torch.from_numpy(policy_np).float().to(self._device)
        value_tensor = torch.from_numpy(value_np.reshape(-1)).float().to(self._device)

        return policy_tensor, value_tensor

    def eval(self) -> None:
        """Set to evaluation mode (no-op for JAX).

        JAX/Equinox models are stateless and always operate in a
        deterministic mode when inference=True (the default in the wrapper).
        """

    def train(self) -> None:
        """Set to training mode (no-op for JAX at the adapter level).

        Training in JAX uses explicit key passing for dropout, handled
        by the JaxTrainer, not by this adapter.
        """

    def parameters(self) -> Any:
        """Return model parameters (for compatibility checks).

        Returns:
            Equinox parameter pytree.
        """
        return self._wrapper.get_params()
