"""Learner module for neural network gradient updates.

Wraps a network, optimizer, and loss function into a single training
step abstraction.  Supports gradient accumulation and optional mixed
precision.  All torch usage is guarded behind HAS_TORCH.

Layer 3 -- depends on networks (optional torch), config, types, logging.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.core.config import TrainingConfig
from src.core.networks import HAS_TORCH

logger = logging.getLogger(__name__)


class Learner:
    """Manages gradient updates for a dual-head network.

    The learner abstracts away the optimizer setup, gradient
    accumulation, and loss computation, exposing a single
    ``train_step`` method.

    When torch is not available, operates in "dry-run" mode:
    train_step returns synthetic metrics without doing real computation.

    Args:
        config: Training configuration.
        network: Neural network (DualHeadNetwork or NetworkProtocol).
            Only used when HAS_TORCH is True.
        loss_fn: Optional custom loss function.  If None, uses
            ``compute_combined_loss`` from the networks module.
        optimizer_factory: Optional callable to create optimizer.
            Receives network parameters and returns an optimizer.
            If None, uses AdamW with config settings.
    """

    def __init__(
        self,
        config: TrainingConfig,
        network: Any = None,
        loss_fn: Callable[..., tuple[Any, dict[str, float]]] | None = None,
        optimizer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._network = network
        self._custom_loss_fn = loss_fn
        self._step_count = 0
        self._accumulated_steps = 0
        self._optimizer: Any = None
        self._loss_fn: Callable[..., tuple[Any, dict[str, float]]] | None = None

        if HAS_TORCH and network is not None:
            self._setup_torch(optimizer_factory)
        else:
            logger.warning(
                "Learner running in dry-run mode (torch %s, network %s)",
                "available" if HAS_TORCH else "unavailable",
                "provided" if network is not None else "None",
            )

    def _setup_torch(self, optimizer_factory: Callable[..., Any] | None) -> None:
        """Initialize torch-specific training components."""
        import torch

        # Setup loss function
        if self._custom_loss_fn is not None:
            self._loss_fn = self._custom_loss_fn
        else:
            from src.core.networks.losses import compute_combined_loss
            self._loss_fn = compute_combined_loss

        # Setup optimizer
        if optimizer_factory is not None:
            self._optimizer = optimizer_factory(self._network.parameters())
        else:
            self._optimizer = torch.optim.AdamW(
                self._network.parameters(),
                lr=self._config.learning_rate,
                weight_decay=self._config.weight_decay,
            )

        # Move network to device
        device_str = self._resolve_device()
        self._device = torch.device(device_str)
        self._network.to(self._device)

        logger.info(
            "Learner initialized: device=%s, lr=%e, weight_decay=%e, "
            "grad_accum_steps=%d",
            self._device,
            self._config.learning_rate,
            self._config.weight_decay,
            self._config.gradient_accumulation_steps,
        )

    def _resolve_device(self) -> str:
        """Resolve the training device from config."""
        import torch

        device = self._config.device.value
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @property
    def step_count(self) -> int:
        """Total number of optimizer steps taken."""
        return self._step_count

    @property
    def network(self) -> Any:
        """The underlying neural network."""
        return self._network

    def train_step(self, batch: dict[str, NDArray[np.float32]]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with numpy arrays.  Expected keys:
                ``spatial``, ``vector``, ``action_mask``, ``policy``, ``value``.

        Returns:
            Dictionary of scalar metrics (total_loss, policy_loss, value_loss).
        """
        if not HAS_TORCH or self._network is None:
            return self._dry_run_step(batch)

        return self._torch_train_step(batch)

    def _torch_train_step(self, batch: dict[str, NDArray[np.float32]]) -> dict[str, float]:
        """Execute a real torch training step."""
        import torch

        self._network.train()

        # Convert numpy batch to torch tensors
        spatial = torch.from_numpy(batch["spatial"]).float().to(self._device)
        vector = torch.from_numpy(batch["vector"]).float().to(self._device)
        action_mask = torch.from_numpy(batch["action_mask"]).bool().to(self._device)
        target_policy = torch.from_numpy(batch["policy"]).float().to(self._device)
        target_value = torch.from_numpy(batch["value"]).float().to(self._device)

        # Forward pass
        policy_logits, predicted_value = self._network(spatial, vector, action_mask)

        # Compute loss
        if self._loss_fn is None:
            msg = "Loss function not initialized; call _setup_torch() first"
            raise RuntimeError(msg)
        total_loss, metrics = self._loss_fn(
            policy_logits,
            target_policy,
            predicted_value,
            target_value,
            action_mask,
        )

        # Scale loss for gradient accumulation
        accum_steps = self._config.gradient_accumulation_steps
        scaled_loss = total_loss / accum_steps
        scaled_loss.backward()

        self._accumulated_steps += 1

        # Optimizer step when accumulation is complete
        if self._accumulated_steps >= accum_steps:
            self._optimizer.step()
            self._optimizer.zero_grad()
            self._accumulated_steps = 0
            self._step_count += 1

        metrics["step"] = float(self._step_count)
        metrics["learning_rate"] = self._get_current_lr()
        return metrics

    def _dry_run_step(self, batch: dict[str, NDArray[np.float32]]) -> dict[str, float]:
        """Simulate a training step when torch is unavailable."""
        self._step_count += 1
        batch_size = batch.get("spatial", np.empty(0)).shape[0] if "spatial" in batch else 0
        return {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "step": float(self._step_count),
            "batch_size": float(batch_size),
            "dry_run": 1.0,
        }

    def _get_current_lr(self) -> float:
        """Get the current learning rate from the optimizer."""
        if self._optimizer is not None and self._optimizer.param_groups:
            return float(self._optimizer.param_groups[0]["lr"])
        return self._config.learning_rate

    def update_learning_rate(self, new_lr: float) -> None:
        """Update the learning rate of the optimizer.

        Args:
            new_lr: New learning rate value.
        """
        if self._optimizer is not None:
            for param_group in self._optimizer.param_groups:
                param_group["lr"] = new_lr
            logger.debug("Learning rate updated to %e", new_lr)

    def get_eval_fn(self) -> Callable[
        [NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]],
        tuple[NDArray[np.float32], float],
    ]:
        """Create an evaluation function from the current network.

        Returns a function suitable for MCTS and evaluation that converts
        numpy inputs to torch, runs inference, and converts back.

        Returns:
            Evaluation function with signature
            ``(spatial, vector, action_mask) -> (policy, value)``.
        """
        if not HAS_TORCH or self._network is None:
            return _dummy_eval_fn

        return self._make_torch_eval_fn()

    def _make_torch_eval_fn(self) -> Callable[
        [NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]],
        tuple[NDArray[np.float32], float],
    ]:
        """Build a torch-backed eval function."""
        import torch

        network = self._network
        device = self._device

        def eval_fn(
            spatial: NDArray[np.float32],
            vector: NDArray[np.float32],
            action_mask: NDArray[np.bool_],
        ) -> tuple[NDArray[np.float32], float]:
            sp = torch.from_numpy(spatial).float().unsqueeze(0).to(device)
            vec = torch.from_numpy(vector).float().unsqueeze(0).to(device)
            mask = torch.from_numpy(action_mask).bool().unsqueeze(0).to(device)

            probs, value = network.predict(sp, vec, mask)
            return probs.squeeze(0).cpu().numpy(), value.item()

        return eval_fn

    def save_state(self, path: str) -> None:
        """Save learner state (network + optimizer) to a file.

        Args:
            path: Destination file path.
        """
        if not HAS_TORCH or self._network is None:
            logger.warning("Cannot save state: torch unavailable or no network")
            return

        import torch

        state = {
            "network_state_dict": self._network.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "step_count": self._step_count,
        }
        torch.save(state, path)
        logger.info("Saved learner state to %s (step %d)", path, self._step_count)

    def load_state(self, path: str) -> None:
        """Load learner state from a file.

        Args:
            path: Source file path.
        """
        if not HAS_TORCH or self._network is None:
            logger.warning("Cannot load state: torch unavailable or no network")
            return

        import torch

        state = torch.load(path, map_location=self._device, weights_only=False)
        self._network.load_state_dict(state["network_state_dict"])
        self._optimizer.load_state_dict(state["optimizer_state_dict"])
        self._step_count = state.get("step_count", 0)
        logger.info("Loaded learner state from %s (step %d)", path, self._step_count)


def _dummy_eval_fn(
    spatial: NDArray[np.float32],
    vector: NDArray[np.float32],
    action_mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float32], float]:
    """Dummy evaluation function returning uniform policy."""
    valid = action_mask.astype(np.float32)
    total = valid.sum()
    if total > 0:
        policy = valid / total
    else:
        policy = np.ones_like(valid) / len(valid)
    return policy, 0.0
