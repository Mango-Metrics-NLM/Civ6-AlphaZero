"""Auxiliary policy targets for improved training.

Implements additional training targets that have been shown to improve
AlphaZero-style training by providing richer learning signals.

Available auxiliary targets:
- attention_targets: Board position attention maps
- opponent_modeling: Predict opponent's next move
- phase_prediction: Early/mid/late game classification
- future_value: Value prediction N moves ahead

Usage:
    from alpha_zero.optimizations.auxiliary_targets import (
        AuxiliaryTargetsConfig,
        AuxiliaryTargetGenerator,
        AuxiliaryLossCalculator,
    )

    config = AuxiliaryTargetsConfig(
        enable_attention_targets=True,
        enable_opponent_modeling=True,
        attention_weight=0.1,
        opponent_weight=0.2,
    )
    generator = AuxiliaryTargetGenerator(config)
    loss_calc = AuxiliaryLossCalculator(config)

    # During data generation
    aux_targets = generator.generate_targets(game_state, game_history)

    # During training
    aux_loss = loss_calc.compute_loss(model_outputs, aux_targets)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GamePhase(str, Enum):
    """Game phase classification."""

    EARLY = "early"
    MID = "mid"
    LATE = "late"


@dataclass
class AuxiliaryTargetsConfig:
    """Configuration for auxiliary training targets.

    Attributes:
        enable_attention_targets: Enable board attention map targets
        enable_opponent_modeling: Enable opponent move prediction
        enable_phase_prediction: Enable game phase classification
        enable_future_value: Enable future value prediction
        attention_weight: Loss weight for attention targets
        opponent_weight: Loss weight for opponent modeling
        phase_weight: Loss weight for phase prediction
        future_value_weight: Loss weight for future value
        future_value_steps: How many moves ahead to predict value
        phase_thresholds: Turn thresholds for phase classification
        attention_blur_sigma: Gaussian blur sigma for attention maps
        opponent_history_length: Moves to consider for opponent modeling
    """

    enable_attention_targets: bool = True
    enable_opponent_modeling: bool = True
    enable_phase_prediction: bool = True
    enable_future_value: bool = True
    attention_weight: float = 0.1
    opponent_weight: float = 0.2
    phase_weight: float = 0.05
    future_value_weight: float = 0.1
    future_value_steps: int = 5
    phase_thresholds: tuple[int, int] = field(default=(20, 60))
    attention_blur_sigma: float = 1.0
    opponent_history_length: int = 5

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.attention_weight < 0:
            raise ValueError(
                f"attention_weight must be >= 0, got {self.attention_weight}"
            )
        if self.opponent_weight < 0:
            raise ValueError(f"opponent_weight must be >= 0, got {self.opponent_weight}")
        if self.phase_weight < 0:
            raise ValueError(f"phase_weight must be >= 0, got {self.phase_weight}")
        if self.future_value_weight < 0:
            raise ValueError(
                f"future_value_weight must be >= 0, got {self.future_value_weight}"
            )
        if self.future_value_steps < 1:
            raise ValueError(
                f"future_value_steps must be >= 1, got {self.future_value_steps}"
            )
        if len(self.phase_thresholds) != 2:
            raise ValueError("phase_thresholds must have exactly 2 values")
        if self.phase_thresholds[0] >= self.phase_thresholds[1]:
            raise ValueError(
                f"phase_thresholds[0] must be < phase_thresholds[1], "
                f"got {self.phase_thresholds}"
            )
        if self.attention_blur_sigma < 0:
            raise ValueError(
                f"attention_blur_sigma must be >= 0, got {self.attention_blur_sigma}"
            )
        if self.opponent_history_length < 1:
            raise ValueError(
                f"opponent_history_length must be >= 1, got {self.opponent_history_length}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuxiliaryTargetsConfig:
        """Create from dictionary."""
        # Handle phase_thresholds tuple conversion
        if "phase_thresholds" in data and isinstance(data["phase_thresholds"], list):
            data = data.copy()
            data["phase_thresholds"] = tuple(data["phase_thresholds"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enable_attention_targets": self.enable_attention_targets,
            "enable_opponent_modeling": self.enable_opponent_modeling,
            "enable_phase_prediction": self.enable_phase_prediction,
            "enable_future_value": self.enable_future_value,
            "attention_weight": self.attention_weight,
            "opponent_weight": self.opponent_weight,
            "phase_weight": self.phase_weight,
            "future_value_weight": self.future_value_weight,
            "future_value_steps": self.future_value_steps,
            "phase_thresholds": list(self.phase_thresholds),
            "attention_blur_sigma": self.attention_blur_sigma,
            "opponent_history_length": self.opponent_history_length,
        }

    def get_total_weight(self) -> float:
        """Get total weight of all enabled auxiliary targets."""
        total = 0.0
        if self.enable_attention_targets:
            total += self.attention_weight
        if self.enable_opponent_modeling:
            total += self.opponent_weight
        if self.enable_phase_prediction:
            total += self.phase_weight
        if self.enable_future_value:
            total += self.future_value_weight
        return total

    def any_enabled(self) -> bool:
        """Check if any auxiliary target is enabled."""
        return (
            self.enable_attention_targets
            or self.enable_opponent_modeling
            or self.enable_phase_prediction
            or self.enable_future_value
        )


@dataclass
class AuxiliaryTargets:
    """Container for auxiliary target values.

    Attributes:
        attention_map: Attention weights over board positions
        opponent_action: Predicted opponent action distribution
        game_phase: One-hot encoded game phase
        future_value: Predicted value N steps ahead
    """

    attention_map: np.ndarray | None = None
    opponent_action: np.ndarray | None = None
    game_phase: np.ndarray | None = None
    future_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        if self.attention_map is not None:
            result["attention_map"] = self.attention_map.tolist()
        if self.opponent_action is not None:
            result["opponent_action"] = self.opponent_action.tolist()
        if self.game_phase is not None:
            result["game_phase"] = self.game_phase.tolist()
        if self.future_value is not None:
            result["future_value"] = self.future_value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuxiliaryTargets:
        """Create from dictionary."""
        return cls(
            attention_map=np.array(data["attention_map"])
            if "attention_map" in data
            else None,
            opponent_action=np.array(data["opponent_action"])
            if "opponent_action" in data
            else None,
            game_phase=np.array(data["game_phase"]) if "game_phase" in data else None,
            future_value=data.get("future_value"),
        )


class AuxiliaryTargetGenerator:
    """Generates auxiliary training targets from game states.

    This class creates additional training signals beyond the standard
    policy and value targets used in AlphaZero.

    Example:
        >>> config = AuxiliaryTargetsConfig(enable_attention_targets=True)
        >>> generator = AuxiliaryTargetGenerator(config)
        >>> targets = generator.generate_targets(state, history)
    """

    def __init__(
        self,
        config: AuxiliaryTargetsConfig,
        board_size: int = 16,
        action_space_size: int = 1000,
        seed: int | None = None,
    ):
        """Initialize auxiliary target generator.

        Args:
            config: Auxiliary targets configuration
            board_size: Size of the game board (for attention maps)
            action_space_size: Total number of possible actions
            seed: Random seed for reproducibility
        """
        self.config = config
        self.board_size = board_size
        self.action_space_size = action_space_size
        self._rng = np.random.default_rng(seed)
        self._targets_generated = 0

    def generate_targets(
        self,
        game_state: dict[str, Any],
        game_history: list[dict[str, Any]] | None = None,
        future_states: list[dict[str, Any]] | None = None,
    ) -> AuxiliaryTargets:
        """Generate all enabled auxiliary targets.

        Args:
            game_state: Current game state dictionary
            game_history: List of previous game states (for opponent modeling)
            future_states: List of future game states (for future value)

        Returns:
            AuxiliaryTargets container with computed targets
        """
        targets = AuxiliaryTargets()

        if self.config.enable_attention_targets:
            targets.attention_map = self._generate_attention_targets(game_state)

        if self.config.enable_opponent_modeling:
            targets.opponent_action = self._generate_opponent_targets(
                game_state, game_history
            )

        if self.config.enable_phase_prediction:
            targets.game_phase = self._generate_phase_targets(game_state)

        if self.config.enable_future_value:
            targets.future_value = self._generate_future_value_targets(
                game_state, future_states
            )

        self._targets_generated += 1
        return targets

    def _generate_attention_targets(self, game_state: dict[str, Any]) -> np.ndarray:
        """Generate attention map targets based on game state.

        Creates a soft attention distribution over board positions based on
        strategic importance (e.g., units, cities, contested areas).

        Args:
            game_state: Current game state

        Returns:
            Attention weights of shape (board_size, board_size)
        """
        attention = np.zeros((self.board_size, self.board_size), dtype=np.float32)

        # Extract relevant features from game state
        board = game_state.get("board", None)
        units = game_state.get("units", [])
        cities = game_state.get("cities", [])

        # Base attention from board state
        if board is not None and isinstance(board, np.ndarray):
            # Higher attention on non-empty cells
            if board.ndim == 2:
                attention += (board != 0).astype(np.float32) * 0.3
            elif board.ndim == 3:
                attention += (board.sum(axis=-1) != 0).astype(np.float32) * 0.3

        # High attention on unit positions
        for unit in units:
            if isinstance(unit, dict) and "position" in unit:
                x, y = unit["position"]
                if 0 <= x < self.board_size and 0 <= y < self.board_size:
                    attention[x, y] += 0.5

        # Very high attention on city positions
        for city in cities:
            if isinstance(city, dict) and "position" in city:
                x, y = city["position"]
                if 0 <= x < self.board_size and 0 <= y < self.board_size:
                    attention[x, y] += 1.0

        # Apply Gaussian blur for smooth attention
        if self.config.attention_blur_sigma > 0:
            attention = self._gaussian_blur(attention, self.config.attention_blur_sigma)

        # Normalize to sum to 1
        total = attention.sum()
        if total > 0:
            attention /= total
        else:
            # Uniform attention if no features
            attention = np.ones_like(attention) / attention.size

        return attention

    def _gaussian_blur(self, array: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to 2D array.

        Simple implementation without scipy dependency.

        Args:
            array: Input 2D array
            sigma: Gaussian blur sigma

        Returns:
            Blurred array
        """
        # Create Gaussian kernel
        kernel_size = int(4 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = np.arange(kernel_size) - kernel_size // 2
        kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
        kernel_1d /= kernel_1d.sum()

        # Apply separable convolution
        result = array.copy()

        # Horizontal pass
        padded = np.pad(result, ((0, 0), (kernel_size // 2, kernel_size // 2)), "edge")
        for i in range(array.shape[0]):
            result[i, :] = np.convolve(padded[i, :], kernel_1d, mode="valid")

        # Vertical pass
        padded = np.pad(result, ((kernel_size // 2, kernel_size // 2), (0, 0)), "edge")
        for j in range(array.shape[1]):
            result[:, j] = np.convolve(padded[:, j], kernel_1d, mode="valid")

        return result

    def _generate_opponent_targets(
        self,
        game_state: dict[str, Any],
        game_history: list[dict[str, Any]] | None,
    ) -> np.ndarray:
        """Generate opponent action prediction targets.

        Creates a probability distribution over actions representing
        the predicted opponent behavior based on history.

        Args:
            game_state: Current game state
            game_history: Previous game states with opponent actions

        Returns:
            Action probability distribution of shape (action_space_size,)
        """
        opponent_probs = np.zeros(self.action_space_size, dtype=np.float32)

        if game_history is None or len(game_history) == 0:
            # No history - return uniform over valid actions
            action_mask = game_state.get("action_mask", None)
            if action_mask is not None:
                valid_actions = action_mask > 0
                if valid_actions.any():
                    opponent_probs[valid_actions] = 1.0 / valid_actions.sum()
                    return opponent_probs

            # Fallback to uniform
            opponent_probs[:] = 1.0 / self.action_space_size
            return opponent_probs

        # Analyze recent opponent actions
        history_length = min(len(game_history), self.config.opponent_history_length)
        recent_history = game_history[-history_length:]

        action_counts = np.zeros(self.action_space_size, dtype=np.float32)
        for state in recent_history:
            opponent_action = state.get("opponent_action", None)
            if opponent_action is not None and 0 <= opponent_action < self.action_space_size:
                action_counts[opponent_action] += 1

        # Convert to probabilities with smoothing
        if action_counts.sum() > 0:
            # Add small smoothing factor
            smoothed = action_counts + 0.1
            opponent_probs = smoothed / smoothed.sum()
        else:
            opponent_probs[:] = 1.0 / self.action_space_size

        return opponent_probs

    def _generate_phase_targets(self, game_state: dict[str, Any]) -> np.ndarray:
        """Generate game phase classification targets.

        Creates a one-hot (or soft) encoding of the current game phase.

        Args:
            game_state: Current game state

        Returns:
            One-hot encoded phase of shape (3,) for [early, mid, late]
        """
        turn = game_state.get("turn", 0)
        early_threshold, late_threshold = self.config.phase_thresholds

        phase = np.zeros(3, dtype=np.float32)

        if turn < early_threshold:
            phase[0] = 1.0  # Early
        elif turn < late_threshold:
            phase[1] = 1.0  # Mid
        else:
            phase[2] = 1.0  # Late

        return phase

    def _generate_future_value_targets(
        self,
        game_state: dict[str, Any],
        future_states: list[dict[str, Any]] | None,
    ) -> float:
        """Generate future value prediction targets.

        Creates a target value representing the game state N moves ahead.

        Args:
            game_state: Current game state
            future_states: List of future game states

        Returns:
            Predicted future value in range [-1, 1]
        """
        if future_states is None or len(future_states) == 0:
            # No future information - use current value if available
            return game_state.get("value", 0.0)

        # Get value from N steps ahead
        target_step = min(self.config.future_value_steps, len(future_states))
        future_state = future_states[target_step - 1]

        # Return future value, discounted slightly
        future_value = future_state.get("value", 0.0)
        discount = 0.99**target_step
        return float(future_value * discount)

    def get_stats(self) -> dict[str, Any]:
        """Get generator statistics.

        Returns:
            Dictionary with generation statistics
        """
        return {
            "targets_generated": self._targets_generated,
            "config": self.config.to_dict(),
            "board_size": self.board_size,
            "action_space_size": self.action_space_size,
        }


class AuxiliaryLossCalculator:
    """Calculates auxiliary losses for training.

    Computes weighted losses for each enabled auxiliary target type.

    Example:
        >>> config = AuxiliaryTargetsConfig()
        >>> calculator = AuxiliaryLossCalculator(config)
        >>> loss = calculator.compute_loss(predictions, targets)
    """

    def __init__(self, config: AuxiliaryTargetsConfig):
        """Initialize loss calculator.

        Args:
            config: Auxiliary targets configuration
        """
        self.config = config
        self._loss_history: list[dict[str, float]] = []

    def compute_loss(
        self,
        predictions: dict[str, np.ndarray],
        targets: AuxiliaryTargets,
    ) -> dict[str, float]:
        """Compute all auxiliary losses.

        Args:
            predictions: Model predictions dictionary with keys:
                - 'attention': Predicted attention map
                - 'opponent': Predicted opponent action distribution
                - 'phase': Predicted phase probabilities
                - 'future_value': Predicted future value
            targets: Ground truth auxiliary targets

        Returns:
            Dictionary with individual and total losses
        """
        losses: dict[str, float] = {}
        total_loss = 0.0

        if self.config.enable_attention_targets and targets.attention_map is not None:
            pred_attention = predictions.get("attention")
            if pred_attention is not None:
                attention_loss = self._attention_loss(pred_attention, targets.attention_map)
                losses["attention_loss"] = attention_loss
                total_loss += attention_loss * self.config.attention_weight

        if self.config.enable_opponent_modeling and targets.opponent_action is not None:
            pred_opponent = predictions.get("opponent")
            if pred_opponent is not None:
                opponent_loss = self._cross_entropy_loss(
                    pred_opponent, targets.opponent_action
                )
                losses["opponent_loss"] = opponent_loss
                total_loss += opponent_loss * self.config.opponent_weight

        if self.config.enable_phase_prediction and targets.game_phase is not None:
            pred_phase = predictions.get("phase")
            if pred_phase is not None:
                phase_loss = self._cross_entropy_loss(pred_phase, targets.game_phase)
                losses["phase_loss"] = phase_loss
                total_loss += phase_loss * self.config.phase_weight

        if self.config.enable_future_value and targets.future_value is not None:
            pred_future = predictions.get("future_value")
            if pred_future is not None:
                future_loss = self._mse_loss(pred_future, targets.future_value)
                losses["future_value_loss"] = future_loss
                total_loss += future_loss * self.config.future_value_weight

        losses["total_auxiliary_loss"] = total_loss

        # Track history
        self._loss_history.append(losses.copy())
        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-500:]

        return losses

    def _attention_loss(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute attention map loss (KL divergence).

        Args:
            predicted: Predicted attention distribution
            target: Target attention distribution

        Returns:
            KL divergence loss value
        """
        # Flatten if needed
        pred_flat = predicted.flatten()
        target_flat = target.flatten()

        # Add small epsilon for numerical stability
        eps = 1e-8
        pred_flat = np.clip(pred_flat, eps, 1.0)
        target_flat = np.clip(target_flat, eps, 1.0)

        # Normalize
        pred_flat = pred_flat / pred_flat.sum()
        target_flat = target_flat / target_flat.sum()

        # KL divergence: sum(target * log(target / pred))
        kl_div = np.sum(target_flat * np.log(target_flat / pred_flat))
        return float(kl_div)

    def _cross_entropy_loss(
        self,
        predicted: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute cross-entropy loss.

        Args:
            predicted: Predicted probability distribution
            target: Target probability distribution

        Returns:
            Cross-entropy loss value
        """
        eps = 1e-8
        predicted = np.clip(predicted, eps, 1.0)

        # Cross-entropy: -sum(target * log(pred))
        ce = -np.sum(target * np.log(predicted))
        return float(ce)

    def _mse_loss(
        self,
        predicted: float | np.ndarray,
        target: float,
    ) -> float:
        """Compute mean squared error loss.

        Args:
            predicted: Predicted value(s)
            target: Target value

        Returns:
            MSE loss value
        """
        if isinstance(predicted, np.ndarray):
            predicted = float(predicted.mean())
        return float((predicted - target) ** 2)

    def get_average_losses(self, window: int = 100) -> dict[str, float]:
        """Get average losses over recent history.

        Args:
            window: Number of recent samples to average

        Returns:
            Dictionary with average loss values
        """
        if not self._loss_history:
            return {}

        recent = self._loss_history[-window:]
        avg_losses: dict[str, float] = {}

        for key in recent[0].keys():
            values = [h[key] for h in recent if key in h]
            if values:
                avg_losses[f"avg_{key}"] = sum(values) / len(values)

        return avg_losses

    def reset_history(self) -> None:
        """Clear loss history."""
        self._loss_history.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get calculator statistics.

        Returns:
            Dictionary with calculator statistics
        """
        return {
            "history_length": len(self._loss_history),
            "average_losses": self.get_average_losses(),
            "config": self.config.to_dict(),
        }


def create_auxiliary_system(
    config: AuxiliaryTargetsConfig | dict[str, Any] | None = None,
    board_size: int = 16,
    action_space_size: int = 1000,
    seed: int | None = None,
) -> tuple[AuxiliaryTargetGenerator, AuxiliaryLossCalculator]:
    """Create auxiliary target generator and loss calculator.

    Convenience function to create both components with shared config.

    Args:
        config: Configuration (dict, AuxiliaryTargetsConfig, or None for defaults)
        board_size: Size of game board
        action_space_size: Number of possible actions
        seed: Random seed for generator

    Returns:
        Tuple of (AuxiliaryTargetGenerator, AuxiliaryLossCalculator)
    """
    if config is None:
        config = AuxiliaryTargetsConfig()
    elif isinstance(config, dict):
        config = AuxiliaryTargetsConfig.from_dict(config)

    generator = AuxiliaryTargetGenerator(
        config=config,
        board_size=board_size,
        action_space_size=action_space_size,
        seed=seed,
    )
    calculator = AuxiliaryLossCalculator(config)

    return generator, calculator
