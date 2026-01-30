"""Centralized loss functions for MuZero/EfficientZero training.

This module provides all loss functions needed for MuZero and EfficientZero training:

1. **MuZero Core Losses:**
   - Policy loss: Cross-entropy between predicted and target policy distributions
   - Value loss: Categorical or MSE loss for value prediction
   - Reward loss: Categorical or MSE loss for reward prediction

2. **EfficientZero Auxiliary Losses:**
   - Consistency loss: SimSiam-style negative cosine similarity
   - Value prefix loss: For predicting cumulative rewards

3. **Combined Loss Functions:**
   - Step loss: Loss for a single unroll step
   - Trajectory loss: Loss for entire trajectory with unrolling

4. **MuZeroLossCalculator:**
   - Orchestrates multi-step loss computation with dynamics unrolling
   - Handles both AlphaZero and MuZero networks

5. **Utility Functions:**
   - Scalar to categorical conversion
   - Categorical to scalar conversion

Reference:
    - MuZero: "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
    - EfficientZero: "Mastering Atari Games with Limited Data"
    - SimSiam: "Exploring Simple Siamese Representation Learning"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import NamedTuple, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.core.checkpoint_utils import validate_reduction

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================


@dataclass
class CategoricalSupportConfig:
    """Configuration for scalar-to-categorical conversion.

    Used by value and reward heads in MuZero networks for representing
    continuous values as categorical distributions over a discrete support.

    The default configuration supports values from -300 to +300 with
    601 bins (step size of 1.0), which is suitable for most game scores.

    Attributes:
        support_size: Number of discrete bins in the support (default 601).
        min_value: Minimum value in support range (default -300.0).
        max_value: Maximum value in support range (default 300.0).

    Example:
        >>> config = CategoricalSupportConfig(support_size=201, min_value=-100, max_value=100)
        >>> # Now pass to scalar_to_categorical or categorical_to_scalar
    """

    support_size: int = 601
    min_value: float = -300.0
    max_value: float = 300.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.support_size < 3:
            raise ValueError(f"support_size must be at least 3, got {self.support_size}")
        if self.min_value >= self.max_value:
            raise ValueError(
                f"min_value ({self.min_value}) must be less than max_value ({self.max_value})"
            )

    @classmethod
    def from_env(cls, prefix: str = "CIV6AZ_CATEGORICAL_") -> CategoricalSupportConfig:
        """Create configuration from environment variables.

        Environment variables:
            {prefix}SUPPORT_SIZE: Number of support bins (default: 601)
            {prefix}MIN_VALUE: Minimum value in range (default: -300.0)
            {prefix}MAX_VALUE: Maximum value in range (default: 300.0)

        Args:
            prefix: Environment variable prefix (default: "CIV6AZ_CATEGORICAL_").

        Returns:
            Configuration instance with values from environment or defaults.
        """
        return cls(
            support_size=int(os.environ.get(f"{prefix}SUPPORT_SIZE", "601")),
            min_value=float(os.environ.get(f"{prefix}MIN_VALUE", "-300.0")),
            max_value=float(os.environ.get(f"{prefix}MAX_VALUE", "300.0")),
        )

    @property
    def step_size(self) -> float:
        """Calculate the step size between support bins."""
        return (self.max_value - self.min_value) / (self.support_size - 1)


# Global default instance for backwards compatibility
_DEFAULT_CATEGORICAL_CONFIG = CategoricalSupportConfig()


# =============================================================================
# Utility Functions: Scalar-Categorical Conversion
# =============================================================================


def scalar_to_categorical(
    scalar: torch.Tensor,
    support_size: int | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    *,
    config: CategoricalSupportConfig | None = None,
) -> torch.Tensor:
    """Convert scalar values to categorical distribution.

    Uses linear interpolation between support points to create a two-hot
    encoding. This is the transformation from the MuZero paper to handle
    large value ranges while maintaining gradient flow.

    Args:
        scalar: Scalar values of shape (B,) or (B, 1).
        support_size: Number of discrete bins in the support.
            Deprecated: Use config parameter instead.
        min_value: Minimum value in support range.
            Deprecated: Use config parameter instead.
        max_value: Maximum value in support range.
            Deprecated: Use config parameter instead.
        config: Configuration object. If provided, individual parameters
            are used as overrides. If None, uses global defaults.

    Returns:
        Categorical distribution of shape (B, support_size).
        The distribution sums to 1 and represents a soft encoding
        of the scalar value.

    Example:
        >>> scalars = torch.tensor([0.0, 10.5, -50.0])
        >>> categorical = scalar_to_categorical(scalars, support_size=601)
        >>> categorical.shape
        torch.Size([3, 601])
        >>> categorical.sum(dim=-1)  # Each sums to 1
        tensor([1., 1., 1.])

        >>> # Preferred: use config
        >>> config = CategoricalSupportConfig(support_size=201, min_value=-100, max_value=100)
        >>> categorical = scalar_to_categorical(scalars, config=config)
    """
    # Resolve configuration: individual params override config, which overrides global default
    if config is None:
        config = _DEFAULT_CATEGORICAL_CONFIG
    support_size = support_size if support_size is not None else config.support_size
    min_value = min_value if min_value is not None else config.min_value
    max_value = max_value if max_value is not None else config.max_value

    logger.debug(
        "Converting scalars to categorical: shape=%s, support_size=%d, range=[%f, %f]",
        scalar.shape, support_size, min_value, max_value
    )

    # Handle (B, 1) input shape
    if scalar.dim() == 2 and scalar.size(-1) == 1:
        scalar = scalar.squeeze(-1)

    # Clamp to valid range
    scalar = torch.clamp(scalar, min_value, max_value)

    # Create support values
    support = torch.linspace(min_value, max_value, support_size, device=scalar.device)

    # Find two nearest support points using searchsorted
    # searchsorted returns index where scalar would be inserted to keep sorted
    lower_idx = torch.searchsorted(support, scalar) - 1
    lower_idx = torch.clamp(lower_idx, 0, support_size - 2)
    upper_idx = lower_idx + 1

    # Compute interpolation weights (linear interpolation)
    lower_val = support[lower_idx]
    upper_val = support[upper_idx]
    upper_weight = (scalar - lower_val) / (upper_val - lower_val + 1e-8)
    lower_weight = 1.0 - upper_weight

    # Create categorical distribution (two-hot encoding)
    categorical = torch.zeros(scalar.size(0), support_size, device=scalar.device)
    categorical.scatter_(1, lower_idx.unsqueeze(1), lower_weight.unsqueeze(1))
    categorical.scatter_(1, upper_idx.unsqueeze(1), upper_weight.unsqueeze(1))

    logger.debug("Created categorical distribution with shape %s", categorical.shape)
    return categorical


def categorical_to_scalar(
    categorical: torch.Tensor,
    support_size: int | None = None,
    min_value: float | None = None,
    max_value: float | None = None,
    apply_softmax: bool = False,
    *,
    config: CategoricalSupportConfig | None = None,
) -> torch.Tensor:
    """Convert categorical distribution back to scalar value.

    Computes the expected value under the categorical distribution.

    Args:
        categorical: Categorical distribution or logits of shape (B, support_size).
        support_size: Number of discrete bins in the support.
            Deprecated: Use config parameter instead.
        min_value: Minimum value in support range.
            Deprecated: Use config parameter instead.
        max_value: Maximum value in support range.
            Deprecated: Use config parameter instead.
        apply_softmax: If True, apply softmax to convert logits to probabilities.
            If False, assume input is already a probability distribution.
        config: Configuration object. If provided, individual parameters
            are used as overrides. If None, uses global defaults.

    Returns:
        Scalar values of shape (B,).

    Example:
        >>> # Round-trip test
        >>> scalars = torch.tensor([0.0, 10.5, -50.0])
        >>> categorical = scalar_to_categorical(scalars, support_size=601)
        >>> recovered = categorical_to_scalar(categorical, support_size=601)
        >>> torch.allclose(scalars, recovered, atol=1.0)
        True

        >>> # Preferred: use config
        >>> config = CategoricalSupportConfig(support_size=201, min_value=-100, max_value=100)
        >>> recovered = categorical_to_scalar(categorical, config=config)
    """
    # Resolve configuration: individual params override config, which overrides global default
    if config is None:
        config = _DEFAULT_CATEGORICAL_CONFIG
    support_size = support_size if support_size is not None else config.support_size
    min_value = min_value if min_value is not None else config.min_value
    max_value = max_value if max_value is not None else config.max_value

    logger.debug(
        "Converting categorical to scalar: shape=%s, apply_softmax=%s",
        categorical.shape, apply_softmax
    )

    support = torch.linspace(min_value, max_value, support_size, device=categorical.device)

    if apply_softmax:
        probs = F.softmax(categorical, dim=-1)
    else:
        probs = categorical

    scalar = (probs * support.unsqueeze(0)).sum(dim=-1)
    logger.debug("Converted to scalars with shape %s", scalar.shape)
    return scalar


# =============================================================================
# MuZero Core Losses
# =============================================================================


def compute_policy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute cross-entropy loss between predicted and target policy.

    The target policy is typically the MCTS visit count distribution.
    Uses KL divergence / cross-entropy formulation.

    Args:
        policy_logits: Predicted policy logits of shape (B, A).
        target_policy: Target policy distribution of shape (B, A).
            Should sum to 1 (or close to 1 after masking).
        mask: Optional valid action mask of shape (B, A).
            If provided, target policy is renormalized to valid actions.
        reduction: Reduction method ('mean', 'sum', or 'none').

    Returns:
        Policy loss as a scalar tensor (or per-sample if reduction='none').

    Note:
        The loss is computed as -sum(target * log(softmax(logits))).
        This is equivalent to cross-entropy when targets are probabilities.
    """
    validate_reduction(reduction)

    logger.debug(
        "Computing policy loss: logits=%s, target=%s, mask=%s, reduction=%s",
        policy_logits.shape, target_policy.shape,
        mask.shape if mask is not None else None, reduction
    )

    # Apply action mask to target policy if provided
    if mask is not None:
        # Zero out invalid actions and renormalize
        target_policy = target_policy * mask
        target_policy = target_policy / (target_policy.sum(dim=-1, keepdim=True) + 1e-8)
        logger.debug("Applied mask and renormalized target policy")

    # Compute log probabilities from logits
    log_probs = F.log_softmax(policy_logits, dim=-1)

    # Cross-entropy: -sum(target * log_prob)
    per_sample_loss = -torch.sum(target_policy * log_probs, dim=-1)

    if reduction == "mean":
        loss = per_sample_loss.mean()
    elif reduction == "sum":
        loss = per_sample_loss.sum()
    else:  # reduction == "none"
        loss = per_sample_loss

    logger.debug("Policy loss computed: %f", loss.item() if loss.dim() == 0 else loss.mean().item())
    return loss


def compute_value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss for value prediction.

    Supports both categorical (cross-entropy) and scalar (MSE) formulations.
    Categorical formulation is recommended for large value ranges as it provides
    better gradient properties.

    Args:
        pred: Predicted value.
            - If use_categorical=True: shape (B, support_size) (logits)
            - If use_categorical=False: shape (B,) (scalar)
        target: Target value of shape (B,).
        config: Loss configuration. Uses defaults if None.
        reduction: Reduction method ('mean', 'sum', or 'none').

    Returns:
        Value loss as a scalar tensor (or per-sample if reduction='none').
    """
    validate_reduction(reduction)

    if config is None:
        config = LossConfig()

    logger.debug(
        "Computing value loss: pred=%s, target=%s, categorical=%s",
        pred.shape, target.shape, config.use_categorical
    )

    if config.use_categorical:
        # Convert target to categorical distribution
        target_categorical = scalar_to_categorical(
            target,
            support_size=config.support_size,
            min_value=config.min_value,
            max_value=config.max_value,
        )
        # Cross-entropy loss
        value_loss = F.cross_entropy(pred, target_categorical, reduction=reduction)
    else:
        # Mean squared error for scalar predictions
        # Ensure pred and target have matching shapes
        if pred.dim() == 2 and pred.shape[1] == 1:
            pred = pred.squeeze(-1)
        if target.dim() == 2 and target.shape[1] == 1:
            target = target.squeeze(-1)
        # Valid reduction already validated at function start
        value_loss = F.mse_loss(pred, target, reduction=reduction)

    logger.debug("Value loss computed: %f", value_loss.item() if value_loss.dim() == 0 else value_loss.mean().item())
    return value_loss


def compute_reward_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss for reward prediction.

    Supports both categorical (cross-entropy) and scalar (MSE) formulations.
    The reward is the immediate reward from taking an action.

    Args:
        pred: Predicted reward.
            - If use_categorical=True: shape (B, support_size) (logits)
            - If use_categorical=False: shape (B,) (scalar)
        target: Target reward of shape (B,).
        config: Loss configuration. Uses defaults if None.
        reduction: Reduction method ('mean', 'sum', or 'none').

    Returns:
        Reward loss as a scalar tensor (or per-sample if reduction='none').
    """
    validate_reduction(reduction)

    if config is None:
        config = LossConfig()

    logger.debug(
        "Computing reward loss: pred=%s, target=%s, categorical=%s",
        pred.shape, target.shape, config.use_categorical
    )

    if config.use_categorical:
        # Convert target to categorical distribution
        target_categorical = scalar_to_categorical(
            target,
            support_size=config.support_size,
            min_value=config.min_value,
            max_value=config.max_value,
        )
        # Cross-entropy loss
        reward_loss = F.cross_entropy(pred, target_categorical, reduction=reduction)
    else:
        # Mean squared error for scalar predictions
        # Valid reduction already validated at function start
        reward_loss = F.mse_loss(pred, target, reduction=reduction)

    logger.debug("Reward loss computed: %f", reward_loss.item() if reward_loss.dim() == 0 else reward_loss.mean().item())
    return reward_loss


# =============================================================================
# EfficientZero Auxiliary Losses
# =============================================================================


def compute_consistency_loss(
    proj1: torch.Tensor,
    proj2: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute EfficientZero consistency loss (SimSiam-style).

    This loss ensures that the dynamics model's predicted next state
    is consistent with the actual next observation's representation.
    Uses negative cosine similarity as the loss function.

    The consistency loss is a key component of EfficientZero that provides
    self-supervised signal to improve the learned world model without
    requiring additional environment interactions.

    Args:
        proj1: Projection of dynamics-predicted next state (B, D).
            This comes from: project(predict(dynamics(state, action)))
        proj2: Projection of actual next observation (B, D).
            This should be detached (stop gradient) for SimSiam training.
            This comes from: project(representation(next_observation)).detach()
        reduction: Reduction method ('mean', 'sum', or 'none').

    Returns:
        Consistency loss as a scalar tensor (or per-sample if reduction='none').
        Loss is in range [-1, 1] where -1 means perfect alignment.

    Note:
        Following SimSiam, only one branch should have gradients.
        The proj2 should typically be detached before passing to this function.

    Example:
        >>> proj_dyn = torch.randn(4, 256)
        >>> proj_obs = torch.randn(4, 256)
        >>> loss = compute_consistency_loss(proj_dyn, proj_obs.detach())
    """
    validate_reduction(reduction)

    logger.debug(
        "Computing consistency loss: proj1=%s, proj2=%s, reduction=%s",
        proj1.shape, proj2.shape, reduction
    )

    # L2 normalize the projections
    proj1 = F.normalize(proj1, dim=-1)
    proj2 = F.normalize(proj2, dim=-1)

    # Negative cosine similarity
    # cosine_sim = (a * b).sum(dim=-1) where ||a|| = ||b|| = 1
    # loss = -cosine_sim (we want to maximize similarity, so minimize negative)
    per_sample_loss = -(proj1 * proj2).sum(dim=-1)

    if reduction == "mean":
        loss = per_sample_loss.mean()
    elif reduction == "sum":
        loss = per_sample_loss.sum()
    else:  # reduction == "none"
        loss = per_sample_loss

    logger.debug("Consistency loss computed: %f", loss.item() if loss.dim() == 0 else loss.mean().item())
    return loss


def compute_value_prefix_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    config: LossConfig | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute loss for value prefix (cumulative reward) prediction.

    The value prefix is the sum of rewards over the next K steps.
    This provides better credit assignment in environments with sparse rewards.

    Args:
        pred: Predicted cumulative reward.
            - If use_categorical=True: shape (B, support_size) (logits)
            - If use_categorical=False: shape (B,) (scalar)
        target: Target cumulative reward of shape (B,).
            This is sum(rewards[t:t+K]) where K is the horizon.
        config: Loss configuration. Uses defaults if None.
        reduction: Reduction method ('mean', 'sum', or 'none').

    Returns:
        Value prefix loss as a scalar tensor (or per-sample if reduction='none').
    """
    validate_reduction(reduction)

    if config is None:
        config = LossConfig()

    logger.debug(
        "Computing value prefix loss: pred=%s, target=%s, categorical=%s",
        pred.shape, target.shape, config.use_categorical
    )

    if config.use_categorical:
        target_categorical = scalar_to_categorical(
            target,
            support_size=config.support_size,
            min_value=config.min_value,
            max_value=config.max_value,
        )
        prefix_loss = F.cross_entropy(pred, target_categorical, reduction=reduction)
    else:
        # Valid reduction already validated at function start
        prefix_loss = F.mse_loss(pred, target, reduction=reduction)

    logger.debug("Value prefix loss computed: %f", prefix_loss.item() if prefix_loss.dim() == 0 else prefix_loss.mean().item())
    return prefix_loss


# =============================================================================
# Data Structures for Combined Losses
# =============================================================================


@dataclass
class LossConfig:
    """Configuration for loss computation.

    All parameters are configurable to allow experimentation with
    different loss weights and formulations.

    Attributes:
        policy_loss_weight: Weight for policy loss (default: 1.0).
        value_loss_weight: Weight for value loss (default: 1.0).
        reward_loss_weight: Weight for reward loss (default: 1.0).
        consistency_loss_weight: Weight for EfficientZero consistency loss (default: 2.0).
        value_prefix_loss_weight: Weight for value prefix loss (default: 1.0).
        use_categorical: Whether to use categorical distributions for value/reward.
        categorical_support: Configuration for categorical encoding of values/rewards.

    Example:
        >>> # New style with CategoricalSupportConfig
        >>> config = LossConfig(
        ...     policy_loss_weight=1.0,
        ...     value_loss_weight=0.5,
        ...     categorical_support=CategoricalSupportConfig(support_size=201),
        ... )
        >>> # Backwards-compatible style with individual parameters
        >>> config = LossConfig(support_size=201, min_value=-100, max_value=100)
    """

    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    consistency_loss_weight: float = 2.0  # EfficientZero default
    value_prefix_loss_weight: float = 1.0

    use_categorical: bool = True
    categorical_support: CategoricalSupportConfig | None = None

    # Backwards-compatible parameters (use categorical_support for new code)
    _support_size: int | None = field(default=None, repr=False)
    _min_value: float | None = field(default=None, repr=False)
    _max_value: float | None = field(default=None, repr=False)

    def __init__(
        self,
        policy_loss_weight: float = 1.0,
        value_loss_weight: float = 1.0,
        reward_loss_weight: float = 1.0,
        consistency_loss_weight: float = 2.0,
        value_prefix_loss_weight: float = 1.0,
        use_categorical: bool = True,
        categorical_support: CategoricalSupportConfig | None = None,
        # Backwards-compatible parameters
        support_size: int | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        """Initialize LossConfig with backwards compatibility.

        Args:
            policy_loss_weight: Weight for policy loss.
            value_loss_weight: Weight for value loss.
            reward_loss_weight: Weight for reward loss.
            consistency_loss_weight: Weight for consistency loss.
            value_prefix_loss_weight: Weight for value prefix loss.
            use_categorical: Whether to use categorical distributions.
            categorical_support: Configuration for categorical encoding.
            support_size: (Deprecated) Use categorical_support instead.
            min_value: (Deprecated) Use categorical_support instead.
            max_value: (Deprecated) Use categorical_support instead.
        """
        self.policy_loss_weight = policy_loss_weight
        self.value_loss_weight = value_loss_weight
        self.reward_loss_weight = reward_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.value_prefix_loss_weight = value_prefix_loss_weight
        self.use_categorical = use_categorical

        # Handle backwards compatibility
        if categorical_support is not None:
            self.categorical_support = categorical_support
        elif support_size is not None or min_value is not None or max_value is not None:
            # Create from individual parameters
            self.categorical_support = CategoricalSupportConfig(
                support_size=support_size if support_size is not None else 601,
                min_value=min_value if min_value is not None else -300.0,
                max_value=max_value if max_value is not None else 300.0,
            )
        else:
            self.categorical_support = CategoricalSupportConfig()

        # Store for repr compatibility
        self._support_size = None
        self._min_value = None
        self._max_value = None

    @property
    def support_size(self) -> int:
        """Size of categorical support (backwards-compatible property)."""
        return self.categorical_support.support_size

    @property
    def min_value(self) -> float:
        """Minimum value in support range (backwards-compatible property)."""
        return self.categorical_support.min_value

    @property
    def max_value(self) -> float:
        """Maximum value in support range (backwards-compatible property)."""
        return self.categorical_support.max_value


class StepLossOutput(NamedTuple):
    """Output from computing loss for a single unroll step.

    Attributes:
        total_loss: Weighted sum of all losses.
        policy_loss: Cross-entropy policy loss.
        value_loss: Value prediction loss.
        reward_loss: Reward prediction loss (None for initial step).
        consistency_loss: EfficientZero consistency loss (None if not used).
        value_prefix_loss: Value prefix loss (None if not used).
    """

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    reward_loss: torch.Tensor | None
    consistency_loss: torch.Tensor | None
    value_prefix_loss: torch.Tensor | None


class TrajectoryLossOutput(NamedTuple):
    """Output from computing loss for entire trajectory.

    Attributes:
        total_loss: Weighted sum of all losses across all steps.
        policy_loss: Sum of policy losses.
        value_loss: Sum of value losses.
        reward_loss: Sum of reward losses.
        consistency_loss: Sum of consistency losses (None if not used).
        value_prefix_loss: Sum of value prefix losses (None if not used).
        step_losses: List of per-step losses.
    """

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    reward_loss: torch.Tensor
    consistency_loss: torch.Tensor | None
    value_prefix_loss: torch.Tensor | None
    step_losses: list[StepLossOutput]


# =============================================================================
# Combined Loss Functions
# =============================================================================


def compute_muzero_step_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    reward: torch.Tensor | None = None,
    target_reward: torch.Tensor | None = None,
    projected_dynamics: torch.Tensor | None = None,
    projected_observation: torch.Tensor | None = None,
    predicted_value_prefix: torch.Tensor | None = None,
    target_value_prefix: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    config: LossConfig | None = None,
) -> StepLossOutput:
    """Compute loss for a single unroll step.

    This function computes all losses for one step of MuZero/EfficientZero training:
    - Policy loss (always)
    - Value loss (always)
    - Reward loss (if reward predictions provided, typically for k > 0)
    - Consistency loss (if projections provided, EfficientZero)
    - Value prefix loss (if value prefix provided, EfficientZero)

    Args:
        policy_logits: Predicted policy logits (B, A).
        value: Predicted value (B, support_size) or (B,).
        target_policy: Target policy distribution (B, A).
        target_value: Target value (B,).
        reward: Predicted reward (B, support_size) or (B,), or None for initial step.
        target_reward: Target reward (B,), or None for initial step.
        projected_dynamics: Projection of dynamics prediction (B, D), or None.
        projected_observation: Projection of observation (B, D), or None.
        predicted_value_prefix: Predicted value prefix (B, support_size) or (B,), or None.
        target_value_prefix: Target value prefix (B,), or None.
        action_mask: Valid action mask (B, A), or None.
        config: Loss configuration. Uses defaults if None.

    Returns:
        StepLossOutput containing all individual losses and total.
    """
    if config is None:
        config = LossConfig()

    logger.debug("Computing MuZero step loss with config: %s", config)

    # Policy loss (always computed)
    policy_loss = compute_policy_loss(policy_logits, target_policy, action_mask)

    # Value loss (always computed)
    value_loss = compute_value_loss(pred=value, target=target_value, config=config)

    # Reward loss (optional, typically not for initial step k=0)
    reward_loss = None
    if reward is not None and target_reward is not None:
        reward_loss = compute_reward_loss(pred=reward, target=target_reward, config=config)

    # Consistency loss (EfficientZero, optional)
    consistency_loss = None
    if projected_dynamics is not None and projected_observation is not None:
        consistency_loss = compute_consistency_loss(projected_dynamics, projected_observation)

    # Value prefix loss (EfficientZero, optional)
    value_prefix_loss = None
    if predicted_value_prefix is not None and target_value_prefix is not None:
        value_prefix_loss = compute_value_prefix_loss(
            pred=predicted_value_prefix, target=target_value_prefix, config=config
        )

    # Compute total weighted loss
    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_loss_weight * value_loss
    )

    if reward_loss is not None:
        total_loss = total_loss + config.reward_loss_weight * reward_loss

    if consistency_loss is not None:
        total_loss = total_loss + config.consistency_loss_weight * consistency_loss

    if value_prefix_loss is not None:
        total_loss = total_loss + config.value_prefix_loss_weight * value_prefix_loss

    logger.debug(
        "Step loss computed: total=%f, policy=%f, value=%f",
        total_loss.item(), policy_loss.item(), value_loss.item()
    )

    return StepLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        reward_loss=reward_loss,
        consistency_loss=consistency_loss,
        value_prefix_loss=value_prefix_loss,
    )


def compute_muzero_trajectory_loss(
    initial_policy_logits: torch.Tensor,
    initial_value: torch.Tensor,
    target_policies: list[torch.Tensor],
    target_values: list[torch.Tensor],
    target_rewards: list[torch.Tensor],
    unrolled_policy_logits: list[torch.Tensor],
    unrolled_values: list[torch.Tensor],
    unrolled_rewards: list[torch.Tensor],
    action_masks: list[torch.Tensor] | None = None,
    projected_dynamics_list: list[torch.Tensor] | None = None,
    projected_observation_list: list[torch.Tensor] | None = None,
    predicted_value_prefixes: list[torch.Tensor] | None = None,
    target_value_prefixes: list[torch.Tensor] | None = None,
    config: LossConfig | None = None,
) -> TrajectoryLossOutput:
    """Compute loss for entire trajectory with unrolling.

    This function computes the MuZero/EfficientZero loss for a full trajectory:
    1. Initial inference loss (k=0): policy + value
    2. Unrolled steps loss (k=1..K): policy + value + reward + (optional) consistency

    The total loss is summed across all K+1 steps.

    Args:
        initial_policy_logits: Policy logits from initial inference (B, A).
        initial_value: Value from initial inference (B, support) or (B,).
        target_policies: Target policies for steps 0..K, list of (B, A).
        target_values: Target values for steps 0..K, list of (B,).
        target_rewards: Target rewards for steps 1..K, list of (B,).
        unrolled_policy_logits: Policy logits for steps 1..K, list of (B, A).
        unrolled_values: Values for steps 1..K, list of (B, support) or (B,).
        unrolled_rewards: Rewards for steps 1..K, list of (B, support) or (B,).
        action_masks: Optional action masks for steps 0..K, list of (B, A).
        projected_dynamics_list: Projections of dynamics states for steps 1..K.
        projected_observation_list: Projections of observations for steps 1..K.
        predicted_value_prefixes: Value prefix predictions for steps 0..K.
        target_value_prefixes: Target value prefixes for steps 0..K.
        config: Loss configuration.

    Returns:
        TrajectoryLossOutput containing aggregated and per-step losses.

    Note:
        - len(target_policies) == len(target_values) == K + 1
        - len(target_rewards) == len(unrolled_*) == K
        - Step 0 is initial inference, steps 1..K are unrolled dynamics
    """
    if config is None:
        config = LossConfig()

    num_unroll_steps = len(target_rewards)
    logger.info(
        "Computing trajectory loss: num_unroll_steps=%d, batch_size=%d",
        num_unroll_steps, initial_policy_logits.size(0)
    )

    step_losses: list[StepLossOutput] = []

    # Step 0: Initial inference
    step0_loss = compute_muzero_step_loss(
        policy_logits=initial_policy_logits,
        value=initial_value,
        target_policy=target_policies[0],
        target_value=target_values[0],
        reward=None,  # No reward prediction for initial step
        target_reward=None,
        projected_dynamics=None,
        projected_observation=None,
        predicted_value_prefix=(
            predicted_value_prefixes[0] if predicted_value_prefixes else None
        ),
        target_value_prefix=(
            target_value_prefixes[0] if target_value_prefixes else None
        ),
        action_mask=action_masks[0] if action_masks else None,
        config=config,
    )
    step_losses.append(step0_loss)

    # Steps 1..K: Unrolled dynamics
    for k in range(num_unroll_steps):
        logger.debug("Computing loss for unroll step %d", k + 1)
        step_k_loss = compute_muzero_step_loss(
            policy_logits=unrolled_policy_logits[k],
            value=unrolled_values[k],
            target_policy=target_policies[k + 1],
            target_value=target_values[k + 1],
            reward=unrolled_rewards[k],
            target_reward=target_rewards[k],
            projected_dynamics=(
                projected_dynamics_list[k] if projected_dynamics_list else None
            ),
            projected_observation=(
                projected_observation_list[k] if projected_observation_list else None
            ),
            predicted_value_prefix=(
                predicted_value_prefixes[k + 1] if predicted_value_prefixes else None
            ),
            target_value_prefix=(
                target_value_prefixes[k + 1] if target_value_prefixes else None
            ),
            action_mask=action_masks[k + 1] if action_masks else None,
            config=config,
        )
        step_losses.append(step_k_loss)

    # Aggregate losses across all steps
    total_loss = sum(s.total_loss for s in step_losses)
    policy_loss = sum(s.policy_loss for s in step_losses)
    value_loss = sum(s.value_loss for s in step_losses)

    # Reward loss (only for k >= 1)
    reward_losses = [s.reward_loss for s in step_losses if s.reward_loss is not None]
    reward_loss = sum(reward_losses) if reward_losses else torch.tensor(0.0)

    # Consistency loss (only if any step has it)
    consistency_losses = [
        s.consistency_loss for s in step_losses if s.consistency_loss is not None
    ]
    consistency_loss = sum(consistency_losses) if consistency_losses else None

    # Value prefix loss (only if any step has it)
    value_prefix_losses = [
        s.value_prefix_loss for s in step_losses if s.value_prefix_loss is not None
    ]
    value_prefix_loss = sum(value_prefix_losses) if value_prefix_losses else None

    logger.info(
        "Trajectory loss computed: total=%f, policy=%f, value=%f, reward=%f",
        total_loss.item(), policy_loss.item(), value_loss.item(),
        reward_loss.item() if isinstance(reward_loss, torch.Tensor) else reward_loss
    )

    return TrajectoryLossOutput(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        reward_loss=reward_loss,
        consistency_loss=consistency_loss,
        value_prefix_loss=value_prefix_loss,
        step_losses=step_losses,
    )


# =============================================================================
# MuZeroLossCalculator Class
# =============================================================================


@runtime_checkable
class MuZeroNetworkProtocol(Protocol):
    """Protocol for MuZero-compatible networks."""

    def initial_inference(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple: ...

    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple: ...


@runtime_checkable
class AlphaZeroNetworkProtocol(Protocol):
    """Protocol for AlphaZero-compatible networks."""

    def forward(
        self,
        spatial: torch.Tensor,
        vector: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@dataclass
class TrajectoryBatch:
    """Batch of trajectory data for MuZero training.

    Attributes:
        observations_spatial: Spatial observations for each step (K+1, B, C, H, W).
        observations_vector: Vector observations for each step (K+1, B, V).
        actions: Actions taken at each step (K, B).
        target_policies: Target policy distributions (K+1, B, A).
        target_values: Target values (K+1, B).
        target_rewards: Target rewards (K, B).
        action_masks: Optional valid action masks (K+1, B, A).
        target_value_prefixes: Optional target value prefixes (K+1, B).
    """

    observations_spatial: list[torch.Tensor]
    observations_vector: list[torch.Tensor]
    actions: list[torch.Tensor]
    target_policies: list[torch.Tensor]
    target_values: list[torch.Tensor]
    target_rewards: list[torch.Tensor]
    action_masks: list[torch.Tensor] | None = None
    target_value_prefixes: list[torch.Tensor] | None = None


class MuZeroLossCalculator:
    """Orchestrates multi-step loss computation for MuZero/EfficientZero training.

    This class handles the complete loss computation pipeline:
    1. Initial inference from observations
    2. Dynamics unrolling for K steps
    3. Loss computation at each step
    4. Optional EfficientZero consistency loss

    The calculator is designed to work with both MuZero and AlphaZero networks,
    providing backward compatibility.

    Example:
        >>> calculator = MuZeroLossCalculator(config=LossConfig())
        >>> loss_output = calculator.compute_loss(network, trajectory_batch)
        >>> loss_output.total_loss.backward()
    """

    def __init__(
        self,
        config: LossConfig | None = None,
        num_unroll_steps: int = 5,
        use_consistency_loss: bool = True,
        use_value_prefix: bool = True,
    ):
        """Initialize the loss calculator.

        Args:
            config: Loss configuration. Uses defaults if None.
            num_unroll_steps: Number of dynamics unroll steps (K).
            use_consistency_loss: Whether to compute EfficientZero consistency loss.
            use_value_prefix: Whether to compute value prefix loss.
        """
        self.config = config if config is not None else LossConfig()
        self.num_unroll_steps = num_unroll_steps
        self.use_consistency_loss = use_consistency_loss
        self.use_value_prefix = use_value_prefix

        logger.info(
            "MuZeroLossCalculator initialized: unroll_steps=%d, consistency=%s, value_prefix=%s",
            num_unroll_steps, use_consistency_loss, use_value_prefix
        )

    def compute_loss(
        self,
        network: nn.Module,
        batch: TrajectoryBatch,
    ) -> TrajectoryLossOutput:
        """Compute the complete MuZero/EfficientZero loss for a trajectory batch.

        This method:
        1. Performs initial inference on the first observation
        2. Unrolls dynamics for K steps using the actions
        3. Computes all losses (policy, value, reward, consistency, value prefix)
        4. Returns aggregated and per-step losses

        Args:
            network: The MuZero or AlphaZero network.
            batch: Batch of trajectory data.

        Returns:
            TrajectoryLossOutput with all computed losses.

        Raises:
            ValueError: If the network doesn't implement required methods.
        """
        logger.info("Computing MuZero loss for batch")

        # Determine network type and compute accordingly
        if self._is_muzero_network(network):
            return self._compute_muzero_loss(network, batch)
        elif self._is_alphazero_network(network):
            return self._compute_alphazero_loss(network, batch)
        else:
            raise ValueError(
                "Network must implement either MuZero (initial_inference, recurrent_inference) "
                "or AlphaZero (forward) interface"
            )

    def _is_muzero_network(self, network: nn.Module) -> bool:
        """Check if network has MuZero interface."""
        return (
            hasattr(network, "initial_inference")
            and hasattr(network, "recurrent_inference")
        )

    def _is_alphazero_network(self, network: nn.Module) -> bool:
        """Check if network has AlphaZero interface."""
        return hasattr(network, "forward") and callable(network.forward)

    def _compute_muzero_loss(
        self,
        network: nn.Module,
        batch: TrajectoryBatch,
    ) -> TrajectoryLossOutput:
        """Compute loss using MuZero network interface with dynamics unrolling."""
        logger.debug("Using MuZero network interface")

        # Initial inference
        initial_output = network.initial_inference(
            batch.observations_spatial[0],
            batch.observations_vector[0],
            batch.action_masks[0] if batch.action_masks else None,
        )

        # Handle different output formats
        if hasattr(initial_output, "hidden_state"):
            hidden_state = initial_output.hidden_state
            initial_policy_logits = initial_output.policy_logits
            initial_value = initial_output.value
        else:
            hidden_state, initial_policy_logits, initial_value = initial_output[:3]

        # Unroll dynamics
        unrolled_policy_logits: list[torch.Tensor] = []
        unrolled_values: list[torch.Tensor] = []
        unrolled_rewards: list[torch.Tensor] = []
        projected_dynamics_list: list[torch.Tensor] | None = (
            [] if self.use_consistency_loss else None
        )
        projected_observation_list: list[torch.Tensor] | None = (
            [] if self.use_consistency_loss else None
        )
        predicted_value_prefixes: list[torch.Tensor] | None = (
            [self._get_value_prefix(network, hidden_state)] if self.use_value_prefix else None
        )

        current_hidden = hidden_state

        for k in range(len(batch.actions)):
            logger.debug("Unrolling dynamics step %d", k)

            # Recurrent inference
            recurrent_output = network.recurrent_inference(
                current_hidden,
                batch.actions[k],
                batch.action_masks[k + 1] if batch.action_masks else None,
            )

            # Handle different output formats
            if isinstance(recurrent_output, tuple):
                if len(recurrent_output) >= 4:
                    next_hidden, reward, policy_logits, value = recurrent_output[:4]
                else:
                    next_hidden, reward, policy_logits, value = recurrent_output
            else:
                next_hidden = recurrent_output.next_hidden_state
                reward = recurrent_output.reward
                policy_logits = recurrent_output.policy_logits
                value = recurrent_output.value

            unrolled_policy_logits.append(policy_logits)
            unrolled_values.append(value)
            unrolled_rewards.append(reward)

            # Consistency loss projections
            if self.use_consistency_loss and projected_dynamics_list is not None:
                proj_dyn = self._get_projection(network, next_hidden)
                proj_obs = self._get_projection(
                    network,
                    self._get_representation(
                        network,
                        batch.observations_spatial[k + 1],
                        batch.observations_vector[k + 1],
                    ),
                ).detach()  # Stop gradient for SimSiam
                projected_dynamics_list.append(proj_dyn)
                projected_observation_list.append(proj_obs)

            # Value prefix predictions
            if self.use_value_prefix and predicted_value_prefixes is not None:
                predicted_value_prefixes.append(
                    self._get_value_prefix(network, next_hidden)
                )

            current_hidden = next_hidden

        # Handle value tensor shape for categorical
        if self.config.use_categorical:
            # If value is scalar, we need logits - check network output format
            if initial_value.dim() == 1:
                # Network outputs scalar values, need to handle differently
                # This shouldn't happen for categorical networks
                logger.warning("Categorical config but network outputs scalar values")

        return compute_muzero_trajectory_loss(
            initial_policy_logits=initial_policy_logits,
            initial_value=initial_value if initial_value.dim() > 1 else self._to_categorical_logits(initial_value),
            target_policies=batch.target_policies,
            target_values=batch.target_values,
            target_rewards=batch.target_rewards,
            unrolled_policy_logits=unrolled_policy_logits,
            unrolled_values=[v if v.dim() > 1 else self._to_categorical_logits(v) for v in unrolled_values],
            unrolled_rewards=[r if r.dim() > 1 else self._to_categorical_logits(r) for r in unrolled_rewards],
            action_masks=batch.action_masks,
            projected_dynamics_list=projected_dynamics_list,
            projected_observation_list=projected_observation_list,
            predicted_value_prefixes=predicted_value_prefixes,
            target_value_prefixes=batch.target_value_prefixes,
            config=self.config,
        )

    def _compute_alphazero_loss(
        self,
        network: nn.Module,
        batch: TrajectoryBatch,
    ) -> TrajectoryLossOutput:
        """Compute loss using AlphaZero network interface (no dynamics unrolling)."""
        logger.debug("Using AlphaZero network interface (no dynamics)")

        # AlphaZero only does single-step inference
        # Compute loss for each observation independently
        step_losses: list[StepLossOutput] = []

        for i, (spatial, vector, target_policy, target_value) in enumerate(
            zip(
                batch.observations_spatial,
                batch.observations_vector,
                batch.target_policies,
                batch.target_values, strict=False,
            )
        ):
            action_mask = batch.action_masks[i] if batch.action_masks else None
            policy_logits, value = network.forward(spatial, vector, action_mask)

            # For AlphaZero, we don't have reward predictions
            step_loss = compute_muzero_step_loss(
                policy_logits=policy_logits,
                value=value if value.dim() > 1 else value.unsqueeze(-1),
                target_policy=target_policy,
                target_value=target_value,
                reward=None,
                target_reward=None,
                action_mask=action_mask,
                config=self.config,
            )
            step_losses.append(step_loss)

        # Aggregate losses
        total_loss = sum(s.total_loss for s in step_losses)
        policy_loss = sum(s.policy_loss for s in step_losses)
        value_loss = sum(s.value_loss for s in step_losses)

        return TrajectoryLossOutput(
            total_loss=total_loss,
            policy_loss=policy_loss,
            value_loss=value_loss,
            reward_loss=torch.tensor(0.0),
            consistency_loss=None,
            value_prefix_loss=None,
            step_losses=step_losses,
        )

    def _get_projection(self, network: nn.Module, hidden_state: torch.Tensor) -> torch.Tensor:
        """Get projection from hidden state for consistency loss."""
        if hasattr(network, "project"):
            return network.project(hidden_state)
        else:
            logger.warning("Network doesn't have project method, using identity")
            return hidden_state.mean(dim=(2, 3)) if hidden_state.dim() == 4 else hidden_state

    def _get_representation(
        self,
        network: nn.Module,
        spatial: torch.Tensor,
        vector: torch.Tensor,
    ) -> torch.Tensor:
        """Get representation from observation."""
        if hasattr(network, "representation"):
            return network.representation(spatial, vector)
        else:
            logger.warning("Network doesn't have representation method")
            return spatial

    def _get_value_prefix(
        self,
        network: nn.Module,
        hidden_state: torch.Tensor,
    ) -> torch.Tensor:
        """Get value prefix prediction from hidden state."""
        if hasattr(network, "predict_value_prefix"):
            return network.predict_value_prefix(hidden_state)
        else:
            logger.warning("Network doesn't have predict_value_prefix method")
            batch_size = hidden_state.size(0)
            if self.config.use_categorical:
                return torch.zeros(batch_size, self.config.support_size, device=hidden_state.device)
            else:
                return torch.zeros(batch_size, device=hidden_state.device)

    def _to_categorical_logits(self, scalar: torch.Tensor) -> torch.Tensor:
        """Convert scalar values to categorical logits (for compatibility)."""
        # Create sharp categorical distribution around the value
        categorical = scalar_to_categorical(
            scalar,
            support_size=self.config.support_size,
            min_value=self.config.min_value,
            max_value=self.config.max_value,
        )
        # Convert to logits by taking log (add small epsilon to avoid -inf)
        return torch.log(categorical + 1e-8)


# =============================================================================
# Backward Compatibility: compute_muzero_loss function
# =============================================================================


def compute_muzero_loss(
    policy_logits: torch.Tensor,
    value: torch.Tensor,
    reward: torch.Tensor | None,
    target_policy: torch.Tensor,
    target_value: torch.Tensor,
    target_reward: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    use_categorical: bool = True,
    support_size: int = 601,
) -> dict[str, torch.Tensor]:
    """Compute MuZero losses for a single step.

    This is a convenience wrapper that returns losses as a dictionary,
    maintaining backward compatibility with the original implementation
    in muzero_network.py.

    Args:
        policy_logits: Predicted policy logits (B, A).
        value: Predicted value (B,) or (B, support_size) if categorical.
        reward: Predicted reward (B,) or (B, support_size), or None for initial.
        target_policy: Target MCTS policy (B, A).
        target_value: Target value (B,).
        target_reward: Target reward (B,), or None for initial.
        action_mask: Valid action mask (B, A).
        use_categorical: Whether predictions are categorical.
        support_size: Size of categorical support.

    Returns:
        Dictionary of losses: policy_loss, value_loss, reward_loss (if applicable).
    """
    logger.debug("Computing MuZero loss (backward-compatible interface)")

    config = LossConfig(use_categorical=use_categorical, support_size=support_size)
    losses: dict[str, torch.Tensor] = {}

    # Policy loss
    losses["policy_loss"] = compute_policy_loss(policy_logits, target_policy, action_mask)

    # Value loss
    losses["value_loss"] = compute_value_loss(
        pred=value, target=target_value, config=config
    )

    # Reward loss (if applicable)
    if reward is not None and target_reward is not None:
        losses["reward_loss"] = compute_reward_loss(
            pred=reward, target=target_reward, config=config
        )

    return losses


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Utility functions
    "scalar_to_categorical",
    "categorical_to_scalar",
    # Core losses
    "compute_policy_loss",
    "compute_value_loss",
    "compute_reward_loss",
    # EfficientZero losses
    "compute_consistency_loss",
    "compute_value_prefix_loss",
    # Combined losses
    "compute_muzero_step_loss",
    "compute_muzero_trajectory_loss",
    "compute_muzero_loss",
    # Data structures
    "LossConfig",
    "StepLossOutput",
    "TrajectoryLossOutput",
    "TrajectoryBatch",
    # Calculator class
    "MuZeroLossCalculator",
    # Protocols
    "MuZeroNetworkProtocol",
    "AlphaZeroNetworkProtocol",
]
