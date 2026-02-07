"""Loss functions for AlphaZero-style training."""
from __future__ import annotations

import torch
import torch.nn.functional as f


def compute_policy_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy policy loss: -sum(target * log_softmax(logits)).

    Args:
        policy_logits: (B, A) raw logits from the policy head.
        target_policy: (B, A) target distribution from MCTS visit counts.
        action_mask: Optional (B, A) boolean mask. If provided, target is
            renormalized over valid actions.

    Returns:
        Scalar loss tensor.
    """
    if action_mask is not None:
        target_policy = target_policy * action_mask.float()
        target_sum = target_policy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        target_policy = target_policy / target_sum

    log_probs = f.log_softmax(policy_logits, dim=-1)
    return -(target_policy * log_probs).sum(dim=-1).mean()


def compute_value_loss(
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor:
    """MSE value loss.

    Args:
        predicted_value: (B, 1) or (B,) predicted values.
        target_value: (B, 1) or (B,) target values from game outcomes.

    Returns:
        Scalar loss tensor.
    """
    predicted = predicted_value.view(-1)
    target = target_value.view(-1)
    return f.mse_loss(predicted, target)


def compute_combined_loss(
    policy_logits: torch.Tensor,
    target_policy: torch.Tensor,
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    value_loss_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined AlphaZero loss: policy_loss + weight * value_loss.

    Args:
        policy_logits: (B, A) raw logits.
        target_policy: (B, A) target policy distribution.
        predicted_value: (B, 1) predicted value.
        target_value: (B, 1) or (B,) target value.
        action_mask: Optional action mask.
        value_loss_weight: Weight for value loss term.

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains
        individual loss components as floats.
    """
    p_loss = compute_policy_loss(policy_logits, target_policy, action_mask)
    v_loss = compute_value_loss(predicted_value, target_value)
    total = p_loss + value_loss_weight * v_loss

    metrics = {
        "total_loss": total.item(),
        "policy_loss": p_loss.item(),
        "value_loss": v_loss.item(),
    }

    return total, metrics
