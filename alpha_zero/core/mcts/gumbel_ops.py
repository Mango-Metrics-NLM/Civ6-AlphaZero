"""Gumbel-Top-k sampling primitives shared by the Gumbel and MuZero searches.

These live in their own module because :class:`~alpha_zero.core.mcts.muzero.MuZeroMCTS`
uses them as well; keeping them in ``gumbel`` would make the MuZero variant
depend on the Gumbel variant.
"""

from __future__ import annotations

import numpy as np

from alpha_zero.core.mcts_constants import EPSILON


def gumbel_sample_actions(
    policy: np.ndarray,
    action_mask: np.ndarray,
    num_actions: int,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample actions using Gumbel-Top-k trick.

    The Gumbel-Top-k trick provides a way to sample k elements from a categorical
    distribution without replacement, while maintaining the policy improvement
    guarantee of Gumbel MuZero.

    Args:
        policy: Policy probabilities for all actions (A,).
        action_mask: Mask of valid actions (A,).
        num_actions: Number of actions to sample (k).
        temperature: Temperature for sampling (higher = more exploration).

    Returns:
        Tuple of (selected_actions, gumbel_scores).
        - selected_actions: Indices of selected actions (k,).
        - gumbel_scores: Gumbel scores for selected actions (k,).
    """
    valid_actions = np.where(action_mask > 0)[0]

    if len(valid_actions) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    # Get log probabilities for valid actions
    valid_policy = policy[valid_actions]
    # Ensure numerical stability
    valid_policy = np.clip(valid_policy, EPSILON, 1.0)
    log_probs = np.log(valid_policy)

    # Sample Gumbel noise: g ~ Gumbel(0, 1)
    # Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
    uniform = np.random.uniform(size=len(valid_actions))
    gumbel_noise = -np.log(-np.log(np.clip(uniform, EPSILON, 1 - EPSILON)))

    # Gumbel-Top-k: score = log(pi) / temperature + g
    scores = log_probs / temperature + gumbel_noise

    # Select top-k by score
    k = min(num_actions, len(valid_actions))
    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(-scores[top_indices])]

    selected_actions = valid_actions[top_indices]
    selected_scores = scores[top_indices]

    return selected_actions, selected_scores


def compute_completed_q(
    q_value: float,
    visit_count: int,
    max_visit_count: int,
    root_value: float,
) -> float:
    """Compute completed Q-value with sigma penalty for low visits.

    In Gumbel MuZero, we "complete" Q-values by hallucinating that missing
    visits would have returned the root value. This provides a principled
    way to compare actions with different visit counts.

    Args:
        q_value: Current Q-value estimate for the action.
        visit_count: Number of visits to this action.
        max_visit_count: Maximum visits among all candidate actions.
        root_value: Value estimate at the root node.

    Returns:
        Completed Q-value.
    """
    if max_visit_count == 0:
        return q_value

    # sigma = fraction of "missing" visits
    sigma = (max_visit_count - visit_count) / max_visit_count

    # Completed Q assumes missing visits would return root value
    completed_q = q_value + sigma * root_value

    return completed_q


