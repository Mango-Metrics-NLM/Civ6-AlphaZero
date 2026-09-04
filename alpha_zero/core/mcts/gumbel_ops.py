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
    rng: np.random.Generator | None = None,
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
        rng: Generator for the Gumbel noise. ``None`` draws from a fresh
            unseeded generator, matching the historical behaviour of drawing
            from the global ``np.random`` state.

            This is a *parameter* rather than module state because all three
            searches that call this function -- ``GumbelMCTS``, ``MuZeroMCTS``
            and ``BatchGumbelMCTS`` -- own their own generator, and a caller
            that seeds its search must have that seed reach the noise drawn
            here. It previously did not: ``run_batch_selfplay(seed=...)`` seeded
            the environments while this line drew from the global RNG, so a
            "seeded" self-play run was only half determined.

    Returns:
        Tuple of (selected_actions, gumbel_scores).
        - selected_actions: Indices of selected actions (k,).
        - gumbel_scores: Gumbel scores for selected actions (k,).
    """
    if rng is None:
        rng = np.random.default_rng()
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
    uniform = rng.uniform(size=len(valid_actions))
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


def build_completed_q_policy(
    *,
    priors: np.ndarray,
    visit_counts: np.ndarray,
    q_values: np.ndarray,
    root_value: float,
    c_visit: float,
    c_scale: float,
) -> np.ndarray:
    """Return Gumbel MuZero's improved policy ``softmax(logits + sigma(completedQ))``.

    Why not visit counts
    --------------------
    Normalised visit counts are AlphaZero's policy-improvement operator, and they
    are the wrong one here. Sequential halving allocates the budget *equally*
    among the survivors of each round, so the root visit distribution is close to
    uniform over the candidate set by construction and the final survivors are
    typically tied. Measured on a synthetic forced-win position, the visit
    distribution was exactly ``[1/3, 1/3, 1/3]`` while the search had in fact
    identified the winning move: the answer lives in the completed Q values and
    the Gumbel scores, and a visit-count target throws it away.

    Danihelka et al. (2022) define the improved policy over the *completed* Q
    values instead, which is what this builds.

    Args:
        priors: Network prior over the whole action space.
        visit_counts: Root child visit counts, aligned with ``priors``. Actions
            the search never considered must be zero.
        q_values: Root child Q values from the acting player's perspective,
            aligned with ``priors``.
        root_value: The root's own value estimate, used to complete the Q value
            of an action the search did not visit.
        c_visit: Visit constant of the monotonic transform.
        c_scale: Scale constant of the monotonic transform. Previously declared
            on the config and read nowhere; this is its intended use.

    Returns:
        A probability vector over the action space. Mass is placed only on
        actions the search considered; if it considered none, an empty (all
        zero) vector is returned and the caller decides the fallback.
    """
    considered = visit_counts > 0
    if not considered.any():
        return np.zeros_like(priors, dtype=np.float32)

    max_visits = float(visit_counts.max())
    completed = np.array(
        [
            compute_completed_q(float(q), int(n), int(max_visits), root_value)
            for q, n in zip(q_values, visit_counts, strict=False)
        ],
        dtype=np.float64,
    )

    # The paper's monotonic transform: larger budgets let the searched values
    # outweigh the prior, which is exactly the behaviour a policy target wants.
    sigma = (c_visit + max_visits) * c_scale
    logits = np.log(np.maximum(priors.astype(np.float64), EPSILON))
    scores = logits + sigma * completed

    # Restrict to the considered set before the softmax: an action the search
    # never looked at has no completed Q to speak of, and including it would let
    # the raw prior leak into a target that is supposed to be search-improved.
    scores = np.where(considered, scores, -np.inf)
    scores -= scores[considered].max()  # shift for numerical stability
    exp = np.where(considered, np.exp(scores), 0.0)
    total = exp.sum()
    if total <= 0.0:
        return np.zeros_like(priors, dtype=np.float32)
    normalised: np.ndarray = (exp / total).astype(np.float32)
    return normalised
