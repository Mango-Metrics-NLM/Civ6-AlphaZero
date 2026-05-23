"""Stateless metric computation helpers for MLflow logging.

These helpers compute scalar summaries from raw evaluation data so the
``MLflowLogger`` can log them with consistent semantics. All functions are
NaN-safe and return well-defined values on degenerate input.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from alpha_zero.tournament.elo import RatingLeaderboard

DEFAULT_MIN_VALUE_SAMPLES = 10


def compute_value_correlation(
    predictions: Sequence[tuple[float, float]],
    min_samples: int = DEFAULT_MIN_VALUE_SAMPLES,
) -> float:
    """Pearson correlation between predicted and actual game outcomes.

    Args:
        predictions: Sequence of ``(predicted_value, actual_outcome)`` tuples.
        min_samples: Minimum sample count required; returns 0.0 otherwise.

    Returns:
        Pearson r in ``[-1.0, 1.0]``. Returns ``0.0`` if either side has zero
        variance or fewer than ``min_samples`` data points.
    """
    if len(predictions) < min_samples:
        return 0.0
    preds = np.asarray([p[0] for p in predictions], dtype=np.float64)
    actuals = np.asarray([p[1] for p in predictions], dtype=np.float64)
    if preds.std() == 0.0 or actuals.std() == 0.0:
        return 0.0
    corr_matrix = np.corrcoef(preds, actuals)
    corr = float(corr_matrix[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def compute_elo_spread(leaderboard: RatingLeaderboard) -> dict[str, float]:
    """Compute aggregate ELO statistics from a leaderboard.

    Args:
        leaderboard: Tournament rating leaderboard.

    Returns:
        ``{"spread", "max", "min", "mean", "stddev"}``. All zero when empty.
    """
    zeros = {"spread": 0.0, "max": 0.0, "min": 0.0, "mean": 0.0, "stddev": 0.0}
    top = leaderboard.get_top(n=10_000)
    if not top:
        return zeros
    ratings = np.asarray([rating for (_, rating, _) in top], dtype=np.float64)
    return {
        "spread": float(ratings.max() - ratings.min()),
        "max": float(ratings.max()),
        "min": float(ratings.min()),
        "mean": float(ratings.mean()),
        "stddev": float(ratings.std()) if ratings.size > 1 else 0.0,
    }


def compute_policy_entropy(
    policy: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Shannon entropy of a policy distribution in nats.

    Args:
        policy: 1-D probability vector. May not be normalized; will be
            renormalized inside the function.
        mask: Optional binary mask; entropy is computed only over legal
            actions where ``mask > 0``.
        eps: Small constant added before the log to guard against ``log(0)``.

    Returns:
        Entropy in nats. ``0.0`` if all probability mass is on one action or
        the policy is empty.
    """
    if policy.size == 0:
        return 0.0
    p = np.asarray(policy, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=np.float64) > 0
        if not m.any():
            return 0.0
        p = np.where(m, p, 0.0)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(-np.sum(p * np.log(p + eps)))
