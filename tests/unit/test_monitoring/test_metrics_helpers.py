"""Tests for ``alpha_zero.monitoring.metrics``."""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.monitoring.metrics import (
    DEFAULT_MIN_VALUE_SAMPLES,
    compute_elo_spread,
    compute_policy_entropy,
    compute_value_correlation,
)

# ---------------------------------------------------------------------
# compute_value_correlation
# ---------------------------------------------------------------------


def test_value_correlation_empty_returns_zero():
    assert compute_value_correlation([]) == 0.0


def test_value_correlation_below_min_samples_returns_zero():
    preds = [(0.1, 0.2)] * (DEFAULT_MIN_VALUE_SAMPLES - 1)
    assert compute_value_correlation(preds) == 0.0


def test_value_correlation_perfect_positive():
    preds = [(float(i), float(i)) for i in range(20)]
    assert compute_value_correlation(preds) == pytest.approx(1.0)


def test_value_correlation_perfect_negative():
    preds = [(float(i), float(-i)) for i in range(20)]
    assert compute_value_correlation(preds) == pytest.approx(-1.0)


def test_value_correlation_zero_variance_returns_zero():
    preds = [(0.5, float(i)) for i in range(20)]  # predicted constant
    assert compute_value_correlation(preds) == 0.0


def test_value_correlation_both_zero_variance_returns_zero():
    preds = [(0.5, 0.5)] * 20
    assert compute_value_correlation(preds) == 0.0


def test_value_correlation_uncorrelated_near_zero():
    rng = np.random.default_rng(42)
    preds = [(float(rng.normal()), float(rng.normal())) for _ in range(500)]
    # Random noise should yield near-zero correlation.
    assert abs(compute_value_correlation(preds)) < 0.2


def test_value_correlation_custom_min_samples():
    preds = [(float(i), float(i)) for i in range(5)]
    assert compute_value_correlation(preds, min_samples=3) == pytest.approx(1.0)


# ---------------------------------------------------------------------
# compute_elo_spread
# ---------------------------------------------------------------------


class _FakeLeaderboard:
    """Minimal stand-in for RatingLeaderboard.get_top(n)."""

    def __init__(self, ratings: dict[str, float]):
        self._ratings = ratings

    def get_top(self, n: int = 10):
        items = sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)
        return [(pid, rating, i + 1) for i, (pid, rating) in enumerate(items[:n])]


def test_elo_spread_empty():
    spread = compute_elo_spread(_FakeLeaderboard({}))
    assert spread == {"spread": 0.0, "max": 0.0, "min": 0.0, "mean": 0.0, "stddev": 0.0}


def test_elo_spread_single_player():
    spread = compute_elo_spread(_FakeLeaderboard({"a": 1500.0}))
    assert spread["spread"] == 0.0
    assert spread["max"] == 1500.0
    assert spread["min"] == 1500.0
    assert spread["mean"] == 1500.0
    assert spread["stddev"] == 0.0


def test_elo_spread_multi_player():
    spread = compute_elo_spread(_FakeLeaderboard({"a": 1400.0, "b": 1500.0, "c": 1700.0}))
    assert spread["spread"] == 300.0
    assert spread["max"] == 1700.0
    assert spread["min"] == 1400.0
    assert spread["mean"] == pytest.approx(1533.333, rel=1e-3)
    assert spread["stddev"] > 0.0


# ---------------------------------------------------------------------
# compute_policy_entropy
# ---------------------------------------------------------------------


def test_policy_entropy_empty():
    assert compute_policy_entropy(np.array([])) == 0.0


def test_policy_entropy_uniform():
    n = 8
    policy = np.full(n, 1.0 / n)
    assert compute_policy_entropy(policy) == pytest.approx(np.log(n), rel=1e-4)


def test_policy_entropy_peaked():
    policy = np.array([1.0, 0.0, 0.0, 0.0])
    assert compute_policy_entropy(policy) == pytest.approx(0.0, abs=1e-10)


def test_policy_entropy_masked():
    policy = np.array([0.25, 0.25, 0.25, 0.25])
    mask = np.array([1, 1, 0, 0])
    # Two legal actions, renormalized → 0.5 each → entropy = ln(2)
    assert compute_policy_entropy(policy, mask=mask) == pytest.approx(np.log(2))


def test_policy_entropy_all_masked_out():
    policy = np.array([0.5, 0.5])
    mask = np.array([0, 0])
    assert compute_policy_entropy(policy, mask=mask) == 0.0


def test_policy_entropy_handles_zero_total():
    policy = np.array([0.0, 0.0, 0.0])
    assert compute_policy_entropy(policy) == 0.0


def test_policy_entropy_unnormalized_policy():
    # Unnormalized: should renormalize internally.
    policy = np.array([2.0, 2.0, 2.0, 2.0])  # uniform after renorm
    assert compute_policy_entropy(policy) == pytest.approx(np.log(4))
