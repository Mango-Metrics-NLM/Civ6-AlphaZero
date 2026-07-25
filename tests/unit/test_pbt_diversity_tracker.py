"""Tests for PBT policy diversity tracker.

Tests cover:
- DiversityMetrics dataclass (defaults, custom values)
- PolicyDiversityTracker initialization (valid/invalid action_space_size)
- action_space_size property
- tracked_agents property (empty and populated)
- record_actions (new agent, accumulation, out-of-range, empty)
- _get_distribution (unknown, zero-count, recorded agent)
- compute_diversity (no agents, 1 agent, 2 same, 2 different, multiple)
- get_diversity_bonus (unknown, single, identical, diverse, weight scaling)
- reset (clears all data)
- _entropy (uniform, concentrated, single-element)
- _kl_divergence (identical, different, asymmetry)
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.pbt.diversity_tracker import (
    DiversityMetrics,
    PolicyDiversityTracker,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tracker() -> PolicyDiversityTracker:
    """Create a tracker with action space size 5."""
    return PolicyDiversityTracker(action_space_size=5)


@pytest.fixture
def large_tracker() -> PolicyDiversityTracker:
    """Create a tracker with a larger action space for entropy tests."""
    return PolicyDiversityTracker(action_space_size=100)


# =============================================================================
# DiversityMetrics dataclass
# =============================================================================


class TestDiversityMetrics:
    """Tests for the DiversityMetrics dataclass."""

    def test_defaults_all_zero(self) -> None:
        """Default-constructed metrics should have all fields 0.0."""
        metrics = DiversityMetrics()
        assert metrics.action_entropy == 0.0
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0

    def test_custom_values(self) -> None:
        """Custom values should be stored correctly."""
        metrics = DiversityMetrics(
            action_entropy=1.5,
            pairwise_kl_divergence=2.3,
            behavioral_diversity=0.75,
        )
        assert metrics.action_entropy == 1.5
        assert metrics.pairwise_kl_divergence == 2.3
        assert metrics.behavioral_diversity == 0.75

    def test_partial_custom_values(self) -> None:
        """Providing some values leaves others at default."""
        metrics = DiversityMetrics(action_entropy=0.5)
        assert metrics.action_entropy == 0.5
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0


# =============================================================================
# PolicyDiversityTracker.__init__
# =============================================================================


class TestInit:
    """Tests for PolicyDiversityTracker constructor."""

    def test_valid_action_space_size(self) -> None:
        """Constructor accepts positive integer action_space_size."""
        tracker = PolicyDiversityTracker(action_space_size=10)
        assert tracker.action_space_size == 10

    def test_action_space_size_one(self) -> None:
        """Edge case: action_space_size of 1 is valid."""
        tracker = PolicyDiversityTracker(action_space_size=1)
        assert tracker.action_space_size == 1

    def test_invalid_zero_raises(self) -> None:
        """action_space_size of 0 raises ValueError."""
        with pytest.raises(ValueError, match="action_space_size must be >= 1"):
            PolicyDiversityTracker(action_space_size=0)

    def test_invalid_negative_raises(self) -> None:
        """Negative action_space_size raises ValueError."""
        with pytest.raises(ValueError, match="action_space_size must be >= 1"):
            PolicyDiversityTracker(action_space_size=-5)


# =============================================================================
# action_space_size property
# =============================================================================


class TestActionSpaceSizeProperty:
    """Tests for the action_space_size property."""

    def test_returns_correct_value(self) -> None:
        """Property returns the value passed to constructor."""
        for size in [1, 5, 100, 1000]:
            tracker = PolicyDiversityTracker(action_space_size=size)
            assert tracker.action_space_size == size


# =============================================================================
# tracked_agents property
# =============================================================================


class TestTrackedAgents:
    """Tests for the tracked_agents property."""

    def test_empty_initially(self, tracker: PolicyDiversityTracker) -> None:
        """No agents tracked before any recording."""
        assert tracker.tracked_agents == []

    def test_populated_after_recording(self, tracker: PolicyDiversityTracker) -> None:
        """Agent appears after recording actions."""
        tracker.record_actions("agent_1", [0, 1])
        assert "agent_1" in tracker.tracked_agents

    def test_multiple_agents(self, tracker: PolicyDiversityTracker) -> None:
        """Multiple agents are all tracked."""
        tracker.record_actions("a", [0])
        tracker.record_actions("b", [1])
        tracker.record_actions("c", [2])
        agents = tracker.tracked_agents
        assert sorted(agents) == ["a", "b", "c"]

    def test_returns_list_copy(self, tracker: PolicyDiversityTracker) -> None:
        """Returned list is a fresh copy, not a reference to internal state."""
        tracker.record_actions("agent_1", [0])
        agents1 = tracker.tracked_agents
        agents2 = tracker.tracked_agents
        assert agents1 == agents2
        assert agents1 is not agents2


# =============================================================================
# record_actions
# =============================================================================


class TestRecordActions:
    """Tests for record_actions method."""

    def test_new_agent_creates_entry(self, tracker: PolicyDiversityTracker) -> None:
        """First call for an agent creates a new entry."""
        tracker.record_actions("agent_1", [0])
        assert "agent_1" in tracker.tracked_agents

    def test_accumulates_counts(self, tracker: PolicyDiversityTracker) -> None:
        """Multiple calls accumulate action counts."""
        tracker.record_actions("a", [0, 0, 1])
        tracker.record_actions("a", [0, 2])
        dist = tracker._get_distribution("a")
        # Action 0: 3 times, action 1: 1 time, action 2: 1 time -> total 5
        assert dist[0] == pytest.approx(3.0 / 5.0)
        assert dist[1] == pytest.approx(1.0 / 5.0)
        assert dist[2] == pytest.approx(1.0 / 5.0)
        assert dist[3] == pytest.approx(0.0)
        assert dist[4] == pytest.approx(0.0)

    def test_out_of_range_actions_ignored(self, tracker: PolicyDiversityTracker) -> None:
        """Actions outside [0, action_space_size) are silently ignored."""
        tracker.record_actions("a", [-1, 5, 100, -100])
        # Agent entry created but no valid counts
        assert "a" in tracker.tracked_agents
        dist = tracker._get_distribution("a")
        # All zeros -> should return uniform
        expected = np.ones(5) / 5
        np.testing.assert_array_almost_equal(dist, expected)

    def test_empty_actions_list(self, tracker: PolicyDiversityTracker) -> None:
        """Empty action list creates entry but no counts."""
        tracker.record_actions("a", [])
        assert "a" in tracker.tracked_agents
        dist = tracker._get_distribution("a")
        expected = np.ones(5) / 5
        np.testing.assert_array_almost_equal(dist, expected)

    def test_boundary_actions(self, tracker: PolicyDiversityTracker) -> None:
        """Actions at boundaries (0 and action_space_size-1) are recorded."""
        tracker.record_actions("a", [0, 4])  # action_space_size is 5
        dist = tracker._get_distribution("a")
        assert dist[0] == pytest.approx(0.5)
        assert dist[4] == pytest.approx(0.5)

    def test_multiple_agents_independent(self, tracker: PolicyDiversityTracker) -> None:
        """Recording for one agent doesn't affect another."""
        tracker.record_actions("a", [0, 0, 0])
        tracker.record_actions("b", [4, 4, 4])
        dist_a = tracker._get_distribution("a")
        dist_b = tracker._get_distribution("b")
        assert dist_a[0] == pytest.approx(1.0)
        assert dist_b[4] == pytest.approx(1.0)


# =============================================================================
# _get_distribution
# =============================================================================


class TestGetDistribution:
    """Tests for _get_distribution method."""

    def test_unknown_agent_returns_uniform(self, tracker: PolicyDiversityTracker) -> None:
        """Unknown agent returns uniform distribution."""
        dist = tracker._get_distribution("nonexistent")
        expected = np.ones(5) / 5
        np.testing.assert_array_almost_equal(dist, expected)
        assert dist.sum() == pytest.approx(1.0)

    def test_zero_count_agent_returns_uniform(self, tracker: PolicyDiversityTracker) -> None:
        """Agent with zero total counts returns uniform distribution."""
        # Record only out-of-range actions so counts stay zero
        tracker.record_actions("a", [-1, 10, 999])
        dist = tracker._get_distribution("a")
        expected = np.ones(5) / 5
        np.testing.assert_array_almost_equal(dist, expected)

    def test_recorded_agent_returns_normalized(self, tracker: PolicyDiversityTracker) -> None:
        """Agent with recorded actions returns normalized distribution."""
        tracker.record_actions("a", [0, 0, 1, 2, 2])
        dist = tracker._get_distribution("a")
        assert dist.sum() == pytest.approx(1.0)
        assert dist[0] == pytest.approx(2.0 / 5.0)
        assert dist[1] == pytest.approx(1.0 / 5.0)
        assert dist[2] == pytest.approx(2.0 / 5.0)
        assert dist[3] == pytest.approx(0.0)
        assert dist[4] == pytest.approx(0.0)

    def test_single_action_deterministic(self, tracker: PolicyDiversityTracker) -> None:
        """Agent with all actions the same has a deterministic distribution."""
        tracker.record_actions("a", [3, 3, 3, 3])
        dist = tracker._get_distribution("a")
        assert dist[3] == pytest.approx(1.0)
        assert dist.sum() == pytest.approx(1.0)


# =============================================================================
# compute_diversity
# =============================================================================


class TestComputeDiversity:
    """Tests for compute_diversity method."""

    def test_no_agents_returns_zeros(self, tracker: PolicyDiversityTracker) -> None:
        """No tracked agents returns zeroed DiversityMetrics."""
        metrics = tracker.compute_diversity()
        assert metrics.action_entropy == 0.0
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0

    def test_one_agent_no_kl(self, tracker: PolicyDiversityTracker) -> None:
        """Single agent: entropy computed, KL is 0."""
        tracker.record_actions("a", [0, 1, 2, 3, 4])
        metrics = tracker.compute_diversity()
        # Uniform over 5 actions -> entropy = ln(5)
        expected_entropy = float(np.log(5))
        assert metrics.action_entropy == pytest.approx(expected_entropy, abs=1e-6)
        assert metrics.pairwise_kl_divergence == pytest.approx(0.0)
        # behavioral_diversity = 0.5 * norm_entropy + 0.5 * 0
        max_entropy = float(np.log(5))
        norm_entropy = expected_entropy / max_entropy
        assert metrics.behavioral_diversity == pytest.approx(0.5 * norm_entropy, abs=1e-6)

    def test_two_agents_same_actions_low_kl(self, tracker: PolicyDiversityTracker) -> None:
        """Two agents with identical action distributions have near-zero KL."""
        tracker.record_actions("a", [0, 1, 2, 3, 4])
        tracker.record_actions("b", [0, 1, 2, 3, 4])
        metrics = tracker.compute_diversity()
        assert metrics.pairwise_kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_two_agents_different_actions_high_kl(self, tracker: PolicyDiversityTracker) -> None:
        """Two agents with disjoint action distributions have high KL."""
        # Agent a only does action 0, agent b only does action 4
        tracker.record_actions("a", [0] * 100)
        tracker.record_actions("b", [4] * 100)
        metrics = tracker.compute_diversity()
        assert metrics.pairwise_kl_divergence > 1.0
        assert metrics.behavioral_diversity > 0.0

    def test_multiple_agents(self, tracker: PolicyDiversityTracker) -> None:
        """Three agents with varying distributions."""
        tracker.record_actions("a", [0, 0, 0, 1])
        tracker.record_actions("b", [2, 2, 3])
        tracker.record_actions("c", [0, 1, 2, 3, 4])
        metrics = tracker.compute_diversity()
        # All metrics should be positive
        assert metrics.action_entropy > 0.0
        assert metrics.pairwise_kl_divergence > 0.0
        assert metrics.behavioral_diversity > 0.0
        # Behavioral diversity in [0, 1]
        assert 0.0 <= metrics.behavioral_diversity <= 1.0

    def test_behavioral_diversity_bounded(self, tracker: PolicyDiversityTracker) -> None:
        """Behavioral diversity should be between 0 and 1."""
        tracker.record_actions("a", [0] * 50)
        tracker.record_actions("b", [4] * 50)
        tracker.record_actions("c", [0, 1, 2, 3, 4] * 10)
        metrics = tracker.compute_diversity()
        assert 0.0 <= metrics.behavioral_diversity <= 1.0

    def test_action_space_size_one_no_crash(self) -> None:
        """Edge case: action_space_size=1 should not crash."""
        t = PolicyDiversityTracker(action_space_size=1)
        t.record_actions("a", [0, 0, 0])
        t.record_actions("b", [0, 0])
        metrics = t.compute_diversity()
        # Only one action possible -> entropy is 0, KL is 0
        assert metrics.action_entropy == pytest.approx(0.0, abs=1e-6)
        assert metrics.pairwise_kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_single_agent_with_concentrated_distribution(
        self, tracker: PolicyDiversityTracker
    ) -> None:
        """Single agent choosing only one action has low entropy."""
        tracker.record_actions("a", [2] * 100)
        metrics = tracker.compute_diversity()
        # Only action 2 chosen -> entropy = 0
        assert metrics.action_entropy == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# get_diversity_bonus
# =============================================================================


class TestGetDiversityBonus:
    """Tests for get_diversity_bonus method."""

    def test_unknown_agent_returns_zero(self, tracker: PolicyDiversityTracker) -> None:
        """Unknown agent gets 0 bonus."""
        tracker.record_actions("a", [0])
        tracker.record_actions("b", [1])
        bonus = tracker.get_diversity_bonus("nonexistent")
        assert bonus == 0.0

    def test_single_agent_returns_zero(self, tracker: PolicyDiversityTracker) -> None:
        """With only one agent, bonus is 0."""
        tracker.record_actions("a", [0, 1, 2])
        bonus = tracker.get_diversity_bonus("a")
        assert bonus == 0.0

    def test_identical_agents_returns_zero_or_near_zero(
        self, tracker: PolicyDiversityTracker
    ) -> None:
        """Two agents with identical distributions have near-zero bonus."""
        tracker.record_actions("a", [0, 1, 2, 3, 4])
        tracker.record_actions("b", [0, 1, 2, 3, 4])
        bonus_a = tracker.get_diversity_bonus("a")
        bonus_b = tracker.get_diversity_bonus("b")
        assert bonus_a == pytest.approx(0.0, abs=1e-6)
        assert bonus_b == pytest.approx(0.0, abs=1e-6)

    def test_diverse_agent_gets_bonus(self, tracker: PolicyDiversityTracker) -> None:
        """Agent with unique strategy gets positive bonus."""
        # Most agents do action 0
        tracker.record_actions("a", [0] * 100)
        tracker.record_actions("b", [0] * 100)
        tracker.record_actions("c", [0] * 100)
        # One agent is unique
        tracker.record_actions("unique", [4] * 100)
        bonus = tracker.get_diversity_bonus("unique")
        assert bonus > 0.0

    def test_weight_parameter_scales_bonus(self, tracker: PolicyDiversityTracker) -> None:
        """Higher weight produces proportionally larger bonus."""
        tracker.record_actions("a", [0] * 100)
        tracker.record_actions("b", [4] * 100)

        bonus_low = tracker.get_diversity_bonus("a", weight=0.1)
        bonus_high = tracker.get_diversity_bonus("a", weight=1.0)
        assert bonus_high > bonus_low
        # Since bonus = weight * f(kl), ratio should be exactly weight ratio
        assert bonus_high == pytest.approx(bonus_low * 10.0, rel=1e-6)

    def test_bonus_non_negative(self, tracker: PolicyDiversityTracker) -> None:
        """Bonus should always be non-negative."""
        tracker.record_actions("a", [0, 1])
        tracker.record_actions("b", [2, 3])
        tracker.record_actions("c", [4, 0])
        for agent in ["a", "b", "c"]:
            bonus = tracker.get_diversity_bonus(agent)
            assert bonus >= 0.0

    def test_weight_zero_returns_zero(self, tracker: PolicyDiversityTracker) -> None:
        """Weight of 0 always gives 0 bonus regardless of diversity."""
        tracker.record_actions("a", [0] * 100)
        tracker.record_actions("b", [4] * 100)
        bonus = tracker.get_diversity_bonus("a", weight=0.0)
        assert bonus == pytest.approx(0.0)

    def test_no_other_agents_after_filtering(self) -> None:
        """Edge case: if filtering removes all others, bonus is 0.

        This tests the `if not other_dists` guard. Practically unreachable
        when len(agent_ids) >= 2 and agent_id is in the dict, but we test
        the logic path nonetheless.
        """
        t = PolicyDiversityTracker(action_space_size=5)
        t.record_actions("solo", [0, 1, 2])
        # Only one agent -> less than 2 -> returns 0
        assert t.get_diversity_bonus("solo") == 0.0


# =============================================================================
# reset
# =============================================================================


class TestReset:
    """Tests for reset method."""

    def test_clears_all_data(self, tracker: PolicyDiversityTracker) -> None:
        """Reset removes all tracked agents."""
        tracker.record_actions("a", [0, 1])
        tracker.record_actions("b", [2, 3])
        assert len(tracker.tracked_agents) == 2

        tracker.reset()
        assert tracker.tracked_agents == []

    def test_after_reset_compute_diversity_returns_zeros(
        self, tracker: PolicyDiversityTracker
    ) -> None:
        """After reset, compute_diversity returns zeroed metrics."""
        tracker.record_actions("a", [0])
        tracker.reset()
        metrics = tracker.compute_diversity()
        assert metrics.action_entropy == 0.0
        assert metrics.pairwise_kl_divergence == 0.0
        assert metrics.behavioral_diversity == 0.0

    def test_can_record_after_reset(self, tracker: PolicyDiversityTracker) -> None:
        """Recording works normally after a reset."""
        tracker.record_actions("a", [0])
        tracker.reset()
        tracker.record_actions("b", [1, 2])
        assert tracker.tracked_agents == ["b"]
        dist = tracker._get_distribution("b")
        assert dist[1] == pytest.approx(0.5)
        assert dist[2] == pytest.approx(0.5)

    def test_action_space_size_preserved(self, tracker: PolicyDiversityTracker) -> None:
        """Reset does not change action_space_size."""
        tracker.record_actions("a", [0])
        tracker.reset()
        assert tracker.action_space_size == 5


# =============================================================================
# _entropy (static method)
# =============================================================================


class TestEntropy:
    """Tests for the _entropy static method."""

    def test_uniform_distribution_max_entropy(self) -> None:
        """Uniform distribution gives maximum entropy = ln(n)."""
        n = 10
        dist = np.ones(n) / n
        expected = float(np.log(n))
        result = PolicyDiversityTracker._entropy(dist)
        assert result == pytest.approx(expected, abs=1e-10)

    def test_concentrated_distribution_low_entropy(self) -> None:
        """Deterministic distribution has entropy 0."""
        dist = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = PolicyDiversityTracker._entropy(dist)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_single_element_distribution(self) -> None:
        """Distribution of length 1 ([1.0]) has entropy 0."""
        dist = np.array([1.0])
        result = PolicyDiversityTracker._entropy(dist)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_binary_uniform_entropy(self) -> None:
        """Binary uniform distribution: entropy = ln(2)."""
        dist = np.array([0.5, 0.5])
        result = PolicyDiversityTracker._entropy(dist)
        assert result == pytest.approx(float(np.log(2)), abs=1e-10)

    def test_entropy_non_negative(self) -> None:
        """Entropy should always be non-negative."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            raw = rng.random(10)
            dist = raw / raw.sum()
            assert PolicyDiversityTracker._entropy(dist) >= 0.0

    def test_near_deterministic_low_entropy(self) -> None:
        """Nearly deterministic distribution has very low entropy."""
        dist = np.array([0.99, 0.005, 0.005])
        result = PolicyDiversityTracker._entropy(dist)
        uniform_entropy = float(np.log(3))
        assert result < 0.1 * uniform_entropy

    def test_zeros_handled_gracefully(self) -> None:
        """Distribution with zeros does not cause NaN or error."""
        dist = np.array([0.5, 0.0, 0.5, 0.0, 0.0])
        result = PolicyDiversityTracker._entropy(dist)
        assert np.isfinite(result)
        assert result > 0.0


# =============================================================================
# _kl_divergence (static method)
# =============================================================================


class TestKLDivergence:
    """Tests for the _kl_divergence static method."""

    def test_identical_distributions_near_zero(self) -> None:
        """KL(p || p) should be approximately 0."""
        dist = np.array([0.2, 0.3, 0.5])
        kl = PolicyDiversityTracker._kl_divergence(dist, dist)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_positive(self) -> None:
        """KL divergence between different distributions is positive."""
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        kl = PolicyDiversityTracker._kl_divergence(p, q)
        assert kl > 0.0

    def test_asymmetry(self) -> None:
        """KL(p || q) != KL(q || p) in general (asymmetric)."""
        p = np.array([0.9, 0.05, 0.05])
        q = np.array([0.1, 0.1, 0.8])
        kl_pq = PolicyDiversityTracker._kl_divergence(p, q)
        kl_qp = PolicyDiversityTracker._kl_divergence(q, p)
        assert kl_pq != pytest.approx(kl_qp, abs=0.01)

    def test_kl_non_negative(self) -> None:
        """KL divergence is always non-negative (Gibbs' inequality)."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            raw_p = rng.random(5)
            raw_q = rng.random(5)
            p = raw_p / raw_p.sum()
            q = raw_q / raw_q.sum()
            kl = PolicyDiversityTracker._kl_divergence(p, q)
            assert kl >= -1e-10  # Allow tiny numerical noise

    def test_uniform_distributions_near_zero(self) -> None:
        """KL between two uniform distributions is near zero."""
        n = 10
        p = np.ones(n) / n
        q = np.ones(n) / n
        kl = PolicyDiversityTracker._kl_divergence(p, q)
        assert kl == pytest.approx(0.0, abs=1e-6)

    def test_handles_zeros_in_distributions(self) -> None:
        """Distributions with zeros are handled via epsilon clipping."""
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        kl = PolicyDiversityTracker._kl_divergence(p, q)
        assert np.isfinite(kl)
        assert kl > 0.0

    def test_known_kl_value(self) -> None:
        """Verify KL against manually computed value.

        For p = [0.5, 0.5] and q = [0.25, 0.75]:
        KL(p||q) = 0.5 * ln(0.5/0.25) + 0.5 * ln(0.5/0.75)
                 = 0.5 * ln(2) + 0.5 * ln(2/3)
        """
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * np.log(2.0) + 0.5 * np.log(2.0 / 3.0)
        kl = PolicyDiversityTracker._kl_divergence(p, q)
        assert kl == pytest.approx(float(expected), abs=1e-6)


# =============================================================================
# Integration / end-to-end scenarios
# =============================================================================


class TestEndToEnd:
    """Integration-style tests combining multiple methods."""

    def test_full_workflow(self) -> None:
        """Full cycle: create, record, compute, bonus, reset."""
        tracker = PolicyDiversityTracker(action_space_size=10)

        # Record diverse actions for three agents
        tracker.record_actions("explorer", list(range(10)) * 5)
        tracker.record_actions("specialist_a", [0] * 50)
        tracker.record_actions("specialist_b", [9] * 50)

        # Compute metrics
        metrics = tracker.compute_diversity()
        assert metrics.action_entropy > 0.0
        assert metrics.pairwise_kl_divergence > 0.0
        assert 0.0 < metrics.behavioral_diversity <= 1.0

        # Specialist should get bonus relative to others
        bonus_a = tracker.get_diversity_bonus("specialist_a")
        bonus_b = tracker.get_diversity_bonus("specialist_b")
        bonus_exp = tracker.get_diversity_bonus("explorer")
        assert bonus_a > 0.0
        assert bonus_b > 0.0
        assert bonus_exp > 0.0

        # Reset clears everything
        tracker.reset()
        assert tracker.tracked_agents == []
        metrics_after = tracker.compute_diversity()
        assert metrics_after.action_entropy == 0.0

    def test_incremental_recording(self) -> None:
        """Recording actions in batches produces consistent distributions."""
        tracker = PolicyDiversityTracker(action_space_size=4)

        # Record in small batches
        for _ in range(10):
            tracker.record_actions("a", [0, 1])
            tracker.record_actions("a", [2, 3])

        dist = tracker._get_distribution("a")
        # Should be uniform over 4 actions
        expected = np.ones(4) / 4
        np.testing.assert_array_almost_equal(dist, expected)

    def test_symmetric_kl_in_compute_diversity(self) -> None:
        """Verify that compute_diversity uses symmetric KL averaging."""
        tracker = PolicyDiversityTracker(action_space_size=3)
        tracker.record_actions("a", [0] * 100)
        tracker.record_actions("b", [2] * 100)

        metrics = tracker.compute_diversity()
        # Manually compute symmetric KL for verification
        dist_a = tracker._get_distribution("a")
        dist_b = tracker._get_distribution("b")
        kl_ab = PolicyDiversityTracker._kl_divergence(dist_a, dist_b)
        kl_ba = PolicyDiversityTracker._kl_divergence(dist_b, dist_a)
        expected_avg_kl = (kl_ab + kl_ba) / 2.0

        assert metrics.pairwise_kl_divergence == pytest.approx(expected_avg_kl, abs=1e-6)

    def test_compute_diversity_entropy_is_average(self) -> None:
        """Action entropy is the average of per-agent entropies."""
        tracker = PolicyDiversityTracker(action_space_size=4)
        tracker.record_actions("uniform", [0, 1, 2, 3])  # max entropy
        tracker.record_actions("deterministic", [0] * 100)  # zero entropy

        metrics = tracker.compute_diversity()

        h_uniform = PolicyDiversityTracker._entropy(tracker._get_distribution("uniform"))
        h_deterministic = PolicyDiversityTracker._entropy(
            tracker._get_distribution("deterministic")
        )
        expected_avg = (h_uniform + h_deterministic) / 2.0

        assert metrics.action_entropy == pytest.approx(expected_avg, abs=1e-6)

    def test_large_population(self, large_tracker: PolicyDiversityTracker) -> None:
        """Tracker handles larger populations gracefully."""
        rng = np.random.default_rng(99)
        for i in range(20):
            actions = rng.integers(0, 100, size=50).tolist()
            large_tracker.record_actions(f"agent_{i}", actions)

        metrics = large_tracker.compute_diversity()
        assert metrics.action_entropy > 0.0
        assert metrics.pairwise_kl_divergence >= 0.0
        assert 0.0 <= metrics.behavioral_diversity <= 1.0

        # All agents should get non-negative bonus
        for i in range(20):
            bonus = large_tracker.get_diversity_bonus(f"agent_{i}")
            assert bonus >= 0.0
