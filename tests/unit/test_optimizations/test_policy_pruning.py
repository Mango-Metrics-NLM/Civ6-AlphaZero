"""Unit tests for policy target pruning."""

import numpy as np
import pytest

from alpha_zero.optimizations.policy_pruning import (
    PolicyPruningConfig,
    PolicyTargetPruner,
    PruningStats,
)


class TestPolicyPruningConfig:
    """Tests for PolicyPruningConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PolicyPruningConfig()
        assert config.min_probability == 0.001
        assert config.max_actions == 100
        assert config.preserve_top_k == 5

    def test_validation_min_probability(self):
        """Test validation of min_probability."""
        with pytest.raises(ValueError, match="min_probability must be >= 0"):
            PolicyPruningConfig(min_probability=-0.1)

    def test_validation_max_actions(self):
        """Test validation of max_actions."""
        with pytest.raises(ValueError, match="max_actions must be >= 1"):
            PolicyPruningConfig(max_actions=0)

    def test_validation_preserve_top_k(self):
        """Test validation of preserve_top_k."""
        with pytest.raises(ValueError, match="preserve_top_k must be >= 0"):
            PolicyPruningConfig(preserve_top_k=-1)

    def test_validation_temperature_scaling(self):
        """Test validation of temperature_scaling."""
        with pytest.raises(ValueError, match="temperature_scaling must be > 0"):
            PolicyPruningConfig(temperature_scaling=0)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "min_probability": 0.01,
            "max_actions": 50,
            "preserve_top_k": 3,
        }
        config = PolicyPruningConfig.from_dict(data)
        assert config.min_probability == 0.01
        assert config.max_actions == 50
        assert config.preserve_top_k == 3

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = PolicyPruningConfig(max_actions=75)
        data = config.to_dict()
        assert data["max_actions"] == 75


class TestPruningStats:
    """Tests for PruningStats."""

    def test_initial_values(self):
        """Test initial statistics values."""
        stats = PruningStats()
        assert stats.total_policies == 0
        assert stats.avg_original_actions == 0.0

    def test_record(self):
        """Test recording pruning operations."""
        stats = PruningStats()
        stats.record(original_actions=100, pruned_actions=50, kl_divergence=0.1)
        stats.record(original_actions=80, pruned_actions=40, kl_divergence=0.2)

        assert stats.total_policies == 2
        assert stats.avg_original_actions == 90.0
        assert stats.avg_pruned_actions == 45.0

    def test_reset(self):
        """Test resetting statistics."""
        stats = PruningStats()
        stats.record(100, 50, 0.1)
        stats.reset()

        assert stats.total_policies == 0
        assert stats.avg_original_actions == 0.0


class TestPolicyTargetPruner:
    """Tests for PolicyTargetPruner."""

    @pytest.fixture
    def pruner(self):
        """Create pruner with default config."""
        return PolicyTargetPruner(PolicyPruningConfig())

    def test_initialization(self, pruner):
        """Test pruner initialization."""
        assert pruner.config is not None

    def test_prune_preserves_normalization(self, pruner):
        """Test that pruned policy sums to 1."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        pruned = pruner.prune(policy, mask)
        assert abs(pruned.sum() - 1.0) < 1e-6

    def test_prune_respects_mask(self, pruner):
        """Test that masked actions are zeroed."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.zeros(100, dtype=bool)
        mask[:50] = True  # Only first 50 valid

        pruned = pruner.prune(policy, mask)
        assert np.all(pruned[50:] == 0)

    def test_prune_preserves_top_k(self):
        """Test that top-k actions are always preserved."""
        config = PolicyPruningConfig(
            min_probability=0.5,  # Very aggressive
            preserve_top_k=3,
        )
        pruner = PolicyTargetPruner(config)

        # Create policy with clear top actions
        policy = np.zeros(100)
        policy[0] = 0.4
        policy[1] = 0.3
        policy[2] = 0.2
        policy[3:10] = 0.01  # These are below min

        mask = np.ones(100, dtype=bool)
        pruned = pruner.prune(policy, mask)

        # Top 3 should be preserved
        assert pruned[0] > 0
        assert pruned[1] > 0
        assert pruned[2] > 0

    def test_prune_max_actions(self):
        """Test that max_actions limit is enforced."""
        config = PolicyPruningConfig(
            min_probability=0.0,  # No minimum
            max_actions=10,
        )
        pruner = PolicyTargetPruner(config)

        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        pruned = pruner.prune(policy, mask)
        non_zero = (pruned > 0).sum()
        assert non_zero <= 10

    def test_prune_removes_low_probability(self):
        """Test that low probability actions are removed."""
        config = PolicyPruningConfig(
            min_probability=0.1,
            preserve_top_k=0,  # Don't preserve any
        )
        pruner = PolicyTargetPruner(config)

        policy = np.zeros(100)
        policy[0] = 0.5
        policy[1] = 0.3
        policy[2:10] = 0.02  # Below threshold

        mask = np.ones(100, dtype=bool)
        pruned = pruner.prune(policy, mask)

        # Only first two should remain (after renormalization)
        assert (pruned > 0).sum() == 2

    def test_temperature_scaling(self):
        """Test temperature affects distribution sharpness."""
        config_low = PolicyPruningConfig(temperature_scaling=0.5)
        config_high = PolicyPruningConfig(temperature_scaling=2.0)

        pruner_low = PolicyTargetPruner(config_low)
        pruner_high = PolicyTargetPruner(config_high)

        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        pruned_low = pruner_low.prune(policy, mask)
        pruned_high = pruner_high.prune(policy, mask)

        # Lower temperature should have higher max probability
        assert pruned_low.max() > pruned_high.max()

    def test_compute_pruning_stats(self, pruner):
        """Test computing pruning statistics."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        pruned = pruner.prune(policy, mask)
        stats = pruner.compute_pruning_stats(policy, pruned)

        assert "original_actions" in stats
        assert "pruned_actions" in stats
        assert "kl_divergence" in stats
        assert stats["kl_divergence"] >= 0

    def test_entropy_threshold(self):
        """Test dynamic threshold based on entropy."""
        config = PolicyPruningConfig(
            use_entropy_threshold=True,
        )
        pruner = PolicyTargetPruner(config)

        # High entropy policy (uniform)
        high_entropy = np.ones(100) / 100
        mask = np.ones(100, dtype=bool)
        pruned_high = pruner.prune(high_entropy, mask)

        # Low entropy policy (concentrated)
        low_entropy = np.zeros(100)
        low_entropy[:5] = 0.2
        pruned_low = pruner.prune(low_entropy, mask)

        # Both should be pruned (number of non-zero may vary)
        assert (pruned_high > 0).sum() > 0
        assert (pruned_low > 0).sum() > 0

    def test_get_stats(self, pruner):
        """Test getting statistics."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        for _ in range(50):
            pruner.prune(policy, mask)

        stats = pruner.get_stats()
        assert isinstance(stats, PruningStats)
        assert stats.total_policies == 50

    def test_reset_stats(self, pruner):
        """Test resetting statistics."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        for _ in range(25):
            pruner.prune(policy, mask)

        pruner.reset_stats()
        stats = pruner.get_stats()
        assert stats.total_policies == 0

    def test_action_reduction_ratio(self, pruner):
        """Test action reduction is achieved."""
        policy = np.random.dirichlet(np.ones(100))
        mask = np.ones(100, dtype=bool)

        pruned = pruner.prune(policy, mask)

        original_actions = mask.sum()
        remaining_actions = (pruned > 0).sum()

        # Should have meaningful reduction
        assert remaining_actions < original_actions

    def test_empty_valid_actions(self, pruner):
        """Test handling of no valid actions."""
        policy = np.ones(100) / 100
        mask = np.zeros(100, dtype=bool)  # No valid actions

        pruned = pruner.prune(policy, mask)
        # Should either be all zeros or normalized
        assert pruned.sum() == 0 or abs(pruned.sum() - 1.0) < 1e-6

    def test_repr(self, pruner):
        """Test string representation."""
        repr_str = repr(pruner)
        assert "PolicyTargetPruner" in repr_str


class TestPolicyPruningIntegration:
    """Integration tests for policy pruning."""

    def test_realistic_scenario(self):
        """Test with realistic game-like policy."""
        config = PolicyPruningConfig(
            min_probability=0.001,
            max_actions=50,
            preserve_top_k=5,
        )
        pruner = PolicyTargetPruner(config)

        # Simulate MCTS-like output
        policy = np.zeros(1000)
        # Few high-probability actions
        policy[:10] = np.random.dirichlet(np.ones(10) * 5)[:10] * 0.8
        # Many low-probability actions
        policy[10:100] = 0.002
        policy[100:] = 0.0

        mask = np.zeros(1000, dtype=bool)
        mask[:200] = True

        pruned = pruner.prune(policy, mask)

        # Should be normalized
        assert abs(pruned.sum() - 1.0) < 1e-6

        # Should have reduced actions significantly
        assert (pruned > 0).sum() <= 50
