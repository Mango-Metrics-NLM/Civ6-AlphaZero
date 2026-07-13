"""Unit tests for playout cap randomization."""

import numpy as np
import pytest

from alpha_zero.optimizations.playout_cap import (
    PlayoutCapConfig,
    PlayoutCapRandomizer,
    PlayoutCapStats,
)


class TestPlayoutCapConfig:
    """Tests for PlayoutCapConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PlayoutCapConfig()
        assert config.min_playouts == 50
        assert config.max_playouts == 800
        assert config.cap_distribution == "geometric"

    def test_validation_min_playouts(self):
        """Test validation of min_playouts."""
        with pytest.raises(ValueError, match="min_playouts must be >= 1"):
            PlayoutCapConfig(min_playouts=0)

    def test_validation_max_playouts(self):
        """Test validation of max_playouts."""
        with pytest.raises(ValueError, match=r"max_playouts.*must be >= min_playouts"):
            PlayoutCapConfig(min_playouts=100, max_playouts=50)

    def test_validation_geometric_p(self):
        """Test validation of geometric_p."""
        with pytest.raises(ValueError, match="geometric_p must be in"):
            PlayoutCapConfig(geometric_p=0)
        with pytest.raises(ValueError, match="geometric_p must be in"):
            PlayoutCapConfig(geometric_p=1.5)

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "min_playouts": 25,
            "max_playouts": 400,
            "cap_distribution": "uniform",
        }
        config = PlayoutCapConfig.from_dict(data)
        assert config.min_playouts == 25
        assert config.max_playouts == 400
        assert config.cap_distribution == "uniform"

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = PlayoutCapConfig(min_playouts=100)
        data = config.to_dict()
        assert data["min_playouts"] == 100
        assert "cap_distribution" in data


class TestPlayoutCapStats:
    """Tests for PlayoutCapStats."""

    def test_initial_values(self):
        """Test initial statistics values."""
        stats = PlayoutCapStats()
        assert stats.total_samples == 0
        assert stats.average_cap == 0.0

    def test_record(self):
        """Test recording samples."""
        stats = PlayoutCapStats()
        stats.record(100)
        stats.record(200)
        stats.record(150)

        assert stats.total_samples == 3
        assert stats.average_cap == 150.0
        assert stats.min_sampled == 100
        assert stats.max_sampled == 200

    def test_reset(self):
        """Test resetting statistics."""
        stats = PlayoutCapStats()
        stats.record(100)
        stats.record(200)
        stats.reset()

        assert stats.total_samples == 0
        assert stats.average_cap == 0.0


class TestPlayoutCapRandomizer:
    """Tests for PlayoutCapRandomizer."""

    @pytest.fixture
    def randomizer(self):
        """Create randomizer with default config."""
        rng = np.random.default_rng(42)
        return PlayoutCapRandomizer(PlayoutCapConfig(), rng=rng)

    def test_initialization(self, randomizer):
        """Test randomizer initialization."""
        assert randomizer.config is not None

    def test_sample_within_bounds(self, randomizer):
        """Test that samples are within configured bounds."""
        for _ in range(100):
            cap = randomizer.sample_playout_cap()
            assert randomizer.config.min_playouts <= cap <= randomizer.config.max_playouts

    def test_geometric_distribution(self):
        """Test geometric distribution favors lower values."""
        config = PlayoutCapConfig(
            min_playouts=10,
            max_playouts=1000,
            cap_distribution="geometric",
            geometric_p=0.3,
        )
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(config, rng=rng)

        samples = [randomizer.sample_playout_cap() for _ in range(1000)]
        mean = np.mean(samples)

        # Geometric should have mean below midpoint
        midpoint = (config.min_playouts + config.max_playouts) / 2
        assert mean < midpoint

    def test_uniform_distribution(self):
        """Test uniform distribution is roughly centered."""
        config = PlayoutCapConfig(
            min_playouts=10,
            max_playouts=100,
            cap_distribution="uniform",
        )
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(config, rng=rng)

        samples = [randomizer.sample_playout_cap() for _ in range(1000)]
        mean = np.mean(samples)

        # Uniform should be close to midpoint
        midpoint = (config.min_playouts + config.max_playouts) / 2
        assert abs(mean - midpoint) < 10

    def test_exponential_distribution(self):
        """Test exponential distribution exists and works."""
        config = PlayoutCapConfig(
            min_playouts=10,
            max_playouts=1000,
            cap_distribution="exponential",
        )
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(config, rng=rng)

        # Should not raise
        cap = randomizer.sample_playout_cap()
        assert config.min_playouts <= cap <= config.max_playouts

    def test_complexity_scaling(self):
        """Test complexity scaling adjusts cap based on observation."""
        config = PlayoutCapConfig(
            min_playouts=50,
            max_playouts=500,
            use_position_complexity=True,
        )

        # Dense mask (many valid actions) should scale up
        dense_obs = {"action_mask": np.ones(100, dtype=bool)}

        # Sparse mask should scale down
        sparse_mask = np.zeros(100, dtype=bool)
        sparse_mask[:5] = True
        sparse_obs = {"action_mask": sparse_mask}

        # Note: Due to randomness, we test multiple times
        rng1 = np.random.default_rng(42)
        randomizer1 = PlayoutCapRandomizer(config, rng=rng1)
        dense_caps = [randomizer1.sample_playout_cap(dense_obs) for _ in range(50)]

        rng2 = np.random.default_rng(42)
        randomizer2 = PlayoutCapRandomizer(config, rng=rng2)
        sparse_caps = [randomizer2.sample_playout_cap(sparse_obs) for _ in range(50)]

        # Average should show trend
        assert np.mean(dense_caps) > np.mean(sparse_caps)

    def test_no_complexity_scaling(self):
        """Test without complexity scaling."""
        config = PlayoutCapConfig(
            use_position_complexity=False,
        )
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(config, rng=rng)

        # Both should produce similar caps (within random variation)
        caps = [randomizer.sample_playout_cap() for _ in range(100)]
        assert len(caps) == 100

    def test_reproducibility(self):
        """Test that same seed produces same sequence."""
        config = PlayoutCapConfig()

        rng1 = np.random.default_rng(12345)
        r1 = PlayoutCapRandomizer(config, rng=rng1)

        rng2 = np.random.default_rng(12345)
        r2 = PlayoutCapRandomizer(config, rng=rng2)

        seq1 = [r1.sample_playout_cap() for _ in range(10)]
        seq2 = [r2.sample_playout_cap() for _ in range(10)]

        assert seq1 == seq2

    def test_get_stats(self, randomizer):
        """Test getting statistics."""
        # Generate some samples
        for _ in range(100):
            randomizer.sample_playout_cap()

        stats = randomizer.get_stats()
        assert isinstance(stats, PlayoutCapStats)
        assert stats.total_samples == 100
        assert stats.average_cap > 0

    def test_reset_stats(self, randomizer):
        """Test resetting statistics."""
        for _ in range(50):
            randomizer.sample_playout_cap()

        randomizer.reset_stats()
        stats = randomizer.get_stats()
        assert stats.total_samples == 0

    def test_effective_speedup(self):
        """Test that effective speedup is calculated."""
        config = PlayoutCapConfig(
            min_playouts=50,
            max_playouts=800,
        )
        rng = np.random.default_rng(42)
        randomizer = PlayoutCapRandomizer(config, rng=rng)

        # Generate samples
        for _ in range(100):
            randomizer.sample_playout_cap()

        speedup = randomizer.get_effective_speedup()

        # Speedup should be > 1 since geometric favors lower values
        assert speedup >= 1.0

    def test_repr(self, randomizer):
        """Test string representation."""
        repr_str = repr(randomizer)
        assert "PlayoutCapRandomizer" in repr_str
        assert "50" in repr_str  # min_playouts
        assert "800" in repr_str  # max_playouts
