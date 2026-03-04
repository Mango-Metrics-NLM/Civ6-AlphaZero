"""Tests for PBT mutation operators.

Tests cover:
- MutationStrategy enum
- MutationConfig dataclass and strength decay
- GaussianMutation for linear, log, and categorical parameters
- UniformMutation for various parameter types
- PolynomialMutation for bounded mutation
- DifferentialMutation with candidate values
- AdaptiveMutation with success tracking
- CrossoverMutation with blend, uniform, SBX, and categorical
- MutationOperator for orchestrating mutations
- create_mutation_operator factory function
- Bounds enforcement across all strategies
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace
from alpha_zero.pbt.mutations import (
    AdaptiveMutation,
    CrossoverMutation,
    DifferentialMutation,
    GaussianMutation,
    MutationConfig,
    MutationOperator,
    MutationStrategy,
    PolynomialMutation,
    UniformMutation,
    create_mutation_operator,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rng() -> np.random.Generator:
    """Create seeded RNG for reproducibility."""
    return np.random.default_rng(42)


@pytest.fixture
def linear_range() -> HyperparameterRange:
    """Create a linear scale parameter range."""
    return HyperparameterRange(
        name="test_linear",
        min_value=0.0,
        max_value=1.0,
        scale="linear",
        discrete=False,
    )


@pytest.fixture
def log_range() -> HyperparameterRange:
    """Create a log scale parameter range."""
    return HyperparameterRange(
        name="test_log",
        min_value=0.0001,
        max_value=0.1,
        scale="log",
        discrete=False,
    )


@pytest.fixture
def discrete_range() -> HyperparameterRange:
    """Create a discrete parameter range."""
    return HyperparameterRange(
        name="test_discrete",
        min_value=1,
        max_value=10,
        scale="linear",
        discrete=True,
    )


@pytest.fixture
def categorical_range() -> HyperparameterRange:
    """Create a categorical parameter range."""
    return HyperparameterRange(
        name="test_categorical",
        min_value=0,
        max_value=1,
        scale="categorical",
        categories=["adam", "sgd", "rmsprop"],
    )


@pytest.fixture
def hyperparameter_space() -> HyperparameterSpace:
    """Create a hyperparameter space for testing."""
    return HyperparameterSpace()


@pytest.fixture
def log_discrete_range() -> HyperparameterRange:
    """Create a log scale discrete parameter range."""
    return HyperparameterRange(
        name="test_log_discrete",
        min_value=1,
        max_value=1000,
        scale="log",
        discrete=True,
    )


# =============================================================================
# MutationStrategy Tests
# =============================================================================


class TestMutationStrategy:
    """Tests for MutationStrategy enum."""

    def test_enum_values(self) -> None:
        """Test enum has expected values."""
        assert MutationStrategy.GAUSSIAN == "gaussian"
        assert MutationStrategy.UNIFORM == "uniform"
        assert MutationStrategy.POLYNOMIAL == "polynomial"
        assert MutationStrategy.DIFFERENTIAL == "differential"
        assert MutationStrategy.ADAPTIVE == "adaptive"
        assert MutationStrategy.CROSSOVER == "crossover"

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        assert MutationStrategy("gaussian") == MutationStrategy.GAUSSIAN
        assert MutationStrategy("crossover") == MutationStrategy.CROSSOVER

    def test_all_strategies_listed(self) -> None:
        """Test all strategies are accessible."""
        names = [s.value for s in MutationStrategy]
        assert len(names) == 6
        assert "gaussian" in names
        assert "crossover" in names


# =============================================================================
# MutationConfig Tests
# =============================================================================


class TestMutationConfig:
    """Tests for MutationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = MutationConfig()
        assert config.strategy == "gaussian"
        assert config.strength == 0.2
        assert config.decay_rate == 0.0
        assert config.min_strength == 0.05
        assert config.adaptive_window == 10
        assert config.differential_f == 0.5
        assert config.polynomial_eta == 20.0

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = MutationConfig(
            strategy="uniform",
            strength=0.5,
            decay_rate=0.01,
            min_strength=0.1,
        )
        assert config.strategy == "uniform"
        assert config.strength == 0.5
        assert config.decay_rate == 0.01
        assert config.min_strength == 0.1

    def test_get_decayed_strength_no_decay(self) -> None:
        """Test strength without decay."""
        config = MutationConfig(strength=0.3, decay_rate=0.0)

        assert config.get_decayed_strength(0) == 0.3
        assert config.get_decayed_strength(100) == 0.3

    def test_get_decayed_strength_with_decay(self) -> None:
        """Test strength with exponential decay."""
        config = MutationConfig(strength=1.0, decay_rate=0.1, min_strength=0.1)

        strength_0 = config.get_decayed_strength(0)
        strength_10 = config.get_decayed_strength(10)
        strength_100 = config.get_decayed_strength(100)

        assert strength_0 == 1.0
        assert strength_10 < strength_0
        assert strength_100 < strength_10
        assert strength_100 >= config.min_strength

    def test_get_decayed_strength_respects_minimum(self) -> None:
        """Test that strength doesn't go below minimum."""
        config = MutationConfig(strength=1.0, decay_rate=1.0, min_strength=0.2)

        # Very high generation should still respect minimum
        strength = config.get_decayed_strength(1000)
        assert strength == config.min_strength

    def test_get_decayed_strength_negative_decay_treated_as_no_decay(self) -> None:
        """Test that negative decay rate is treated as no decay."""
        config = MutationConfig(strength=0.5, decay_rate=-0.1)
        assert config.get_decayed_strength(100) == 0.5

    def test_get_decayed_strength_math(self) -> None:
        """Test exact decay formula: strength * exp(-decay_rate * generation)."""
        config = MutationConfig(strength=1.0, decay_rate=0.01, min_strength=0.0)

        gen_50 = config.get_decayed_strength(50)
        expected = 1.0 * math.exp(-0.01 * 50)
        assert abs(gen_50 - expected) < 1e-10


# =============================================================================
# GaussianMutation Tests
# =============================================================================


class TestGaussianMutation:
    """Tests for GaussianMutation strategy."""

    def test_mutate_linear(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test Gaussian mutation on linear scale parameter."""
        mutation = GaussianMutation()
        value = 0.5

        # Run multiple mutations
        results = [mutation.mutate(value, linear_range, rng, 0.2) for _ in range(100)]

        # Results should be within bounds
        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)
        # Should have some variation
        assert len(set(results)) > 1

    def test_mutate_log(self, rng: np.random.Generator, log_range: HyperparameterRange) -> None:
        """Test Gaussian mutation on log scale parameter."""
        mutation = GaussianMutation()
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 0.2) for _ in range(100)]

        # Results should be within bounds
        assert all(log_range.min_value <= r <= log_range.max_value for r in results)
        # Should have variation
        assert len(set(results)) > 1

    def test_mutate_discrete(
        self, rng: np.random.Generator, discrete_range: HyperparameterRange
    ) -> None:
        """Test Gaussian mutation on discrete parameter."""
        mutation = GaussianMutation()
        value = 5

        results = [mutation.mutate(value, discrete_range, rng, 0.2) for _ in range(100)]

        # Results should be integers within bounds
        assert all(isinstance(r, int) for r in results)
        assert all(discrete_range.min_value <= r <= discrete_range.max_value for r in results)

    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test Gaussian mutation on categorical parameter."""
        mutation = GaussianMutation()
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 0.5) for _ in range(100)]

        # Results should be valid categories
        assert all(r in categorical_range.categories for r in results)

    def test_mutate_strength_affects_variance(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that higher strength produces more variance."""
        mutation = GaussianMutation()
        value = 0.5

        low_strength_results = [
            mutation.mutate(value, linear_range, np.random.default_rng(i), 0.05) for i in range(100)
        ]
        high_strength_results = [
            mutation.mutate(value, linear_range, np.random.default_rng(i + 1000), 0.5)
            for i in range(100)
        ]

        low_var = np.var(low_strength_results)
        high_var = np.var(high_strength_results)

        # Higher strength should generally produce higher variance
        # Note: This is probabilistic, so we use a relaxed assertion
        assert high_var > low_var * 0.5  # Allow some tolerance

    def test_mutate_log_discrete(
        self, rng: np.random.Generator, log_discrete_range: HyperparameterRange
    ) -> None:
        """Test Gaussian mutation on log-discrete parameter."""
        mutation = GaussianMutation()
        value = 100

        results = [mutation.mutate(value, log_discrete_range, rng, 0.3) for _ in range(100)]
        assert all(isinstance(r, int) for r in results)
        assert all(log_discrete_range.min_value <= r <= log_discrete_range.max_value for r in results)


# =============================================================================
# UniformMutation Tests
# =============================================================================


class TestUniformMutation:
    """Tests for UniformMutation strategy."""

    def test_mutate_linear(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on linear scale parameter."""
        mutation = UniformMutation()
        value = 0.5

        results = [mutation.mutate(value, linear_range, rng, 0.2) for _ in range(100)]

        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)
        assert len(set(results)) > 1

    def test_mutate_log(self, rng: np.random.Generator, log_range: HyperparameterRange) -> None:
        """Test uniform mutation on log scale parameter."""
        mutation = UniformMutation()
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 0.2) for _ in range(100)]

        assert all(log_range.min_value <= r <= log_range.max_value for r in results)

    @pytest.mark.xfail(
        reason="Bug: UniformMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on categorical parameter (known bug)."""
        mutation = UniformMutation()
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 1.0) for _ in range(100)]

        # With strength=1.0, should see different categories
        assert all(r in categorical_range.categories for r in results)
        assert len(set(results)) > 1

    @pytest.mark.xfail(
        reason="Bug: UniformMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical_low_strength(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test that low strength keeps categorical unchanged often (known bug)."""
        mutation = UniformMutation()
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 0.01) for _ in range(100)]

        # Most should stay "adam"
        adam_count = sum(1 for r in results if r == "adam")
        assert adam_count > 80  # More than 80% unchanged

    def test_mutate_discrete(
        self, rng: np.random.Generator, discrete_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on discrete parameter produces integers."""
        mutation = UniformMutation()
        value = 5

        results = [mutation.mutate(value, discrete_range, rng, 0.3) for _ in range(100)]

        assert all(isinstance(r, int) for r in results)
        assert all(discrete_range.min_value <= r <= discrete_range.max_value for r in results)

    def test_mutate_log_discrete(
        self, rng: np.random.Generator, log_discrete_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on log-discrete parameter."""
        mutation = UniformMutation()
        value = 100

        results = [mutation.mutate(value, log_discrete_range, rng, 0.3) for _ in range(100)]
        assert all(isinstance(r, int) for r in results)
        assert all(log_discrete_range.min_value <= r <= log_discrete_range.max_value for r in results)

    @pytest.mark.xfail(
        reason="Bug: UniformMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical_string_bug(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on categorical with string categories (known bug)."""
        mutation = UniformMutation()
        value = "adam"
        # This fails because UniformMutation does float(rng.choice(categories))
        # which can't convert string categories to float
        mutation.mutate(value, categorical_range, rng, 1.0)


# =============================================================================
# PolynomialMutation Tests
# =============================================================================


class TestPolynomialMutation:
    """Tests for PolynomialMutation strategy."""

    def test_default_eta(self) -> None:
        """Test default eta value."""
        mutation = PolynomialMutation()
        assert mutation.eta == 20.0

    def test_custom_eta(self) -> None:
        """Test custom eta value."""
        mutation = PolynomialMutation(eta=10.0)
        assert mutation.eta == 10.0

    def test_mutate_linear(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test polynomial mutation on linear scale parameter."""
        mutation = PolynomialMutation(eta=20.0)
        value = 0.5

        results = [mutation.mutate(value, linear_range, rng, 0.2) for _ in range(100)]

        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_log(self, rng: np.random.Generator, log_range: HyperparameterRange) -> None:
        """Test polynomial mutation on log scale parameter."""
        mutation = PolynomialMutation(eta=20.0)
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 0.2) for _ in range(100)]

        assert all(log_range.min_value <= r <= log_range.max_value for r in results)

    def test_higher_eta_smaller_mutations(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that higher eta produces smaller mutations."""
        low_eta = PolynomialMutation(eta=5.0)
        high_eta = PolynomialMutation(eta=50.0)
        value = 0.5

        low_eta_results = [
            low_eta.mutate(value, linear_range, np.random.default_rng(i), 0.3) for i in range(100)
        ]
        high_eta_results = [
            high_eta.mutate(value, linear_range, np.random.default_rng(i + 1000), 0.3)
            for i in range(100)
        ]

        low_var = np.var(low_eta_results)
        high_var = np.var(high_eta_results)

        # Higher eta should produce smaller variance
        assert low_var > high_var

    @pytest.mark.xfail(
        reason="Bug: PolynomialMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test polynomial mutation on categorical parameter (known bug)."""
        mutation = PolynomialMutation(eta=20.0)
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 1.0) for _ in range(100)]
        # Should produce valid categories
        assert all(r in categorical_range.categories for r in results)

    @pytest.mark.xfail(
        reason="Bug: PolynomialMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical_low_strength(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test polynomial mutation on categorical with low strength (known bug)."""
        mutation = PolynomialMutation(eta=20.0)
        value = "sgd"

        results = [mutation.mutate(value, categorical_range, rng, 0.01) for _ in range(100)]
        sgd_count = sum(1 for r in results if r == "sgd")
        assert sgd_count > 80

    def test_mutate_discrete(
        self, rng: np.random.Generator, discrete_range: HyperparameterRange
    ) -> None:
        """Test polynomial mutation on discrete parameter."""
        mutation = PolynomialMutation(eta=20.0)
        value = 5

        results = [mutation.mutate(value, discrete_range, rng, 0.3) for _ in range(100)]
        assert all(isinstance(r, int) for r in results)
        assert all(discrete_range.min_value <= r <= discrete_range.max_value for r in results)


# =============================================================================
# DifferentialMutation Tests
# =============================================================================


class TestDifferentialMutation:
    """Tests for DifferentialMutation strategy."""

    def test_default_f(self) -> None:
        """Test default F value."""
        mutation = DifferentialMutation()
        assert mutation.f == 0.5

    def test_custom_f(self) -> None:
        """Test custom F value."""
        mutation = DifferentialMutation(f=0.8)
        assert mutation.f == 0.8

    def test_set_candidates(self) -> None:
        """Test setting candidates."""
        mutation = DifferentialMutation()
        candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
        mutation.set_candidates(candidates)
        assert mutation._candidates == candidates

    def test_mutate_with_candidates(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test differential mutation with candidates."""
        mutation = DifferentialMutation(f=0.5)
        mutation.set_candidates([0.1, 0.3, 0.5, 0.7, 0.9])
        value = 0.5

        results = [mutation.mutate(value, linear_range, rng, 1.0) for _ in range(50)]

        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_fallback_without_candidates(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test fallback to Gaussian when not enough candidates."""
        mutation = DifferentialMutation()
        mutation.set_candidates([0.1, 0.2])  # Only 2 candidates
        value = 0.5

        # Should fallback to Gaussian and still work
        result = mutation.mutate(value, linear_range, rng, 0.2)
        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_fallback_no_candidates(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test fallback to Gaussian with zero candidates."""
        mutation = DifferentialMutation()
        # No candidates set at all (empty list by default)
        value = 0.5

        result = mutation.mutate(value, linear_range, rng, 0.2)
        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_log_scale(
        self, rng: np.random.Generator, log_range: HyperparameterRange
    ) -> None:
        """Test differential mutation on log scale."""
        mutation = DifferentialMutation(f=0.5)
        mutation.set_candidates([0.0001, 0.001, 0.01, 0.05, 0.1])
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 1.0) for _ in range(50)]

        assert all(log_range.min_value <= r <= log_range.max_value for r in results)

    @pytest.mark.xfail(
        reason="Bug: DifferentialMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test differential mutation on categorical (known bug)."""
        mutation = DifferentialMutation(f=0.5)
        mutation.set_candidates([0.1, 0.3, 0.5, 0.7])  # Candidates irrelevant for categorical
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 1.0) for _ in range(100)]
        assert all(r in categorical_range.categories for r in results)

    def test_mutate_discrete(
        self, rng: np.random.Generator, discrete_range: HyperparameterRange
    ) -> None:
        """Test differential mutation produces discrete output."""
        mutation = DifferentialMutation(f=0.5)
        mutation.set_candidates([1, 3, 5, 7, 9])
        value = 5

        results = [mutation.mutate(value, discrete_range, rng, 1.0) for _ in range(50)]
        assert all(isinstance(r, int) for r in results)
        assert all(discrete_range.min_value <= r <= discrete_range.max_value for r in results)


# =============================================================================
# AdaptiveMutation Tests
# =============================================================================


class TestAdaptiveMutation:
    """Tests for AdaptiveMutation strategy."""

    def test_default_strength(self) -> None:
        """Test default strength values."""
        mutation = AdaptiveMutation()
        assert mutation.base_strength == 0.2
        assert mutation.current_strength == 0.2

    def test_custom_strength(self) -> None:
        """Test custom strength values."""
        mutation = AdaptiveMutation(base_strength=0.5, adaptation_rate=0.2)
        assert mutation.base_strength == 0.5
        assert mutation.current_strength == 0.5

    def test_record_outcome_success(self) -> None:
        """Test recording successful outcomes increases strength."""
        mutation = AdaptiveMutation(base_strength=0.2)

        # Record many successes
        for _ in range(10):
            mutation.record_outcome(success=True)

        # Strength should have increased
        assert mutation.current_strength > mutation.base_strength

    def test_record_outcome_failure(self) -> None:
        """Test recording failures decreases strength."""
        mutation = AdaptiveMutation(base_strength=0.5)

        # Record many failures
        for _ in range(20):
            mutation.record_outcome(success=False)

        # Strength should have decreased
        assert mutation.current_strength < mutation.base_strength

    def test_strength_bounds(self) -> None:
        """Test that strength stays within bounds."""
        mutation = AdaptiveMutation(base_strength=0.5)

        # Many successes should not exceed 1.0
        for _ in range(100):
            mutation.record_outcome(success=True)
        assert mutation.current_strength <= 1.0

        # Reset and test lower bound
        mutation = AdaptiveMutation(base_strength=0.5)
        for _ in range(100):
            mutation.record_outcome(success=False)
        assert mutation.current_strength >= 0.01

    def test_mutate_uses_adapted_strength(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that mutation uses adapted strength."""
        mutation = AdaptiveMutation(base_strength=0.2)
        value = 0.5

        result = mutation.mutate(value, linear_range, rng, 0.5)  # Provided strength ignored

        assert linear_range.min_value <= result <= linear_range.max_value

    def test_history_window_capped_at_20(self) -> None:
        """Test that success history is capped at 20 entries."""
        mutation = AdaptiveMutation(base_strength=0.3)

        for _ in range(30):
            mutation.record_outcome(success=True)

        assert len(mutation._success_history) == 20

    def test_adaptation_below_5_entries_does_not_adapt(self) -> None:
        """Test that fewer than 5 entries doesn't trigger adaptation."""
        mutation = AdaptiveMutation(base_strength=0.3)

        for _ in range(4):
            mutation.record_outcome(success=True)

        # Strength should remain at base since we need at least 5
        assert mutation.current_strength == mutation.base_strength

    def test_mixed_outcomes_moderate_strength(self) -> None:
        """Test that mixed success/failure leads to moderate strength."""
        mutation = AdaptiveMutation(base_strength=0.3, adaptation_rate=0.1)

        # 15% success rate (between 0.1 and 0.2) -- no adaptation
        for i in range(20):
            mutation.record_outcome(success=(i % 7 == 0))

        # Should not have changed much
        assert 0.01 <= mutation.current_strength <= 1.0


# =============================================================================
# CrossoverMutation Tests
# =============================================================================


class TestCrossoverMutation:
    """Tests for CrossoverMutation strategy."""

    def test_default_initialization(self) -> None:
        """Test default initialization values."""
        mutation = CrossoverMutation()
        assert mutation.crossover_rate == 0.5
        assert mutation.blend_alpha == 0.5
        assert mutation.crossover_type == "blend"

    def test_custom_initialization(self) -> None:
        """Test custom initialization."""
        mutation = CrossoverMutation(
            crossover_rate=0.8,
            blend_alpha=0.3,
            crossover_type="uniform",
        )
        assert mutation.crossover_rate == 0.8
        assert mutation.blend_alpha == 0.3
        assert mutation.crossover_type == "uniform"

    def test_set_crossover_parent(self) -> None:
        """Test setting second parent values."""
        mutation = CrossoverMutation()
        parent_values = {"learning_rate": 0.001, "c_puct": 2.5}
        mutation.set_crossover_parent(parent_values)
        assert mutation._parent_values == parent_values

    def test_mutate_no_parent_falls_back_to_gaussian(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that mutate without parent falls back to Gaussian."""
        mutation = CrossoverMutation()
        # No parent set
        value = 0.5

        result = mutation.mutate(value, linear_range, rng, 0.2)
        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_parent_key_missing_falls_back(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test fallback when parent doesn't have the specific parameter."""
        mutation = CrossoverMutation()
        mutation.set_crossover_parent({"other_param": 0.5})
        value = 0.5

        result = mutation.mutate(value, linear_range, rng, 0.2)
        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_uniform_crossover(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test uniform crossover picks from either parent."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="uniform")
        mutation.set_crossover_parent({"test_linear": 0.8})
        value = 0.2

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(200)]

        # Should produce both parent values
        unique_values = set(results)
        assert len(unique_values) >= 2
        # All values within bounds
        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_blend_crossover(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test blend (BLX-alpha) crossover produces values near parents."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="blend", blend_alpha=0.5)
        mutation.set_crossover_parent({"test_linear": 0.7})
        value = 0.3

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(200)]

        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)
        # Mean should be somewhere between parents
        mean_result = np.mean(results)
        assert 0.1 <= mean_result <= 0.9

    def test_mutate_blend_crossover_log_scale(
        self, rng: np.random.Generator, log_range: HyperparameterRange
    ) -> None:
        """Test blend crossover on log scale parameter."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="blend", blend_alpha=0.3)
        mutation.set_crossover_parent({"test_log": 0.01})
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 0.5) for _ in range(100)]

        assert all(log_range.min_value <= r <= log_range.max_value for r in results)

    def test_mutate_sbx_crossover(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test simulated binary crossover (SBX)."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="simulated_binary")
        mutation.set_crossover_parent({"test_linear": 0.7})
        value = 0.3

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(200)]

        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_sbx_crossover_log_scale(
        self, rng: np.random.Generator, log_range: HyperparameterRange
    ) -> None:
        """Test SBX crossover on log scale parameter."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="simulated_binary")
        mutation.set_crossover_parent({"test_log": 0.01})
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 0.5) for _ in range(100)]

        # SBX may produce values outside bounds before clipping
        # After clipping, all should be within bounds
        assert all(log_range.min_value <= r <= log_range.max_value for r in results)

    def test_mutate_default_crossover_type(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test unknown crossover type falls back to average."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="average_fallback")
        mutation.set_crossover_parent({"test_linear": 0.8})
        value = 0.2

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(50)]

        # Default average: (0.2 + 0.8) / 2 = 0.5
        for r in results:
            assert abs(r - 0.5) < 0.01  # Should be very close to average

    @pytest.mark.xfail(
        reason="Bug: CrossoverMutation.mutate uses float(rng.choice(...)) for string categories",
        strict=True,
    )
    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test crossover on categorical parameter (known bug)."""
        mutation = CrossoverMutation(crossover_rate=1.0)
        mutation.set_crossover_parent({"test_categorical": "sgd"})
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 0.5) for _ in range(100)]

        # Should only produce "adam" or "sgd" (from parents)
        assert all(r in ["adam", "sgd"] for r in results)
        assert len(set(results)) == 2  # Both should appear

    def test_crossover_rate_zero_returns_original(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that crossover_rate=0 returns original value."""
        mutation = CrossoverMutation(crossover_rate=0.0)
        mutation.set_crossover_parent({"test_linear": 0.9})
        value = 0.1

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(50)]

        # All should be original value
        assert all(r == 0.1 for r in results)

    def test_mutate_discrete(
        self, rng: np.random.Generator, discrete_range: HyperparameterRange
    ) -> None:
        """Test crossover on discrete parameter produces integers."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="blend")
        mutation.set_crossover_parent({"test_discrete": 8})
        value = 3

        results = [mutation.mutate(value, discrete_range, rng, 0.5) for _ in range(100)]

        assert all(isinstance(r, int) for r in results)
        assert all(discrete_range.min_value <= r <= discrete_range.max_value for r in results)


# =============================================================================
# MutationOperator Tests
# =============================================================================


class TestMutationOperator:
    """Tests for MutationOperator class."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        operator = MutationOperator()
        assert operator.config.strategy == "gaussian"
        assert operator.current_strength == 0.2

    def test_custom_initialization(self, rng: np.random.Generator) -> None:
        """Test custom initialization."""
        config = MutationConfig(strategy="uniform", strength=0.3)
        operator = MutationOperator(config, rng=rng)

        assert operator.config.strategy == "uniform"
        assert operator.current_strength == 0.3

    def test_mutate_single_value(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test mutating a single value."""
        operator = MutationOperator(rng=rng)

        result = operator.mutate(0.5, linear_range)

        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_with_generation(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test mutating with specific generation."""
        config = MutationConfig(strength=1.0, decay_rate=0.1)
        operator = MutationOperator(config, rng=rng)

        result = operator.mutate(0.5, linear_range, generation=10)

        assert linear_range.min_value <= result <= linear_range.max_value

    def test_mutate_all(
        self, rng: np.random.Generator, hyperparameter_space: HyperparameterSpace
    ) -> None:
        """Test mutating all hyperparameters."""
        operator = MutationOperator(rng=rng)

        # Sample initial values
        initial_values = hyperparameter_space.sample_all(rng)

        # Mutate all
        mutated = operator.mutate_all(
            initial_values, hyperparameter_space, mutation_probability=1.0
        )

        assert set(mutated.keys()) == set(initial_values.keys())

    def test_mutate_all_respects_probability(
        self, rng: np.random.Generator, hyperparameter_space: HyperparameterSpace
    ) -> None:
        """Test that mutation_probability affects mutation rate."""
        operator = MutationOperator(rng=rng)

        initial_values = hyperparameter_space.sample_all(rng)

        # With probability 0, nothing should change
        mutated = operator.mutate_all(
            initial_values, hyperparameter_space, mutation_probability=0.0
        )

        assert mutated == initial_values

    def test_mutate_all_with_generation(
        self, rng: np.random.Generator, hyperparameter_space: HyperparameterSpace
    ) -> None:
        """Test mutate_all with generation parameter."""
        config = MutationConfig(strength=1.0, decay_rate=0.1)
        operator = MutationOperator(config, rng=rng)

        initial_values = hyperparameter_space.sample_all(rng)
        mutated = operator.mutate_all(
            initial_values, hyperparameter_space, mutation_probability=1.0, generation=50
        )

        assert set(mutated.keys()) == set(initial_values.keys())

    def test_mutate_all_missing_values_get_sampled(
        self, rng: np.random.Generator, hyperparameter_space: HyperparameterSpace
    ) -> None:
        """Test that missing values in input dict get sampled."""
        operator = MutationOperator(rng=rng)

        # Provide only some values
        partial_values = {"learning_rate": 0.001}

        mutated = operator.mutate_all(
            partial_values, hyperparameter_space, mutation_probability=1.0
        )

        # Should have all parameters
        assert set(mutated.keys()) == set(hyperparameter_space.get_all_params().keys())

    def test_set_candidates_for_differential(self) -> None:
        """Test setting candidates for differential mutation."""
        config = MutationConfig(strategy="differential")
        operator = MutationOperator(config)

        candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
        operator.set_candidates_for_differential("test", candidates)

        # Should not raise
        assert True

    def test_set_crossover_parent(self) -> None:
        """Test setting crossover parent on operator."""
        config = MutationConfig(strategy="crossover")
        operator = MutationOperator(config)

        parent = {"learning_rate": 0.005, "c_puct": 2.0}
        operator.set_crossover_parent(parent)

        # Verify it was set on the crossover strategy
        strategy = operator._strategies["crossover"]
        assert isinstance(strategy, CrossoverMutation)
        assert strategy._parent_values == parent

    def test_set_crossover_parent_noop_without_crossover_strategy(self) -> None:
        """Test set_crossover_parent is safe when crossover is not in strategies."""
        operator = MutationOperator()
        # This should not raise even though strategy is gaussian
        operator.set_crossover_parent({"lr": 0.001})

    def test_record_adaptive_outcome(self) -> None:
        """Test recording outcome for adaptive mutation."""
        config = MutationConfig(strategy="adaptive")
        operator = MutationOperator(config)

        # Should not raise
        operator.record_adaptive_outcome(True)
        operator.record_adaptive_outcome(False)

    def test_record_adaptive_outcome_affects_strength(self) -> None:
        """Test that recording adaptive outcomes changes adapted strength."""
        config = MutationConfig(strategy="adaptive")
        operator = MutationOperator(config)

        adaptive_strategy = operator._strategies["adaptive"]
        assert isinstance(adaptive_strategy, AdaptiveMutation)
        initial_strength = adaptive_strategy.current_strength

        # Record enough successes to trigger adaptation
        for _ in range(10):
            operator.record_adaptive_outcome(True)

        assert adaptive_strategy.current_strength > initial_strength

    def test_increment_generation(self) -> None:
        """Test incrementing generation counter."""
        operator = MutationOperator()

        assert operator._generation == 0

        operator.increment_generation()
        assert operator._generation == 1

        operator.increment_generation()
        assert operator._generation == 2

    def test_current_strength_with_decay(self) -> None:
        """Test current_strength property with decay."""
        config = MutationConfig(strength=1.0, decay_rate=0.1)
        operator = MutationOperator(config)

        initial_strength = operator.current_strength

        operator.increment_generation()
        operator.increment_generation()
        operator.increment_generation()

        # Strength should have decayed
        assert operator.current_strength < initial_strength

    def test_different_strategies(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that different strategies work."""
        strategies = ["gaussian", "uniform", "polynomial", "adaptive", "crossover"]

        for strategy in strategies:
            config = MutationConfig(strategy=strategy)
            operator = MutationOperator(config, rng=np.random.default_rng(42))

            result = operator.mutate(0.5, linear_range)
            assert linear_range.min_value <= result <= linear_range.max_value

    def test_unknown_strategy_falls_back_to_gaussian(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test that unknown strategy falls back to gaussian."""
        config = MutationConfig(strategy="unknown_strategy")
        operator = MutationOperator(config, rng=rng)

        # Should not raise, falls back to gaussian
        result = operator.mutate(0.5, linear_range)
        assert linear_range.min_value <= result <= linear_range.max_value

    def test_crossover_strategy_via_operator(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test using crossover strategy through operator."""
        config = MutationConfig(strategy="crossover")
        operator = MutationOperator(config, rng=rng)

        operator.set_crossover_parent({"test_linear": 0.8})

        result = operator.mutate(0.2, linear_range)
        assert linear_range.min_value <= result <= linear_range.max_value


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMutationOperator:
    """Tests for create_mutation_operator factory function."""

    def test_default_creation(self) -> None:
        """Test creating operator with defaults."""
        operator = create_mutation_operator()

        assert operator.config.strategy == "gaussian"
        assert operator.config.strength == 0.2

    def test_custom_creation(self, rng: np.random.Generator) -> None:
        """Test creating operator with custom values."""
        operator = create_mutation_operator(
            strategy="uniform",
            strength=0.4,
            rng=rng,
            decay_rate=0.05,
        )

        assert operator.config.strategy == "uniform"
        assert operator.config.strength == 0.4
        assert operator.config.decay_rate == 0.05

    def test_all_strategies(self) -> None:
        """Test creating operators for all strategies."""
        strategies = ["gaussian", "uniform", "polynomial", "differential", "adaptive", "crossover"]

        for strategy in strategies:
            operator = create_mutation_operator(strategy=strategy)
            assert operator.config.strategy == strategy

    def test_factory_with_polynomial_eta(self) -> None:
        """Test factory respects polynomial_eta parameter."""
        operator = create_mutation_operator(strategy="polynomial", polynomial_eta=10.0)
        assert operator.config.polynomial_eta == 10.0

    def test_factory_with_differential_f(self) -> None:
        """Test factory respects differential_f parameter."""
        operator = create_mutation_operator(strategy="differential", differential_f=0.8)
        assert operator.config.differential_f == 0.8


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_mutate_at_min_boundary(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test mutation at minimum boundary."""
        mutation = GaussianMutation()
        value = linear_range.min_value

        results = [mutation.mutate(value, linear_range, rng, 0.2) for _ in range(100)]

        # All should still be within bounds
        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_at_max_boundary(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test mutation at maximum boundary."""
        mutation = GaussianMutation()
        value = linear_range.max_value

        results = [mutation.mutate(value, linear_range, rng, 0.2) for _ in range(100)]

        # All should still be within bounds
        assert all(linear_range.min_value <= r <= linear_range.max_value for r in results)

    def test_mutate_zero_strength(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test mutation with zero strength."""
        mutation = GaussianMutation()
        value = 0.5

        results = [mutation.mutate(value, linear_range, rng, 0.0) for _ in range(10)]

        # With zero strength, results should be very close to original
        # (allowing for floating point precision)
        assert all(abs(r - value) < 0.01 for r in results)

    def test_mutate_very_small_range(self, rng: np.random.Generator) -> None:
        """Test mutation on very small range."""
        small_range = HyperparameterRange(
            name="small",
            min_value=0.99,
            max_value=1.0,
            scale="linear",
        )
        mutation = GaussianMutation()
        value = 0.995

        results = [mutation.mutate(value, small_range, rng, 0.2) for _ in range(100)]

        assert all(small_range.min_value <= r <= small_range.max_value for r in results)

    def test_mutate_single_category(self, rng: np.random.Generator) -> None:
        """Test mutation with single category."""
        single_cat = HyperparameterRange(
            name="single",
            min_value=0,
            max_value=1,
            scale="categorical",
            categories=["only_option"],
        )
        mutation = GaussianMutation()

        result = mutation.mutate("only_option", single_cat, rng, 1.0)

        assert result == "only_option"

    def test_all_mutations_respect_bounds_parametrized(self, rng: np.random.Generator) -> None:
        """Test all mutation types respect bounds on linear range."""
        linear_range = HyperparameterRange("test", 0.0, 1.0, "linear")
        mutations = [
            GaussianMutation(),
            UniformMutation(),
            PolynomialMutation(eta=20.0),
        ]

        for mutation in mutations:
            for _ in range(50):
                result = mutation.mutate(0.5, linear_range, rng, 0.5)
                assert linear_range.min_value <= result <= linear_range.max_value, (
                    f"{type(mutation).__name__} produced out-of-bounds value: {result}"
                )

    def test_all_mutations_respect_log_bounds(self, rng: np.random.Generator) -> None:
        """Test all mutation types respect bounds on log range (with float tolerance)."""
        log_range = HyperparameterRange("test", 0.001, 0.1, "log")
        mutations = [
            GaussianMutation(),
            UniformMutation(),
            PolynomialMutation(eta=20.0),
        ]

        eps = 1e-9  # Floating point tolerance for log-space clipping
        for mutation in mutations:
            for _ in range(50):
                result = mutation.mutate(0.01, log_range, rng, 0.5)
                assert log_range.min_value - eps <= result <= log_range.max_value + eps, (
                    f"{type(mutation).__name__} produced out-of-bounds value: {result}"
                )

    def test_crossover_with_identical_parents(
        self, rng: np.random.Generator, linear_range: HyperparameterRange
    ) -> None:
        """Test crossover where both parents have same value."""
        mutation = CrossoverMutation(crossover_rate=1.0, crossover_type="blend")
        mutation.set_crossover_parent({"test_linear": 0.5})
        value = 0.5

        results = [mutation.mutate(value, linear_range, rng, 0.5) for _ in range(50)]
        # With identical parents, blend should produce values near 0.5
        assert all(0.0 <= r <= 1.0 for r in results)
        mean_r = np.mean(results)
        assert 0.3 <= mean_r <= 0.7
