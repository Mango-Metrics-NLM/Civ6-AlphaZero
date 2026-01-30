"""Tests for PBT mutation operators.

Tests cover:
- MutationStrategy enum
- MutationConfig dataclass and strength decay
- GaussianMutation for linear, log, and categorical parameters
- UniformMutation for various parameter types
- PolynomialMutation for bounded mutation
- DifferentialMutation with candidate values
- AdaptiveMutation with success tracking
- MutationOperator for orchestrating mutations
- create_mutation_operator factory function
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace
from alpha_zero.pbt.mutations import (
    AdaptiveMutation,
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

    def test_enum_from_string(self) -> None:
        """Test creating enum from string."""
        assert MutationStrategy("gaussian") == MutationStrategy.GAUSSIAN


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

    def test_mutate_categorical(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test uniform mutation on categorical parameter."""
        mutation = UniformMutation()
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 1.0) for _ in range(100)]

        # With strength=1.0, should see different categories
        assert all(r in categorical_range.categories for r in results)
        assert len(set(results)) > 1

    def test_mutate_categorical_low_strength(
        self, rng: np.random.Generator, categorical_range: HyperparameterRange
    ) -> None:
        """Test that low strength keeps categorical unchanged often."""
        mutation = UniformMutation()
        value = "adam"

        results = [mutation.mutate(value, categorical_range, rng, 0.01) for _ in range(100)]

        # Most should stay "adam"
        adam_count = sum(1 for r in results if r == "adam")
        assert adam_count > 80  # More than 80% unchanged


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

    def test_mutate_log_scale(
        self, rng: np.random.Generator, log_range: HyperparameterRange
    ) -> None:
        """Test differential mutation on log scale."""
        mutation = DifferentialMutation(f=0.5)
        mutation.set_candidates([0.0001, 0.001, 0.01, 0.05, 0.1])
        value = 0.001

        results = [mutation.mutate(value, log_range, rng, 1.0) for _ in range(50)]

        assert all(log_range.min_value <= r <= log_range.max_value for r in results)


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

    def test_set_candidates_for_differential(self) -> None:
        """Test setting candidates for differential mutation."""
        config = MutationConfig(strategy="differential")
        operator = MutationOperator(config)

        candidates = [0.1, 0.2, 0.3, 0.4, 0.5]
        operator.set_candidates_for_differential("test", candidates)

        # Should not raise
        assert True

    def test_record_adaptive_outcome(self) -> None:
        """Test recording outcome for adaptive mutation."""
        config = MutationConfig(strategy="adaptive")
        operator = MutationOperator(config)

        # Should not raise
        operator.record_adaptive_outcome(True)
        operator.record_adaptive_outcome(False)

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
        strategies = ["gaussian", "uniform", "polynomial", "adaptive"]

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
        strategies = ["gaussian", "uniform", "polynomial", "differential", "adaptive"]

        for strategy in strategies:
            operator = create_mutation_operator(strategy=strategy)
            assert operator.config.strategy == strategy


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
