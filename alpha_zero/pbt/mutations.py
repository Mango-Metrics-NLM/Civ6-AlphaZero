"""Mutation operators for Population-Based Training.

This module provides various mutation strategies for hyperparameter
exploration during PBT.

Usage:
    from alpha_zero.pbt.mutations import MutationOperator, GaussianMutation

    mutator = MutationOperator(
        strategy=GaussianMutation(strength=0.2)
    )
    new_value = mutator.mutate(current_value, param_range)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from alpha_zero.pbt.config import HyperparameterRange, HyperparameterSpace


class MutationStrategy(str, Enum):
    """Available mutation strategies."""

    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    POLYNOMIAL = "polynomial"
    DIFFERENTIAL = "differential"
    ADAPTIVE = "adaptive"


@dataclass
class MutationConfig:
    """Configuration for mutation operations.

    Attributes:
        strategy: Mutation strategy to use
        strength: Base mutation strength (0-1)
        decay_rate: Rate at which strength decays over generations
        min_strength: Minimum mutation strength after decay
        adaptive_window: Window size for adaptive mutation
        differential_f: Scaling factor for differential mutation
        polynomial_eta: Distribution index for polynomial mutation
    """

    strategy: str = "gaussian"
    strength: float = 0.2
    decay_rate: float = 0.0
    min_strength: float = 0.05
    adaptive_window: int = 10
    differential_f: float = 0.5
    polynomial_eta: float = 20.0

    def get_decayed_strength(self, generation: int) -> float:
        """Get mutation strength after decay.

        Args:
            generation: Current generation number

        Returns:
            Decayed mutation strength.
        """
        if self.decay_rate <= 0:
            return self.strength

        decayed = self.strength * math.exp(-self.decay_rate * generation)
        return max(self.min_strength, decayed)


class BaseMutation(ABC):
    """Abstract base class for mutation strategies."""

    @abstractmethod
    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Mutate a single value.

        Args:
            value: Current value to mutate
            param_range: Valid range for the parameter
            rng: Random number generator
            strength: Mutation strength

        Returns:
            Mutated value within valid bounds.
        """
        pass


class GaussianMutation(BaseMutation):
    """Gaussian (normal) mutation strategy.

    Perturbs values by adding Gaussian noise scaled by the parameter range.
    """

    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Apply Gaussian mutation."""
        if param_range.scale == "categorical":
            return param_range.mutate(value, rng, strength)

        if param_range.scale == "log":
            # Mutate in log space
            log_value = math.log(max(value, 1e-10))
            log_min = math.log(param_range.min_value)
            log_max = math.log(param_range.max_value)
            log_range = log_max - log_min

            perturbation = rng.normal(0, strength * log_range)
            new_log_value = log_value + perturbation
            new_value = math.exp(np.clip(new_log_value, log_min, log_max))
        else:
            value_range = param_range.max_value - param_range.min_value
            perturbation = rng.normal(0, strength * value_range)
            new_value = np.clip(
                value + perturbation, param_range.min_value, param_range.max_value
            )

        if param_range.discrete:
            new_value = int(round(new_value))

        return new_value


class UniformMutation(BaseMutation):
    """Uniform random mutation strategy.

    Perturbs values by adding uniform noise within a specified range.
    """

    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Apply uniform mutation."""
        if param_range.scale == "categorical":
            if rng.random() < strength:
                return rng.choice(param_range.categories)
            return value

        if param_range.scale == "log":
            log_value = math.log(max(value, 1e-10))
            log_min = math.log(param_range.min_value)
            log_max = math.log(param_range.max_value)
            log_range = log_max - log_min

            delta = strength * log_range
            new_log_value = rng.uniform(log_value - delta, log_value + delta)
            new_value = math.exp(np.clip(new_log_value, log_min, log_max))
        else:
            value_range = param_range.max_value - param_range.min_value
            delta = strength * value_range
            new_value = rng.uniform(value - delta, value + delta)
            new_value = np.clip(new_value, param_range.min_value, param_range.max_value)

        if param_range.discrete:
            new_value = int(round(new_value))

        return new_value


class PolynomialMutation(BaseMutation):
    """Polynomial mutation strategy.

    Uses polynomial distribution for mutation, common in evolutionary algorithms.
    """

    def __init__(self, eta: float = 20.0):
        """Initialize polynomial mutation.

        Args:
            eta: Distribution index (higher = smaller mutations)
        """
        self.eta = eta

    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Apply polynomial mutation."""
        if param_range.scale == "categorical":
            if rng.random() < strength:
                return rng.choice(param_range.categories)
            return value

        # Normalize value to [0, 1]
        if param_range.scale == "log":
            log_val = math.log(max(value, 1e-10))
            log_min = math.log(param_range.min_value)
            log_max = math.log(param_range.max_value)
            y = (log_val - log_min) / (log_max - log_min)
        else:
            y = (value - param_range.min_value) / (
                param_range.max_value - param_range.min_value
            )

        # Apply polynomial mutation
        u = rng.random()
        if u < 0.5:
            delta = (2 * u) ** (1 / (self.eta + 1)) - 1
        else:
            delta = 1 - (2 * (1 - u)) ** (1 / (self.eta + 1))

        # Scale by strength
        delta *= strength

        # Apply mutation
        y_new = np.clip(y + delta, 0, 1)

        # Denormalize
        if param_range.scale == "log":
            new_value = math.exp(log_min + y_new * (log_max - log_min))
        else:
            new_value = param_range.min_value + y_new * (
                param_range.max_value - param_range.min_value
            )

        if param_range.discrete:
            new_value = int(round(new_value))

        return new_value


class DifferentialMutation(BaseMutation):
    """Differential evolution mutation strategy.

    Requires multiple candidate values to compute mutation direction.
    """

    def __init__(self, f: float = 0.5):
        """Initialize differential mutation.

        Args:
            f: Scaling factor for difference vector
        """
        self.f = f
        self._candidates: list[float | int] = []

    def set_candidates(self, candidates: list[float | int]) -> None:
        """Set candidate values for differential mutation.

        Args:
            candidates: List of candidate values from population
        """
        self._candidates = candidates

    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Apply differential mutation."""
        if param_range.scale == "categorical":
            if rng.random() < strength:
                return rng.choice(param_range.categories)
            return value

        if len(self._candidates) < 3:
            # Fall back to Gaussian mutation
            return GaussianMutation().mutate(value, param_range, rng, strength)

        # Select three distinct candidates
        indices = rng.choice(len(self._candidates), size=3, replace=False)
        a, b, c = [self._candidates[i] for i in indices]

        # Differential mutation: a + F * (b - c)
        if param_range.scale == "log":
            log_a = math.log(max(a, 1e-10))
            log_b = math.log(max(b, 1e-10))
            log_c = math.log(max(c, 1e-10))
            log_new = log_a + self.f * strength * (log_b - log_c)
            new_value = math.exp(log_new)
        else:
            new_value = a + self.f * strength * (b - c)

        new_value = np.clip(new_value, param_range.min_value, param_range.max_value)

        if param_range.discrete:
            new_value = int(round(new_value))

        return new_value


class AdaptiveMutation(BaseMutation):
    """Adaptive mutation strategy.

    Adjusts mutation strength based on fitness improvement history.
    """

    def __init__(self, base_strength: float = 0.2, adaptation_rate: float = 0.1):
        """Initialize adaptive mutation.

        Args:
            base_strength: Initial mutation strength
            adaptation_rate: Rate of strength adaptation
        """
        self.base_strength = base_strength
        self.adaptation_rate = adaptation_rate
        self._success_history: list[bool] = []
        self._adapted_strength: float = base_strength

    def record_outcome(self, success: bool) -> None:
        """Record whether a mutation was successful.

        Args:
            success: Whether the mutation improved fitness
        """
        self._success_history.append(success)
        if len(self._success_history) > 20:
            self._success_history = self._success_history[-20:]

        # Adapt strength based on success rate
        if len(self._success_history) >= 5:
            success_rate = sum(self._success_history) / len(self._success_history)
            if success_rate > 0.2:
                # Successful mutations, increase strength
                self._adapted_strength = min(
                    1.0, self._adapted_strength * (1 + self.adaptation_rate)
                )
            elif success_rate < 0.1:
                # Few successful mutations, decrease strength
                self._adapted_strength = max(
                    0.01, self._adapted_strength * (1 - self.adaptation_rate)
                )

    def mutate(
        self,
        value: float | int,
        param_range: HyperparameterRange,
        rng: np.random.Generator,
        strength: float = 0.2,
    ) -> float | int:
        """Apply adaptive mutation using adapted strength."""
        # Use adapted strength instead of provided strength
        effective_strength = self._adapted_strength

        # Use Gaussian mutation with adapted strength
        return GaussianMutation().mutate(value, param_range, rng, effective_strength)

    @property
    def current_strength(self) -> float:
        """Get current adapted mutation strength."""
        return self._adapted_strength


class MutationOperator:
    """Main mutation operator combining strategy and configuration.

    Usage:
        >>> config = MutationConfig(strategy="gaussian", strength=0.2)
        >>> operator = MutationOperator(config)
        >>> new_value = operator.mutate(0.001, learning_rate_range, rng)
    """

    def __init__(
        self,
        config: MutationConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        """Initialize mutation operator.

        Args:
            config: Mutation configuration
            rng: Random number generator
        """
        self.config = config or MutationConfig()
        self.rng = rng or np.random.default_rng()

        self._strategies: dict[str, BaseMutation] = {
            "gaussian": GaussianMutation(),
            "uniform": UniformMutation(),
            "polynomial": PolynomialMutation(self.config.polynomial_eta),
            "differential": DifferentialMutation(self.config.differential_f),
            "adaptive": AdaptiveMutation(self.config.strength, adaptation_rate=0.1),
        }

        self._generation: int = 0

    def mutate(
        self,
        value: float | int | Any,
        param_range: HyperparameterRange,
        generation: int | None = None,
    ) -> float | int | Any:
        """Mutate a single value.

        Args:
            value: Current value
            param_range: Valid range for parameter
            generation: Current generation (for strength decay)

        Returns:
            Mutated value.
        """
        gen = generation if generation is not None else self._generation
        strength = self.config.get_decayed_strength(gen)

        strategy = self._strategies.get(self.config.strategy)
        if strategy is None:
            strategy = self._strategies["gaussian"]

        return strategy.mutate(value, param_range, self.rng, strength)

    def mutate_all(
        self,
        values: dict[str, float | int | Any],
        hyperparameter_space: HyperparameterSpace,
        mutation_probability: float = 0.8,
        generation: int | None = None,
    ) -> dict[str, float | int | Any]:
        """Mutate all hyperparameters.

        Args:
            values: Current hyperparameter values
            hyperparameter_space: Space defining valid ranges
            mutation_probability: Probability of mutating each parameter
            generation: Current generation

        Returns:
            Dictionary of mutated values.
        """
        gen = generation if generation is not None else self._generation
        result = {}

        for name, param_range in hyperparameter_space.get_all_params().items():
            current_value = values.get(name, param_range.sample(self.rng))

            if self.rng.random() < mutation_probability:
                result[name] = self.mutate(current_value, param_range, gen)
            else:
                result[name] = current_value

        return result

    def set_candidates_for_differential(
        self, param_name: str, candidates: list[float | int]
    ) -> None:
        """Set candidates for differential mutation.

        Args:
            param_name: Parameter name
            candidates: List of candidate values from population
        """
        if "differential" in self._strategies:
            strategy = self._strategies["differential"]
            if isinstance(strategy, DifferentialMutation):
                strategy.set_candidates(candidates)

    def record_adaptive_outcome(self, success: bool) -> None:
        """Record outcome for adaptive mutation.

        Args:
            success: Whether the mutation improved fitness
        """
        if "adaptive" in self._strategies:
            strategy = self._strategies["adaptive"]
            if isinstance(strategy, AdaptiveMutation):
                strategy.record_outcome(success)

    def increment_generation(self) -> None:
        """Increment the generation counter."""
        self._generation += 1

    @property
    def current_strength(self) -> float:
        """Get current effective mutation strength."""
        return self.config.get_decayed_strength(self._generation)


def create_mutation_operator(
    strategy: str = "gaussian",
    strength: float = 0.2,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> MutationOperator:
    """Factory function to create a mutation operator.

    Args:
        strategy: Mutation strategy name
        strength: Mutation strength
        rng: Random number generator
        **kwargs: Additional configuration parameters

    Returns:
        Configured MutationOperator.
    """
    config = MutationConfig(strategy=strategy, strength=strength, **kwargs)
    return MutationOperator(config, rng)
