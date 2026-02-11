"""
Unit tests for Phase 14 selfplay_worker.compute_temperature function.

Tests temperature scheduling logic for action selection during self-play:
- Threshold schedule: Step function at threshold turn
- Linear schedule: Linear decay from full temperature to 0
- Cosine schedule: Cosine annealing from full temperature to 0
- Edge cases: Unknown schedules, zero max_turns
"""

import math

import pytest

from alpha_zero.cloud.gcp.selfplay_worker import compute_temperature


class TestComputeTemperatureThreshold:
    """Test threshold schedule (step function at temperature_threshold)."""

    def test_threshold_before(self) -> None:
        """Temperature before threshold returns full temperature."""
        result = compute_temperature(
            turn=10,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="threshold",
        )
        assert result == 1.0

    def test_threshold_at(self) -> None:
        """Temperature at threshold returns 0.0."""
        result = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="threshold",
        )
        assert result == 0.0

    def test_threshold_after(self) -> None:
        """Temperature after threshold returns 0.0."""
        result = compute_temperature(
            turn=20,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="threshold",
        )
        assert result == 0.0


class TestComputeTemperatureLinear:
    """Test linear schedule (smooth decay from temperature to 0)."""

    def test_linear_start(self) -> None:
        """At turn 0, linear schedule returns full temperature."""
        result = compute_temperature(
            turn=0,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for linear
            schedule="linear",
        )
        assert result == pytest.approx(1.0)

    def test_linear_midpoint(self) -> None:
        """At max_turns/2, linear schedule returns ~temperature/2."""
        result = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for linear
            schedule="linear",
        )
        # progress = 15/30 = 0.5
        # temperature * (1.0 - 0.5) = 0.5
        assert result == pytest.approx(0.5)

    def test_linear_end(self) -> None:
        """At max_turns, linear schedule returns 0.0."""
        result = compute_temperature(
            turn=30,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for linear
            schedule="linear",
        )
        # progress = 30/30 = 1.0
        # temperature * (1.0 - 1.0) = 0.0
        assert result == pytest.approx(0.0)


class TestComputeTemperatureCosine:
    """Test cosine schedule (cosine annealing from temperature to 0)."""

    def test_cosine_start(self) -> None:
        """At turn 0, cosine schedule returns full temperature."""
        result = compute_temperature(
            turn=0,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for cosine
            schedule="cosine",
        )
        # progress = 0/30 = 0.0
        # temperature * 0.5 * (1.0 + cos(pi * 0.0)) = 1.0 * 0.5 * (1 + 1) = 1.0
        assert result == pytest.approx(1.0)

    def test_cosine_end(self) -> None:
        """At max_turns, cosine schedule returns ~0.0."""
        result = compute_temperature(
            turn=30,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for cosine
            schedule="cosine",
        )
        # progress = 30/30 = 1.0
        # temperature * 0.5 * (1.0 + cos(pi * 1.0)) = 1.0 * 0.5 * (1 + (-1)) = 0.0
        assert result == pytest.approx(0.0)

    def test_cosine_midpoint(self) -> None:
        """At max_turns/2, cosine schedule returns ~temperature/2."""
        result = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,  # Ignored for cosine
            schedule="cosine",
        )
        # progress = 15/30 = 0.5
        # temperature * 0.5 * (1.0 + cos(pi * 0.5))
        # = 1.0 * 0.5 * (1.0 + cos(pi/2))
        # = 1.0 * 0.5 * (1.0 + 0.0)
        # = 0.5
        assert result == pytest.approx(0.5)

    def test_cosine_quarter_point(self) -> None:
        """At max_turns/4, cosine schedule returns expected value."""
        result = compute_temperature(
            turn=15,
            max_turns=60,
            temperature=2.0,
            temperature_threshold=30,  # Ignored for cosine
            schedule="cosine",
        )
        # progress = 15/60 = 0.25
        # temperature * 0.5 * (1.0 + cos(pi * 0.25))
        # = 2.0 * 0.5 * (1.0 + cos(pi/4))
        # = 1.0 * (1.0 + 0.707...)
        # ≈ 1.707
        expected = 2.0 * 0.5 * (1.0 + math.cos(math.pi * 0.25))
        assert result == pytest.approx(expected)


class TestComputeTemperatureEdgeCases:
    """Test edge cases and unknown schedules."""

    def test_unknown_schedule(self) -> None:
        """Unknown schedule string returns temperature unchanged."""
        result = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=1.5,
            temperature_threshold=15,
            schedule="invalid_schedule",
        )
        # Should fallback to returning temperature
        assert result == 1.5

    def test_zero_max_turns_linear(self) -> None:
        """Linear schedule with max_turns=0 does not cause division by zero."""
        result = compute_temperature(
            turn=5,
            max_turns=0,
            temperature=1.0,
            temperature_threshold=15,
            schedule="linear",
        )
        # Implementation uses max(max_turns, 1), so:
        # progress = 5/1 = 5.0, clamped to 1.0
        # temperature * (1.0 - 1.0) = 0.0
        assert result == pytest.approx(0.0)

    def test_zero_max_turns_cosine(self) -> None:
        """Cosine schedule with max_turns=0 does not cause division by zero."""
        result = compute_temperature(
            turn=5,
            max_turns=0,
            temperature=1.0,
            temperature_threshold=15,
            schedule="cosine",
        )
        # Implementation uses max(max_turns, 1), so:
        # progress = 5/1 = 5.0, clamped to 1.0
        # temperature * 0.5 * (1.0 + cos(pi * 1.0)) = 0.0
        assert result == pytest.approx(0.0)

    def test_threshold_with_different_temperature(self) -> None:
        """Threshold schedule works with non-default temperature values."""
        result_before = compute_temperature(
            turn=5,
            max_turns=30,
            temperature=2.5,
            temperature_threshold=10,
            schedule="threshold",
        )
        assert result_before == 2.5

        result_after = compute_temperature(
            turn=15,
            max_turns=30,
            temperature=2.5,
            temperature_threshold=10,
            schedule="threshold",
        )
        assert result_after == 0.0

    def test_linear_beyond_max_turns(self) -> None:
        """Linear schedule clamps progress when turn > max_turns."""
        result = compute_temperature(
            turn=50,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="linear",
        )
        # progress = min(50/30, 1.0) = 1.0
        # temperature * (1.0 - 1.0) = 0.0
        assert result == pytest.approx(0.0)

    def test_cosine_beyond_max_turns(self) -> None:
        """Cosine schedule clamps progress when turn > max_turns."""
        result = compute_temperature(
            turn=50,
            max_turns=30,
            temperature=1.0,
            temperature_threshold=15,
            schedule="cosine",
        )
        # progress = min(50/30, 1.0) = 1.0
        # temperature * 0.5 * (1.0 + cos(pi * 1.0)) = 0.0
        assert result == pytest.approx(0.0)
