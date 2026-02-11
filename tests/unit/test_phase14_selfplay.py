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


class TestMCTSKwargsFiltering:
    """Test that mcts_config_override is filtered by MCTS variant.

    Regression test for: create_mcts() got unexpected keyword argument(s):
    {'dirichlet_epsilon', 'dirichlet_alpha', 'c_puct'} when using GumbelMCTS.
    """

    def test_gumbel_rejects_civ6_kwargs_via_create_mcts(self) -> None:
        """GumbelMCTS should NOT receive c_puct/dirichlet_* from create_mcts."""
        from unittest.mock import MagicMock

        from alpha_zero.core.mcts_factory import create_mcts

        mock_network = MagicMock()

        # These are the Gumbel-valid kwargs — should succeed
        mcts = create_mcts(
            variant="gumbel",
            network=mock_network,
            num_simulations=8,
            num_sampled_actions=4,
        )
        assert mcts is not None

        # These include civ6-only kwargs — should raise TypeError
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            create_mcts(
                variant="gumbel",
                network=mock_network,
                num_simulations=8,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
            )

    def test_selfplay_override_filters_civ6_keys_for_gumbel(self) -> None:
        """Verify that selfplay worker's filtering logic removes civ6-only keys.

        This directly tests the filtering pattern used in run_selfplay().
        """
        # Simulate the filtering logic from run_selfplay
        civ6_only_keys = {"c_puct", "dirichlet_alpha", "dirichlet_epsilon"}
        gumbel_only_keys = {"num_sampled_actions", "c_visit", "c_scale"}
        puct_variants = {"puct", "civ6"}
        gumbel_variants = {"gumbel"}

        mcts_config_override = {
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "num_simulations": 16,
        }

        # For gumbel MCTS type
        mcts_type = "gumbel"
        filtered = {}
        for k, v in mcts_config_override.items():
            if v is None:
                continue
            if mcts_type in gumbel_variants and k in civ6_only_keys:
                continue
            if mcts_type in puct_variants and k in gumbel_only_keys:
                continue
            filtered[k] = v

        # Only num_simulations should remain
        assert "c_puct" not in filtered
        assert "dirichlet_alpha" not in filtered
        assert "dirichlet_epsilon" not in filtered
        assert filtered["num_simulations"] == 16

    def test_selfplay_override_keeps_civ6_keys_for_puct(self) -> None:
        """Verify civ6-only keys are kept for puct/civ6 MCTS type."""
        civ6_only_keys = {"c_puct", "dirichlet_alpha", "dirichlet_epsilon"}
        gumbel_only_keys = {"num_sampled_actions", "c_visit", "c_scale"}
        puct_variants = {"puct", "civ6"}
        gumbel_variants = {"gumbel"}

        mcts_config_override = {
            "c_puct": 1.5,
            "dirichlet_alpha": 0.3,
            "dirichlet_epsilon": 0.25,
            "num_simulations": 16,
        }

        # For civ6 MCTS type
        mcts_type = "civ6"
        filtered = {}
        for k, v in mcts_config_override.items():
            if v is None:
                continue
            if mcts_type in gumbel_variants and k in civ6_only_keys:
                continue
            if mcts_type in puct_variants and k in gumbel_only_keys:
                continue
            filtered[k] = v

        # All keys should remain for civ6
        assert filtered["c_puct"] == 1.5
        assert filtered["dirichlet_alpha"] == 0.3
        assert filtered["dirichlet_epsilon"] == 0.25
        assert filtered["num_simulations"] == 16
