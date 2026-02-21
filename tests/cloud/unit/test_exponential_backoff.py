"""Unit tests for exponential backoff with jitter in vertex_orchestrator.

Tests the ``compute_retry_delay()`` utility function and the new CLI
arguments ``--initial_retry_delay`` and ``--max_retry_delay``.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from alpha_zero.cloud.gcp.vertex_orchestrator import (
    _BACKOFF_MULTIPLIER,
    _INITIAL_RETRY_DELAY_S,
    _JITTER_FRACTION,
    _MAX_RETRY_DELAY_S,
    compute_retry_delay,
)


# =============================================================================
# compute_retry_delay — deterministic (jitter=0)
# =============================================================================


class TestComputeRetryDelayDeterministic:
    """Tests for compute_retry_delay with jitter disabled."""

    def test_base_case_attempt_zero(self) -> None:
        """attempt=0 should return exactly the base delay."""
        delay = compute_retry_delay(0, base=30.0, jitter=0.0)
        assert delay == 30.0

    def test_exponential_growth(self) -> None:
        """Each successive attempt doubles (default multiplier=2)."""
        d0 = compute_retry_delay(0, base=10.0, jitter=0.0, multiplier=2.0)
        d1 = compute_retry_delay(1, base=10.0, jitter=0.0, multiplier=2.0)
        d2 = compute_retry_delay(2, base=10.0, jitter=0.0, multiplier=2.0)
        assert d0 == 10.0
        assert d1 == 20.0
        assert d2 == 40.0

    def test_capped_at_maximum(self) -> None:
        """Delay is capped at the maximum, even for high attempt numbers."""
        delay = compute_retry_delay(
            100, base=30.0, maximum=300.0, jitter=0.0, multiplier=2.0
        )
        assert delay == 300.0

    def test_custom_multiplier(self) -> None:
        """Supports a custom multiplier (e.g. 3x)."""
        delay = compute_retry_delay(2, base=5.0, jitter=0.0, multiplier=3.0)
        assert delay == 45.0  # 5 * 3^2

    def test_always_non_negative(self) -> None:
        """Delay is never negative, even with extreme parameters."""
        delay = compute_retry_delay(0, base=0.0, jitter=0.0)
        assert delay >= 0.0


# =============================================================================
# compute_retry_delay — with jitter
# =============================================================================


class TestComputeRetryDelayWithJitter:
    """Tests for compute_retry_delay with jitter enabled."""

    def test_jitter_produces_variable_results(self) -> None:
        """With jitter>0, repeated calls produce different values."""
        results = {compute_retry_delay(0, base=30.0, jitter=0.25) for _ in range(100)}
        # Should have more than 1 unique value
        assert len(results) > 1

    def test_jitter_stays_within_bounds(self) -> None:
        """All results should be within [base*(1-jitter), base*(1+jitter)]."""
        base = 30.0
        jitter = 0.25
        lower = base * (1 - jitter)
        upper = base * (1 + jitter)
        for _ in range(1000):
            delay = compute_retry_delay(0, base=base, jitter=jitter)
            assert lower <= delay <= upper, f"delay {delay} out of [{lower}, {upper}]"

    def test_default_jitter_matches_constant(self) -> None:
        """Default jitter parameter should match _JITTER_FRACTION."""
        # No jitter arg → uses default
        delays = [compute_retry_delay(0, base=30.0) for _ in range(100)]
        # With default jitter of 0.25, range is [22.5, 37.5]
        for d in delays:
            assert 30.0 * (1 - _JITTER_FRACTION) <= d <= 30.0 * (1 + _JITTER_FRACTION)


# =============================================================================
# Module-level constants
# =============================================================================


class TestBackoffConstants:
    """Verify module-level constants have expected values."""

    def test_initial_retry_delay(self) -> None:
        assert _INITIAL_RETRY_DELAY_S == 30.0

    def test_max_retry_delay(self) -> None:
        assert _MAX_RETRY_DELAY_S == 300.0

    def test_backoff_multiplier(self) -> None:
        assert _BACKOFF_MULTIPLIER == 2.0

    def test_jitter_fraction(self) -> None:
        assert _JITTER_FRACTION == 0.25


# =============================================================================
# CLI argument parsing
# =============================================================================


class TestBackoffCLIArgs:
    """Test that parse_args accepts the new backoff CLI arguments."""

    def test_initial_retry_delay_default(self) -> None:
        """Default --initial_retry_delay matches _INITIAL_RETRY_DELAY_S."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator"]):
            args = parse_args()
        assert args.initial_retry_delay == _INITIAL_RETRY_DELAY_S

    def test_initial_retry_delay_custom(self) -> None:
        """--initial_retry_delay accepts a custom float."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator", "--initial_retry_delay=10.5"]):
            args = parse_args()
        assert args.initial_retry_delay == 10.5

    def test_max_retry_delay_default(self) -> None:
        """Default --max_retry_delay matches _MAX_RETRY_DELAY_S."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator"]):
            args = parse_args()
        assert args.max_retry_delay == _MAX_RETRY_DELAY_S

    def test_max_retry_delay_custom(self) -> None:
        """--max_retry_delay accepts a custom float."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator", "--max_retry_delay=600.0"]):
            args = parse_args()
        assert args.max_retry_delay == 600.0

    def test_skip_image_validation_default_false(self) -> None:
        """Default --skip_image_validation is False."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator"]):
            args = parse_args()
        assert args.skip_image_validation is False

    def test_skip_image_validation_flag(self) -> None:
        """--skip_image_validation sets the flag to True."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        with patch.object(sys, "argv", ["orchestrator", "--skip_image_validation"]):
            args = parse_args()
        assert args.skip_image_validation is True
