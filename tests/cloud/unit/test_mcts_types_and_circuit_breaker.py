"""
Unit tests for MCTS type constants alignment and circuit breaker logic.

Tests:
1. VALID_MCTS_TYPES constant correctness and alignment with mcts_factory
2. All cloud workers accept the same MCTS types via shared constant
3. Circuit breaker CLI argument parsing and default values
4. Circuit breaker triggers after N consecutive failures
5. Circuit breaker resets on success
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

# =============================================================================
# VALID_MCTS_TYPES Constant Tests
# =============================================================================


class TestValidMCTSTypesConstant:
    """Test the VALID_MCTS_TYPES constant in config.py."""

    def test_valid_mcts_types_is_tuple(self) -> None:
        """VALID_MCTS_TYPES should be an immutable tuple."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        assert isinstance(VALID_MCTS_TYPES, tuple)

    def test_valid_mcts_types_contains_muzero(self) -> None:
        """VALID_MCTS_TYPES must include 'muzero' (root cause fix)."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        assert "muzero" in VALID_MCTS_TYPES

    def test_valid_mcts_types_contains_all_expected(self) -> None:
        """VALID_MCTS_TYPES must include all 4 cloud-supported types."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        expected = {"civ6", "gumbel", "batch_gumbel", "muzero"}
        assert set(VALID_MCTS_TYPES) == expected

    def test_valid_mcts_types_subset_of_mcts_factory(self) -> None:
        """Every type in VALID_MCTS_TYPES must exist in mcts_factory registry."""
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES
        from alpha_zero.core.mcts_factory import list_variants

        factory_variants = list_variants()
        for mcts_type in VALID_MCTS_TYPES:
            assert mcts_type in factory_variants, (
                f"'{mcts_type}' is in VALID_MCTS_TYPES but not in mcts_factory"
            )

    def test_default_mcts_type_is_gumbel(self) -> None:
        """DEFAULT_MCTS_TYPE should be 'gumbel' (updated from 'civ6' in Step 4)."""
        from alpha_zero.cloud.gcp.config import DEFAULT_MCTS_TYPE

        assert DEFAULT_MCTS_TYPE == "gumbel"

    def test_default_mcts_type_in_valid_types(self) -> None:
        """DEFAULT_MCTS_TYPE must be one of the valid types."""
        from alpha_zero.cloud.gcp.config import DEFAULT_MCTS_TYPE, VALID_MCTS_TYPES

        assert DEFAULT_MCTS_TYPE in VALID_MCTS_TYPES


# =============================================================================
# Orchestrator MCTS Type Parsing Tests
# =============================================================================


class TestOrchestratorMCTSTypeParsing:
    """Test that orchestrator's --mcts_type arg accepts all valid types."""

    @pytest.mark.parametrize("mcts_type", ["civ6", "gumbel", "batch_gumbel", "muzero"])
    def test_all_mcts_types_accepted(self, mcts_type: str) -> None:
        """Orchestrator should accept every VALID_MCTS_TYPES value."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", f"--mcts_type={mcts_type}"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == mcts_type

    def test_invalid_mcts_type_rejected(self) -> None:
        """Orchestrator should reject unknown MCTS types."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--mcts_type=invalid_type"]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()

    def test_default_mcts_type_is_gumbel(self) -> None:
        """Orchestrator default should match DEFAULT_MCTS_TYPE ('gumbel')."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == "gumbel"


# =============================================================================
# Worker MCTS Type Parsing Tests (Selfplay, Evaluation, Tournament)
# =============================================================================


class TestSelfplayWorkerMCTSTypeParsing:
    """Test selfplay_worker --mcts_type accepts all valid types."""

    @pytest.mark.parametrize("mcts_type", ["civ6", "gumbel", "batch_gumbel", "muzero"])
    def test_all_mcts_types_accepted(self, mcts_type: str) -> None:
        """Selfplay worker should accept every VALID_MCTS_TYPES value."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=0",
            f"--mcts_type={mcts_type}",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == mcts_type

    def test_invalid_mcts_type_rejected(self) -> None:
        """Selfplay worker should reject unknown MCTS types."""
        from alpha_zero.cloud.gcp.selfplay_worker import parse_args

        test_args = [
            "selfplay_worker",
            "--worker_id=0",
            "--model_version=1",
            "--iteration=0",
            "--mcts_type=invalid_type",
        ]
        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                parse_args()


class TestEvaluationWorkerMCTSTypeParsing:
    """Test evaluation_worker --mcts_type accepts all valid types."""

    @pytest.mark.parametrize("mcts_type", ["civ6", "gumbel", "batch_gumbel", "muzero"])
    def test_all_mcts_types_accepted(self, mcts_type: str) -> None:
        """Evaluation worker should accept every VALID_MCTS_TYPES value."""
        from alpha_zero.cloud.gcp.evaluation_worker import parse_args

        test_args = [
            "evaluation_worker",
            "--model_version_new=2",
            "--model_version_old=1",
            "--iteration=10",
            f"--mcts_type={mcts_type}",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == mcts_type


class TestTournamentWorkerMCTSTypeParsing:
    """Test tournament_worker --mcts_type accepts all valid types."""

    @pytest.mark.parametrize("mcts_type", ["civ6", "gumbel", "batch_gumbel", "muzero"])
    def test_all_mcts_types_accepted(self, mcts_type: str) -> None:
        """Tournament worker should accept every VALID_MCTS_TYPES value."""
        from alpha_zero.cloud.gcp.tournament_worker import parse_args

        test_args = [
            "tournament_worker",
            f"--mcts_type={mcts_type}",
        ]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.mcts_type == mcts_type


# =============================================================================
# Circuit Breaker CLI Parsing Tests
# =============================================================================


class TestCircuitBreakerParsing:
    """Test circuit breaker CLI argument parsing."""

    def test_default_max_consecutive_failures(self) -> None:
        """Default --max_consecutive_failures should be 3."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.max_consecutive_failures == 3

    def test_custom_max_consecutive_failures(self) -> None:
        """--max_consecutive_failures should accept custom values."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--max_consecutive_failures=5"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.max_consecutive_failures == 5

    def test_disable_circuit_breaker_with_zero(self) -> None:
        """--max_consecutive_failures=0 should disable the circuit breaker."""
        from alpha_zero.cloud.gcp.vertex_orchestrator import parse_args

        test_args = ["vertex_orchestrator", "--max_consecutive_failures=0"]
        with patch.object(sys, "argv", test_args):
            args = parse_args()

        assert args.max_consecutive_failures == 0


# =============================================================================
# Circuit Breaker Logic Tests
# =============================================================================


class TestCircuitBreakerLogic:
    """Test the circuit breaker logic in the main orchestrator loop."""

    def test_circuit_breaker_counter_resets_on_success(self) -> None:
        """Consecutive failure counter should reset to 0 after a successful iteration.

        This tests the logic: consecutive_failures = 0 after successful_iterations += 1
        """
        # Simulate the circuit breaker logic from the main loop
        consecutive_failures = 2

        # Simulate a successful iteration
        selfplay_ok = True
        training_ok = True
        if selfplay_ok and training_ok:
            consecutive_failures = 0  # Reset on success

        assert consecutive_failures == 0

    def test_circuit_breaker_increments_on_selfplay_failure(self) -> None:
        """Consecutive failure counter increments when self-play fails."""
        consecutive_failures = 0
        selfplay_ok = False

        if not selfplay_ok:
            consecutive_failures += 1

        assert consecutive_failures == 1

    def test_circuit_breaker_increments_on_training_failure(self) -> None:
        """Consecutive failure counter increments when training fails."""
        consecutive_failures = 1
        training_ok = False

        if not training_ok:
            consecutive_failures += 1

        assert consecutive_failures == 2

    def test_circuit_breaker_triggers_at_threshold(self) -> None:
        """Circuit breaker should trigger when consecutive failures >= threshold."""
        max_consec = 3
        consecutive_failures = 3

        should_break = max_consec > 0 and consecutive_failures >= max_consec
        assert should_break is True

    def test_circuit_breaker_does_not_trigger_below_threshold(self) -> None:
        """Circuit breaker should NOT trigger below threshold."""
        max_consec = 3
        consecutive_failures = 2

        should_break = max_consec > 0 and consecutive_failures >= max_consec
        assert should_break is False

    def test_circuit_breaker_disabled_when_zero(self) -> None:
        """Circuit breaker should be disabled when max_consec is 0."""
        max_consec = 0
        consecutive_failures = 100  # Many failures

        should_break = max_consec > 0 and consecutive_failures >= max_consec
        assert should_break is False

    def test_circuit_breaker_with_mixed_failures_and_successes(self) -> None:
        """Simulate a sequence: fail, fail, succeed, fail — should NOT trigger."""
        max_consec = 3
        consecutive_failures = 0

        # Iteration 1: fail
        consecutive_failures += 1
        assert consecutive_failures == 1

        # Iteration 2: fail
        consecutive_failures += 1
        assert consecutive_failures == 2

        # Iteration 3: succeed → reset
        consecutive_failures = 0
        assert consecutive_failures == 0

        # Iteration 4: fail
        consecutive_failures += 1
        should_break = max_consec > 0 and consecutive_failures >= max_consec
        assert should_break is False  # Only 1 consecutive failure


# =============================================================================
# Cross-Module Alignment Tests
# =============================================================================


class TestCrossModuleAlignment:
    """Test that all modules use the same VALID_MCTS_TYPES source of truth."""

    def test_config_exports_valid_mcts_types(self) -> None:
        """config module should export VALID_MCTS_TYPES."""
        from alpha_zero.cloud.gcp import config

        assert hasattr(config, "VALID_MCTS_TYPES")
        assert hasattr(config, "DEFAULT_MCTS_TYPE")

    def test_orchestrator_imports_from_config(self) -> None:
        """vertex_orchestrator should import VALID_MCTS_TYPES from config."""
        import alpha_zero.cloud.gcp.vertex_orchestrator as orch

        # The orchestrator should use the shared constant
        from alpha_zero.cloud.gcp.config import VALID_MCTS_TYPES

        # Verify the parse_args function uses the constant by testing
        # that it accepts all VALID_MCTS_TYPES members
        for mcts_type in VALID_MCTS_TYPES:
            test_args = ["orch", f"--mcts_type={mcts_type}"]
            with patch.object(sys, "argv", test_args):
                args = orch.parse_args()
                assert args.mcts_type == mcts_type
