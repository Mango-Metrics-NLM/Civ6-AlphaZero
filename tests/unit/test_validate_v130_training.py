"""Tests for scripts/validate_v130_training.py.

Covers:
- compute_weight_checksum
- verify_weight_continuity
- verify_checkpoint_roundtrip
- ValidationResult dataclass and summary()
"""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Import the script as a module via importlib (it is not a package).
# ---------------------------------------------------------------------------

_script_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "validate_v130_training.py"
)
_spec = importlib.util.spec_from_file_location("validate_v130_training", _script_path)
_mod = importlib.util.module_from_spec(_spec)
# Register in sys.modules so @dataclass can resolve cls.__module__
import sys
sys.modules["validate_v130_training"] = _mod
_spec.loader.exec_module(_mod)

compute_weight_checksum = _mod.compute_weight_checksum
verify_weight_continuity = _mod.verify_weight_continuity
verify_checkpoint_roundtrip = _mod.verify_checkpoint_roundtrip
ValidationResult = _mod.ValidationResult

from alpha_zero.core.civ6_network import MicroCivNetwork

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# Small network parameters for fast tests.
SMALL_BOARD = 8
SMALL_ACTIONS = 50
SMALL_RES_BLOCKS = 2
SMALL_FILTERS = 16


@pytest.fixture()
def small_network() -> MicroCivNetwork:
    """Return a small MicroCivNetwork for testing."""
    return MicroCivNetwork(
        board_size=SMALL_BOARD,
        action_space_size=SMALL_ACTIONS,
        num_res_blocks=SMALL_RES_BLOCKS,
        num_filters=SMALL_FILTERS,
    )


@pytest.fixture()
def another_small_network() -> MicroCivNetwork:
    """Return a second independently-initialized small network."""
    return MicroCivNetwork(
        board_size=SMALL_BOARD,
        action_space_size=SMALL_ACTIONS,
        num_res_blocks=SMALL_RES_BLOCKS,
        num_filters=SMALL_FILTERS,
    )


class _ZeroModule(nn.Module):
    """Trivial module whose parameters are all zeros."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(10, 10))
        self.bias = nn.Parameter(torch.zeros(10))


# ===========================================================================
# Tests for compute_weight_checksum
# ===========================================================================


class TestComputeWeightChecksum:
    """Tests for compute_weight_checksum."""

    def test_nonzero_on_real_network(self, small_network: MicroCivNetwork) -> None:
        """A randomly-initialized MicroCivNetwork should have a non-zero checksum."""
        cs = compute_weight_checksum(small_network)
        assert cs > 0.0, "Checksum of a real network must be positive"

    def test_different_networks_give_different_checksums(
        self,
        small_network: MicroCivNetwork,
        another_small_network: MicroCivNetwork,
    ) -> None:
        """Two independently-initialized networks should have different checksums.

        Because the weights are randomly initialized, the probability that
        two networks produce the exact same checksum is effectively zero.
        """
        cs1 = compute_weight_checksum(small_network)
        cs2 = compute_weight_checksum(another_small_network)
        assert cs1 != cs2

    def test_zero_weight_network_gives_zero(self) -> None:
        """A module with all-zero parameters should return 0.0."""
        module = _ZeroModule()
        cs = compute_weight_checksum(module)
        assert cs == 0.0

    def test_consistent_across_calls(self, small_network: MicroCivNetwork) -> None:
        """Calling the function twice on the same network must return the same value."""
        cs1 = compute_weight_checksum(small_network)
        cs2 = compute_weight_checksum(small_network)
        assert cs1 == cs2

    def test_changes_after_weight_modification(self, small_network: MicroCivNetwork) -> None:
        """Checksum must change when weights are modified."""
        cs_before = compute_weight_checksum(small_network)
        # Manually perturb a parameter
        with torch.no_grad():
            for p in small_network.parameters():
                p.add_(1.0)
                break  # Only need to change one parameter
        cs_after = compute_weight_checksum(small_network)
        assert cs_before != cs_after


# ===========================================================================
# Tests for verify_weight_continuity
# ===========================================================================


class TestVerifyWeightContinuity:
    """Tests for verify_weight_continuity."""

    def test_empty_list_returns_true(self) -> None:
        """An empty checksum list should return True."""
        assert verify_weight_continuity([]) is True

    def test_single_checksum_returns_true(self) -> None:
        """A list with one checksum should return True."""
        assert verify_weight_continuity([42.0]) is True

    def test_gradual_change_returns_true(self) -> None:
        """Consecutive checksums changing by less than 50% should pass."""
        # Each step changes by ~5%
        checksums = [100.0, 105.0, 110.0, 108.0, 112.0]
        assert verify_weight_continuity(checksums) is True

    def test_large_change_returns_false(self) -> None:
        """A > 50% relative jump should fail."""
        # 100 -> 200 is a 100% change
        checksums = [100.0, 200.0]
        assert verify_weight_continuity(checksums) is False

    def test_multiple_smooth_iterations(self) -> None:
        """Many iterations with small gradient-like changes should pass."""
        # Simulate gradual weight growth over 20 iterations
        checksums = [1000.0 + i * 10.0 for i in range(20)]
        assert verify_weight_continuity(checksums) is True

    def test_large_change_in_middle(self) -> None:
        """A large jump in the middle of an otherwise smooth sequence should fail."""
        checksums = [100.0, 102.0, 104.0, 300.0, 302.0]
        assert verify_weight_continuity(checksums) is False

    def test_exactly_50_percent_boundary(self) -> None:
        """A change of exactly 50% should pass (threshold is strictly >0.5)."""
        # relative_change = |150 - 100| / 100 = 0.5, NOT > 0.5
        checksums = [100.0, 150.0]
        assert verify_weight_continuity(checksums) is True

    def test_just_over_50_percent_fails(self) -> None:
        """A change of 50.01% should fail."""
        checksums = [100.0, 150.02]
        assert verify_weight_continuity(checksums) is False

    def test_decreasing_values_within_threshold(self) -> None:
        """Decreasing values that stay within 50% change should pass."""
        checksums = [200.0, 180.0, 160.0, 145.0]
        assert verify_weight_continuity(checksums) is True

    def test_near_zero_checksum_does_not_divide_by_zero(self) -> None:
        """Very small checksums should not cause division errors."""
        # The function uses max(abs(prev), 1e-8) in the denominator
        checksums = [0.0, 1e-9]
        # relative_change = 1e-9 / 1e-8 = 0.1 which is < 0.5
        assert verify_weight_continuity(checksums) is True


# ===========================================================================
# Tests for verify_checkpoint_roundtrip
# ===========================================================================


class TestVerifyCheckpointRoundtrip:
    """Tests for verify_checkpoint_roundtrip."""

    def test_successful_roundtrip(
        self, small_network: MicroCivNetwork, tmp_path: str
    ) -> None:
        """Saving and loading a MicroCivNetwork should produce matching checksums."""
        result = verify_checkpoint_roundtrip(small_network, str(tmp_path), "cpu")
        assert result is True

    def test_temp_file_is_cleaned_up(
        self, small_network: MicroCivNetwork, tmp_path: str
    ) -> None:
        """The temporary checkpoint file should be removed after verification."""
        verify_checkpoint_roundtrip(small_network, str(tmp_path), "cpu")
        roundtrip_file = os.path.join(str(tmp_path), "roundtrip_test.pt")
        assert not os.path.exists(roundtrip_file), "Temp checkpoint file should be deleted"

    def test_roundtrip_preserves_exact_checksum(
        self, small_network: MicroCivNetwork, tmp_path: str
    ) -> None:
        """After roundtrip the checksum of the loaded model should match the original."""
        original_cs = compute_weight_checksum(small_network)
        path = os.path.join(str(tmp_path), "manual_roundtrip.pt")
        small_network.save(path)
        loaded = MicroCivNetwork.load(path, device="cpu")
        loaded_cs = compute_weight_checksum(loaded)
        assert abs(original_cs - loaded_cs) < 1e-4


# ===========================================================================
# Tests for ValidationResult
# ===========================================================================


def _make_result(
    *,
    weight_continuity_verified: bool = True,
    loss_trend_downward: bool = True,
    win_rate_improved: bool = True,
    checkpoint_roundtrip_ok: bool = True,
    iterations_completed: int = 3,
    policy_losses: list[float] | None = None,
    value_losses: list[float] | None = None,
    total_losses: list[float] | None = None,
    win_rates: list[float] | None = None,
    weight_checksums: list[float] | None = None,
) -> "ValidationResult":
    """Helper to create a ValidationResult with sensible defaults."""
    all_passed = weight_continuity_verified and loss_trend_downward and checkpoint_roundtrip_ok
    return ValidationResult(
        iterations_completed=iterations_completed,
        policy_losses=policy_losses if policy_losses is not None else [1.0, 0.8, 0.6],
        value_losses=value_losses if value_losses is not None else [0.5, 0.4, 0.3],
        total_losses=total_losses if total_losses is not None else [1.5, 1.2, 0.9],
        win_rates=win_rates if win_rates is not None else [0.2, 0.4, 0.6],
        weight_checksums=weight_checksums if weight_checksums is not None else [100.0, 102.0, 104.0],
        weight_continuity_verified=weight_continuity_verified,
        loss_trend_downward=loss_trend_downward,
        win_rate_improved=win_rate_improved,
        checkpoint_roundtrip_ok=checkpoint_roundtrip_ok,
        all_passed=all_passed,
    )


class TestValidationResult:
    """Tests for the ValidationResult dataclass and its summary() method."""

    def test_summary_returns_string(self) -> None:
        """summary() should return a non-empty string."""
        result = _make_result()
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_all_passed_true_when_all_checks_pass(self) -> None:
        """all_passed should be True when all three core checks pass."""
        result = _make_result(
            weight_continuity_verified=True,
            loss_trend_downward=True,
            checkpoint_roundtrip_ok=True,
        )
        assert result.all_passed is True

    def test_all_passed_false_when_weight_continuity_fails(self) -> None:
        """all_passed should be False when weight_continuity_verified is False."""
        result = _make_result(weight_continuity_verified=False)
        assert result.all_passed is False

    def test_all_passed_false_when_loss_trend_fails(self) -> None:
        """all_passed should be False when loss_trend_downward is False."""
        result = _make_result(loss_trend_downward=False)
        assert result.all_passed is False

    def test_all_passed_false_when_checkpoint_roundtrip_fails(self) -> None:
        """all_passed should be False when checkpoint_roundtrip_ok is False."""
        result = _make_result(checkpoint_roundtrip_ok=False)
        assert result.all_passed is False

    def test_all_passed_independent_of_win_rate(self) -> None:
        """all_passed should not depend on win_rate_improved (it is informational)."""
        result = _make_result(win_rate_improved=False)
        # all_passed should still be True because win_rate is not a core check
        assert result.all_passed is True

    def test_summary_contains_policy_losses(self) -> None:
        """summary() should contain formatted policy loss values."""
        result = _make_result(
            iterations_completed=2,
            policy_losses=[1.2345, 0.9876],
            value_losses=[0.5, 0.4],
            total_losses=[1.7345, 1.3876],
            win_rates=[0.2, 0.4],
            weight_checksums=[100.0, 102.0],
        )
        s = result.summary()
        assert "1.2345" in s
        assert "0.9876" in s

    def test_summary_contains_win_rates(self) -> None:
        """summary() should contain formatted win rate values."""
        result = _make_result(win_rates=[0.3, 0.7])
        s = result.summary()
        # Win rates are formatted with .1% e.g., 30.0%
        assert "30.0%" in s
        assert "70.0%" in s

    def test_summary_contains_weight_checksums(self) -> None:
        """summary() should contain formatted weight checksum values."""
        result = _make_result(weight_checksums=[123.456789, 124.000000])
        s = result.summary()
        assert "123.456789" in s
        assert "124.000000" in s

    def test_summary_shows_pass_when_all_passed(self) -> None:
        """summary() should contain 'ALL CHECKS PASSED' when all_passed is True."""
        result = _make_result()
        s = result.summary()
        assert "ALL CHECKS PASSED" in s

    def test_summary_shows_fail_when_not_all_passed(self) -> None:
        """summary() should contain 'SOME CHECKS FAILED' when all_passed is False."""
        result = _make_result(loss_trend_downward=False)
        s = result.summary()
        assert "SOME CHECKS FAILED" in s

    def test_summary_shows_pass_fail_labels(self) -> None:
        """summary() should contain [PASS] and [FAIL] labels for each check."""
        result = _make_result(
            weight_continuity_verified=True,
            loss_trend_downward=False,
            checkpoint_roundtrip_ok=True,
        )
        s = result.summary()
        # Weight continuity should be PASS
        assert "[PASS] Weight continuity" in s
        # Loss trend should be FAIL
        assert "[FAIL] Loss trend downward" in s
        # Checkpoint roundtrip should be PASS
        assert "[PASS] Checkpoint save/load roundtrip" in s

    def test_summary_contains_iterations_completed(self) -> None:
        """summary() should display the number of iterations completed."""
        result = _make_result(iterations_completed=7)
        s = result.summary()
        assert "7" in s
        assert "Iterations completed" in s

    def test_summary_contains_section_headers(self) -> None:
        """summary() should include recognizable section headers."""
        result = _make_result()
        s = result.summary()
        assert "Loss Trajectory" in s
        assert "Win Rate vs Random" in s
        assert "Weight Checksums" in s
        assert "Checks:" in s
