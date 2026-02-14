"""Tests for alpha_zero.cloud.gcp.model_selection — shared subsampling utilities.

Covers:
- subsample_model_versions: stride, range, combined, edge cases
"""

from __future__ import annotations

from alpha_zero.cloud.gcp.model_selection import subsample_model_versions

# =============================================================================
# subsample_model_versions
# =============================================================================


class TestSubsampleModelVersions:
    """Tests for subsample_model_versions()."""

    # --- identity / no-op cases ---------------------------------------------

    def test_no_filters_returns_sorted_input(self):
        """Default stride=1 and no range returns the sorted input."""
        versions = [5, 3, 1, 4, 2, 0]
        assert subsample_model_versions(versions) == [0, 1, 2, 3, 4, 5]

    def test_stride_one_is_identity(self):
        """Explicit stride=1 is equivalent to no subsampling."""
        versions = list(range(10))
        assert subsample_model_versions(versions, stride=1) == versions

    def test_stride_none_is_identity(self):
        """stride=None should be treated as stride=1."""
        versions = list(range(5))
        assert subsample_model_versions(versions, stride=None) == versions

    # --- stride subsampling -------------------------------------------------

    def test_stride_3_on_10_elements(self):
        """Stride=3 picks every 3rd element; first and last preserved."""
        result = subsample_model_versions(list(range(10)), stride=3)
        assert 0 in result  # first always preserved
        assert 9 in result  # last always preserved
        assert result == [0, 3, 6, 9]

    def test_stride_2_on_5_elements(self):
        """Stride=2 on [0,1,2,3,4] -> [0,2,4]."""
        result = subsample_model_versions([0, 1, 2, 3, 4], stride=2)
        assert result == [0, 2, 4]

    def test_stride_4_preserves_last(self):
        """Stride=4 on [0..9] -> [0,4,8] + last=9 -> [0,4,8,9]."""
        result = subsample_model_versions(list(range(10)), stride=4)
        assert result == [0, 4, 8, 9]

    def test_stride_larger_than_list(self):
        """Stride larger than list size returns first and last."""
        result = subsample_model_versions([0, 1, 2, 3, 4], stride=100)
        assert result == [0, 4]

    def test_stride_equals_list_size(self):
        """Stride == len(versions) returns first and last."""
        versions = [10, 20, 30, 40, 50]
        result = subsample_model_versions(versions, stride=5)
        assert result == [10, 50]

    # --- range filtering ----------------------------------------------------

    def test_range_filters_inclusive(self):
        """Range '3:7' keeps versions 3 through 7 inclusive."""
        result = subsample_model_versions(list(range(10)), version_range="3:7")
        assert result == [3, 4, 5, 6, 7]

    def test_range_excludes_all_returns_empty(self):
        """Range with no matching versions returns []."""
        result = subsample_model_versions(list(range(10)), version_range="100:200")
        assert result == []

    def test_range_single_match(self):
        """Range that matches exactly one version."""
        result = subsample_model_versions([0, 5, 10, 15], version_range="5:5")
        assert result == [5]

    def test_range_with_unordered_input(self):
        """Range filter works correctly with unordered input."""
        result = subsample_model_versions([9, 3, 7, 1, 5], version_range="3:7")
        assert result == [3, 5, 7]

    # --- combined stride + range --------------------------------------------

    def test_stride_plus_range(self):
        """Stride=2 + range='2:8' on [0..9] -> filter to [2..8], then stride."""
        result = subsample_model_versions(list(range(10)), stride=2, version_range="2:8")
        # [2,3,4,5,6,7,8] stride=2 -> [2,4,6] + last=8 -> [2,4,6,8]
        assert 2 in result  # first in range
        assert 8 in result  # last in range
        assert result == [2, 4, 6, 8]

    def test_stride_plus_range_preserves_endpoints(self):
        """Combined filter always preserves first and last of the filtered set."""
        result = subsample_model_versions(list(range(20)), stride=5, version_range="3:17")
        assert result[0] == 3
        assert result[-1] == 17

    # --- edge cases ---------------------------------------------------------

    def test_empty_input(self):
        """Empty version list returns []."""
        assert subsample_model_versions([]) == []

    def test_single_version(self):
        """Single version returns that version."""
        assert subsample_model_versions([42]) == [42]

    def test_two_versions_stride_2(self):
        """Two versions with stride=2 returns both (first=last after stride)."""
        result = subsample_model_versions([10, 20], stride=2)
        assert result == [10, 20]

    def test_duplicates_in_input(self):
        """Duplicate versions in input are deduplicated."""
        result = subsample_model_versions([1, 3, 3, 5, 5, 7])
        assert result == [1, 3, 5, 7]

    def test_invalid_range_format_returns_all(self):
        """Invalid range format (e.g., missing colon) should return all versions."""
        result = subsample_model_versions(list(range(5)), version_range="invalid")
        assert result == [0, 1, 2, 3, 4]

    def test_non_integer_range_warns_and_returns_all(self):
        """Non-integer range values should warn and return all versions."""
        result = subsample_model_versions(list(range(5)), version_range="a:b")
        assert result == [0, 1, 2, 3, 4]

    def test_negative_versions(self):
        """Negative version numbers handled correctly."""
        result = subsample_model_versions([-5, -3, -1, 0, 1, 3, 5], stride=2)
        assert result[0] == -5
        assert result[-1] == 5

    def test_large_stride_single_element(self):
        """Single element list with large stride returns that element."""
        assert subsample_model_versions([42], stride=100) == [42]


# =============================================================================
# Integration with vertex_orchestrator._apply_model_subsampling
# =============================================================================


class TestApplyModelSubsamplingIntegration:
    """Verify _apply_model_subsampling delegates to subsample_model_versions."""

    def test_orchestrator_delegates_to_shared_function(self):
        """_apply_model_subsampling should produce same results as direct call."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        args = argparse.Namespace(
            tournament_model_stride=3,
            tournament_model_range="2:8",
        )
        versions = list(range(10))

        orchestrator_result = _apply_model_subsampling(versions, args)
        direct_result = subsample_model_versions(versions, stride=3, version_range="2:8")

        assert orchestrator_result == direct_result

    def test_orchestrator_backwards_compatible_no_attrs(self):
        """Works with old args that don't have tournament_model_* fields."""
        import argparse

        from alpha_zero.cloud.gcp.vertex_orchestrator import _apply_model_subsampling

        old_args = argparse.Namespace(tournament_games=10, mcts_type="gumbel")
        versions = [0, 1, 2, 3, 4]
        assert _apply_model_subsampling(versions, old_args) == versions


# =============================================================================
# Integration with tournament_worker.resolve_model_versions
# =============================================================================


class TestResolveModelVersionsIntegration:
    """Verify resolve_model_versions delegates subsampling to shared function."""

    def test_tournament_worker_stride_delegates(self):
        """Stride in tournament_worker produces same result as direct call."""
        import argparse

        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=3,
            model_range=None,
        )
        # dry_run gives [0..9]
        result = resolve_model_versions(args, None, dry_run=True)
        direct = subsample_model_versions(list(range(10)), stride=3)
        assert result == direct

    def test_tournament_worker_range_delegates(self):
        """Range in tournament_worker produces same result as direct call."""
        import argparse

        from alpha_zero.cloud.gcp.tournament_worker import resolve_model_versions

        args = argparse.Namespace(
            model_versions=None,
            model_stride=1,
            model_range="3:7",
        )
        result = resolve_model_versions(args, None, dry_run=True)
        direct = subsample_model_versions(list(range(10)), version_range="3:7")
        assert result == direct
