"""Pair-clustered interval, interval hull and the one-sided 2.5% MDE (LF-D6, plan §5.2).

The benchmark plays every map twice with the seats swapped. Games within a pair
share a map, so they are not independent, and the iid ternary interval
(``score_interval``) can be too narrow (positively correlated pairs) or too wide
(negatively correlated pairs). The A1 decision rule therefore takes the hull of
the iid interval and the pair-clustered one. These tests pin the formula by hand
on two fixtures: one where the pair interval is wider on both sides, and one
where each side of the hull comes from a different interval.
"""

from __future__ import annotations

import math

import pytest

from alpha_zero.evaluation.scoring import (
    interval_hull,
    minimum_detectable_effect,
    pair_score_interval,
    score_interval,
)

Z95 = 1.959963984540054
Z99 = 2.5758293035489004


def _by_hand(pair_scores: list[float], z: float) -> tuple[float, float]:
    """The docstring formula, written out independently of the implementation."""
    k = z * z / 2.0
    n_adj = len(pair_scores) + 2.0 * k
    mean = (sum(pair_scores) + k) / n_adj
    var = (sum(s * s for s in pair_scores) + k) / n_adj - mean * mean
    half = z * math.sqrt(var / n_adj)
    return (max(0.0, mean - half), min(1.0, mean + half))


# Fixture A: 10 pairs, six won from both seats and four lost from both seats.
# Every pair is decisive and internally consistent, so the per-pair variance is
# as large as it can be: the pair interval is wider than the iid one on BOTH sides.
FIXTURE_A = [1.0] * 6 + [0.0] * 4  # games: 12 W / 0 D / 8 L

# Fixture B: 8 pairs -- five split W/L, two won twice, one lost and drawn.
# Games: 9 W / 1 D / 6 L. Here the pair interval's low is below the iid low but
# its high is below the iid high, so each side of the hull comes from a
# different interval.
FIXTURE_B = [0.5] * 5 + [1.0] * 2 + [0.25]


class TestPairScoreInterval:
    def test_fixture_a_by_hand(self) -> None:
        low, high = pair_score_interval(FIXTURE_A, 0.95)
        assert (low, high) == pytest.approx(_by_hand(FIXTURE_A, Z95), abs=1e-12)
        # Literal values, so a change to the shared formula cannot move both sides.
        assert low == pytest.approx(0.311604066, abs=1e-8)
        assert high == pytest.approx(0.832889374, abs=1e-8)

    def test_fixture_a_at_99_percent(self) -> None:
        low, high = pair_score_interval(FIXTURE_A, 0.99)
        assert (low, high) == pytest.approx(_by_hand(FIXTURE_A, Z99), abs=1e-12)
        assert low == pytest.approx(0.246630686, abs=1e-8)

    def test_fixture_b_by_hand(self) -> None:
        low, high = pair_score_interval(FIXTURE_B, 0.95)
        assert low == pytest.approx(0.362292291, abs=1e-8)
        assert high == pytest.approx(0.764381290, abs=1e-8)

    def test_single_games_reduce_to_the_ternary_interval(self) -> None:
        """Same estimator as ``score_interval``: units of 0/0.5/1 give its exact answer."""
        units = [1.0] * 9 + [0.5] * 4 + [0.0] * 7
        for confidence in (0.90, 0.95, 0.99):
            assert pair_score_interval(units, confidence) == pytest.approx(
                score_interval(9, 4, 7, confidence), abs=1e-12
            )

    def test_empty_is_uninformative(self) -> None:
        assert pair_score_interval([], 0.95) == (0.0, 1.0)

    @pytest.mark.parametrize("bad", [[1.5], [-0.25], [float("nan")]])
    def test_rejects_scores_outside_unit_interval(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match="pair score"):
            pair_score_interval(bad, 0.95)

    def test_rejects_unsupported_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            pair_score_interval(FIXTURE_A, 0.8)


class TestIntervalHull:
    def test_fixture_a_hull_is_the_pair_interval(self) -> None:
        iid = score_interval(12, 0, 8, 0.95)
        assert iid == pytest.approx((0.386030379, 0.781744589), abs=1e-8)
        pair = pair_score_interval(FIXTURE_A, 0.95)
        assert interval_hull(iid, pair) == pair

    def test_fixture_b_hull_takes_each_side_from_a_different_interval(self) -> None:
        iid = score_interval(9, 1, 6, 0.95)
        pair = pair_score_interval(FIXTURE_B, 0.95)
        assert iid == pytest.approx((0.363806999, 0.787391560), abs=1e-8)
        assert interval_hull(iid, pair) == (pair[0], iid[1])

    def test_hull_needs_an_interval(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            interval_hull()


class TestMdeAlpha025:
    def test_one_sided_2_5_percent(self) -> None:
        # (z_0.975 + z_0.80) * sqrt((1 - d) / 4) / sqrt(n)
        expected = (1.959963984540054 + 0.8416212335729143) * math.sqrt(0.53 / 4) / math.sqrt(200)
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) == pytest.approx(expected)
        # The plan's quoted figure: ~0.072 at n=200, d=0.47.
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) == pytest.approx(0.0721, abs=5e-4)

    def test_is_wider_than_alpha_05(self) -> None:
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) > minimum_detectable_effect(
            200, 0.47, alpha=0.05
        )
