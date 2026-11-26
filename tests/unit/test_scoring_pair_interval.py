"""Pair-clustered interval, interval hull and the one-sided 2.5% MDE (LF-D6, plan §5.2).

The benchmark plays every map twice with the seats swapped. Games within a pair
share a map, so they are not independent, and the iid ternary interval
(``score_interval``) can be too narrow (positively correlated pairs) or too wide
(negatively correlated pairs). The A1 decision rule therefore takes the hull of
the iid interval and the pair-clustered one.

The pair interval is the plain pentanomial interval: the mean of the pair scores
plus or minus ``z * sd / sqrt(P)``, with no pseudo-counts (T11 review C1). Fixture
values below are the reviewer's hand computations (``task-T11-review.md``),
re-derived for the plain estimator. Formula tests cannot tell a right estimator
from a wrong one, so ``TestOperatingCharacteristics`` measures what the A1
verdict rule actually does with it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from alpha_zero.evaluation.benchmark import interval_summary
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
    p = len(pair_scores)
    mean = sum(pair_scores) / p
    var = sum((s - mean) ** 2 for s in pair_scores) / (p - 1)
    half = z * math.sqrt(var / p)
    return (max(0.0, mean - half), min(1.0, mean + half))


# Fixture A: 10 pairs, six won from both seats and four lost from both seats --
# every pair decisive and internally consistent (maximal map clustering).
FIXTURE_A = [1.0] * 6 + [0.0] * 4  # games: 12 W / 0 D / 8 L
# Fixture B: 8 pairs -- five split W/L, two won twice, one lost and drawn.
FIXTURE_B = [0.5] * 5 + [1.0] * 2 + [0.25]  # games: 9 W / 1 D / 6 L
# F1: 20 drawn pairs (symmetry-arm shape) -- zero pair variance.
FIXTURE_F1 = [0.5] * 20  # games: 0 W / 40 D / 0 L
# F2: strong seat asymmetry -- seat 0 always wins, 20 W/L pairs.
FIXTURE_F2 = [0.5] * 20  # games: 20 W / 0 D / 20 L
# F3: map-clustered, 12 WW + 8 LL.
FIXTURE_F3 = [1.0] * 12 + [0.0] * 8  # games: 24 W / 0 D / 16 L

# (pair scores, (w, d, l), confidence, iid, pair, hull) -- values to 1e-6.
FIXTURES = [
    ("A", FIXTURE_A, (12, 0, 8), 0.95,
     (0.386030, 0.781745), (0.279939, 0.920061), (0.279939, 0.920061)),
    ("A", FIXTURE_A, (12, 0, 8), 0.99,
     (0.328367, 0.821812), (0.179369, 1.0), (0.179369, 1.0)),
    ("B", FIXTURE_B, (9, 1, 6), 0.95,
     (0.363807, 0.787392), (0.410003, 0.777497), (0.363807, 0.787392)),
    ("B", FIXTURE_B, (9, 1, 6), 0.99,
     (0.304054, 0.828485), (0.352266, 0.835234), (0.304054, 0.835234)),
    ("F1", FIXTURE_F1, (0, 40, 0), 0.95,
     (0.456189, 0.543811), (0.5, 0.5), (0.456189, 0.543811)),
    ("F1", FIXTURE_F1, (0, 40, 0), 0.99,
     (0.428863, 0.571137), (0.5, 0.5), (0.428863, 0.571137)),
    ("F2", FIXTURE_F2, (20, 0, 20), 0.99,
     (0.311404, 0.688596), (0.5, 0.5), (0.311404, 0.688596)),
    ("F3", FIXTURE_F3, (24, 0, 16), 0.95,
     (0.445718, 0.736758), (0.379719, 0.820281), (0.379719, 0.820281)),
    ("F3", FIXTURE_F3, (24, 0, 16), 0.99,
     (0.399973, 0.771573), (0.310502, 0.889498), (0.310502, 0.889498)),
]  # fmt: skip


class TestPairScoreInterval:
    @pytest.mark.parametrize(
        ("name", "scores", "wdl", "confidence", "iid", "pair", "hull"),
        FIXTURES,
        ids=[f"{f[0]}-{f[3]}" for f in FIXTURES],
    )
    def test_reviewer_fixtures(
        self,
        name: str,
        scores: list[float],
        wdl: tuple[int, int, int],
        confidence: float,
        iid: tuple[float, float],
        pair: tuple[float, float],
        hull: tuple[float, float],
    ) -> None:
        got_iid = score_interval(*wdl, confidence)
        got_pair = pair_score_interval(scores, confidence)
        assert got_iid == pytest.approx(iid, abs=1e-6)
        assert got_pair == pytest.approx(pair, abs=1e-6)
        assert interval_hull(got_iid, got_pair) == pytest.approx(hull, abs=1e-6)

    def test_matches_the_docstring_formula(self) -> None:
        for scores in (FIXTURE_A, FIXTURE_B, FIXTURE_F3):
            for confidence, z in ((0.95, Z95), (0.99, Z99)):
                assert pair_score_interval(scores, confidence) == pytest.approx(
                    _by_hand(scores, z), abs=1e-12
                )

    def test_zero_variance_has_zero_width_and_the_hull_keeps_the_iid_width(self) -> None:
        assert pair_score_interval(FIXTURE_F1, 0.99) == (0.5, 0.5)
        iid = score_interval(0, 40, 0, 0.99)
        assert interval_hull(iid, pair_score_interval(FIXTURE_F1, 0.99)) == iid

    @pytest.mark.parametrize("scores", [[], [0.75]])
    def test_fewer_than_two_pairs_is_uninformative(self, scores: list[float]) -> None:
        assert pair_score_interval(scores, 0.95) == (0.0, 1.0)

    @pytest.mark.parametrize("bad", [[1.5, 0.5], [-0.25, 0.5], [float("nan"), 0.5]])
    def test_rejects_scores_outside_unit_interval(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match="pair score"):
            pair_score_interval(bad, 0.95)

    def test_rejects_unsupported_confidence(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            pair_score_interval(FIXTURE_A, 0.8)


class TestIntervalHull:
    def test_each_side_may_come_from_a_different_interval(self) -> None:
        iid = score_interval(9, 1, 6, 0.99)
        pair = pair_score_interval(FIXTURE_B, 0.99)
        assert interval_hull(iid, pair) == (iid[0], pair[1])

    def test_hull_needs_an_interval(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            interval_hull()


class TestMdeAlpha025:
    def test_one_sided_2_5_percent(self) -> None:
        expected = (Z95 + 0.8416212335729143) * math.sqrt(0.53 / 4) / math.sqrt(200)
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) == pytest.approx(expected)
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) == pytest.approx(
            0.072110, abs=1e-6
        )

    def test_is_wider_than_alpha_05(self) -> None:
        assert minimum_detectable_effect(200, 0.47, alpha=0.025) > minimum_detectable_effect(
            200, 0.47, alpha=0.05
        )


# ---------------------------------------------------------------------------
# Operating characteristics of the A1 verdict rule (T11 review C1)
# ---------------------------------------------------------------------------

_RESULTS = np.array(["win", "draw", "loss"])


def _simulate(
    mu: float, draw_rate: float, pairs: int, reps: int, confidence: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(hull, iid)`` bounds for ``reps`` simulated arms of independent ternary games.

    Game outcomes are drawn with ``P(win) = mu - d/2``, ``P(draw) = d`` and
    ``P(loss) = 1 - mu - d/2``, the reviewer's model; every replicate goes
    through the benchmark's own ``interval_summary``.
    """
    rng = np.random.default_rng(seed)
    p = [mu - draw_rate / 2, draw_rate, 1.0 - mu - draw_rate / 2]
    draws = rng.choice(3, size=(reps, 2 * pairs), p=p)
    hull = np.empty((reps, 2))
    iid = np.empty((reps, 2))
    for r in range(reps):
        games = [
            {"pair": g // 2, "seat": g % 2, "result": res}
            for g, res in enumerate(_RESULTS[draws[r]])
        ]
        block = interval_summary(games, confidence)
        hull[r], iid[r] = block["hull"], block["iid"]
    return hull, iid


def _void(bounds: np.ndarray) -> float:
    """Fraction of intervals that exclude 0.5."""
    return float(np.mean((bounds[:, 0] > 0.5) | (bounds[:, 1] < 0.5)))


class TestOperatingCharacteristics:
    """Seeded Monte Carlo of the registered verdict conditions (~2 s, 44,000 arms).

    Reviewer's 40,000-replicate figures for the hull with the plain pentanomial
    interval: symmetry detection of a 0.20 asymmetry 83%, symmetry false VOID
    0.47%, positive-control VOID 2.9% (iid alone 2.6%). With z^2/2 pseudo-counts
    per pair they were 40%, 0.02% and 6.2-6.4% -- the negative control: restoring
    that estimator fails the detection and positive-control tests.
    """

    def test_symmetry_arm_detects_a_0_20_asymmetry(self) -> None:
        # 20 pairs, 99%: VOID when the hull excludes 0.5. mu = 0.70 is a 0.20 bias.
        hull, _ = _simulate(0.70, 0.47, pairs=20, reps=4000, confidence=0.99, seed=1)
        detected = _void(hull)
        assert detected >= 0.80, f"symmetry detection {detected:.4f} (reviewer: 0.83)"

    def test_symmetry_arm_rarely_voids_a_fair_arena(self) -> None:
        # The plan's stated property: chance of a VOID <= 0.8% when the arena is fair.
        hull, _ = _simulate(0.50, 0.47, pairs=20, reps=20000, confidence=0.99, seed=2)
        false_void = _void(hull)
        assert false_void <= 0.008, f"symmetry false VOID {false_void:.4f} (reviewer: 0.0047)"

    def test_positive_control_void_rate_stays_near_the_iid_rate(self) -> None:
        # 40 games (20 pairs), 95%: VOID when L <= 0.5 at the pre-B1 reference mu.
        hull, iid = _simulate(0.7375, 0.30, pairs=20, reps=20000, confidence=0.95, seed=3)
        void, void_iid = float(np.mean(hull[:, 0] <= 0.5)), float(np.mean(iid[:, 0] <= 0.5))
        message = (
            f"positive-control VOID hull {void:.4f}, iid {void_iid:.4f} (reviewer: 0.029 / 0.026)"
        )
        assert 0.020 <= void <= 0.040, message
        assert void - void_iid <= 0.01, message
