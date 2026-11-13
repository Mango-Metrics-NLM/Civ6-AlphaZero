"""The section 7.1 verdict: ``a1_protocol.verdict_from_bounds`` and ``compute_verdict`` (T14).

The table is tested on exact bounds (``verdict_from_bounds``), where the edges --
a bound of exactly 0.5 -- can be stated; ``compute_verdict`` is tested on
per-game records, against the benchmark's own interval code.

Negative control (run when written, then reverted): making the rule ignore the
symmetry hull failed ``test_symmetry_void_beats_a_primary_pass`` and
``test_symmetry_hull_below_half_voids``.
"""

from __future__ import annotations

import json

import pytest

from alpha_zero.evaluation import a1_protocol as a1
from alpha_zero.evaluation.benchmark_evidence import facing_hull_ratio, interval_summary
from alpha_zero.evaluation.scoring import minimum_detectable_effect

FAIR = (0.40, 0.60)
STRONG = (0.60, 0.90)
WEAK = (0.10, 0.40)


def decide(
    *,
    interruptions: tuple[str, ...] = (),
    symmetry: tuple[float, float] = FAIR,
    positive: tuple[float, float] = STRONG,
    primary: tuple[float, float] = FAIR,
) -> tuple[str, tuple[str, ...]]:
    return a1.verdict_from_bounds(
        interruptions=interruptions,
        symmetry_hull=symmetry,
        positive_control_hull=positive,
        primary_hull=primary,
    )


def games(pairs: list[tuple[str, str]], n: int) -> list[dict[str, object]]:
    out = []
    for p in range(n // 2):
        first, second = pairs[p % len(pairs)]
        out += [{"pair": p, "seat": 0, "result": first}, {"pair": p, "seat": 1, "result": second}]
    return out


SYMMETRIC = [("win", "loss"), ("loss", "win")]
LOPSIDED = [("win", "win")]
STRONG_GAMES = [("win", "win"), ("win", "draw"), ("win", "win"), ("draw", "loss")]
WEAK_GAMES = [("loss", "loss"), ("loss", "draw"), ("loss", "loss"), ("draw", "win")]
EVEN = [("win", "loss"), ("draw", "draw"), ("loss", "win"), ("draw", "draw")]


class TestVerdictTable:
    def test_order(self) -> None:
        assert a1.VERDICT_ORDER == ("VOID", "PASS", "FAIL", "INCONCLUSIVE")

    def test_pass_fail_inconclusive(self) -> None:
        assert decide(primary=STRONG)[0] == "PASS"
        assert decide(primary=WEAK)[0] == "FAIL"
        assert decide(primary=FAIR)[0] == "INCONCLUSIVE"

    def test_primary_low_exactly_half_is_not_a_pass(self) -> None:
        assert decide(primary=(0.5, 0.7))[0] == "INCONCLUSIVE"

    def test_primary_high_exactly_half_is_not_a_fail(self) -> None:
        assert decide(primary=(0.3, 0.5))[0] == "INCONCLUSIVE"

    def test_positive_control_low_exactly_half_voids(self) -> None:
        verdict, reasons = decide(positive=(0.5, 0.8), primary=STRONG)
        assert verdict == "VOID"
        assert "positive control" in reasons[0]

    def test_positive_control_just_above_half_does_not_void(self) -> None:
        assert decide(positive=(0.5000001, 0.8), primary=STRONG)[0] == "PASS"

    def test_symmetry_bound_exactly_half_does_not_void(self) -> None:
        assert decide(symmetry=(0.5, 0.7), primary=STRONG)[0] == "PASS"
        assert decide(symmetry=(0.3, 0.5), primary=STRONG)[0] == "PASS"

    def test_symmetry_void_beats_a_primary_pass(self) -> None:
        verdict, reasons = decide(symmetry=(0.51, 0.8), primary=STRONG)
        assert verdict == "VOID"
        assert any("symmetry" in r for r in reasons)

    def test_symmetry_hull_below_half_voids(self) -> None:
        assert decide(symmetry=(0.2, 0.49), primary=STRONG)[0] == "VOID"

    def test_an_interrupted_run_is_void_whatever_the_arms_say(self) -> None:
        verdict, reasons = decide(interruptions=("run.json status is 'VOID'",), primary=STRONG)
        assert verdict == "VOID"
        assert reasons == ("interrupted or resumed: run.json status is 'VOID'",)

    def test_every_void_condition_is_reported(self) -> None:
        verdict, reasons = decide(
            interruptions=("resumed",), symmetry=(0.6, 0.9), positive=(0.3, 0.6)
        )
        assert verdict == "VOID"
        assert len(reasons) == 3


class TestComputeVerdict:
    def test_a_pass(self) -> None:
        v = a1.compute_verdict(
            games(STRONG_GAMES, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40)
        )
        assert v.verdict == "PASS"
        assert v.primary_hull[0] > 0.5
        assert v.mde_alpha025_80pct is None and v.hull_ratio is None

    def test_a_fail(self) -> None:
        v = a1.compute_verdict(
            games(WEAK_GAMES, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40)
        )
        assert v.verdict == "FAIL"

    def test_inconclusive_reports_the_registered_mde(self) -> None:
        primary = games(EVEN, 200)
        v = a1.compute_verdict(primary, games(SYMMETRIC, 40), games(STRONG_GAMES, 40))
        assert v.verdict == "INCONCLUSIVE"
        mde = minimum_detectable_effect(200, v.primary_draw_rate, alpha=0.025)
        block = interval_summary(primary, 0.95)
        ratio = facing_hull_ratio(v.primary_score, block["iid"], block["hull"])
        assert v.mde_alpha025_80pct == pytest.approx(mde)
        assert v.hull_ratio == pytest.approx(ratio)
        assert v.mde_alpha025_80pct_hull == pytest.approx(mde * ratio)

    def test_hulls_are_read_at_the_registered_levels(self) -> None:
        sym, pc = games(SYMMETRIC, 40), games(STRONG_GAMES, 40)
        v = a1.compute_verdict(games(STRONG_GAMES, 200), sym, pc)
        assert list(v.symmetry_hull) == interval_summary(sym, 0.99)["hull"]
        assert list(v.positive_control_hull) == interval_summary(pc, 0.95)["hull"]

    def test_an_asymmetric_symmetry_arm_voids_a_primary_pass(self) -> None:
        v = a1.compute_verdict(
            games(STRONG_GAMES, 200), games(LOPSIDED, 40), games(STRONG_GAMES, 40)
        )
        assert v.verdict == "VOID"
        assert any("symmetry" in r for r in v.reasons)

    def test_a_weak_positive_control_voids(self) -> None:
        v = a1.compute_verdict(games(STRONG_GAMES, 200), games(SYMMETRIC, 40), games(EVEN, 40))
        assert v.verdict == "VOID"

    def test_an_interrupted_run_voids(self) -> None:
        v = a1.compute_verdict(
            games(STRONG_GAMES, 200),
            games(SYMMETRIC, 40),
            games(STRONG_GAMES, 40),
            interruptions=["iterations.jsonl line 11 was resumed"],
        )
        assert v.verdict == "VOID"

    def test_an_unknown_result_is_refused(self) -> None:
        bad = [{"pair": 0, "seat": 0, "result": "won"}, {"pair": 0, "seat": 1, "result": "win"}]
        with pytest.raises(ValueError, match="unknown"):
            a1.compute_verdict(bad, games(SYMMETRIC, 40), games(STRONG_GAMES, 40))

    @pytest.mark.parametrize("arm", [1, 2])
    def test_an_unknown_result_in_a_control_arm_is_a_value_error(self, arm: int) -> None:
        """T14 review M-7: every arm is validated, not only the primary."""
        arms = [games(STRONG_GAMES, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40)]
        arms[arm][0] = {**arms[arm][0], "result": "won"}
        with pytest.raises(ValueError, match="unknown"):
            a1.compute_verdict(*arms)

    def test_a_half_pair_is_refused(self) -> None:
        with pytest.raises(ValueError, match="incomplete"):
            a1.compute_verdict(
                games(STRONG_GAMES, 200)[:-1], games(SYMMETRIC, 40), games(STRONG_GAMES, 40)
            )

    def test_to_dict_is_json(self) -> None:
        v = a1.compute_verdict(games(EVEN, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40))
        data = json.loads(json.dumps(v.to_dict()))
        assert data["verdict"] == "INCONCLUSIVE"
        assert data["primary_hull"] == list(v.primary_hull)
