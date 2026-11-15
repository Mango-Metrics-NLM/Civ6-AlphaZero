"""The section 7.1 verdict: ``a1_protocol.verdict_from_bounds`` and ``compute_verdict`` (T14).

The table is tested on exact bounds (``verdict_from_bounds``), where the edges --
a bound of exactly 0.5 -- can be stated; ``compute_verdict`` is tested on
per-game records, against the benchmark's own interval code.

Negative control (run when written, then reverted): making the rule ignore the
symmetry hull failed ``test_symmetry_void_beats_a_primary_pass`` and
``test_symmetry_hull_below_half_voids``.

Negative controls for ``TestNonReproducingReRun`` (2026-11-15; each defect was
planted in ``compute_verdict`` / ``A1Verdict.to_dict``, the listed tests were seen
to FAIL, and the file was restored and checked byte-identical before the next):

1. a non-reproducing PASS is not demoted (the ``verdict = VERDICT_INCONCLUSIVE``
   line removed) -> ``test_a_pass_is_read_as_inconclusive_and_reports_its_mde``
   failed;
2. the MDE block skipped for a non-reproducing re-run (``and not
   non_reproducing`` added to its condition; for a demoted PASS that is what
   placing the demotion below the block does) ->
   ``test_a_pass_is_read_as_inconclusive_and_reports_its_mde`` and the
   INCONCLUSIVE case of ``test_other_verdicts_keep_their_verdict_and_carry_the_label``
   failed (the MDE fields were ``None``);
3. the label set only when a PASS was demoted -> the three
   ``test_other_verdicts_keep_their_verdict_and_carry_the_label`` cases failed;
4. ``to_dict`` always emits ``label`` -> ``test_a_plain_verdict_dict_has_no_label_key``
   failed.
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


#: Why a re-run is non-reproducing, as ``scripts/release_reproduction.py`` words it.
WHY = ("it does not reproduce a1-20261111: checkpoint checkpoint_5.pt: 1 of 5 tensors differ",)
#: Every key a verdict dict had before the label existed.
PLAIN_KEYS = {
    "verdict",
    "reasons",
    "primary_score",
    "primary_draw_rate",
    "primary_hull",
    "symmetry_hull",
    "positive_control_hull",
    "mde_alpha025_80pct",
    "hull_ratio",
    "mde_alpha025_80pct_hull",
}


class TestNonReproducingReRun:
    """Section 7.1, "After a VOID": a non-reproducing re-run is labelled and cannot PASS."""

    def test_a_pass_is_read_as_inconclusive_and_reports_its_mde(self) -> None:
        arms = (games(STRONG_GAMES, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40))
        plain = a1.compute_verdict(*arms)
        assert plain.verdict == "PASS" and plain.label is None
        v = a1.compute_verdict(*arms, non_reproducing=WHY)
        assert v.verdict == "INCONCLUSIVE"
        assert v.label == a1.NON_REPRODUCING_LABEL == "non-reproducing re-run"
        # The arm is what it was: only how its row is read changed.
        assert v.primary_hull == plain.primary_hull and v.primary_hull[0] > 0.5
        mde = minimum_detectable_effect(200, v.primary_draw_rate, alpha=0.025)
        block = interval_summary(arms[0], 0.95)
        ratio = facing_hull_ratio(v.primary_score, block["iid"], block["hull"])
        assert v.mde_alpha025_80pct == pytest.approx(mde)
        assert v.hull_ratio == pytest.approx(ratio)
        assert v.mde_alpha025_80pct_hull == pytest.approx(mde * ratio)
        assert plain.reasons[0] in v.reasons, "the PASS row it would have been is still reported"
        assert any(WHY[0] in reason for reason in v.reasons)
        data = v.to_dict()
        assert data["label"] == "non-reproducing re-run" and data["verdict"] == "INCONCLUSIVE"
        assert set(data) == PLAIN_KEYS | {"label"}

    @pytest.mark.parametrize(
        ("verdict", "primary", "symmetry"),
        [
            ("VOID", STRONG_GAMES, LOPSIDED),
            ("FAIL", WEAK_GAMES, SYMMETRIC),
            ("INCONCLUSIVE", EVEN, SYMMETRIC),
        ],
    )
    def test_other_verdicts_keep_their_verdict_and_carry_the_label(
        self, verdict: str, primary: list[tuple[str, str]], symmetry: list[tuple[str, str]]
    ) -> None:
        arms = (games(primary, 200), games(symmetry, 40), games(STRONG_GAMES, 40))
        plain = a1.compute_verdict(*arms)
        v = a1.compute_verdict(*arms, non_reproducing=WHY)
        assert plain.verdict == v.verdict == verdict
        assert plain.label is None and v.label == a1.NON_REPRODUCING_LABEL
        assert v.reasons[: len(plain.reasons)] == plain.reasons
        assert any(WHY[0] in reason for reason in v.reasons)
        assert (v.mde_alpha025_80pct, v.hull_ratio) == (plain.mde_alpha025_80pct, plain.hull_ratio)
        assert v.to_dict()["label"] == a1.NON_REPRODUCING_LABEL

    @pytest.mark.parametrize("primary", [STRONG_GAMES, WEAK_GAMES, EVEN])
    def test_a_plain_verdict_dict_has_no_label_key(self, primary: list[tuple[str, str]]) -> None:
        """Additive: a verdict dict written before the label existed is still what is written."""
        v = a1.compute_verdict(games(primary, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40))
        assert v.label is None
        assert set(v.to_dict()) == PLAIN_KEYS
        empty = a1.compute_verdict(
            games(primary, 200), games(SYMMETRIC, 40), games(STRONG_GAMES, 40), non_reproducing=()
        )
        assert empty == v
