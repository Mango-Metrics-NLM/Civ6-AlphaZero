"""Aggregation, the stop rule and the sampling of the expressivity census (GM-B11).

Spec: ``kb/specs/20261123-expressivity-census-preregistration.md``, sections "Aggregation",
"Stop rule" and "Input required for a real answer". Records are written by hand here, so
every median, quartile and rank is counted from the specification.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.census_states import agg, load_census, record


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


# -- the order-statistic interval ----------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "ranks"),
    [(40, (14, 27)), (20, (6, 15)), (6, (1, 6)), (10, (2, 9))],
)
def test_median_interval_ranks_match_the_registered_values(
    mod: Any, n: int, ranks: tuple[int, int]
) -> None:
    # N=40 -> 14 and 27, N=20 -> 6 and 15 (the specification). N=6: P(X<=0) = 1/64 <= 0.025
    # and P(X<=1) = 7/64 > 0.025, so rank 1 and 6. N=10: P(X<=1) = 11/1024 <= 0.025 and
    # P(X<=2) = 56/1024 > 0.025, so rank 2 and 9.
    assert mod.median_interval_ranks(n) == ranks


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_median_interval_has_no_ranks_below_six_values(mod: Any, n: int) -> None:
    # Even the extremes cover the median with probability 1 - 2/2**n < 0.95 for n < 6
    # (n = 5: 1 - 1/16 = 0.9375): there is no 95% interval, and the script says None.
    assert mod.median_interval_ranks(n) is None


def test_aggregate_of_one_to_forty(mod: Any) -> None:
    result = mod.aggregate([float(i) for i in range(1, 41)])
    # Median of 1..40 = 20.5; numpy linear percentiles: 1 + 0.25 * 39 = 10.75 and
    # 1 + 0.75 * 39 = 30.25; ranks 14 and 27 are the values 14 and 27.
    assert result["median"] == 20.5
    assert result["q25"] == pytest.approx(10.75) and result["q75"] == pytest.approx(30.25)
    assert tuple(result["median_ci95_ranks"]) == (14, 27)
    assert result["median_ci95"] == [14.0, 27.0]
    assert result["states_with_value"] == 40 and result["states_excluded"] == 0


def test_aggregate_of_one_to_twenty_uses_ranks_six_and_fifteen(mod: Any) -> None:
    assert mod.aggregate([float(i) for i in range(1, 21)])["median_ci95"] == [6.0, 15.0]


def test_aggregate_skips_none_and_ignores_input_order(mod: Any) -> None:
    values: list[float | None] = [None, 0.9, 0.1, None, 0.5, 0.3, 0.7]
    result = mod.aggregate(values)
    # Five values {0.1, 0.3, 0.5, 0.7, 0.9}: median 0.5; q25 at 0.25*4 = 1 -> 0.3; q75 -> 0.7.
    assert result["median"] == 0.5 and result["q25"] == pytest.approx(0.3)
    assert result["q75"] == pytest.approx(0.7)
    assert (result["states_with_value"], result["states_excluded"]) == (5, 2)
    assert result["median_ci95"] is None  # five values: no 95% interval


def test_aggregate_of_nothing_has_no_median(mod: Any) -> None:
    for values in ([], [None, None]):
        result = mod.aggregate(values)
        assert result["median"] is None and result["median_ci95"] is None
        assert result["states_with_value"] == 0


# -- aggregate_board: after turn 30, per source, void, by kind ------------------------------


def _board_records() -> list[dict[str, Any]]:
    kinds = {"unit_move": (1, 2)}
    recs = [
        record(0, 28, e1=0.1, u=0.2, by_kind=kinds),
        record(0, 29, e1=0.2, u=0.3, by_kind=kinds),
        record(0, 30, e1=0.3, u=0.4, by_kind=kinds),  # turn 30 is not after turn 30
        record(0, 31, e1=0.4, u=0.5, by_kind=kinds),
        record(0, 32, e1=0.5, u=0.6, by_kind=kinds),
        record(1, 29, e1=0.6, u=0.7, by_kind=kinds),
        record(1, 31, e1=0.7, u=0.8, by_kind={"unit_move": (1, 2), "unit_skip": (1, 1)}),
        record(1, 40, e1=0.8, u=0.9, by_kind=kinds),
        record(1, 35, e1=None, u=None, by_kind=kinds),  # no E1 value: excluded, not 0
        record(1, 33, void=True),
    ]
    return recs


def test_aggregate_board_all_states(mod: Any) -> None:
    out = mod.aggregate_board(_board_records(), 8)
    assert out["states"] == 9 and out["states_void"] == 1
    e1 = out["E1"]["all"]
    # 8 values {0.1..0.8}: median 0.45; q25 at 0.25 * 7 = 1.75 -> 0.2 + 0.75 * 0.1 = 0.275;
    # q75 at 5.25 -> 0.6 + 0.25 * 0.1 = 0.625; one state (turn 35) has no value.
    assert e1["median"] == pytest.approx(0.45)
    assert e1["q25"] == pytest.approx(0.275) and e1["q75"] == pytest.approx(0.625)
    assert (e1["states_with_value"], e1["states_excluded"]) == (8, 1)
    # U is aggregated the same way: {0.2..0.9} -> median 0.55.
    assert out["U"]["all"]["median"] == pytest.approx(0.55)


def test_aggregate_board_after_turn_30_is_strictly_after(mod: Any) -> None:
    late = mod.aggregate_board(_board_records(), 8)["E1"]["after_turn_30"]
    # Turns 31 (0.4), 32 (0.5), 31 (0.7), 40 (0.8) and 35 (no value); turn 30 (0.3) is out
    # and so is the void turn 33. Median of {0.4, 0.5, 0.7, 0.8} = 0.6.
    assert late["median"] == pytest.approx(0.6)
    assert (late["states_with_value"], late["states_excluded"]) == (4, 1)


def test_aggregate_board_per_source_medians(mod: Any) -> None:
    per = mod.aggregate_board(_board_records(), 8)["E1"]["per_source_median"]
    # Source 0: {0.1, 0.2, 0.3, 0.4, 0.5} -> 0.3. Source 1: {0.6, 0.7, 0.8} -> 0.7.
    assert per == {"0": pytest.approx(0.3), "1": pytest.approx(0.7)}


def test_aggregate_board_pools_the_e1_split_by_kind(mod: Any) -> None:
    pooled = mod.aggregate_board(_board_records(), 8)["E1_by_kind_pooled"]
    # Nine usable states, each 1/2 for unit_move -> 9/18; one state adds skip 1/1.
    assert (pooled["unit_move"]["numerator"], pooled["unit_move"]["denominator"]) == (9, 18)
    assert (pooled["unit_skip"]["numerator"], pooled["unit_skip"]["denominator"]) == (1, 1)


def test_aggregate_board_reads_only_the_requested_board(mod: Any) -> None:
    rec = record(0, 1, board="8", e1=0.1)
    rec["boards"] |= record(0, 1, board="16", e1=0.9)["boards"]
    assert mod.aggregate_board([rec], 8)["E1"]["all"]["median"] == 0.1
    assert mod.aggregate_board([rec], 16)["E1"]["all"]["median"] == 0.9


# -- the stop rule -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("e1", "e2", "verdict"),
    [
        (0.29, 0.8, "stop"),
        (0.3, 0.8, "no_stop"),  # exactly 0.3 does not stop: the rule is "< 0.3"
        (0.31, 0.8, "no_stop"),
        (0.8, 0.49, "stop"),
        (0.8, 0.5, "no_stop"),  # exactly 0.5 does not stop
        (0.8, 0.51, "no_stop"),
        (0.1, 0.1, "stop"),
        (0.3, 0.5, "no_stop"),
    ],
)
def test_stop_rule_thresholds(mod: Any, e1: float, e2: float, verdict: str) -> None:
    result = mod.stop_rule_verdict(agg(e1), agg(e2))
    assert result["verdict"] == verdict
    assert result["board"] == 8


def test_stop_rule_reads_which_condition_fired(mod: Any) -> None:
    result = mod.stop_rule_verdict(agg(0.2), agg(0.9))
    assert result["E1"]["fires"] is True and result["E2"]["fires"] is False
    result = mod.stop_rule_verdict(agg(0.9), agg(0.2))
    assert result["E1"]["fires"] is False and result["E2"]["fires"] is True


def test_borderline_flags_but_the_rule_applies_as_written(mod: Any) -> None:
    # Interval [0.25, 0.45] holds 0.3 but the median 0.35 is above it: no stop, borderline.
    above = mod.stop_rule_verdict(agg(0.35, 0.25, 0.45), agg(0.8, 0.7, 0.9))
    assert above["verdict"] == "no_stop" and above["borderline"] is True
    # Median 0.25 under the threshold with the interval across it: still a stop, borderline.
    below = mod.stop_rule_verdict(agg(0.25, 0.2, 0.4), agg(0.8, 0.7, 0.9))
    assert below["verdict"] == "stop" and below["borderline"] is True
    # E2's threshold is 0.5.
    e2 = mod.stop_rule_verdict(agg(0.9, 0.8, 1.0), agg(0.45, 0.4, 0.6))
    assert e2["verdict"] == "stop" and e2["borderline"] is True


def test_an_interval_clear_of_both_thresholds_is_not_borderline(mod: Any) -> None:
    result = mod.stop_rule_verdict(agg(0.35, 0.31, 0.4), agg(0.8, 0.6, 0.9))
    assert result["verdict"] == "no_stop" and result["borderline"] is False
    no_interval = mod.stop_rule_verdict(agg(0.35), agg(0.8))
    assert no_interval["borderline"] is False


def test_a_missing_median_is_never_read_as_no_stop(mod: Any) -> None:
    assert mod.stop_rule_verdict(agg(None), agg(0.8))["verdict"] not in ("no_stop", "stop")
    # Either condition firing decides it, whatever the other one has.
    assert mod.stop_rule_verdict(agg(None), agg(0.4))["verdict"] == "stop"
    assert mod.stop_rule_verdict(agg(None), agg(None))["verdict"] not in ("no_stop", "stop")


@pytest.mark.parametrize(("median_u", "fires"), [(0.29, True), (0.3, False), (0.9, False)])
def test_offer_mix_alone_flag_follows_median_u_below_0_3(
    mod: Any, median_u: float, fires: bool
) -> None:
    result = mod.stop_rule_verdict(agg(0.2), agg(0.9), agg(median_u))
    assert result["e1_fires_from_the_offer_mix_alone"] is fires
    assert result["median_u"] == median_u
    assert mod.stop_rule_verdict(agg(0.2), agg(0.9))["e1_fires_from_the_offer_mix_alone"] is None


# -- sampling: the first complete state of each game turn ----------------------------------


def _s(turn: Any, tag: str) -> dict[str, Any]:
    return {"turn_id": turn, "tag": tag}


def test_first_state_of_each_turn_is_kept_in_order(mod: Any) -> None:
    messages = [_s(1, "a"), _s(1, "b"), _s(2, "c"), _s(2, "d"), _s(2, "e"), _s(3, "f")]
    kept, counts = mod.first_state_per_turn(messages)
    assert [s["tag"] for s in kept] == ["a", "c", "f"]  # the first of each, never the last
    assert counts["states_read"] == 6 and counts["states_without_turn"] == 0


def test_a_turn_is_never_sampled_twice_and_order_is_the_log_order(mod: Any) -> None:
    # A reload goes back to turn 2 and plays on; turn 2 is already sampled.
    messages = [_s(5, "a"), _s(6, "b"), _s(7, "c"), _s(6, "d"), _s(8, "e")]
    kept, counts = mod.first_state_per_turn(messages)
    assert [s["tag"] for s in kept] == ["a", "b", "c", "e"]
    assert counts["turn_went_back"] == 1
    out_of_order, _ = mod.first_state_per_turn([_s(9, "x"), _s(4, "y")])
    assert [s["tag"] for s in out_of_order] == ["x", "y"]  # log order, not turn order


def test_wrapped_states_are_unwrapped_and_turnless_ones_are_not_sampled(mod: Any) -> None:
    messages = [
        {"type": "state", "state": _s(1, "wrapped")},
        {"tag": "no turn"},
        _s("2", "string turn"),
        _s(True, "bool turn"),
        _s(3, "bare"),
    ]
    kept, counts = mod.first_state_per_turn(messages)
    assert [s["tag"] for s in kept] == ["wrapped", "bare"]
    assert counts["states_read"] == 5 and counts["states_without_turn"] == 3


def test_a_message_that_is_not_an_object_is_malformed_input(mod: Any) -> None:
    with pytest.raises(mod.MalformedInputError):
        mod.first_state_per_turn([_s(1, "a"), [1, 2, 3]])


def test_each_source_is_sampled_on_its_own(mod: Any, tmp_path: Any) -> None:
    from tests.fixtures.census_states import end_turn_only, wire, write_json

    paths = []
    for name in ("one", "two"):
        states = [wire(end_turn_only(turn)) for turn in (1, 1, 2)]  # turn 1 twice
        paths.append(write_json(tmp_path / f"{name}.json", states))
    result = mod.census([("states_json", p) for p in paths], [8])
    # Two sources, each: turn 1 once and turn 2 once.
    assert [(r["source"], r["turn"]) for r in result["states"]] == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert [s["states_read"] for s in result["sources"]] == [3, 3]
