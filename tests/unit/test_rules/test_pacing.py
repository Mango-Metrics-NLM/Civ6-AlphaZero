"""``alpha_zero.rules.pacing`` and the ``pace`` subcommand (GM-K2)."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from alpha_zero.rules import pacing
from alpha_zero.rules.__main__ import main
from alpha_zero.rules.builtin import load_builtin
from alpha_zero.rules.loader import parse_ruleset
from alpha_zero.rules.pacing import (
    CAPSTONE_BAND_PERCENT,
    band_science_rates,
    full_tree_cost,
    pacing_report,
    prerequisite_set_cost,
    turns_to_research,
)
from tests.unit.test_rules._fixtures import BASE, mutate

REPO_ROOT = Path(__file__).resolve().parents[3]

DIAMOND = """\
schema_version: 1
id: diamond-test
techs:
  - {id: root, era: 1, cost: 20}
  - {id: left, era: 2, cost: 30, prerequisites: [root]}
  - {id: right, era: 2, cost: 40, prerequisites: [root]}
  - {id: top, era: 3, cost: 60, prerequisites: [left, right]}
  - {id: side, era: 1, cost: 7}
victory:
  capstone_tech: top
"""


def brute_force_rates(cost: int, max_turns: int, low: int, high: int) -> list[int]:
    """Independent oracle: exact rational turn counts, no integer-percent tricks."""
    lo, hi = Fraction(low, 100) * max_turns, Fraction(high, 100) * max_turns
    return [r for r in range(1, cost + 1) if lo <= math.ceil(Fraction(cost, r)) <= hi]


# ---- costs ------------------------------------------------------------------------


def test_builtin_costs_are_the_registered_numbers() -> None:
    spec = load_builtin("science-v1-draft")
    assert prerequisite_set_cost(spec, "scientific_theory") == 580
    assert full_tree_cost(spec) == 780


def test_a_diamond_counts_each_tech_once() -> None:
    spec = parse_ruleset(DIAMOND)
    # root is reached by two paths but paid once: 60 + 30 + 40 + 20, not 60+30+20+40+20.
    assert prerequisite_set_cost(spec, "top") == 150
    assert prerequisite_set_cost(spec, "root") == 20
    assert prerequisite_set_cost(spec, "side") == 7
    assert full_tree_cost(spec) == 157


def test_unknown_tech_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        prerequisite_set_cost(parse_ruleset(DIAMOND), "ghost")


# ---- turns_to_research ------------------------------------------------------------


@pytest.mark.parametrize(
    ("cost", "rate", "turns"),
    [
        (580, 58, 10),
        (580, 59, 10),
        (581, 58, 11),
        (10, 100, 1),
        (10, 10, 1),
        (11, 10, 2),
        (1, 1, 1),
    ],
)
def test_turns_to_research_is_a_ceiling(cost: int, rate: int, turns: int) -> None:
    assert turns_to_research(cost, rate) == turns


@pytest.mark.parametrize("rate", [0, -1, -50])
def test_non_positive_rate_raises(rate: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        turns_to_research(100, rate)


# ---- band_science_rates -----------------------------------------------------------


def test_registered_band_is_14_to_16_and_matches_brute_force() -> None:
    assert CAPSTONE_BAND_PERCENT == (60, 70)
    assert band_science_rates(580, 60) == (14, 16)
    oracle = brute_force_rates(580, 60, 60, 70)
    assert oracle == [14, 15, 16]
    # the neighbours really are outside: 13 -> 45 turns, 17 -> 35 turns
    assert turns_to_research(580, 13) == 45 and turns_to_research(580, 17) == 35


@pytest.mark.parametrize(
    ("cost", "max_turns"), [(100, 10), (580, 60), (780, 60), (37, 20), (1000, 100), (90, 25)]
)
def test_band_matches_brute_force_over_inputs(cost: int, max_turns: int) -> None:
    rates = brute_force_rates(cost, max_turns, 60, 70)
    expected = (rates[0], rates[-1]) if rates else None
    assert band_science_rates(cost, max_turns) == expected


def check_band_boundaries() -> None:
    """Resolved through the module, so the negative control's monkeypatch reaches it."""
    # cost 100, 10 turns: the band is exactly 6..7 turns. Rates 17-19 take 6 turns
    # (the lower end), 15-16 take 7 (the upper end), 14 takes 8 (out), 20 takes 5 (out).
    assert [turns_to_research(100, r) for r in (14, 15, 16, 17, 19, 20)] == [8, 7, 7, 6, 6, 5]
    assert pacing.band_science_rates(100, 10) == (15, 19)
    # a band of exactly one turn count is non-empty only if both ends are inclusive
    assert pacing.band_science_rates(100, 10, (70, 70)) == (15, 16)
    assert pacing.band_science_rates(100, 10, (60, 60)) == (17, 19)
    assert pacing.band_science_rates(580, 60) == (14, 16)


def test_both_band_ends_are_inclusive() -> None:
    check_band_boundaries()


def test_none_when_no_rate_fits() -> None:
    # cost 5 needs at least 5 turns (rate 1) but the band wants 6..7 turns
    assert band_science_rates(5, 10) is None
    # ceil(7/r) takes 7, 4, 3, 2, 2, 2, 1 turns: nothing lands in 5..6
    assert band_science_rates(7, 10, (50, 60)) is None
    assert brute_force_rates(7, 10, 50, 60) == []


# ---- pacing_report ----------------------------------------------------------------


def test_report_for_the_builtin() -> None:
    report = pacing_report(load_builtin("science-v1-draft"), 60)
    assert report.ruleset_id == "science-v1-draft"
    assert report.max_turns == 60
    assert report.full_tree_cost == 780
    assert report.capstone_tech == "scientific_theory"
    assert report.capstone_cost == 580
    assert report.band_percent == (60, 70)
    assert report.band_rates == (14, 16)


def test_report_without_a_capstone_has_only_the_tree_cost() -> None:
    text = mutate("victory:\n  capstone_tech: beta\n", "")
    report = pacing_report(parse_ruleset(text), 60)
    assert report.full_tree_cost == 30
    assert report.capstone_tech is None
    assert report.capstone_cost is None
    assert report.band_rates is None


@pytest.mark.parametrize("max_turns", [0, -1, -60])
def test_non_positive_max_turns_raises(max_turns: int) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        pacing_report(parse_ruleset(BASE), max_turns)


# ---- the CLI ----------------------------------------------------------------------


def test_pace_builtin_exits_zero_and_reports_both_numbers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["pace", "science-v1-draft"]) == 0
    out = capsys.readouterr().out
    assert "580" in out and "780" in out
    assert "14 to 16 science per turn" in out
    assert "60-70% of 60 turns" in out


def test_pace_max_turns_changes_the_band(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pace", "science-v1-draft", "--max_turns", "120"]) == 0
    out = capsys.readouterr().out
    assert "of 120 turns" in out
    expected = brute_force_rates(580, 120, 60, 70)
    assert f"{expected[0]} to {expected[-1]} science per turn" in out


@pytest.mark.parametrize("value", ["0", "-5"])
def test_pace_non_positive_max_turns_exits_two(
    value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["pace", "science-v1-draft", "--max_turns", value]) == 2
    assert "max_turns" in capsys.readouterr().err


def test_pace_unknown_target_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["pace", "no-such-ruleset"]) == 2
    assert "no-such-ruleset" in capsys.readouterr().err


def test_pace_invalid_file_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(mutate("prerequisites: [alpha]", "prerequisites: [ghost]"), encoding="utf-8")
    assert main(["pace", str(path)]) == 1
    assert "unresolved-prerequisite" in capsys.readouterr().err


def test_pace_malformed_file_exits_one(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("key: [unclosed", encoding="utf-8")
    assert main(["pace", str(path)]) == 1


def test_pace_without_a_capstone_exits_zero_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "nocap.yaml"
    path.write_text(mutate("victory:\n  capstone_tech: beta\n", ""), encoding="utf-8")
    assert main(["pace", str(path)]) == 0
    out = capsys.readouterr().out
    assert "capstone: none declared" in out
    assert "full-tree cost: 30" in out


def test_pace_usage_errors_exit_two() -> None:
    for argv in (["pace"], ["pace", "x", "--max_turns", "abc"], ["pace", "x", "--bogus"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2


def test_pace_runs_for_real_as_a_subprocess() -> None:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    done = subprocess.run(
        [sys.executable, "-m", "alpha_zero.rules", "pace", "science-v1-draft"],
        capture_output=True,
        text=True,
        env=env,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert done.returncode == 0, done.stderr
    assert "580" in done.stdout and "780" in done.stdout


# ---- negative control: the boundary test fails on an exclusive band ---------------


def _exclusive_low(
    cost: int, max_turns: int, band: tuple[int, int] = (60, 70)
) -> tuple[int, int] | None:
    rates = [
        r
        for r in range(1, max(cost, 1) + 1)
        if band[0] * max_turns < turns_to_research(cost, r) * 100 <= band[1] * max_turns
    ]
    return (rates[0], rates[-1]) if rates else None


def _exclusive_high(
    cost: int, max_turns: int, band: tuple[int, int] = (60, 70)
) -> tuple[int, int] | None:
    rates = [
        r
        for r in range(1, max(cost, 1) + 1)
        if band[0] * max_turns <= turns_to_research(cost, r) * 100 < band[1] * max_turns
    ]
    return (rates[0], rates[-1]) if rates else None


@pytest.mark.parametrize("mutant", [_exclusive_low, _exclusive_high])
def test_control_boundary_test_fails_when_a_band_end_is_exclusive(
    monkeypatch: pytest.MonkeyPatch, mutant: object
) -> None:
    check_band_boundaries()  # passes against the real implementation
    monkeypatch.setattr(pacing, "band_science_rates", mutant)
    with pytest.raises(AssertionError):
        check_band_boundaries()


def test_pace_says_so_when_no_whole_rate_fits_the_band(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["pace", "science-v1-draft", "--max_turns", "1"]) == 0
    out = capsys.readouterr().out
    assert "no whole rate lands in the band" in out
    assert "580" in out
