"""What may be called a census result, and what the stop-rule field says otherwise (GM-B11).

Spec: ``kb/specs/20261123-expressivity-census-preregistration.md``, implementation notes 8 and
10: the registered words (``stop``, ``no_stop``, ``undetermined``) appear under ``verdict``
only for a census result; on any other input ``verdict`` is ``not_decided`` and the computed
word is kept under ``verdict_on_this_input_unregistered``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.census_states import (
    ATTEST,
    cell,
    load_census,
    read_json,
    run_cli,
    state_line,
    wire_states,
    write_json,
    write_log,
)

REGISTERED_WORDS = {"stop", "no_stop", "undetermined"}


@pytest.fixture(scope="module")
def mod() -> Any:
    return load_census()


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", "0123456789abcdef0123456789abcdef01234567")


def _log(path: Path, turns: Sequence[int], **kw: Any) -> Path:
    return write_log(path, [state_line(s) for s in wire_states(turns, **kw)])


def test_a_qualifying_log_is_a_census_result_and_carries_the_registered_word(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = _log(tmp_path / "Lua.log", range(1, 41))  # 40 states, turns 31..40 after turn 30
    code, target = run_cli(mod, tmp_path, "--log", str(log), *ATTEST)  # both default boards
    assert code == 0
    data = read_json(target)
    info, stop = data["input"], data["stop_rule"]
    assert info["is_census_result"] is True and info["not_a_census_reasons"] == []
    assert info["states_after_turn_30"] == 10 and info["states_measured"] == 40
    # Every state is the hand-counted one: E1 = 8/12, E2 = 2/3; neither stop condition fires.
    at8 = data["aggregates"]["8"]
    assert at8["E1"]["all"]["median"] == pytest.approx(8 / 12)
    assert at8["E2"]["all"]["median"] == pytest.approx(2 / 3)
    assert at8["E1"]["all"]["median_ci95_ranks"] == [14, 27]
    assert "16" in data["aggregates"]
    assert stop["verdict"] == "no_stop" and stop["decided_by_this_input"] is True
    assert "verdict_on_this_input_unregistered" not in stop
    out = capsys.readouterr().out
    assert "NOT a census result" not in out and "Stop rule at board 8" in out
    assert "NOT DECIDED" not in out


def _log_args(tmp: Path, turns: Sequence[int] = range(1, 41), **kw: Any) -> list[str]:
    return ["--log", str(_log(tmp / "Lua.log", turns, **kw))]


# Each case misses exactly one condition; the reason must name it.
MISSING: dict[str, tuple[Callable[[Path], list[str]], str]] = {
    "only board 8": (lambda t: [*_log_args(t), "--boards", "8", *ATTEST], "boards 8 and 16"),
    "no mod version": (lambda t: [*_log_args(t), "--played-by", "greedy policy"], "--mod-version"),
    "empty mod version": (
        lambda t: [*_log_args(t), "--mod-version", "  ", "--played-by", "greedy policy"],
        "--mod-version",
    ),
    "no played-by": (lambda t: [*_log_args(t), "--mod-version", "0.9.1"], "--played-by"),
    "a states json": (
        lambda t: [
            "--states-json",
            str(write_json(t / "s.json", wire_states(range(1, 41)))),
            *ATTEST,
        ],
        "input is states_json",
    ),
    "an older mod": (lambda t: [*_log_args(t, schema=1), *ATTEST], "older mod"),
    "no observes_others": (lambda t: [*_log_args(t, observes=None), *ATTEST], "older mod"),
    "below the minimum": (
        lambda t: [*_log_args(t, range(1, 4)), "--allow-small", *ATTEST],
        "registered minimum",
    ),
}


@pytest.mark.parametrize("case", sorted(MISSING))
def test_each_missing_condition_is_named_and_the_verdict_is_not_decided(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    build, reason = MISSING[case]
    code, target = run_cli(mod, tmp_path, *build(tmp_path))
    assert code == 0
    data = read_json(target)
    info, stop = data["input"], data["stop_rule"]
    assert info["is_census_result"] is False
    assert len(info["not_a_census_reasons"]) == 1 and reason in info["not_a_census_reasons"][0]
    # The registered words are never the verdict of an input that is not a census result.
    assert stop["verdict"] == "not_decided" and stop["decided_by_this_input"] is False
    assert stop["verdict_on_this_input_unregistered"] in REGISTERED_WORDS
    out = capsys.readouterr().out
    would = stop["verdict_on_this_input_unregistered"]
    assert f"Stop rule: NOT DECIDED (this input would give {would}" in out
    # The caveat comes before any number.
    assert out.index("NOT a census result") < out.index("board 8 E1")
    assert out.index("NOT a census result") < out.index("States:")
    assert reason in out


def test_every_missing_condition_gets_its_own_reason(mod: Any, tmp_path: Path) -> None:
    log = _log(tmp_path / "Lua.log", [1, 2, 3])
    code, target = run_cli(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    reasons = read_json(target)["input"]["not_a_census_reasons"]
    assert len(reasons) == 4  # minimum, boards, mod version, who played
    for word in ("registered minimum", "boards 8 and 16", "--mod-version", "--played-by"):
        assert sum(word in r for r in reasons) == 1


def _fake_measure(per_board: dict[int, dict[str, float]]) -> Any:
    def measure(_state: Any, board: int) -> dict[str, Any]:
        return cell(**per_board[board])

    return measure


BOARD_CASES = [
    ({"e1": 0.1, "e2": 0.9}, {"e1": 0.95, "e2": 0.95}, "stop"),  # a good B=16 does not lift
    ({"e1": 0.9, "e2": 0.9}, {"e1": 0.05, "e2": 0.05}, "no_stop"),  # a bad B=16 does not trigger
    ({"e1": 0.9, "e2": 0.2}, {"e1": 0.9, "e2": 0.9}, "stop"),
]


@pytest.mark.parametrize(("low8", "high16", "word"), BOARD_CASES)
def test_board_16_neither_triggers_nor_lifts_the_stop_on_a_states_json(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, low8: Any, high16: Any, word: str
) -> None:
    monkeypatch.setattr(mod, "measure_state", _fake_measure({8: low8, 16: high16}))
    states = write_json(tmp_path / "s.json", wire_states(range(1, 4)))
    code, target = run_cli(mod, tmp_path, "--states-json", str(states), "--allow-small")
    assert code == 0
    data = read_json(target)
    assert data["stop_rule"]["verdict"] == "not_decided"
    assert data["stop_rule"]["verdict_on_this_input_unregistered"] == word
    assert data["stop_rule"]["board"] == 8
    assert data["aggregates"]["16"]["E1"]["all"]["median"] == pytest.approx(high16["e1"])


@pytest.mark.parametrize(("low8", "high16", "word"), BOARD_CASES)
def test_board_16_neither_triggers_nor_lifts_the_stop_on_a_qualifying_log(
    mod: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, low8: Any, high16: Any, word: str
) -> None:
    monkeypatch.setattr(mod, "measure_state", _fake_measure({8: low8, 16: high16}))
    log = _log(tmp_path / "Lua.log", range(1, 41))
    code, target = run_cli(mod, tmp_path, "--log", str(log), *ATTEST)
    assert code == 0
    data = read_json(target)
    assert data["input"]["is_census_result"] is True
    assert data["stop_rule"]["verdict"] == word and data["stop_rule"]["board"] == 8
    assert "verdict_on_this_input_unregistered" not in data["stop_rule"]
    assert data["aggregates"]["16"]["E2"]["all"]["median"] == pytest.approx(high16["e2"])


def test_without_board_8_the_stop_rule_says_why_it_is_not_computed(
    mod: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    states = write_json(tmp_path / "s.json", wire_states(range(1, 4)))
    code, target = run_cli(
        mod, tmp_path, "--states-json", str(states), "--allow-small", "--boards", "16"
    )
    assert code == 0
    data = read_json(target)
    stop = data["stop_rule"]
    assert stop["verdict"] == "not_decided"
    assert stop["verdict_on_this_input_unregistered"] is None
    assert stop["not_computed"] == "board 8 was not requested"
    assert data["exploratory"] is None
    assert "Stop rule: NOT DECIDED (board 8 was not requested)" in capsys.readouterr().out
