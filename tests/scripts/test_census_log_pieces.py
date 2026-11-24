"""Pieces of two exports, unreadable states and their counts (GM-B11, notes 3 and 4).

``iter_log_states`` yields a pieced state only when its pieces are exactly ``1..n`` of its end
line, and gives an export up (counted) when the next export has plainly begun. Every export
here is built so that the wrongly joined text WOULD parse as JSON: only the piece rules can
stop it, not the JSON parser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import DROPPED_MISSING_PIECE, UNREADABLE_STATE, iter_log_states
from tests.fixtures.bridge_search_states import make_state
from tests.fixtures.census_states import (
    chunk_lines,
    load_census,
    read_json,
    run_cli,
    state_line,
    wire,
    write_log,
)


def _export(turn: int, fill: str) -> list[str]:
    """Three pieces and an end line of one long string: a mix of two of them still parses."""
    return chunk_lines({"turn_id": turn, "pad": fill * 900}, pieces=3)


def _body(line: str) -> str:
    return line.split(":", 2)[2]


def _read(path: Path) -> tuple[list[int], dict[str, int]]:
    counts: dict[str, int] = {}
    turns = [s["turn_id"] for s in iter_log_states(path, counts)]
    return turns, counts


def test_pieces_of_two_exports_are_not_joined(tmp_path: Path) -> None:
    a, b = _export(1, "a"), _export(2, "b")
    # Export A: pieces 1/3 2/3 3/3 and no end line. Export B: no piece 1/3.
    wrongly_joined = _body(a[0]) + _body(b[1]) + _body(b[2])
    mixed = json.loads(wrongly_joined)  # sanity: this WOULD have been yielded as a state
    assert mixed["turn_id"] == 1 and "a" in mixed["pad"] and "b" in mixed["pad"]
    lines = [
        *a[:3],  # A without its end line
        *b[1:],  # B's 2/3, 3/3 and END:3
        state_line({"turn_id": 5}),
        "AZ_STATE:{this is not json",
        *chunk_lines({"turn_id": 6, "tiles": [{"x": i} for i in range(12)]}, pieces=2),
        a[0].replace("1/3", "1/2"),  # a lone piece 1/2 and then the log ends
    ]
    turns, counts = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [5, 6]
    assert counts == {DROPPED_MISSING_PIECE: 3, UNREADABLE_STATE: 1}


def test_a_piece_whose_total_differs_ends_the_export_before_it(tmp_path: Path) -> None:
    a = _export(1, "a")
    lines = [a[0], a[1], "AZ_STATE_CHUNK:3/4:" + _body(a[2]), "AZ_STATE_END:4"]
    turns, counts = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [] and counts == {DROPPED_MISSING_PIECE: 2}


def test_a_whole_state_in_the_middle_of_a_pieced_one_ends_it(tmp_path: Path) -> None:
    a = _export(1, "a")
    lines = [a[0], a[1], state_line({"turn_id": 7})]
    turns, counts = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [7] and counts == {DROPPED_MISSING_PIECE: 1}


def test_a_complete_pieced_state_is_unaltered_and_not_counted(tmp_path: Path) -> None:
    lines = [*_export(1, "a"), *_export(2, "b"), state_line({"turn_id": 3})]
    states = list(iter_log_states(write_log(tmp_path / "Lua.log", lines)))
    assert [s["turn_id"] for s in states] == [1, 2, 3]
    assert states[0]["pad"] == "a" * 900 and states[1]["pad"] == "b" * 900


def test_unreadable_text_is_counted_apart_from_a_missing_piece(tmp_path: Path) -> None:
    whole = chunk_lines({"turn_id": 1}, pieces=1)
    text = "{not json at all"
    pieced_but_garbage = [f"AZ_STATE_CHUNK:1/2:{text[:6]}", f"AZ_STATE_CHUNK:2/2:{text[6:]}"]
    lines = ["AZ_STATE:{cut off", *pieced_but_garbage, "AZ_STATE_END:2", *whole]
    turns, counts = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [1] and counts == {UNREADABLE_STATE: 2}


def test_known_limit_an_export_that_lost_its_tail_joined_to_one_that_lost_its_head(
    tmp_path: Path,
) -> None:
    """Pieces carry no export id, so this is NOT detected today and the join is yielded.

    Export A loses piece 3/3 and its end line; export B loses pieces 1/3 and 2/3. What is
    left reads 1/3, 2/3, 3/3, END:3, in order. The text parses, so a state that never
    existed (A's head, B's tail) is yielded and nothing is counted. This test pins that.
    """
    a, b = _export(1, "a"), _export(2, "b")
    lines = [a[0], a[1], b[2], b[3]]
    counts: dict[str, int] = {}
    states = list(iter_log_states(write_log(tmp_path / "Lua.log", lines), counts))
    assert len(states) == 1 and counts == {}
    assert "a" in states[0]["pad"] and "b" in states[0]["pad"]  # a state nobody wrote


# -- through the command line ---------------------------------------------------------------


@pytest.fixture(autouse=True)
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", "0123456789abcdef0123456789abcdef01234567")


def test_unreadable_states_are_counted_per_source_in_the_input_and_in_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mod: Any = load_census()
    good = [wire(make_state(size=8, turn_id=t)) for t in (1, 2)]
    lines = [state_line(good[0]), "AZ_STATE:{cut off here", state_line(good[1])]
    log = write_log(tmp_path / "Lua.log", lines)
    code, target = run_cli(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    data = read_json(target)
    assert data["sources"][0]["states_unreadable"] == 1
    assert data["input"]["states_unreadable"] == 1
    assert [r["turn"] for r in data["states"]] == [1, 2]
    assert "1 unreadable" in capsys.readouterr().out
