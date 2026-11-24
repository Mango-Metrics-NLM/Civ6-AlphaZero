"""Hostile log text must neither allocate by its own say-so nor abort the reading (round 2).

R2: an end line's number is untrusted; ``AZ_STATE_END:100000000`` must be counted as a
missing piece without building anything that size. R3 (offline path only): a complete
payload that is JSON but that the live reader cannot take is counted under
``UNREADABLE_STATE`` (the choice made here: counted, not yielded raw) and the reading goes on.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

try:
    import resource
except ImportError:  # Windows: no address-space limit, the time bound still applies
    resource = None  # type: ignore[assignment]

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

HUGE = 100_000_000
LATER = {"turn_id": 9}


@contextmanager
def _memory_ceiling(extra: int = 768 * 1024 * 1024) -> Iterator[None]:
    """Turn a runaway allocation into a MemoryError instead of an OOM kill (Linux only)."""
    if resource is None:
        yield
        return
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            current = int(handle.read().split()[0]) * resource.getpagesize()
    except OSError:
        yield  # no way to measure here: no ceiling
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (current + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def _read(path: Path) -> tuple[list[int], dict[str, int], float]:
    counts: dict[str, int] = {}
    started = time.perf_counter()
    with _memory_ceiling():
        turns = [s["turn_id"] for s in iter_log_states(path, counts)]
    return turns, counts, time.perf_counter() - started


# -- R2 ---------------------------------------------------------------------------------------


def test_a_huge_end_number_with_nothing_held_is_one_missing_piece_and_fast(tmp_path: Path) -> None:
    log = write_log(tmp_path / "Lua.log", [f"AZ_STATE_END:{HUGE}", state_line(LATER)])
    turns, counts, seconds = _read(log)
    assert turns == [9] and counts == {DROPPED_MISSING_PIECE: 1}
    assert seconds < 2.0


def test_a_huge_end_number_with_two_pieces_held_is_one_missing_piece_and_fast(
    tmp_path: Path,
) -> None:
    pieces = chunk_lines({"turn_id": 1, "pad": "x" * 100}, pieces=2)[:2]
    log = write_log(tmp_path / "Lua.log", [*pieces, f"AZ_STATE_END:{HUGE}", state_line(LATER)])
    turns, counts, seconds = _read(log)
    assert turns == [9] and counts == {DROPPED_MISSING_PIECE: 1}
    assert seconds < 2.0


def test_control_a_correct_end_line_still_yields_its_state(tmp_path: Path) -> None:
    log = write_log(tmp_path / "Lua.log", chunk_lines({"turn_id": 4, "pad": "y" * 100}, pieces=3))
    turns, counts, _ = _read(log)
    assert turns == [4] and counts == {}


# -- R3 ---------------------------------------------------------------------------------------

#: Payloads the live reader accepts: yielded as they are, and the census voids them.
ACCEPTED = ['{"turn_id": 2, "units": null}', '{"turn_id": 2, "legal_actions": 5}']
BAD = [
    '{"turn_id": 2, "map": null}',
    "[]",
    "3",
    '"x"',
    "null",
]


@pytest.mark.parametrize("text", BAD)
def test_a_whole_payload_the_reader_cannot_take_is_counted_and_reading_goes_on(
    tmp_path: Path, text: str
) -> None:
    log = write_log(tmp_path / "Lua.log", [f"AZ_STATE:{text}", state_line(LATER)])
    turns, counts, _ = _read(log)
    assert turns == [9]
    assert counts == {UNREADABLE_STATE: 1}


@pytest.mark.parametrize("text", BAD)
def test_a_pieced_payload_the_reader_cannot_take_is_counted_and_reading_goes_on(
    tmp_path: Path, text: str
) -> None:
    cut = max(1, len(text) // 2)
    lines = [
        f"AZ_STATE_CHUNK:1/2:{text[:cut]}",
        f"AZ_STATE_CHUNK:2/2:{text[cut:]}",
        "AZ_STATE_END:2",
        state_line(LATER),
    ]
    turns, counts, _ = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [9]
    assert counts == {UNREADABLE_STATE: 1}


@pytest.mark.parametrize("text", ACCEPTED)
def test_a_payload_the_live_reader_accepts_is_yielded_and_not_counted(
    tmp_path: Path, text: str
) -> None:
    log = write_log(tmp_path / "Lua.log", [f"AZ_STATE:{text}", state_line(LATER)])
    turns, counts, _ = _read(log)
    assert turns == [2, 9] and counts == {}


def test_one_damaged_payload_does_not_hide_the_states_around_it(tmp_path: Path) -> None:
    lines = [state_line({"turn_id": 1}), 'AZ_STATE:{"turn_id": 2, "map": null}']
    lines += [state_line({"turn_id": 3}), "AZ_STATE:null", state_line({"turn_id": 4})]
    turns, counts, _ = _read(write_log(tmp_path / "Lua.log", lines))
    assert turns == [1, 3, 4] and counts == {UNREADABLE_STATE: 2}


@pytest.fixture
def _pinned_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", "0123456789abcdef0123456789abcdef01234567")


@pytest.mark.usefixtures("_pinned_sha")
@pytest.mark.parametrize("text", ['{"turn_id": 2, "map": null}', "[]", "null"])
def test_the_census_runs_to_the_end_and_counts_the_bad_payload(tmp_path: Path, text: str) -> None:
    mod: Any = load_census()
    good = wire(make_state(size=8, turn_id=3))
    log = write_log(tmp_path / "Lua.log", [f"AZ_STATE:{text}", state_line(good)])
    code, target = run_cli(mod, tmp_path, "--log", str(log), "--allow-small", "--boards", "8")
    assert code == 0
    data = read_json(target)
    assert data["input"]["states_unreadable"] == 1
    assert [r["turn"] for r in data["states"]] == [3]
    assert data["input"]["states_measured"] == 1
    json.dumps(data)  # still plain JSON
