"""``iter_log_states``: every complete state of a ``Lua.log``, read offline (GM-B11).

Spec: "Reading states" in ``kb/specs/20261123-expressivity-census-preregistration.md``: each
line goes through the tailer's own parser, chunk reassembly included, and each completed
state is yielded. The logs are real files in the mod's line formats
(``tests/unit/test_log_tailer.py`` has the formats).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import DROPPED_MISSING_PIECE, LogTailer, iter_log_states
from tests.fixtures.census_states import chunk_lines, state_line, write_log


def _payload(turn: int, pad: int = 300) -> dict[str, Any]:
    return {"turn_id": turn, "player_id": 0, "pad": "x" * pad}


HEARTBEAT = 'AZ_HB:{"lua_alive": true, "turn_id": 7}'
ACK = 'AZ_ACK:{"msg_id": 1, "success": true}'


def _turns(path: Path) -> list[int]:
    return [s["turn_id"] for s in iter_log_states(path)]


def test_whole_line_states_are_yielded_in_order_with_or_without_the_context_prefix(
    tmp_path: Path,
) -> None:
    log = write_log(
        tmp_path / "Lua.log",
        [
            state_line(_payload(1)),
            state_line(_payload(2), context=False),
            state_line(_payload(3)),
        ],
    )
    states = list(iter_log_states(log))
    assert [s["turn_id"] for s in states] == [1, 2, 3]
    assert states[0]["pad"] == "x" * 300  # the state as the mod wrote it


def test_acks_heartbeats_and_noise_are_ignored(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "Lua.log",
        [
            "[1234.5] some engine line",
            HEARTBEAT,
            ACK,
            state_line(_payload(4)),
            "AlphaZeroMain: " + HEARTBEAT,
            "AlphaZeroMain: " + ACK,
            "",
            state_line(_payload(5)),
        ],
    )
    assert _turns(log) == [4, 5]


def test_a_state_sent_in_pieces_is_joined_and_yielded_once(tmp_path: Path) -> None:
    lines = [state_line(_payload(1)), *chunk_lines(_payload(2, pad=900), pieces=4)]
    lines.append(state_line(_payload(3)))
    states = list(iter_log_states(write_log(tmp_path / "Lua.log", lines)))
    assert [s["turn_id"] for s in states] == [1, 2, 3]
    assert states[1]["pad"] == "x" * 900


def test_a_state_in_pieces_with_the_context_prefix_on_each_line(tmp_path: Path) -> None:
    lines = ["AlphaZeroMain: " + line for line in chunk_lines(_payload(6, pad=500))]
    assert _turns(write_log(tmp_path / "Lua.log", lines)) == [6]


def test_a_heartbeat_between_pieces_does_not_stop_the_state(tmp_path: Path) -> None:
    pieces = chunk_lines(_payload(8, pad=600), pieces=3)
    lines = [pieces[0], HEARTBEAT, pieces[1], ACK, *pieces[2:]]
    assert _turns(write_log(tmp_path / "Lua.log", lines)) == [8]


def test_a_broken_line_is_skipped_and_the_states_around_it_survive(tmp_path: Path) -> None:
    log = write_log(
        tmp_path / "Lua.log",
        [
            state_line(_payload(1)),
            'AZ_STATE:{"turn_id": 2, "pad": "cut off here',  # a line cut short
            state_line(_payload(3)),
            "AZ_STATE:not json at all",
            state_line(_payload(4)),
        ],
    )
    assert _turns(log) == [1, 3, 4]


def test_a_state_with_a_lost_piece_is_not_yielded_and_the_next_one_is(tmp_path: Path) -> None:
    # Structured content, so that losing a whole piece cannot leave valid JSON behind.
    lost_state = {"turn_id": 5, "tiles": [{"x": i, "y": i * 7, "t": "plains"} for i in range(40)]}
    pieces = chunk_lines(lost_state, pieces=4)
    lost = [pieces[0], pieces[2], pieces[3], pieces[4]]  # piece 2 of 4 never arrived
    lines = [state_line(_payload(4)), *lost, state_line(_payload(6))]
    assert _turns(write_log(tmp_path / "Lua.log", lines)) == [4, 6]


def test_every_complete_state_is_yielded_exactly_once(tmp_path: Path) -> None:
    lines: list[str] = []
    for turn in range(1, 21):
        lines += (
            chunk_lines(_payload(turn, pad=700)) if turn % 3 == 0 else [state_line(_payload(turn))]
        )
        lines.append(HEARTBEAT)
    assert _turns(write_log(tmp_path / "Lua.log", lines)) == list(range(1, 21))


def test_a_log_written_with_windows_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "Lua.log"
    body = "\r\n".join(
        [state_line(_payload(1)), *chunk_lines(_payload(2)), state_line(_payload(3))]
    )
    path.write_bytes((body + "\r\n").encode("utf-8"))
    assert _turns(path) == [1, 2, 3]


def test_a_log_with_no_state_yields_nothing(tmp_path: Path) -> None:
    assert _turns(write_log(tmp_path / "Lua.log", [HEARTBEAT, ACK, "noise"])) == []
    empty = tmp_path / "empty.log"
    empty.write_text("", encoding="utf-8")
    assert _turns(empty) == []


def test_a_missing_file_is_an_oserror(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        list(iter_log_states(tmp_path / "no_such.log"))


def test_a_path_may_be_a_string(tmp_path: Path) -> None:
    log = write_log(tmp_path / "Lua.log", [state_line(_payload(1))])
    assert [s["turn_id"] for s in iter_log_states(str(log))] == [1]


def test_wrapped_states_are_yielded_as_written(tmp_path: Path) -> None:
    wrapped = {"type": "state", "state": _payload(9)}
    log = write_log(tmp_path / "Lua.log", ["AZ_STATE:" + json.dumps(wrapped)])
    assert list(iter_log_states(log)) == [wrapped]


def _parsing_pieces(turn: int) -> list[str]:
    """Four pieces and an end line of one long string value: without a piece it still parses."""
    return chunk_lines({"turn_id": turn, "pad": "x" * 900}, pieces=4)


def _drop_second(pieces: list[str]) -> list[str]:
    return [pieces[0], *pieces[2:]]


def test_the_surviving_text_of_the_damaged_state_does_parse() -> None:
    text = "".join(p.split(":", 2)[2] for p in _drop_second(_parsing_pieces(2))[:-1])
    assert json.loads(text)["turn_id"] == 2  # the hole this guards against is real


def test_a_state_with_a_piece_missing_is_dropped_even_when_the_rest_parses(
    tmp_path: Path,
) -> None:
    lines = [state_line(_payload(1)), *_drop_second(_parsing_pieces(2)), state_line(_payload(3))]
    counts: dict[str, int] = {}
    log = write_log(tmp_path / "Lua.log", lines)
    assert [s["turn_id"] for s in iter_log_states(log, counts)] == [1, 3]
    assert counts == {DROPPED_MISSING_PIECE: 1}


def test_a_state_with_all_its_pieces_is_yielded_and_nothing_is_counted(tmp_path: Path) -> None:
    lines = [state_line(_payload(1)), *_parsing_pieces(2), state_line(_payload(3))]
    counts: dict[str, int] = {}
    log = write_log(tmp_path / "Lua.log", lines)
    assert [s["turn_id"] for s in iter_log_states(log, counts)] == [1, 2, 3]
    assert counts == {}


def test_leftover_pieces_do_not_leak_into_the_next_state(tmp_path: Path) -> None:
    follower = {"turn_id": 4, "tiles": [{"x": i, "y": i} for i in range(20)]}
    lines = [*_drop_second(_parsing_pieces(2)), *chunk_lines(follower, pieces=2)]
    counts: dict[str, int] = {}
    states = list(iter_log_states(write_log(tmp_path / "Lua.log", lines), counts))
    assert states == [follower]  # unaltered
    assert counts == {DROPPED_MISSING_PIECE: 1}


def test_a_malformed_end_line_is_dropped_and_counted(tmp_path: Path) -> None:
    pieces = _parsing_pieces(2)
    lines = [*pieces[:-1], "AZ_STATE_END:abc", state_line(_payload(3))]
    counts: dict[str, int] = {}
    log = write_log(tmp_path / "Lua.log", lines)
    assert [s["turn_id"] for s in iter_log_states(log, counts)] == [3]
    assert counts == {DROPPED_MISSING_PIECE: 1}


def test_counts_are_optional(tmp_path: Path) -> None:
    log = write_log(tmp_path / "Lua.log", _drop_second(_parsing_pieces(2)))
    assert list(iter_log_states(log)) == []


def test_the_live_tailer_still_publishes_the_state_that_lost_a_piece(tmp_path: Path) -> None:
    tailer = LogTailer(tmp_path / "Lua.log")
    for line in _drop_second(_parsing_pieces(2)):
        tailer._parse_line(line)
    assert tailer.latest_state is not None and tailer.latest_state["turn_id"] == 2
    assert len(tailer.latest_state["pad"]) < 900  # reassembled from what arrived
