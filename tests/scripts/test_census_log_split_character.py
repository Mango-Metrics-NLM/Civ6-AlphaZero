"""A UTF-8 character cut between two pieces is joined whole by the offline reader.

The offline reader joins the pieces it holds exactly as the live tailer's
``_handle_chunk_end`` does (ascending number, same strings, read with ``surrogateescape``)
and then hands the text to the live ``_handle_state``; the shared decoder re-joins the two
halves of a split character. Bytes are written, so the cut can fall inside a character.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import iter_log_states
from tests.fixtures.census_states import load_census, mixed_state, read_json, run_cli, wire

CITY = "Ñandú 東京"
SHA = "0123456789abcdef0123456789abcdef01234567"


def _pieced_bytes(payload: bytes, cut: int) -> bytes:
    parts = [payload[:cut], payload[cut:]]
    lines = [b"AlphaZeroMain: AZ_STATE_CHUNK:%d/2:" % (i + 1) + p for i, p in enumerate(parts)]
    return b"\n".join([*lines, b"AlphaZeroMain: AZ_STATE_END:2"]) + b"\n"


def _write(tmp_path: Path, raw: dict[str, Any], marker: str) -> Path:
    payload = json.dumps(raw, ensure_ascii=False).encode("utf-8")
    cut = payload.index(marker.encode("utf-8")) + 1  # between the two bytes of its first character
    with pytest.raises(UnicodeDecodeError):  # control: the cut really is inside a character
        payload[:cut].decode("utf-8")
    log = tmp_path / "Lua.log"
    log.write_bytes(_pieced_bytes(payload, cut))
    return log


def test_the_offline_reader_yields_a_split_character_whole_and_counts_nothing(
    tmp_path: Path,
) -> None:
    log = _write(tmp_path, {"turn_id": 3, "city": CITY}, "Ñ")
    counts: dict[str, int] = {}
    states = list(iter_log_states(log, counts))
    assert [s["turn_id"] for s in states] == [3]
    assert states[0]["city"] == CITY and "�" not in states[0]["city"]
    assert counts == {}


@pytest.mark.parametrize("marker", ["Ñ", "東"])
def test_the_census_measures_a_state_split_inside_a_character(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str
) -> None:
    monkeypatch.setenv("CIV6AZ_GIT_SHA", SHA)
    raw = wire(mixed_state(3))
    raw["current_player"] = CITY
    log = _write(tmp_path, raw, marker)
    code, target = run_cli(
        load_census(), tmp_path, "--log", str(log), "--allow-small", "--boards", "8"
    )
    assert code == 0
    data = read_json(target)
    assert data["input"]["states_unreadable"] == 0
    assert data["input"]["states_dropped_missing_piece"] == 0
    assert data["input"]["states_measured"] == 1
    assert data["states"][0]["civilization"] == CITY
