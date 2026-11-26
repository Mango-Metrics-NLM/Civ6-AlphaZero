"""A non-ASCII character the mod splits across two chunk lines survives.

The mod cuts a long state into 1,800-byte pieces without regard to character
boundaries (``FileIO.lua``), and the tailer used to decode each log line by
itself with ``errors="replace"``: both halves of a split character became
replacement characters, in JSON that still parsed. A real ``LogTailer`` on a
real file, fed bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_zero.bridge.log_tailer import LogTailer

NAME = "Zürich né São Tomé 東京"


def _chunk_lines(payload: bytes, size: int) -> bytes:
    pieces = [payload[i : i + size] for i in range(0, len(payload), size)]
    out = b""
    for index, piece in enumerate(pieces, start=1):
        out += b"AlphaZeroMain: AZ_STATE_CHUNK:%d/%d:" % (index, len(pieces)) + piece + b"\n"
    return out + b"AlphaZeroMain: AZ_STATE_END:%d\n" % len(pieces)


def _payload() -> bytes:
    return json.dumps({"turn_id": 3, "city": NAME}, ensure_ascii=False).encode("utf-8")


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
async def test_character_split_across_chunks_is_reassembled(tmp_path: Path, size: int) -> None:
    log = tmp_path / "Lua.log"
    log.write_bytes(b"")
    payload = _payload()
    # The control: at these sizes a multi-byte character is cut somewhere.
    assert any(
        (payload[i : i + size]).decode("utf-8", "replace")
        != (payload[i : i + size]).decode("utf-8", "ignore")
        for i in range(0, len(payload), size)
    )
    tailer = LogTailer(log, poll_interval=60.0)
    await tailer.start()
    try:
        with log.open("ab") as handle:
            handle.write(_chunk_lines(payload, size))
        tailer._read_new_lines()
        state = await tailer.wait_for_state(timeout=0.1)
    finally:
        await tailer.stop()
    assert state["city"] == NAME
    assert "�" not in state["city"]


async def test_split_character_in_the_backlog_is_reassembled(tmp_path: Path) -> None:
    log = tmp_path / "Lua.log"
    log.write_bytes(_chunk_lines(_payload(), 3))
    tailer = LogTailer(log, poll_interval=60.0)
    await tailer.start()
    try:
        assert tailer.backlog_state is not None
        assert tailer.backlog_state["city"] == NAME
    finally:
        await tailer.stop()


async def test_single_line_state_keeps_non_ascii_text(tmp_path: Path) -> None:
    log = tmp_path / "Lua.log"
    log.write_bytes(b"")
    tailer = LogTailer(log, poll_interval=60.0)
    await tailer.start()
    try:
        with log.open("ab") as handle:
            handle.write(b"AZ_STATE:" + _payload() + b"\n")
        tailer._read_new_lines()
        state = await tailer.wait_for_state(timeout=0.1)
    finally:
        await tailer.stop()
    assert state["city"] == NAME


async def test_bytes_that_are_not_utf8_do_not_stop_the_tailer(tmp_path: Path) -> None:
    log = tmp_path / "Lua.log"
    log.write_bytes(b"")
    tailer = LogTailer(log, poll_interval=60.0)
    await tailer.start()
    try:
        with log.open("ab") as handle:
            handle.write(b'AZ_STATE:{"turn_id": 4, "city": "a\xffb"}\n')
        tailer._read_new_lines()
        state = await tailer.wait_for_state(timeout=0.1)
    finally:
        await tailer.stop()
    assert state["turn_id"] == 4
    assert state["city"] == "a�b"
