"""Defects an independent review found in the first version of GM-B7c-1.

Each is a way the tailer handed out a state that was not the current one, or
the same state to two callers. A real ``LogTailer`` on a real file.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from alpha_zero.bridge.log_tailer import LogTailer, StateTimeoutError

SHORT = 0.1


def _state(turn: int, **extra: object) -> str:
    return "AlphaZeroMain: AZ_STATE:" + json.dumps({"turn_id": turn, **extra})


def _hb(turn: int) -> str:
    return "AlphaZeroMain: AZ_HB:" + json.dumps({"lua_alive": True, "turn": turn})


def _chunks(turn: int, pieces: int = 3) -> list[str]:
    text = json.dumps({"turn_id": turn, "pad": "x" * 40})
    size = -(-len(text) // pieces)
    parts = [text[i : i + size] for i in range(0, len(text), size)]
    lines = [
        f"AlphaZeroMain: AZ_STATE_CHUNK:{i}/{len(parts)}:{part}" for i, part in enumerate(parts, 1)
    ]
    return [*lines, f"AlphaZeroMain: AZ_STATE_END:{len(parts)}"]


def _append(log: Path, *lines: str) -> None:
    with log.open("a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "Lua.log"
    path.write_text("")
    return path


async def _started(log: Path) -> LogTailer:
    tailer = LogTailer(log, poll_interval=60.0)
    await tailer.start()
    return tailer


async def test_heartbeat_of_another_turn_does_not_release_the_backlog(log: Path) -> None:
    """The mod also writes a heartbeat when a game loads, outside any turn."""
    _append(log, _state(5))
    t = await _started(log)
    try:
        _append(log, _hb(1))
        t._read_new_lines()
        assert t.state_seq == 0
        with pytest.raises(StateTimeoutError):
            await t.wait_for_state(timeout=SHORT)
        _append(log, _hb(5))
        t._read_new_lines()
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 5
    finally:
        await t.stop()


async def test_heartbeat_without_a_turn_does_not_release_the_backlog(log: Path) -> None:
    _append(log, _state(5))
    t = await _started(log)
    try:
        _append(log, "AlphaZeroMain: AZ_HB:" + json.dumps({"lua_alive": True}))
        t._read_new_lines()
        assert t.state_seq == 0
    finally:
        await t.stop()


async def test_a_truncated_log_drops_the_backlog_and_half_read_pieces(log: Path) -> None:
    """A relaunched game starts a new log: the old session's state is not current."""
    _append(log, _state(5), "padding " * 50)
    t = await _started(log)
    try:
        _append(log, *_chunks(5)[:1])
        t._read_new_lines()
        log.write_text("")  # the game was relaunched
        _append(log, _hb(5))
        t._read_new_lines()
        assert t.backlog_state is None
        assert t.state_seq == 0
        assert t._chunk_buffer == {}
    finally:
        await t.stop()


async def test_a_line_written_between_scan_and_tail_is_not_lost(
    log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Turn 6's state lands while start() is between its scan and its tail."""
    _append(log, _state(5))
    t = LogTailer(log, poll_interval=60.0)
    scan = t._scan_backlog

    def scan_then_write() -> tuple[dict[str, object] | None, int]:
        result = scan()
        _append(log, _state(6))
        return result

    monkeypatch.setattr(t, "_scan_backlog", scan_then_write)
    await t.start()
    try:
        t._read_new_lines()
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 6
        _append(log, _hb(6), _hb(5))
        t._read_new_lines()
        assert t.stats["backlog_states_used"] == 0
    finally:
        await t.stop()


async def test_start_in_the_middle_of_a_state_sent_in_pieces(log: Path) -> None:
    """The newest state is half written at start: the older one is not offered."""
    pieces = _chunks(6)
    _append(log, _state(5), *pieces[:2])
    t = await _started(log)
    try:
        assert t.backlog_state is None
        _append(log, *pieces[2:], _hb(5), _hb(6))
        t._read_new_lines()
        assert t.stats["backlog_states_used"] == 0
        # The pieces read before start are joined with the rest: the export
        # that was under way is received, not lost.
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 6
    finally:
        await t.stop()


async def test_a_damaged_live_state_still_retires_the_backlog(log: Path) -> None:
    _append(log, _state(5))
    t = await _started(log)
    try:
        _append(log, 'AlphaZeroMain: AZ_STATE:{"turn_id": 6, "ma', _hb(5))
        t._read_new_lines()
        assert t.backlog_state is None
        assert t.stats["backlog_states_used"] == 0
    finally:
        await t.stop()


async def test_one_state_goes_to_one_of_two_waiters(log: Path) -> None:
    t = await _started(log)
    try:
        first = asyncio.ensure_future(t.wait_for_state(timeout=SHORT))
        second = asyncio.ensure_future(t.wait_for_state(timeout=SHORT))
        await asyncio.sleep(0)
        _append(log, _state(7))
        t._read_new_lines()
        results = await asyncio.gather(first, second, return_exceptions=True)
        got = [r for r in results if isinstance(r, dict)]
        timed_out = [r for r in results if isinstance(r, StateTimeoutError)]
        assert len(got) == 1 and got[0]["turn_id"] == 7
        assert len(timed_out) == 1
    finally:
        await t.stop()


async def test_a_restart_forgets_what_was_read_before(log: Path) -> None:
    t = await _started(log)
    _append(log, _state(7), _hb(7))
    t._read_new_lines()
    assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 7
    assert t.check_heartbeat() is True
    await t.stop()

    await t.start()
    try:
        assert t.current_state() is None
        assert t.check_heartbeat() is False
        assert t.state_seq == 0
        # The state is in the log, so it is the backlog; its turn's heartbeat releases it.
        assert t.backlog_state is not None
    finally:
        await t.stop()


async def test_line_separator_inside_a_name_does_not_cut_the_backlog_state(log: Path) -> None:
    separator = chr(0x2028)  # str.splitlines() cuts here; the live path does not
    name = f"A{separator}B"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            "AZ_STATE:" + json.dumps({"turn_id": 5, "city": name}, ensure_ascii=False) + "\n"
        )
    t = await _started(log)
    try:
        assert t.backlog_state is not None
        assert t.backlog_state["city"] == name
    finally:
        await t.stop()


async def test_a_truncated_log_forgets_a_state_not_yet_read(log: Path) -> None:
    """A state published just before the game rewrote the log is the old game's."""
    _append(log, "padding " * 50)
    t = await _started(log)
    try:
        _append(log, _state(9), _hb(9))
        t._read_new_lines()
        assert t.state_seq == 1 and t.check_heartbeat() is True
        log.write_text("")  # the game was relaunched before anyone asked
        t._read_new_lines()
        assert t.current_state() is None
        assert t.latest_state is None
        assert t.check_heartbeat() is False
        with pytest.raises(StateTimeoutError):
            await t.wait_for_state(timeout=SHORT)
    finally:
        await t.stop()


async def test_a_truncated_log_forgets_a_state_already_read(log: Path) -> None:
    _append(log, "padding " * 50)
    t = await _started(log)
    try:
        _append(log, _state(9))
        t._read_new_lines()
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 9
        log.write_text("")
        t._read_new_lines()
        assert t.current_state() is None  # require_new=False must not see the old game
        _append(log, _state(1))
        t._read_new_lines()
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 1
    finally:
        await t.stop()


async def test_pieces_from_before_start_do_not_complete_a_different_export(log: Path) -> None:
    """Handed-over pieces are only good with their own end line."""
    pieces = _chunks(6, pieces=4)
    _append(log, *pieces[:2])
    t = await _started(log)
    try:
        _append(log, _state(7))  # a newer whole state arrives instead of the rest
        t._read_new_lines()
        assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 7
    finally:
        await t.stop()
