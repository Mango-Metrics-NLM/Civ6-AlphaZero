"""GM-B7c-1: LiveBridge hands out every state (C4).

A real ``LogTailer`` on a real temp file; only the command transport (the tuner)
is faked. The fake plays the game: when a command arrives it appends the ack and
the next state to the log in one write, then lets the tailer read that batch
(``on_sent``), which is the race the defect lost.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.live_bridge import LiveBridge
from alpha_zero.bridge.log_tailer import StateTimeoutError
from alpha_zero.bridge.protocol import CommandMessage, GameStateData, create_end_turn_command

SHORT = 0.1


def _state(turn: int) -> dict[str, Any]:
    return {"turn_id": turn, "map": {"width": 0, "height": 0, "tiles": []}}


def _state_line(turn: int) -> str:
    return f"AlphaZeroMain: AZ_STATE:{json.dumps(_state(turn))}"


def _ack_line(msg_id: int) -> str:
    return f"AlphaZeroMain: AZ_ACK:{json.dumps({'msg_id': msg_id, 'success': True})}"


def _hb_line(turn: int | None = None) -> str:
    """A heartbeat as the mod writes it; ``turn`` is the game turn it reports."""
    body = {"lua_alive": True} if turn is None else {"lua_alive": True, "turn": turn}
    return f"AlphaZeroMain: AZ_HB:{json.dumps(body)}"


def _append(path: Path, *lines: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))


class _FakeTuner:
    """Command transport stand-in that answers like the game: ack + next state, together."""

    def __init__(self, log: Path) -> None:
        self._log = log
        self.next_turn = 6
        self.on_sent: Callable[[], None] = lambda: None
        self.sent: list[CommandMessage] = []

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_command(self, command: CommandMessage) -> bool:
        self.sent.append(command)
        _append(self._log, _ack_line(command.msg_id), _state_line(self.next_turn))
        self.on_sent()
        return True

    @property
    def stats(self) -> dict[str, Any]:
        return {"commands_sent": len(self.sent)}


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "Lua.log"
    path.write_text("")
    return path


def _make(tmp_path: Path, log: Path) -> tuple[LiveBridge, _FakeTuner]:
    bridge = LiveBridge(BridgeConfig(comm_dir=tmp_path), log, poll_interval=60.0)
    tuner = _FakeTuner(log)
    bridge._tuner = tuner
    tuner.on_sent = bridge._tailer._read_new_lines
    return bridge, tuner


@pytest.fixture
async def bridge(tmp_path: Path, log: Path) -> AsyncIterator[tuple[LiveBridge, _FakeTuner]]:
    b, tuner = _make(tmp_path, log)
    await b.start()
    await asyncio.sleep(0)
    yield b, tuner
    await b.stop()


class TestRequireNew:
    async def test_state_written_with_the_ack_is_returned_after_send_and_wait(
        self, bridge: tuple[LiveBridge, _FakeTuner]
    ) -> None:
        """The defect end to end: FAILS on the old code (StateTimeoutError)."""
        b, tuner = bridge
        cmd = create_end_turn_command(b.next_msg_id())

        ack = await b.send_and_wait(cmd, timeout=SHORT)
        state = await b.wait_for_state(timeout=SHORT)

        assert ack["success"] is True
        assert ack["msg_id"] == cmd.msg_id
        assert state.turn_id == 6
        assert len(tuner.sent) == 1

    async def test_a_state_is_not_handed_out_twice(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        _append(log, _state_line(5))
        b._tailer._read_new_lines()

        assert (await b.wait_for_state(timeout=SHORT)).turn_id == 5
        with pytest.raises(StateTimeoutError):
            await b.wait_for_state(timeout=SHORT)

    async def test_two_rounds_each_get_their_own_state(
        self, bridge: tuple[LiveBridge, _FakeTuner]
    ) -> None:
        b, tuner = bridge
        turns = []
        for turn in (6, 7, 8):
            tuner.next_turn = turn
            await b.send_and_wait(create_end_turn_command(b.next_msg_id()), timeout=SHORT)
            turns.append((await b.wait_for_state(timeout=SHORT)).turn_id)
        assert turns == [6, 7, 8]

    async def test_returns_a_parsed_game_state(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        _append(log, _state_line(5))
        b._tailer._read_new_lines()
        assert isinstance(await b.wait_for_state(timeout=SHORT), GameStateData)


class TestRequireNewFalse:
    async def test_returns_the_latest_state_and_marks_it_handed_out(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        _append(log, _state_line(5))
        b._tailer._read_new_lines()

        assert (await b.wait_for_state(timeout=SHORT, require_new=False)).turn_id == 5
        with pytest.raises(StateTimeoutError):
            await b.wait_for_state(timeout=SHORT, require_new=True)

        _append(log, _state_line(6))
        b._tailer._read_new_lines()
        assert (await b.wait_for_state(timeout=SHORT, require_new=True)).turn_id == 6

    async def test_waits_when_there_is_no_state_yet(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        waiter = asyncio.create_task(b.wait_for_state(timeout=2.0, require_new=False))
        for _ in range(5):
            await asyncio.sleep(0)
        _append(log, _state_line(5))
        b._tailer._read_new_lines()
        assert (await waiter).turn_id == 5

    async def test_times_out_when_nothing_ever_arrives(
        self, bridge: tuple[LiveBridge, _FakeTuner]
    ) -> None:
        b, _ = bridge
        with pytest.raises(StateTimeoutError):
            await b.wait_for_state(timeout=SHORT, require_new=False)

    async def test_returns_the_newest_when_several_arrived(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        _append(log, _state_line(1), _state_line(2))
        b._tailer._read_new_lines()
        assert (await b.wait_for_state(timeout=SHORT, require_new=False)).turn_id == 2


class TestStats:
    async def test_new_counters_are_exposed(self, bridge: tuple[LiveBridge, _FakeTuner]) -> None:
        b, _ = bridge
        for key in ("states_skipped", "acks_dropped", "backlog_states_used"):
            assert b.stats[key] == 0
        assert b.get_stats()["states_skipped"] == 0

    async def test_counters_move_with_the_tailer(
        self, bridge: tuple[LiveBridge, _FakeTuner], log: Path
    ) -> None:
        b, _ = bridge
        _append(log, _state_line(1), _state_line(2))
        b._tailer._read_new_lines()
        await b.wait_for_state(timeout=SHORT)
        assert b.stats["states_skipped"] == 1

        _append(log, *[_ack_line(i) for i in range(1, 259)])
        b._tailer._read_new_lines()
        assert b.stats["acks_dropped"] == 2


class TestBacklogState:
    async def test_none_without_a_state_in_the_log(
        self, bridge: tuple[LiveBridge, _FakeTuner]
    ) -> None:
        b, _ = bridge
        assert b.backlog_state() is None

    async def test_parsed_state_in_the_log_at_start_is_pending_until_a_heartbeat(
        self, tmp_path: Path, log: Path
    ) -> None:
        _append(log, _state_line(4))
        b, _ = _make(tmp_path, log)
        await b.start()
        await asyncio.sleep(0)
        try:
            backlog = b.backlog_state()
            assert isinstance(backlog, GameStateData)
            assert backlog.turn_id == 4
            # not handed out by itself
            with pytest.raises(StateTimeoutError):
                await b.wait_for_state(timeout=SHORT, require_new=False)
            assert b.backlog_state() is not None

            _append(log, _hb_line(4))
            b._tailer._read_new_lines()
            assert b.stats["backlog_states_used"] == 1
            assert b.backlog_state() is None
            assert (await b.wait_for_state(timeout=SHORT, require_new=False)).turn_id == 4
        finally:
            await b.stop()

    async def test_a_live_state_discards_it(self, tmp_path: Path, log: Path) -> None:
        _append(log, _state_line(4))
        b, _ = _make(tmp_path, log)
        await b.start()
        await asyncio.sleep(0)
        try:
            _append(log, _state_line(5))
            b._tailer._read_new_lines()
            assert b.backlog_state() is None
            assert (await b.wait_for_state(timeout=SHORT)).turn_id == 5
        finally:
            await b.stop()
