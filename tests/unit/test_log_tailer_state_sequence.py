"""GM-B7c-1: the log tailer keeps every state (sequence numbers, backlog, ack bound).

The defect: ``wait_for_state`` cleared its event and then waited, while
``_read_new_lines`` parses a whole batch synchronously. An ``AZ_ACK`` and the
``AZ_STATE`` behind it read in one batch therefore lost the state: the caller's
next ``wait_for_state`` waited for a state that had already come.

Every test uses a real ``LogTailer`` on a real temp file. The poll loop is given
a long interval so the test alone decides when a batch is read (it calls
``_read_new_lines()`` itself); no wait that must expire exceeds 0.2 s.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.log_tailer import AckTimeoutError, LogTailer, StateTimeoutError

SHORT = 0.1  # a wait that must expire


def _state(turn: int, **extra: Any) -> dict[str, Any]:
    return {"turn_id": turn, "map": {"width": 0, "height": 0, "tiles": []}, **extra}


def _state_line(turn: int) -> str:
    return f"AlphaZeroMain: AZ_STATE:{json.dumps(_state(turn))}"


def _ack_line(msg_id: int, success: bool = True) -> str:
    return f"AlphaZeroMain: AZ_ACK:{json.dumps({'msg_id': msg_id, 'success': success})}"


def _hb_line(turn: int | None = None) -> str:
    """A heartbeat as the mod writes it; ``turn`` is the game turn it reports."""
    body = {"lua_alive": True} if turn is None else {"lua_alive": True, "turn": turn}
    return f"AlphaZeroMain: AZ_HB:{json.dumps(body)}"


def _chunk_lines(turn: int, parts: int = 3) -> list[str]:
    text = json.dumps(_state(turn, padding="P" * 90))
    size = -(-len(text) // parts)
    pieces = [text[i : i + size] for i in range(0, len(text), size)]
    lines = [
        f"AlphaZeroMain: AZ_STATE_CHUNK:{i}/{len(pieces)}:{p}" for i, p in enumerate(pieces, 1)
    ]
    lines.append(f"AlphaZeroMain: AZ_STATE_END:{len(pieces)}")
    return lines


def _append(path: Path, *lines: str) -> None:
    """Append lines in a single write, so they land in one batch."""
    with open(path, "a", encoding="utf-8") as f:
        f.write("".join(line + "\n" for line in lines))


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "Lua.log"
    path.write_text("")
    return path


@pytest.fixture
async def tailer(log: Path) -> AsyncIterator[LogTailer]:
    t = LogTailer(log, poll_interval=60.0)
    await t.start()
    await asyncio.sleep(0)  # the poll task's first (empty) read happens here
    yield t
    await t.stop()


# --- C1: sequence numbers ---


class TestStateSequence:
    async def test_ack_and_state_in_one_batch_state_is_returned(
        self, tailer: LogTailer, log: Path
    ) -> None:
        """The defect, end to end at the tailer: FAILS on the old code (StateTimeoutError)."""
        _append(log, _ack_line(7), _state_line(5))
        tailer._read_new_lines()

        ack = await tailer.wait_for_ack(7, timeout=SHORT)
        state = await tailer.wait_for_state(timeout=SHORT)

        assert ack["success"] is True
        assert state["turn_id"] == 5

    async def test_end_turn_ack_and_next_turn_state_in_one_batch(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _state_line(5))
        tailer._read_new_lines()
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 5

        _append(log, _ack_line(8), _state_line(6))
        tailer._read_new_lines()
        await tailer.wait_for_ack(8, timeout=SHORT)
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 6

    async def test_two_states_in_one_batch_return_the_newer(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _state_line(1), _state_line(2))
        tailer._read_new_lines()

        state = await tailer.wait_for_state(timeout=SHORT)

        assert state["turn_id"] == 2
        assert tailer.stats["states_skipped"] == 1
        assert tailer.state_seq == 2

    async def test_skipped_counts_only_states_never_handed_out(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _state_line(1))
        tailer._read_new_lines()
        await tailer.wait_for_state(timeout=SHORT)
        assert tailer.stats["states_skipped"] == 0

        _append(log, _state_line(2), _state_line(3))
        tailer._read_new_lines()
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 3
        assert tailer.stats["states_skipped"] == 1

    async def test_chunked_state_in_the_same_batch_as_an_ack(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _ack_line(9), *_chunk_lines(12))
        tailer._read_new_lines()

        await tailer.wait_for_ack(9, timeout=SHORT)
        state = await tailer.wait_for_state(timeout=SHORT)

        assert state["turn_id"] == 12
        assert state["padding"] == "P" * 90
        assert tailer.state_seq == 1

    async def test_ack_and_state_in_separate_batches_still_work(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _ack_line(3))
        tailer._read_new_lines()
        await tailer.wait_for_ack(3, timeout=SHORT)

        waiter = asyncio.create_task(tailer.wait_for_state(timeout=2.0))
        for _ in range(5):
            await asyncio.sleep(0)
        _append(log, _state_line(4))
        tailer._read_new_lines()

        assert (await waiter)["turn_id"] == 4

    async def test_state_seq_starts_at_zero_and_counts_every_state(
        self, tailer: LogTailer, log: Path
    ) -> None:
        assert tailer.state_seq == 0
        _append(log, _state_line(1))
        tailer._read_new_lines()
        assert tailer.state_seq == 1
        _append(log, *_chunk_lines(2))
        tailer._read_new_lines()
        assert tailer.state_seq == 2

    async def test_second_wait_without_new_state_times_out(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _state_line(5))
        tailer._read_new_lines()
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 5

        with pytest.raises(StateTimeoutError):
            await tailer.wait_for_state(timeout=SHORT)

        _append(log, _state_line(6))
        tailer._read_new_lines()
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 6

    async def test_current_state_marks_handed_out(self, tailer: LogTailer, log: Path) -> None:
        assert tailer.current_state() is None
        _append(log, _state_line(5))
        tailer._read_new_lines()

        current = tailer.current_state()
        assert current is not None
        assert current["turn_id"] == 5
        with pytest.raises(StateTimeoutError):
            await tailer.wait_for_state(timeout=SHORT)

        _append(log, _state_line(6))
        tailer._read_new_lines()
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 6

    async def test_latest_state_is_a_peek_that_marks_nothing(
        self, tailer: LogTailer, log: Path
    ) -> None:
        _append(log, _state_line(5))
        tailer._read_new_lines()

        peeked = tailer.latest_state
        assert peeked is not None
        assert peeked["turn_id"] == 5
        assert (await tailer.wait_for_state(timeout=SHORT))["turn_id"] == 5

    async def test_wait_started_before_the_state_gets_the_next_state(
        self, tailer: LogTailer, log: Path
    ) -> None:
        waiter = asyncio.create_task(tailer.wait_for_state(timeout=2.0))
        for _ in range(5):
            await asyncio.sleep(0)
        _append(log, _state_line(8))
        tailer._read_new_lines()
        assert (await waiter)["turn_id"] == 8

    async def test_event_set_with_no_state_still_raises(self, tailer: LogTailer) -> None:
        tailer._state_event.set()
        with pytest.raises(StateTimeoutError):
            await tailer.wait_for_state(timeout=SHORT)

    async def test_new_counters_start_at_zero(self, tailer: LogTailer) -> None:
        stats = tailer.stats
        assert stats["states_skipped"] == 0
        assert stats["acks_dropped"] == 0
        assert stats["backlog_states_used"] == 0
        for key in ("states_received", "acks_received", "heartbeats_received", "parse_errors"):
            assert stats[key] == 0


# --- C2: the state already in the log at start ---


async def _started(log: Path, **kwargs: Any) -> LogTailer:
    t = LogTailer(log, poll_interval=60.0, **kwargs)
    await t.start()
    await asyncio.sleep(0)
    return t


class TestBacklogState:
    async def test_last_complete_state_is_found_but_not_handed_out(self, log: Path) -> None:
        _append(log, _state_line(3), _ack_line(1), _state_line(4), _hb_line())
        t = await _started(log)
        try:
            backlog = t.backlog_state
            assert backlog is not None
            assert backlog["turn_id"] == 4
            assert t.state_seq == 0
            assert t.latest_state is None
            with pytest.raises(StateTimeoutError):
                await t.wait_for_state(timeout=SHORT)
        finally:
            await t.stop()

    async def test_live_heartbeat_promotes_the_backlog_state(self, log: Path) -> None:
        _append(log, _state_line(4))
        t = await _started(log)
        try:
            _append(log, _hb_line(4))
            t._read_new_lines()

            assert t.state_seq == 1
            assert t.stats["backlog_states_used"] == 1
            assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 4
            assert t.backlog_state is None

            _append(log, _hb_line(4))
            t._read_new_lines()
            assert t.state_seq == 1
            assert t.stats["backlog_states_used"] == 1
        finally:
            await t.stop()

    async def test_chunked_backlog_state_is_promoted_too(self, log: Path) -> None:
        _append(log, *_chunk_lines(9))
        t = await _started(log)
        try:
            backlog = t.backlog_state
            assert backlog is not None
            assert backlog["turn_id"] == 9
            _append(log, _hb_line(9))
            t._read_new_lines()
            assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 9
        finally:
            await t.stop()

    async def test_live_state_before_any_heartbeat_discards_the_backlog(self, log: Path) -> None:
        _append(log, _state_line(4))
        t = await _started(log)
        try:
            _append(log, _state_line(5))
            t._read_new_lines()
            assert t.backlog_state is None

            _append(log, _hb_line())
            t._read_new_lines()

            assert t.state_seq == 1
            assert t.stats["backlog_states_used"] == 0
            assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 5
            with pytest.raises(StateTimeoutError):
                await t.wait_for_state(timeout=SHORT)
        finally:
            await t.stop()

    async def test_heartbeat_then_live_state_in_one_batch_hands_out_the_live_one(
        self, log: Path
    ) -> None:
        _append(log, _state_line(4))
        t = await _started(log)
        try:
            _append(log, _hb_line(), _state_line(5))
            t._read_new_lines()
            assert (await t.wait_for_state(timeout=SHORT))["turn_id"] == 5
        finally:
            await t.stop()

    async def test_stale_heartbeat_in_the_backlog_is_not_a_live_heartbeat(self, log: Path) -> None:
        _append(log, _hb_line(), _state_line(4), _hb_line())
        t = await _started(log)
        try:
            assert t.check_heartbeat() is False
            assert t.latest_heartbeat is None
            assert t.stats["heartbeats_received"] == 0
            # and it does not promote the backlog state either
            assert t.state_seq == 0
        finally:
            await t.stop()

    async def test_acks_in_the_backlog_are_ignored(self, log: Path) -> None:
        _append(log, _ack_line(11), _state_line(4))
        t = await _started(log)
        try:
            assert t.stats["acks_received"] == 0
            with pytest.raises(AckTimeoutError):
                await t.wait_for_ack(11, timeout=SHORT)
        finally:
            await t.stop()

    async def test_incomplete_chunked_last_state_is_not_replaced_by_an_earlier_one(
        self, log: Path
    ) -> None:
        chunks = _chunk_lines(6)
        _append(log, _state_line(5), *chunks[:-1])  # no END line
        t = await _started(log)
        try:
            # A newer state was being written (or was damaged): the earlier
            # one is not the current one, so nothing is offered.
            assert t.backlog_state is None
        finally:
            await t.stop()

    async def test_truncated_json_last_state_is_not_replaced_by_an_earlier_one(
        self, log: Path
    ) -> None:
        _append(log, _state_line(5), 'AlphaZeroMain: AZ_STATE:{"turn_id": 6, "map": {"wid')
        t = await _started(log)
        try:
            # A newer state was being written (or was damaged): the earlier
            # one is not the current one, so nothing is offered.
            assert t.backlog_state is None
        finally:
            await t.stop()

    @pytest.mark.parametrize(
        "lines",
        [
            [],
            [_ack_line(1), _hb_line(), "Some unrelated log line"],
            ['AlphaZeroMain: AZ_STATE:{"turn_id": 6, "map": {"wid'],
            _chunk_lines(6)[:-1],
        ],
        ids=["empty", "no-state", "only-truncated", "only-incomplete-chunks"],
    )
    async def test_no_complete_state_gives_none_and_never_raises(
        self, log: Path, lines: list[str]
    ) -> None:
        if lines:
            _append(log, *lines)
        t = await _started(log)
        try:
            assert t.backlog_state is None
            assert t.state_seq == 0
            _append(log, _hb_line())
            t._read_new_lines()
            assert t.state_seq == 0
        finally:
            await t.stop()

    async def test_backlog_bytes_zero_turns_the_scan_off(self, log: Path) -> None:
        _append(log, _state_line(4))
        t = await _started(log, backlog_bytes=0)
        try:
            assert t.backlog_state is None
            _append(log, _hb_line())
            t._read_new_lines()
            assert t.state_seq == 0
        finally:
            await t.stop()

    async def test_scan_is_limited_to_the_last_backlog_bytes(self, log: Path) -> None:
        old = _state_line(3)
        _append(log, old, _state_line(4))
        # window smaller than the last line: its first (partial) line is ignored,
        # so nothing complete is left in the window
        t = await _started(log, backlog_bytes=len(_state_line(4)) // 2)
        try:
            assert t.backlog_state is None
        finally:
            await t.stop()

    async def test_window_that_starts_mid_line_ignores_that_partial_line(self, log: Path) -> None:
        first, second = _state_line(3), _state_line(4)
        _append(log, first, second)
        # window covers all of `second` and the tail of `first`
        t = await _started(log, backlog_bytes=len(second) + 1 + 10)
        try:
            backlog = t.backlog_state
            assert backlog is not None
            assert backlog["turn_id"] == 4
        finally:
            await t.stop()

    async def test_tailing_still_starts_at_the_end_of_the_file(self, log: Path) -> None:
        _append(log, _state_line(4), _ack_line(2))
        t = await _started(log)
        try:
            t._read_new_lines()
            assert t.stats["acks_received"] == 0
            assert t.stats["states_received"] == 0
        finally:
            await t.stop()


# --- C3: unread acks are bounded ---


class TestUnreadAckBound:
    def test_bound_is_256(self) -> None:
        assert LogTailer.MAX_UNREAD_ACKS == 256

    async def test_exactly_the_bound_drops_nothing(self, tailer: LogTailer, log: Path) -> None:
        _append(log, *[_ack_line(i) for i in range(1, 257)])
        tailer._read_new_lines()
        assert tailer.stats["acks_dropped"] == 0
        assert (await tailer.wait_for_ack(1, timeout=SHORT))["msg_id"] == 1

    async def test_one_more_drops_the_oldest(self, tailer: LogTailer, log: Path) -> None:
        _append(log, *[_ack_line(i) for i in range(1, 258)])
        tailer._read_new_lines()

        assert tailer.stats["acks_dropped"] == 1
        with pytest.raises(AckTimeoutError):
            await tailer.wait_for_ack(1, timeout=SHORT)
        assert (await tailer.wait_for_ack(2, timeout=SHORT))["msg_id"] == 2
        assert (await tailer.wait_for_ack(257, timeout=SHORT))["msg_id"] == 257

    async def test_many_unread_acks_stay_bounded(self, tailer: LogTailer, log: Path) -> None:
        _append(log, *[_ack_line(i) for i in range(1, 601)])
        tailer._read_new_lines()

        assert tailer.stats["acks_dropped"] == 600 - LogTailer.MAX_UNREAD_ACKS
        kept = []
        for msg_id in range(1, 601):
            try:
                await tailer.wait_for_ack(msg_id, timeout=0.001)
                kept.append(msg_id)
            except AckTimeoutError:
                pass
        # the newest 256 are the ones kept
        assert kept == list(range(600 - LogTailer.MAX_UNREAD_ACKS + 1, 601))

    async def test_an_ack_someone_waits_for_is_never_dropped(
        self, tailer: LogTailer, log: Path
    ) -> None:
        waiter = asyncio.create_task(tailer.wait_for_ack(2, timeout=2.0))
        for _ in range(5):
            await asyncio.sleep(0)

        # ack 2 is the oldest unread-looking ack of the batch, and the bound is exceeded
        _append(log, *[_ack_line(i) for i in range(2, 401)])
        tailer._read_new_lines()

        assert (await waiter)["msg_id"] == 2
        assert tailer.stats["acks_dropped"] >= 1
        with pytest.raises(AckTimeoutError):
            await tailer.wait_for_ack(3, timeout=SHORT)
        assert (await tailer.wait_for_ack(400, timeout=SHORT))["msg_id"] == 400
