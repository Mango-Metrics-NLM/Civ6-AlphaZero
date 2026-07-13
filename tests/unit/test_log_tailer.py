"""Tests for alpha_zero.bridge.log_tailer module.

Covers: line parsing, unicode handling, chunk reassembly, state/ack/heartbeat
waiting, log rotation, error handling, and edge cases.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from alpha_zero.bridge.log_tailer import (
    AckTimeoutError,
    LogTailer,
    LogTailerError,
    StateTimeoutError,
)


# --- Fixtures ---


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    """Create a temporary log file."""
    log_file = tmp_path / "Lua.log"
    log_file.write_text("")
    return log_file


@pytest.fixture
def tailer(tmp_log: Path) -> LogTailer:
    """Create a LogTailer instance (not started)."""
    return LogTailer(tmp_log, poll_interval=0.05)


def _write_line(log_path: Path, line: str) -> None:
    """Append a line to the log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Construction tests ---


class TestLogTailerConstruction:
    def test_init_sets_defaults(self, tmp_log: Path) -> None:
        t = LogTailer(tmp_log)
        assert t._poll_interval == 0.1
        assert not t.is_running
        assert t.latest_state is None
        assert t.latest_heartbeat is None

    def test_init_custom_poll_interval(self, tmp_log: Path) -> None:
        t = LogTailer(tmp_log, poll_interval=0.5)
        assert t._poll_interval == 0.5

    def test_stats_initial(self, tmp_log: Path) -> None:
        t = LogTailer(tmp_log)
        stats = t.stats
        assert stats["states_received"] == 0
        assert stats["acks_received"] == 0
        assert stats["heartbeats_received"] == 0
        assert stats["parse_errors"] == 0
        assert stats["lines_processed"] == 0


# --- Line parsing tests ---


class TestLineParsing:
    def test_parse_state_line(self, tailer: LogTailer) -> None:
        state = {"turn_id": 5, "map": {"tiles": []}}
        line = f'AlphaZeroMain: AZ_STATE:{json.dumps(state)}'
        tailer._parse_line(line)
        assert tailer.latest_state is not None
        assert tailer.latest_state["turn_id"] == 5

    def test_parse_state_without_context_prefix(self, tailer: LogTailer) -> None:
        state = {"turn_id": 10}
        line = f'AZ_STATE:{json.dumps(state)}'
        tailer._parse_line(line)
        assert tailer.latest_state is not None
        assert tailer.latest_state["turn_id"] == 10

    def test_parse_heartbeat(self, tailer: LogTailer) -> None:
        hb = {"lua_alive": True, "timestamp": 12345, "turn": 3}
        line = f'AlphaZeroMain: AZ_HB:{json.dumps(hb)}'
        tailer._parse_line(line)
        assert tailer.latest_heartbeat is not None
        assert tailer.latest_heartbeat["lua_alive"] is True

    def test_parse_ack(self, tailer: LogTailer) -> None:
        ack = {"msg_id": 42, "success": True}
        line = f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}'
        tailer._parse_line(line)
        assert 42 in tailer._ack_queue
        assert tailer._ack_queue[42]["success"] is True

    def test_parse_ack_missing_msg_id_dropped(
        self, tailer: LogTailer, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A malformed ack with no msg_id must not be filed under key None.

        Regression test: previously `data.get("msg_id")` could be None, and
        the ack would be stored as `self._ack_queue[None]`, which no waiter
        keyed on the real integer msg_id could ever match.
        """
        ack = {"success": True}
        line = f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}'
        with caplog.at_level(
            "WARNING", logger="alpha_zero.bridge.log_tailer"
        ):
            tailer._parse_line(line)
        assert None not in tailer._ack_queue
        assert tailer._ack_queue == {}
        assert tailer._stats.acks_received == 0
        assert any("msg_id" in r.message for r in caplog.records)

    def test_parse_ack_non_int_msg_id_dropped(self, tailer: LogTailer) -> None:
        """A malformed ack with a non-integer msg_id is dropped, not queued."""
        ack = {"msg_id": "not-an-int", "success": True}
        line = f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}'
        tailer._parse_line(line)
        assert "not-an-int" not in tailer._ack_queue
        assert tailer._stats.acks_received == 0

    def test_parse_unknown_prefix_ignored(self, tailer: LogTailer) -> None:
        tailer._parse_line("AlphaZeroMain: Some random log line")
        assert tailer.latest_state is None
        assert tailer._stats.states_received == 0

    def test_parse_malformed_json(self, tailer: LogTailer) -> None:
        tailer._parse_line("AlphaZeroMain: AZ_STATE:{invalid json!!!")
        assert tailer.latest_state is None
        assert tailer._stats.parse_errors == 1

    def test_parse_empty_json(self, tailer: LogTailer) -> None:
        tailer._parse_line("AlphaZeroMain: AZ_STATE:")
        assert tailer.latest_state is None

    def test_parse_unicode_escaped_keys(self, tailer: LogTailer) -> None:
        """Lua JSON encoder produces \\uXXXX for all chars; json.loads handles it."""
        raw = '{"\\u0074\\u0075\\u0072\\u006e\\u005f\\u0069\\u0064":7}'
        line = f"AlphaZeroMain: AZ_STATE:{raw}"
        tailer._parse_line(line)
        assert tailer.latest_state is not None
        assert tailer.latest_state["turn_id"] == 7

    def test_stats_increment_on_parse(self, tailer: LogTailer) -> None:
        state = {"turn_id": 1}
        tailer._parse_line(f'AZ_STATE:{json.dumps(state)}')
        tailer._parse_line(f'AZ_HB:{json.dumps({"lua_alive": True})}')
        tailer._parse_line(f'AZ_ACK:{json.dumps({"msg_id": 1})}')
        assert tailer._stats.states_received == 1
        assert tailer._stats.heartbeats_received == 1
        assert tailer._stats.acks_received == 1


# --- Chunk reassembly tests ---


class TestChunkReassembly:
    def test_two_chunk_reassembly(self, tailer: LogTailer) -> None:
        full_state = {"turn_id": 42, "data": "x" * 100}
        full_json = json.dumps(full_state)
        mid = len(full_json) // 2
        chunk1 = full_json[:mid]
        chunk2 = full_json[mid:]

        tailer._parse_line(f"AZ_STATE_CHUNK:1/2:{chunk1}")
        tailer._parse_line(f"AZ_STATE_CHUNK:2/2:{chunk2}")
        tailer._parse_line("AZ_STATE_END:2")

        assert tailer.latest_state is not None
        assert tailer.latest_state["turn_id"] == 42

    def test_three_chunk_reassembly(self, tailer: LogTailer) -> None:
        full_state = {"turn_id": 99, "tiles": list(range(50))}
        full_json = json.dumps(full_state)
        chunk_size = len(full_json) // 3 + 1

        for i in range(3):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(full_json))
            chunk = full_json[start:end]
            tailer._parse_line(f"AZ_STATE_CHUNK:{i+1}/3:{chunk}")

        tailer._parse_line("AZ_STATE_END:3")
        assert tailer.latest_state is not None
        assert tailer.latest_state["turn_id"] == 99

    def test_single_chunk_state(self, tailer: LogTailer) -> None:
        """Small state that fits in one line uses AZ_STATE: prefix directly."""
        state = {"turn_id": 1}
        tailer._parse_line(f'AZ_STATE:{json.dumps(state)}')
        assert tailer.latest_state["turn_id"] == 1
        assert tailer._stats.chunks_received == 0

    def test_missing_chunk_logged(self, tailer: LogTailer) -> None:
        """Missing chunk still attempts reassembly."""
        tailer._parse_line("AZ_STATE_CHUNK:1/3:{'a':")
        tailer._parse_line("AZ_STATE_CHUNK:3/3:}")
        # Chunk 2 missing
        tailer._parse_line("AZ_STATE_END:3")
        # Should fail JSON parse due to missing middle
        assert tailer._stats.parse_errors >= 1

    def test_malformed_chunk_header(self, tailer: LogTailer) -> None:
        tailer._parse_line("AZ_STATE_CHUNK:bad_format")
        assert tailer._stats.parse_errors == 1

    def test_invalid_chunk_end_total(self, tailer: LogTailer) -> None:
        tailer._parse_line("AZ_STATE_END:abc")
        assert tailer._stats.parse_errors == 1

    def test_chunk_buffer_cleared_after_assembly(self, tailer: LogTailer) -> None:
        state = {"x": 1}
        full_json = json.dumps(state)
        tailer._parse_line(f"AZ_STATE_CHUNK:1/1:{full_json}")
        tailer._parse_line("AZ_STATE_END:1")
        assert len(tailer._chunk_buffer) == 0
        assert tailer._chunk_total == 0


# --- Heartbeat tests ---


class TestHeartbeat:
    def test_check_heartbeat_no_data(self, tailer: LogTailer) -> None:
        assert tailer.check_heartbeat() is False

    def test_check_heartbeat_fresh(self, tailer: LogTailer) -> None:
        hb = {"lua_alive": True, "timestamp": int(time.time())}
        tailer._parse_line(f'AZ_HB:{json.dumps(hb)}')
        assert tailer.check_heartbeat(max_age=5.0) is True

    def test_check_heartbeat_stale(self, tailer: LogTailer) -> None:
        hb = {"lua_alive": True, "timestamp": int(time.time())}
        tailer._parse_line(f'AZ_HB:{json.dumps(hb)}')
        # Fake old heartbeat time
        tailer._heartbeat_time = time.monotonic() - 10.0
        assert tailer.check_heartbeat(max_age=5.0) is False


# --- Async tests ---


@pytest.mark.asyncio
class TestAsyncOperations:
    async def test_context_manager(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            assert tailer.is_running
        assert not tailer.is_running

    async def test_start_stop(self, tmp_log: Path) -> None:
        tailer = LogTailer(tmp_log, poll_interval=0.05)
        await tailer.start()
        assert tailer.is_running
        await tailer.stop()
        assert not tailer.is_running

    async def test_double_start_idempotent(self, tmp_log: Path) -> None:
        tailer = LogTailer(tmp_log, poll_interval=0.05)
        await tailer.start()
        await tailer.start()  # Should not error
        assert tailer.is_running
        await tailer.stop()

    async def test_wait_for_state_timeout(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            with pytest.raises(StateTimeoutError) as exc_info:
                await tailer.wait_for_state(timeout=0.2)
            assert exc_info.value.timeout == 0.2

    async def test_wait_for_state_success(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            # Write state after a small delay
            async def write_state() -> None:
                await asyncio.sleep(0.1)
                state = {"turn_id": 15, "map": {"tiles": []}}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_STATE:{json.dumps(state)}')

            asyncio.create_task(write_state())
            result = await tailer.wait_for_state(timeout=2.0)
            assert result["turn_id"] == 15

    async def test_wait_for_ack_timeout(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            with pytest.raises(AckTimeoutError) as exc_info:
                await tailer.wait_for_ack(msg_id=999, timeout=0.2)
            assert exc_info.value.msg_id == 999

    async def test_wait_for_ack_success(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            async def write_ack() -> None:
                await asyncio.sleep(0.1)
                ack = {"msg_id": 42, "success": True}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}')

            asyncio.create_task(write_ack())
            result = await tailer.wait_for_ack(msg_id=42, timeout=2.0)
            assert result["success"] is True

    async def test_wait_for_ack_already_received(self, tmp_log: Path) -> None:
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            # Pre-inject ack
            tailer._ack_queue[10] = {"msg_id": 10, "success": True}
            result = await tailer.wait_for_ack(msg_id=10, timeout=0.5)
            assert result["msg_id"] == 10

    async def test_log_rotation_detected(self, tmp_log: Path) -> None:
        """When log file is truncated, tailer resets position."""
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            # Write some data
            _write_line(tmp_log, f'AZ_STATE:{json.dumps({"turn_id": 1})}')
            await asyncio.sleep(0.15)
            assert tailer.latest_state is not None

            # Truncate the file (simulating game restart)
            tmp_log.write_text("")
            await asyncio.sleep(0.15)

            # Write new data
            _write_line(tmp_log, f'AZ_STATE:{json.dumps({"turn_id": 2})}')
            await asyncio.sleep(0.15)
            assert tailer.latest_state["turn_id"] == 2

    async def test_file_not_found_timeout(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.log"
        tailer = LogTailer(missing, poll_interval=0.05)
        # asyncio.wait_for raises TimeoutError when start()'s internal
        # wait loop exceeds the external timeout
        with pytest.raises((asyncio.TimeoutError, LogTailerError)):
            await asyncio.wait_for(tailer.start(), timeout=1.0)

    async def test_chunked_state_via_file(self, tmp_log: Path) -> None:
        """End-to-end test: write chunked state to file, read via tailer."""
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            state = {"turn_id": 77, "data": "A" * 200}
            full_json = json.dumps(state)
            mid = len(full_json) // 2

            async def write_chunks() -> None:
                await asyncio.sleep(0.1)
                _write_line(tmp_log, f"AlphaZeroMain: AZ_STATE_CHUNK:1/2:{full_json[:mid]}")
                _write_line(tmp_log, f"AlphaZeroMain: AZ_STATE_CHUNK:2/2:{full_json[mid:]}")
                _write_line(tmp_log, "AlphaZeroMain: AZ_STATE_END:2")

            asyncio.create_task(write_chunks())
            result = await tailer.wait_for_state(timeout=2.0)
            assert result["turn_id"] == 77

    async def test_mixed_lines_processed(self, tmp_log: Path) -> None:
        """Test that non-AZ lines are silently skipped."""
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            _write_line(tmp_log, "TutorialUIRoot: Tutorial loaded")
            _write_line(tmp_log, "LoadScreen: OnLoadGameViewStateDone")
            _write_line(tmp_log, f'AlphaZeroMain: AZ_HB:{json.dumps({"lua_alive": True})}')
            _write_line(tmp_log, "Some other log line")
            await asyncio.sleep(0.15)
            assert tailer._stats.lines_processed >= 3
            assert tailer._stats.heartbeats_received == 1
            assert tailer.latest_state is None  # No state was written


# --- Exception tests ---


class TestExceptions:
    def test_state_timeout_error_message(self) -> None:
        err = StateTimeoutError(30.0)
        assert "30.0s" in str(err)
        assert err.timeout == 30.0

    def test_ack_timeout_error_message(self) -> None:
        err = AckTimeoutError(42, 10.0)
        assert "42" in str(err)
        assert "10.0s" in str(err)
        assert err.msg_id == 42
        assert err.timeout == 10.0

    def test_log_tailer_error_base(self) -> None:
        err = LogTailerError("test error")
        assert "test error" in str(err)


# --- Additional coverage: parse-failure and lifecycle edge paths ---


class TestReadNewLinesEdgeCases:
    """Direct (synchronous) exercise of `_read_new_lines()` branches that
    the async end-to-end tests above don't reliably hit."""

    def test_no_open_file_is_a_noop(self, tailer: LogTailer) -> None:
        """Without an open file handle, reading must return immediately
        rather than raising an AttributeError on self._file.tell()."""
        assert tailer._file is None
        tailer._read_new_lines()  # must not raise
        assert tailer._stats.lines_processed == 0

    def test_getsize_oserror_is_swallowed(
        self, tailer: LogTailer, tmp_log: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the log file vanishes between polls (os.path.getsize raises
        OSError), the read is skipped rather than crashing the poll loop."""
        tailer._file = open(tmp_log, "r", encoding="utf-8")
        try:
            def _raise_oserror(path):
                raise OSError("file vanished")

            monkeypatch.setattr(
                "alpha_zero.bridge.log_tailer.os.path.getsize", _raise_oserror
            )
            tailer._read_new_lines()  # must not raise
            assert tailer._stats.lines_processed == 0
        finally:
            tailer._file.close()

    def test_blank_lines_are_skipped_within_batch(
        self, tailer: LogTailer, tmp_log: Path
    ) -> None:
        """A batch containing a blank line alongside real content must skip
        the blank line and continue processing the rest of the batch."""
        with open(tmp_log, "a", encoding="utf-8") as f:
            f.write("\n")
            f.write(f'AZ_HB:{json.dumps({"lua_alive": True})}\n')

        tailer._file = open(tmp_log, "r", encoding="utf-8")
        try:
            tailer._read_new_lines()
        finally:
            tailer._file.close()

        assert tailer._stats.lines_processed == 1
        assert tailer._stats.heartbeats_received == 1


class TestHeartbeatAckParseFailures:
    """Malformed heartbeat/ack payloads must be dropped, not crash or
    populate state with garbage."""

    def test_malformed_heartbeat_json_is_dropped(self, tailer: LogTailer) -> None:
        tailer._handle_heartbeat("{not valid json")
        assert tailer.latest_heartbeat is None
        assert tailer._stats.heartbeats_received == 0
        assert tailer._stats.parse_errors == 1

    def test_malformed_ack_json_is_dropped(self, tailer: LogTailer) -> None:
        tailer._handle_ack("{not valid json")
        assert tailer._ack_queue == {}
        assert tailer._stats.acks_received == 0
        assert tailer._stats.parse_errors == 1


@pytest.mark.asyncio
class TestLifecycleEdgeCases:
    async def test_stop_without_start_is_a_noop(self, tmp_log: Path) -> None:
        """Calling stop() on a tailer that was never started must not raise
        (no poll task to cancel, no file handle to close)."""
        tailer = LogTailer(tmp_log, poll_interval=0.05)
        await tailer.stop()  # must not raise
        assert not tailer.is_running

    async def test_stop_after_poll_task_already_done(self, tmp_log: Path) -> None:
        """If the poll task has already finished by the time stop() runs,
        the cancel-and-await branch is skipped rather than double-awaited."""
        tailer = LogTailer(tmp_log, poll_interval=0.05)
        await tailer.start()

        async def _noop() -> None:
            return None

        # Replace the running task with one that's already finished.
        tailer._poll_task = asyncio.create_task(_noop())
        await tailer._poll_task
        assert tailer._poll_task.done()

        await tailer.stop()  # must not raise
        assert not tailer.is_running

    async def test_poll_loop_exits_immediately_when_not_running(
        self, tmp_log: Path
    ) -> None:
        """_poll_loop()'s while condition is checked before ever entering the
        body when _running is already False."""
        tailer = LogTailer(tmp_log, poll_interval=0.05)
        tailer._running = False
        await asyncio.wait_for(tailer._poll_loop(), timeout=1.0)
        assert tailer._stats.lines_processed == 0

    async def test_poll_loop_survives_read_exception(
        self, tmp_log: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unexpected exception from _read_new_lines() inside the poll
        loop is logged and swallowed, not left to kill the background task."""
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            call_count = 0
            original = tailer._read_new_lines

            def _flaky_read() -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RuntimeError("boom")
                original()

            tailer._read_new_lines = _flaky_read
            await asyncio.sleep(0.2)
            # The poll loop must still be alive after the injected exception.
            assert tailer.is_running
            assert call_count >= 2

    async def test_wait_for_state_raises_if_event_set_without_state(
        self, tmp_log: Path
    ) -> None:
        """Defensive branch: if the state event is set but latest_state is
        still None (e.g. a race), wait_for_state must still raise rather than
        returning None to the caller."""
        async with LogTailer(tmp_log, poll_interval=0.05) as tailer:
            tailer._state_event.set()
            assert tailer._latest_state is None
            with pytest.raises(StateTimeoutError):
                await tailer.wait_for_state(timeout=0.5)

    async def test_start_raises_after_30s_wait_for_missing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the log file never appears, start() raises LogTailerError
        once the 30s wait budget is exhausted (rather than waiting forever)."""
        missing = tmp_path / "never_created.log"
        tailer = LogTailer(missing, poll_interval=0.05)

        # Fast-forward time so the 30s budget is exhausted on the first check.
        real_monotonic = time.monotonic
        fake_time = {"t": real_monotonic()}

        def _fake_monotonic() -> float:
            fake_time["t"] += 40.0
            return fake_time["t"]

        monkeypatch.setattr(
            "alpha_zero.bridge.log_tailer.time.monotonic", _fake_monotonic
        )
        with pytest.raises(LogTailerError, match="not found after 30s"):
            await tailer.start()
