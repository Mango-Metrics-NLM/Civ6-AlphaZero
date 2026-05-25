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
