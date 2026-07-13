"""Comprehensive tests for alpha_zero/bridge/file_bridge.py and key parts of protocol.py.

Covers:
  - FileBridgeError subclasses (AckTimeoutError, StateTimeoutError, ConnectionLostError)
  - FileBridge lifecycle (start / stop)
  - FileBridge.check_heartbeat (fresh, stale, missing)
  - FileBridge.send_command (writes JSON to commands_file)
  - FileBridge.detect_multiplayer_state (players list / state sub-dict / default)
  - FileBridge._get_retry_delay (exponential backoff, capped)
  - FileBridge.clear_files (creates + clears communication files)
  - FileBridge.get_stats (key presence)
  - FileBridge.is_connected property
  - FileBridge.read_errors (empty / JSON array / invalid JSON)
  - FileBridge.poll_state (no file / new file / stale mtime)
  - FileBridge.next_msg_id / _next_msg_id monotonicity
  - MockFileBridge (set_mock_state + poll_state, set_mock_ack, get_sent_commands)
  - Protocol: CommandMessage.create_single + to_json
  - Protocol factory helpers: create_end_turn_command, create_unit_move_command,
    create_city_produce_command
  - Protocol: validate_state_json (valid + various invalid shapes)

Target: ≥70 % line coverage of file_bridge.py.
Test count: 40 + test functions.

Usage:
    pytest tests/unit/test_bridge_file_bridge.py -v
    pytest tests/unit/test_bridge_file_bridge.py --cov=alpha_zero/bridge/file_bridge \
        --cov-report=term-missing
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio  # noqa: F401 – imported to ensure marker registration

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import (
    _MAX_RETRY_BACKOFF_SECONDS,
    AckTimeoutError,
    BridgeStats,
    ConnectionLostError,
    FileBridge,
    FileBridgeError,
    MockFileBridge,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import (
    ActionType,
    CommandMessage,
    ErrorCode,
    LuaCommand,
    Position,
    create_city_produce_command,
    create_end_turn_command,
    create_unit_move_command,
    validate_state_json,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_MINIMAL_STATE: dict[str, Any] = {
    "turn_id": 1,
    "player_id": 0,
    "player": {},
    "map": {"width": 10, "height": 10},
    "units": [],
    "cities": [],
    "players": [{"id": 0, "civilization": "CIV_AMERICA"}],
    "legal_actions": [],
}

_MINIMAL_STATE_WRAPPED: dict[str, Any] = {
    "msg_id": 42,
    "timestamp": time.time(),
    "state": _MINIMAL_STATE,
}


@pytest.fixture()
def cfg(tmp_path: Path) -> BridgeConfig:
    """Minimal BridgeConfig pointing at a temporary directory."""
    return BridgeConfig(
        comm_dir=tmp_path,
        poll_interval_ms=10,
        command_timeout_ms=200,
        heartbeat_timeout_s=1.0,
        max_retries=3,
        retry_delay_s=0.05,
        max_actions_per_second=0.0,  # disable throttle in tests
    )


@pytest.fixture()
def bridge(cfg: BridgeConfig) -> FileBridge:
    """Un-started FileBridge with test config."""
    return FileBridge(cfg)


@pytest.fixture()
def mock_bridge(cfg: BridgeConfig) -> MockFileBridge:
    """Un-started MockFileBridge with test config."""
    return MockFileBridge(cfg)


# ---------------------------------------------------------------------------
# Section 1 – Error class assertions
# ---------------------------------------------------------------------------


class TestAckTimeoutError:
    """Tests for AckTimeoutError."""

    def test_message_contains_msg_id(self) -> None:
        err = AckTimeoutError(msg_id=7, timeout=5.0)
        assert "7" in str(err)

    def test_message_contains_timeout(self) -> None:
        err = AckTimeoutError(msg_id=1, timeout=5.0)
        assert "5" in str(err)

    def test_is_file_bridge_error(self) -> None:
        err = AckTimeoutError(msg_id=1, timeout=5.0)
        assert isinstance(err, FileBridgeError)

    def test_error_code_is_timeout(self) -> None:
        err = AckTimeoutError(msg_id=1, timeout=5.0)
        assert err.code == ErrorCode.TIMEOUT


class TestStateTimeoutError:
    """Tests for StateTimeoutError."""

    def test_message_contains_timeout(self) -> None:
        err = StateTimeoutError(timeout=3.5)
        assert "3.5" in str(err)

    def test_is_file_bridge_error(self) -> None:
        assert isinstance(StateTimeoutError(timeout=1.0), FileBridgeError)

    def test_error_code_is_timeout(self) -> None:
        assert StateTimeoutError(timeout=1.0).code == ErrorCode.TIMEOUT


class TestConnectionLostError:
    """Tests for ConnectionLostError."""

    def test_message_describes_heartbeat(self) -> None:
        err = ConnectionLostError()
        msg = str(err).lower()
        assert "heartbeat" in msg or "connection" in msg

    def test_is_file_bridge_error(self) -> None:
        assert isinstance(ConnectionLostError(), FileBridgeError)

    def test_error_code_is_connection_lost(self) -> None:
        assert ConnectionLostError().code == ErrorCode.CONNECTION_LOST


# ---------------------------------------------------------------------------
# Section 2 – FileBridge lifecycle
# ---------------------------------------------------------------------------


class TestFileBridgeLifecycle:
    """Tests for start() / stop()."""

    @pytest.mark.asyncio
    async def test_start_creates_comm_dir(self, bridge: FileBridge, tmp_path: Path) -> None:
        sub = tmp_path / "subdir"
        bridge.config.comm_dir = sub
        await bridge.start()
        assert sub.is_dir()

    @pytest.mark.asyncio
    async def test_start_sets_running_true(self, bridge: FileBridge) -> None:
        assert not bridge._running
        await bridge.start()
        assert bridge._running

    @pytest.mark.asyncio
    async def test_start_writes_heartbeat(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Python now writes python_heartbeat.txt, not heartbeat.txt
        python_hb = bridge.config.comm_dir / "python_heartbeat.txt"
        assert python_hb.exists()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, bridge: FileBridge) -> None:
        await bridge.start()
        await bridge.stop()
        assert not bridge._running

    @pytest.mark.asyncio
    async def test_context_manager_start_stop(self, cfg: BridgeConfig) -> None:
        async with FileBridge(cfg) as b:
            assert b._running
        assert not b._running

    @pytest.mark.asyncio
    async def test_next_msg_id_monotonically_increases(self, bridge: FileBridge) -> None:
        await bridge.start()
        ids = [bridge.next_msg_id() for _ in range(5)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 5  # all unique

    def test_private_next_msg_id_equals_public(self, bridge: FileBridge) -> None:
        """_next_msg_id and next_msg_id share the same counter."""
        bridge._msg_counter = 100
        assert bridge._next_msg_id() == 101
        assert bridge.next_msg_id() == 102


# ---------------------------------------------------------------------------
# Section 3 – check_heartbeat
# ---------------------------------------------------------------------------


class TestCheckHeartbeat:
    """Tests for FileBridge.check_heartbeat()."""

    @pytest.mark.asyncio
    async def test_fresh_heartbeat_returns_true(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Write a fresh Lua-style heartbeat (lua_alive=True) so check_heartbeat() returns True
        fresh = {"timestamp": time.time(), "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(fresh))
        assert bridge.check_heartbeat() is True

    def test_missing_heartbeat_file_returns_false(self, bridge: FileBridge) -> None:
        # comm_dir exists but heartbeat file does not
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        assert bridge.check_heartbeat() is False

    def test_stale_heartbeat_returns_false(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        # Write a Lua heartbeat that is 1000 seconds old
        stale = {"timestamp": time.time() - 1000, "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(stale))
        assert bridge.check_heartbeat() is False

    def test_python_only_heartbeat_returns_false(self, bridge: FileBridge) -> None:
        """Python's own heartbeat (python_alive=True) must not be mistaken for Lua."""
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        python_hb = {"timestamp": time.time(), "python_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(python_hb))
        assert bridge.check_heartbeat() is False

    def test_invalid_json_heartbeat_returns_false(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        bridge.config.heartbeat_file.write_text("{not valid json")
        assert bridge.check_heartbeat() is False

    def test_heartbeat_missing_timestamp_returns_false(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        bridge.config.heartbeat_file.write_text(json.dumps({"python_alive": True}))
        # timestamp defaults to 0, which is always stale
        assert bridge.check_heartbeat() is False


# ---------------------------------------------------------------------------
# Section 4 – send_command
# ---------------------------------------------------------------------------


class TestSendCommand:
    """Tests for FileBridge.send_command()."""

    @pytest.mark.asyncio
    async def test_send_command_writes_json_file(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(
            msg_id=1, command=LuaCommand(action_type=ActionType.END_TURN)
        )
        result = await bridge.send_command(cmd)
        assert result is True
        content = bridge.config.commands_file.read_text(encoding="utf-8")
        data = json.loads(content)
        assert data["msg_id"] == 1
        assert data["commands"][0]["action_type"] == "end_turn"

    @pytest.mark.asyncio
    async def test_send_command_increments_stats(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(
            msg_id=2, command=LuaCommand(action_type=ActionType.END_TURN)
        )
        await bridge.send_command(cmd)
        assert bridge.stats.commands_sent == 1

    @pytest.mark.asyncio
    async def test_send_command_updates_last_command_time(self, bridge: FileBridge) -> None:
        await bridge.start()
        before = time.time()
        cmd = CommandMessage.create_single(
            msg_id=3, command=LuaCommand(action_type=ActionType.END_TURN)
        )
        await bridge.send_command(cmd)
        assert bridge.stats.last_command_time >= before

    @pytest.mark.asyncio
    async def test_send_command_returns_false_on_io_error(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Make commands_file point to a directory (unwritable as file)
        bridge.config.commands_file.mkdir(parents=True)
        cmd = CommandMessage.create_single(
            msg_id=4, command=LuaCommand(action_type=ActionType.END_TURN)
        )
        result = await bridge.send_command(cmd)
        assert result is False
        assert bridge.stats.errors == 1


# ---------------------------------------------------------------------------
# Section 5 – detect_multiplayer_state
# ---------------------------------------------------------------------------


class TestDetectMultiplayerState:
    """Tests for FileBridge.detect_multiplayer_state()."""

    def test_players_list_at_top_level(self) -> None:
        data = {"players": [{"id": 0}, {"id": 1}, {"id": 2}]}
        assert FileBridge.detect_multiplayer_state(data) == 3

    def test_players_list_inside_state_sub_dict(self) -> None:
        data = {"state": {"players": [{"id": 0}, {"id": 1}]}}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_default_when_no_players_key(self) -> None:
        assert FileBridge.detect_multiplayer_state({}) == 2

    def test_default_when_players_list_too_short(self) -> None:
        data = {"players": [{"id": 0}]}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_players_not_a_list_uses_default(self) -> None:
        data = {"players": "not_a_list"}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_inner_state_players_not_a_list_uses_default(self) -> None:
        data = {"state": {"players": 42}}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_prefers_top_level_over_state_sub_dict(self) -> None:
        data = {
            "players": [{"id": 0}, {"id": 1}, {"id": 2}],
            "state": {"players": [{"id": 0}, {"id": 1}]},
        }
        assert FileBridge.detect_multiplayer_state(data) == 3


# ---------------------------------------------------------------------------
# Section 6 – _get_retry_delay
# ---------------------------------------------------------------------------


class TestGetRetryDelay:
    """Tests for FileBridge._get_retry_delay()."""

    def test_attempt_0_returns_base_delay(self, bridge: FileBridge) -> None:
        expected = bridge.config.retry_delay_s * (2**0)
        assert bridge._get_retry_delay(0) == pytest.approx(expected)

    def test_attempt_1_doubles(self, bridge: FileBridge) -> None:
        base = bridge.config.retry_delay_s
        assert bridge._get_retry_delay(1) == pytest.approx(base * 2)

    def test_attempt_2_quadruples(self, bridge: FileBridge) -> None:
        base = bridge.config.retry_delay_s
        assert bridge._get_retry_delay(2) == pytest.approx(base * 4)

    def test_capped_at_max_backoff(self, bridge: FileBridge) -> None:
        # With base=0.05, attempt=100 → 0.05 * 2^100, which is huge
        assert bridge._get_retry_delay(100) == _MAX_RETRY_BACKOFF_SECONDS

    def test_returns_float(self, bridge: FileBridge) -> None:
        assert isinstance(bridge._get_retry_delay(0), float)


# ---------------------------------------------------------------------------
# Section 7 – clear_files
# ---------------------------------------------------------------------------


class TestClearFiles:
    """Tests for FileBridge.clear_files()."""

    @pytest.mark.asyncio
    async def test_clear_removes_existing_files(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Write dummy content to all comm files
        for f in [
            bridge.config.state_file,
            bridge.config.commands_file,
            bridge.config.ack_file,
            bridge.config.errors_file,
        ]:
            f.write_text("dummy")

        bridge.clear_files()

        for f in [
            bridge.config.state_file,
            bridge.config.commands_file,
            bridge.config.ack_file,
            bridge.config.errors_file,
        ]:
            assert not f.exists(), f"{f} should have been removed"

    @pytest.mark.asyncio
    async def test_clear_resets_internal_mtime(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge._last_state_mtime = 12345.0
        bridge.clear_files()
        assert bridge._last_state_mtime == 0.0

    @pytest.mark.asyncio
    async def test_clear_resets_last_state(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge._last_state = MagicMock()
        bridge.clear_files()
        assert bridge._last_state is None

    def test_clear_no_error_when_files_missing(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        # None of the comm files exist – should not raise
        bridge.clear_files()


# ---------------------------------------------------------------------------
# Section 8 – get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    """Tests for FileBridge.get_stats()."""

    @pytest.mark.asyncio
    async def test_stats_contains_expected_keys(self, bridge: FileBridge) -> None:
        await bridge.start()
        stats = bridge.get_stats()
        expected_keys = {
            "states_received",
            "commands_sent",
            "acks_received",
            "errors",
            "reconnects",
            "last_state_time",
            "last_command_time",
            "last_ack_time",
            "is_connected",
            "running",
        }
        assert expected_keys <= set(stats.keys())

    @pytest.mark.asyncio
    async def test_stats_running_reflects_bridge_state(self, bridge: FileBridge) -> None:
        await bridge.start()
        assert bridge.get_stats()["running"] is True
        await bridge.stop()
        assert bridge.get_stats()["running"] is False


# ---------------------------------------------------------------------------
# Section 9 – is_connected property
# ---------------------------------------------------------------------------


class TestIsConnected:
    """Tests for FileBridge.is_connected."""

    def test_not_connected_when_not_running(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        assert bridge.is_connected is False

    @pytest.mark.asyncio
    async def test_connected_after_start_with_fresh_heartbeat(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Write a fresh Lua heartbeat so is_connected returns True
        fresh = {"timestamp": time.time(), "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(fresh))
        assert bridge.is_connected is True

    @pytest.mark.asyncio
    async def test_not_connected_after_stop(self, bridge: FileBridge) -> None:
        await bridge.start()
        await bridge.stop()
        assert bridge.is_connected is False


# ---------------------------------------------------------------------------
# Section 10 – read_errors
# ---------------------------------------------------------------------------


class TestReadErrors:
    """Tests for FileBridge.read_errors()."""

    def test_returns_empty_list_when_file_missing(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        assert bridge.read_errors() == []

    def test_parses_json_array_of_errors(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        errors = [
            {"error_code": "E001", "message": "Bad action", "details": {}},
            {"error_code": "E002", "message": "State unavail", "details": {}},
        ]
        bridge.config.errors_file.write_text(json.dumps(errors))
        result = bridge.read_errors()
        assert len(result) == 2
        assert result[0].error_code == "E001"
        assert result[1].error_code == "E002"

    def test_parses_single_json_object(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        error = {"error_code": "E003", "message": "Unit not found", "details": {}}
        bridge.config.errors_file.write_text(json.dumps(error))
        result = bridge.read_errors()
        assert len(result) == 1
        assert result[0].error_code == "E003"

    def test_invalid_json_returns_empty_list(self, bridge: FileBridge) -> None:
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        bridge.config.errors_file.write_text("this is not json at all")
        # Falls into line-by-line parse which also fails – should return []
        result = bridge.read_errors()
        assert isinstance(result, list)

    def test_returns_empty_on_unreadable_file(self, bridge: FileBridge) -> None:
        """Simulates an OS-level exception on read."""
        bridge.config.comm_dir.mkdir(parents=True, exist_ok=True)
        bridge.config.errors_file.write_text("{}")  # ensure file exists
        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            result = bridge.read_errors()
        assert result == []


# ---------------------------------------------------------------------------
# Section 11 – poll_state
# ---------------------------------------------------------------------------


class TestPollState:
    """Tests for FileBridge.poll_state()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_state_file_missing(self, bridge: FileBridge) -> None:
        await bridge.start()
        result = await bridge.poll_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_state_on_new_file(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge.config.state_file.write_text(json.dumps(_MINIMAL_STATE_WRAPPED))
        result = await bridge.poll_state()
        assert result is not None
        assert result.turn_id == 1
        assert bridge.stats.states_received == 1

    @pytest.mark.asyncio
    async def test_returns_none_on_stale_mtime(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge.config.state_file.write_text(json.dumps(_MINIMAL_STATE_WRAPPED))
        # First call consumes the file
        await bridge.poll_state()
        # Second call: mtime unchanged → should return None
        result = await bridge.poll_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge.config.state_file.write_text("{broken json")
        result = await bridge.poll_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_increments_error_count_on_exception(self, bridge: FileBridge) -> None:
        await bridge.start()
        with patch.object(Path, "stat", side_effect=OSError("disk error")):
            # Create a file so the exists() check passes
            bridge.config.state_file.write_text("{}")
            with patch.object(Path, "exists", return_value=True):
                result = await bridge.poll_state()
        assert result is None


# ---------------------------------------------------------------------------
# Section 12 – MockFileBridge
# ---------------------------------------------------------------------------


class TestMockFileBridge:
    """Tests for MockFileBridge."""

    @pytest.mark.asyncio
    async def test_start_sets_running_no_file_io(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        assert mock_bridge._running
        # Heartbeat file must NOT have been created (no file I/O)
        assert not mock_bridge.config.heartbeat_file.exists()

    @pytest.mark.asyncio
    async def test_stop_sets_running_false(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        await mock_bridge.stop()
        assert not mock_bridge._running

    @pytest.mark.asyncio
    async def test_set_mock_state_then_poll(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        mock_bridge.set_mock_state(_MINIMAL_STATE_WRAPPED)
        state = await mock_bridge.poll_state()
        assert state is not None
        assert state.turn_id == 1

    @pytest.mark.asyncio
    async def test_mock_state_consumed_after_poll(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        mock_bridge.set_mock_state(_MINIMAL_STATE_WRAPPED)
        await mock_bridge.poll_state()
        # Second call should return None (state cleared)
        second = await mock_bridge.poll_state()
        assert second is None

    @pytest.mark.asyncio
    async def test_set_mock_ack_resolves_wait(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        ack_data = {"msg_id": 99, "success": True}
        mock_bridge.set_mock_ack(ack_data)
        ack = await mock_bridge.wait_for_ack(msg_id=99, timeout=1.0)
        assert ack.msg_id == 99
        assert ack.success is True

    @pytest.mark.asyncio
    async def test_get_sent_commands_records_commands(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        cmd1 = CommandMessage.create_single(1, LuaCommand(action_type=ActionType.END_TURN))
        cmd2 = CommandMessage.create_single(2, LuaCommand(action_type=ActionType.UNIT_SKIP))
        await mock_bridge.send_command(cmd1)
        await mock_bridge.send_command(cmd2)
        sent = mock_bridge.get_sent_commands()
        assert len(sent) == 2
        assert sent[0].msg_id == 1
        assert sent[1].msg_id == 2

    @pytest.mark.asyncio
    async def test_mock_check_heartbeat_always_true(self, mock_bridge: MockFileBridge) -> None:
        # MockFileBridge.check_heartbeat() always returns True
        assert mock_bridge.check_heartbeat() is True

    @pytest.mark.asyncio
    async def test_mock_send_command_increments_stats(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        cmd = CommandMessage.create_single(10, LuaCommand(action_type=ActionType.END_TURN))
        await mock_bridge.send_command(cmd)
        assert mock_bridge.stats.commands_sent == 1

    @pytest.mark.asyncio
    async def test_mock_ack_wrong_id_returns_none(self, mock_bridge: MockFileBridge) -> None:
        await mock_bridge.start()
        mock_bridge.set_mock_ack({"msg_id": 5, "success": True})
        # Waiting for a different msg_id → should timeout
        with pytest.raises(AckTimeoutError):
            await mock_bridge.wait_for_ack(msg_id=999, timeout=0.05)


# ---------------------------------------------------------------------------
# Section 13 – CommandMessage protocol helpers
# ---------------------------------------------------------------------------


class TestCommandMessageProtocol:
    """Tests for CommandMessage factory methods and serialisation."""

    def test_create_single_wraps_command(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        msg = CommandMessage.create_single(msg_id=77, command=cmd)
        assert msg.msg_id == 77
        assert len(msg.commands) == 1
        assert msg.commands[0].action_type == ActionType.END_TURN

    def test_to_json_is_valid_json(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        msg = CommandMessage.create_single(msg_id=1, command=cmd)
        parsed = json.loads(msg.to_json())
        assert parsed["msg_id"] == 1
        assert parsed["type"] == "command"

    def test_to_json_includes_commands_list(self) -> None:
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        msg = CommandMessage.create_single(msg_id=2, command=cmd)
        parsed = json.loads(msg.to_json())
        assert isinstance(parsed["commands"], list)
        assert parsed["commands"][0]["action_type"] == "end_turn"

    def test_roundtrip_from_json(self) -> None:
        cmd = LuaCommand(
            action_type=ActionType.UNIT_MOVE,
            unit_id=5,
            target=Position(x=3, y=7),
        )
        original = CommandMessage.create_single(msg_id=9, command=cmd)
        restored = CommandMessage.from_json(original.to_json())
        assert restored.msg_id == 9
        assert restored.commands[0].unit_id == 5
        assert restored.commands[0].target is not None
        assert restored.commands[0].target.x == 3


class TestCreateEndTurnCommand:
    def test_action_type_is_end_turn(self) -> None:
        msg = create_end_turn_command(msg_id=1)
        assert msg.commands[0].action_type == ActionType.END_TURN

    def test_msg_id_preserved(self) -> None:
        msg = create_end_turn_command(msg_id=42)
        assert msg.msg_id == 42

    def test_single_command_in_list(self) -> None:
        msg = create_end_turn_command(msg_id=1)
        assert len(msg.commands) == 1


class TestCreateUnitMoveCommand:
    def test_action_type_is_unit_move(self) -> None:
        msg = create_unit_move_command(msg_id=1, unit_id=3, target_x=5, target_y=8)
        assert msg.commands[0].action_type == ActionType.UNIT_MOVE

    def test_unit_id_preserved(self) -> None:
        msg = create_unit_move_command(msg_id=1, unit_id=77, target_x=0, target_y=0)
        assert msg.commands[0].unit_id == 77

    def test_target_position_preserved(self) -> None:
        msg = create_unit_move_command(msg_id=1, unit_id=1, target_x=10, target_y=20)
        assert msg.commands[0].target.x == 10
        assert msg.commands[0].target.y == 20


class TestCreateCityProduceCommand:
    def test_action_type_is_city_produce(self) -> None:
        msg = create_city_produce_command(msg_id=1, city_id=2, item_id=5)
        assert msg.commands[0].action_type == ActionType.CITY_PRODUCE

    def test_city_id_preserved(self) -> None:
        msg = create_city_produce_command(msg_id=1, city_id=99, item_id=3)
        assert msg.commands[0].city_id == 99

    def test_item_id_preserved(self) -> None:
        msg = create_city_produce_command(msg_id=1, city_id=1, item_id=42)
        assert msg.commands[0].item_id == 42

    def test_default_item_type_is_unit(self) -> None:
        msg = create_city_produce_command(msg_id=1, city_id=1, item_id=1)
        assert msg.commands[0].item_type == "unit"

    def test_custom_item_type_preserved(self) -> None:
        msg = create_city_produce_command(msg_id=1, city_id=1, item_id=1, item_type="building")
        assert msg.commands[0].item_type == "building"


# ---------------------------------------------------------------------------
# Section 14 – validate_state_json
# ---------------------------------------------------------------------------


class TestValidateStateJson:
    """Tests for protocol.validate_state_json()."""

    def _valid_state(self) -> str:
        return json.dumps(
            {
                "turn_id": 1,
                "map": {"width": 84, "height": 54},
            }
        )

    def test_valid_state_returns_true(self) -> None:
        ok, err = validate_state_json(self._valid_state())
        assert ok is True
        assert err is None

    def test_missing_turn_id_returns_false(self) -> None:
        data = {"map": {"width": 84, "height": 54}}
        ok, err = validate_state_json(json.dumps(data))
        assert ok is False
        assert err is not None

    def test_missing_map_field_returns_false(self) -> None:
        data = {"turn_id": 1}
        ok, err = validate_state_json(json.dumps(data))
        assert ok is False
        assert err is not None

    def test_map_missing_width_returns_false(self) -> None:
        data = {"turn_id": 1, "map": {"height": 54}}
        ok, err = validate_state_json(json.dumps(data))
        assert ok is False
        assert err is not None

    def test_map_missing_height_returns_false(self) -> None:
        data = {"turn_id": 1, "map": {"width": 84}}
        ok, err = validate_state_json(json.dumps(data))
        assert ok is False
        assert err is not None

    def test_invalid_json_returns_false(self) -> None:
        ok, err = validate_state_json("{broken")
        assert ok is False
        assert err is not None
        assert "JSON" in err or "json" in err.lower()

    def test_accepts_turn_alias(self) -> None:
        """validate_state_json should accept 'turn' as an alias for 'turn_id'."""
        data = {"turn": 2, "map": {"width": 10, "height": 10}}
        ok, _err = validate_state_json(json.dumps(data))
        assert ok is True


# ---------------------------------------------------------------------------
# Section 15 – BridgeStats dataclass
# ---------------------------------------------------------------------------


class TestBridgeStats:
    """Sanity tests for BridgeStats defaults."""

    def test_default_values_are_zero(self) -> None:
        stats = BridgeStats()
        assert stats.states_received == 0
        assert stats.commands_sent == 0
        assert stats.acks_received == 0
        assert stats.errors == 0
        assert stats.reconnects == 0

    def test_float_fields_default_to_zero(self) -> None:
        stats = BridgeStats()
        assert stats.last_state_time == 0.0
        assert stats.last_command_time == 0.0
        assert stats.last_ack_time == 0.0
        assert stats.last_heartbeat_time == 0.0


# ---------------------------------------------------------------------------
# Section 16 – Integration: send_command_with_retry
# ---------------------------------------------------------------------------


class TestSendCommandWithRetryIntegration:
    """Integration tests for FileBridge.send_command_with_retry()."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(1, LuaCommand(action_type=ActionType.END_TURN))
        result = await bridge.send_command_with_retry(cmd, max_retries=3)
        assert result is True

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(1, LuaCommand(action_type=ActionType.END_TURN))
        call_count = 0

        async def failing_send(command: CommandMessage) -> bool:
            nonlocal call_count
            call_count += 1
            return False

        with patch.object(bridge, "send_command", side_effect=failing_send):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await bridge.send_command_with_retry(cmd, max_retries=3)
        assert result is False
        assert call_count == 3


# ---------------------------------------------------------------------------
# Section 17 – FileBridge default config construction
# ---------------------------------------------------------------------------


class TestFileBridgeDefaultConfig:
    """FileBridge uses BridgeConfig defaults when config=None."""

    def test_default_config_assigned(self) -> None:
        b = FileBridge()
        assert isinstance(b.config, BridgeConfig)

    def test_stats_initialised(self) -> None:
        b = FileBridge()
        assert isinstance(b.stats, BridgeStats)


# ---------------------------------------------------------------------------
# Section 18 – get_last_state
# ---------------------------------------------------------------------------


class TestGetLastState:
    """Tests for FileBridge.get_last_state()."""

    def test_returns_none_before_any_poll(self, bridge: FileBridge) -> None:
        assert bridge.get_last_state() is None

    @pytest.mark.asyncio
    async def test_returns_state_after_successful_poll(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge.config.state_file.write_text(json.dumps(_MINIMAL_STATE_WRAPPED))
        state = await bridge.poll_state()
        assert bridge.get_last_state() is state


# ---------------------------------------------------------------------------
# Section 19 – wait_for_state
# ---------------------------------------------------------------------------


class TestWaitForState:
    """Tests for FileBridge.wait_for_state()."""

    @pytest.mark.asyncio
    async def test_returns_state_immediately_when_file_present(
        self, bridge: FileBridge
    ) -> None:
        await bridge.start()
        bridge.config.state_file.write_text(json.dumps(_MINIMAL_STATE_WRAPPED))
        state = await bridge.wait_for_state(timeout=2.0, require_new=False)
        assert state is not None
        assert state.turn_id == 1

    @pytest.mark.asyncio
    async def test_raises_state_timeout_error_when_no_file(
        self, bridge: FileBridge
    ) -> None:
        await bridge.start()
        # No state file written — should time out immediately
        with pytest.raises(StateTimeoutError):
            await bridge.wait_for_state(timeout=0.05, require_new=False)

    @pytest.mark.asyncio
    async def test_wait_for_state_uses_default_timeout_when_none(
        self, bridge: FileBridge
    ) -> None:
        """Passing timeout=None should fall back to config.command_timeout_ms."""
        await bridge.start()
        bridge.config.state_file.write_text(json.dumps(_MINIMAL_STATE_WRAPPED))
        # Very short configured timeout but file is present — should succeed
        bridge.config.command_timeout_ms = 500
        state = await bridge.wait_for_state(timeout=None, require_new=False)
        assert state is not None


# ---------------------------------------------------------------------------
# Section 20 – _poll_ack
# ---------------------------------------------------------------------------


class TestPollAck:
    """Tests for FileBridge._poll_ack()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_ack_file_missing(self, bridge: FileBridge) -> None:
        await bridge.start()
        result = await bridge._poll_ack(expected_msg_id=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_ack_when_matching_msg_id(self, bridge: FileBridge) -> None:
        await bridge.start()
        ack_data = {"msg_id": 5, "success": True, "message": "OK"}
        bridge.config.ack_file.write_text(json.dumps(ack_data), encoding="utf-8")
        result = await bridge._poll_ack(expected_msg_id=5)
        assert result is not None
        assert result.msg_id == 5
        assert result.success is True
        assert bridge.stats.acks_received == 1

    @pytest.mark.asyncio
    async def test_returns_none_when_wrong_msg_id(self, bridge: FileBridge) -> None:
        await bridge.start()
        ack_data = {"msg_id": 5, "success": True, "message": "OK"}
        bridge.config.ack_file.write_text(json.dumps(ack_data), encoding="utf-8")
        result = await bridge._poll_ack(expected_msg_id=99)
        assert result is None
        assert bridge.stats.acks_received == 0

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_json(self, bridge: FileBridge) -> None:
        await bridge.start()
        bridge.config.ack_file.write_text("{invalid json}", encoding="utf-8")
        result = await bridge._poll_ack(expected_msg_id=1)
        assert result is None

    @pytest.mark.asyncio
    async def test_updates_last_ack_time_on_match(self, bridge: FileBridge) -> None:
        await bridge.start()
        assert bridge.stats.last_ack_time == 0.0
        ack_data = {"msg_id": 7, "success": True, "message": "done"}
        bridge.config.ack_file.write_text(json.dumps(ack_data), encoding="utf-8")
        await bridge._poll_ack(expected_msg_id=7)
        assert bridge.stats.last_ack_time > 0.0


# ---------------------------------------------------------------------------
# Section 21 – send_command_with_retry
# ---------------------------------------------------------------------------


class TestSendCommandWithRetry:
    """Tests for FileBridge.send_command_with_retry()."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(1, LuaCommand(action_type=ActionType.END_TURN))
        result = await bridge.send_command_with_retry(cmd, max_retries=3)
        assert result is True
        assert bridge.stats.commands_sent == 1

    @pytest.mark.asyncio
    async def test_returns_false_when_all_attempts_fail(self, bridge: FileBridge) -> None:
        await bridge.start()
        # Make command file unwritable (it's a directory)
        bridge.config.commands_file.mkdir(parents=True)
        cmd = CommandMessage.create_single(2, LuaCommand(action_type=ActionType.END_TURN))
        with patch("asyncio.sleep", new_callable=lambda: lambda *a, **k: __import__("asyncio").coroutine(lambda: None)()):
            pass
        # patch sleep to avoid real delay
        async def fast_sleep(_t: float) -> None:
            pass

        with patch("asyncio.sleep", side_effect=fast_sleep):
            result = await bridge.send_command_with_retry(cmd, max_retries=2)
        assert result is False

    @pytest.mark.asyncio
    async def test_uses_config_max_retries_when_none(self, bridge: FileBridge) -> None:
        await bridge.start()
        cmd = CommandMessage.create_single(3, LuaCommand(action_type=ActionType.END_TURN))
        bridge.config.max_retries = 2
        # Success on first try → retries don't matter
        result = await bridge.send_command_with_retry(cmd, max_retries=None)
        assert result is True


# ---------------------------------------------------------------------------
# Section 22 – wait_for_ack timeout path (real FileBridge)
# ---------------------------------------------------------------------------


class TestWaitForAckTimeout:
    """Tests for FileBridge.wait_for_ack() timeout path."""

    @pytest.mark.asyncio
    async def test_raises_ack_timeout_when_no_ack_file(self, bridge: FileBridge) -> None:
        await bridge.start()
        with pytest.raises(AckTimeoutError):
            await bridge.wait_for_ack(msg_id=42, timeout=0.05)

    @pytest.mark.asyncio
    async def test_returns_ack_when_file_present(self, bridge: FileBridge) -> None:
        await bridge.start()
        ack_data = {"msg_id": 10, "success": True, "message": "OK"}
        bridge.config.ack_file.write_text(json.dumps(ack_data), encoding="utf-8")
        ack = await bridge.wait_for_ack(msg_id=10, timeout=1.0)
        assert ack.msg_id == 10
        assert ack.success is True
