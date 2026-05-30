"""Comprehensive tests for alpha_zero.bridge.file_bridge module.

Tests cover:
- BridgeStats defaults
- Exception hierarchy and attributes
- detect_multiplayer_state with various inputs
- FileBridge init, start/stop, context manager
- check_heartbeat (recent, expired, missing, invalid)
- poll_state (no file, new state, old state, invalid JSON)
- wait_for_state timeout
- send_command success
- send_command_with_retry
- _get_retry_delay exponential backoff
- clear_files
- get_stats
- is_connected property
- MockFileBridge operations
- _throttle_action rate limiting
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

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
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state_data(
    msg_id: int = 1,
    turn_id: int = 1,
    player_id: int = 0,
    players: list[dict[str, Any]] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build a minimal valid state dict for writing to state.json."""
    if players is None:
        players = [{"id": 0, "gold": 100}]
    return {
        "msg_id": msg_id,
        "type": "state",
        "timestamp": timestamp or time.time(),
        "turn_id": turn_id,
        "player_id": player_id,
        "player": {"id": player_id, "gold": 100, "science": 50},
        "players": players,
        "map": {"width": 16, "height": 16, "tiles": []},
        "units": [],
        "cities": [],
        "legal_actions": [],
    }


def _make_ack_data(
    msg_id: int = 1,
    success: bool = True,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """Build a minimal valid ack dict for writing to ack.json."""
    return {
        "msg_id": msg_id,
        "type": "ack",
        "timestamp": timestamp or time.time(),
        "success": success,
        "status": "success" if success else "error",
    }


def _make_command(msg_id: int = 1) -> CommandMessage:
    """Build a minimal CommandMessage."""
    return CommandMessage(
        msg_id=msg_id,
        commands=[LuaCommand(action_type=ActionType.END_TURN)],
    )


def _write_state_file(config: BridgeConfig, data: dict[str, Any]) -> None:
    """Write state data to the config's state file."""
    config.state_file.write_text(json.dumps(data), encoding="utf-8")


def _write_ack_file(config: BridgeConfig, data: dict[str, Any]) -> None:
    """Write ack data to the config's ack file."""
    config.ack_file.write_text(json.dumps(data), encoding="utf-8")


def _write_heartbeat(config: BridgeConfig, timestamp: float | None = None) -> None:
    """Write a heartbeat file with the given timestamp."""
    ts = timestamp or time.time()
    config.heartbeat_file.write_text(
        json.dumps({"timestamp": ts, "python_alive": True})
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def bridge_config(tmp_path: Path) -> BridgeConfig:
    """Return a BridgeConfig pointing at a temporary directory."""
    return BridgeConfig(comm_dir=tmp_path)


@pytest.fixture()
def bridge(bridge_config: BridgeConfig) -> FileBridge:
    """Return a FileBridge with tmp_path-based config (not started)."""
    return FileBridge(bridge_config)


# ===========================================================================
# 1. BridgeStats defaults
# ===========================================================================

class TestBridgeStats:
    """Tests for the BridgeStats dataclass."""

    def test_defaults_are_zero(self) -> None:
        stats = BridgeStats()
        assert stats.states_received == 0
        assert stats.commands_sent == 0
        assert stats.acks_received == 0
        assert stats.errors == 0
        assert stats.reconnects == 0
        assert stats.last_state_time == 0.0
        assert stats.last_command_time == 0.0
        assert stats.last_ack_time == 0.0
        assert stats.last_heartbeat_time == 0.0

    def test_mutable_fields(self) -> None:
        stats = BridgeStats()
        stats.states_received = 5
        stats.errors = 2
        assert stats.states_received == 5
        assert stats.errors == 2


# ===========================================================================
# 2. Exception hierarchy and attributes
# ===========================================================================

class TestExceptions:
    """Tests for FileBridgeError and its subclasses."""

    def test_file_bridge_error_default_code(self) -> None:
        err = FileBridgeError("something broke")
        assert str(err) == "something broke"
        assert err.code == ErrorCode.UNKNOWN

    def test_file_bridge_error_custom_code(self) -> None:
        err = FileBridgeError("io problem", code=ErrorCode.FILE_IO_ERROR)
        assert err.code == ErrorCode.FILE_IO_ERROR

    def test_file_bridge_error_is_exception(self) -> None:
        assert issubclass(FileBridgeError, Exception)

    def test_state_timeout_error(self) -> None:
        err = StateTimeoutError(timeout=5.0)
        assert "5.0" in str(err)
        assert err.code == ErrorCode.TIMEOUT
        assert isinstance(err, FileBridgeError)

    def test_ack_timeout_error(self) -> None:
        err = AckTimeoutError(msg_id=42, timeout=3.0)
        assert "42" in str(err)
        assert "3.0" in str(err)
        assert err.code == ErrorCode.TIMEOUT
        assert isinstance(err, FileBridgeError)

    def test_connection_lost_error(self) -> None:
        err = ConnectionLostError()
        assert "heartbeat" in str(err).lower()
        assert err.code == ErrorCode.CONNECTION_LOST
        assert isinstance(err, FileBridgeError)


# ===========================================================================
# 3. detect_multiplayer_state
# ===========================================================================

class TestDetectMultiplayerState:
    """Tests for FileBridge.detect_multiplayer_state static method."""

    def test_players_at_top_level(self) -> None:
        data = {"players": [{"id": 0}, {"id": 1}, {"id": 2}]}
        assert FileBridge.detect_multiplayer_state(data) == 3

    def test_players_nested_in_state(self) -> None:
        data = {"state": {"players": [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]}}
        assert FileBridge.detect_multiplayer_state(data) == 4

    def test_empty_players_defaults_to_two(self) -> None:
        data: dict[str, Any] = {"players": []}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_single_player_defaults_to_two(self) -> None:
        data = {"players": [{"id": 0}]}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_no_players_key_defaults_to_two(self) -> None:
        data: dict[str, Any] = {"some_other_key": "value"}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_players_not_a_list_defaults_to_two(self) -> None:
        data: dict[str, Any] = {"players": "not_a_list"}
        assert FileBridge.detect_multiplayer_state(data) == 2

    def test_top_level_takes_priority_over_nested(self) -> None:
        data: dict[str, Any] = {
            "players": [{"id": 0}, {"id": 1}, {"id": 2}],
            "state": {"players": [{"id": 0}, {"id": 1}]},
        }
        assert FileBridge.detect_multiplayer_state(data) == 3


# ===========================================================================
# 4. FileBridge init
# ===========================================================================

class TestFileBridgeInit:
    """Tests for FileBridge.__init__."""

    def test_default_config(self) -> None:
        bridge = FileBridge()
        assert bridge.config is not None
        assert isinstance(bridge.stats, BridgeStats)
        assert bridge._running is False
        assert bridge._msg_counter == 0

    def test_custom_config(self, bridge_config: BridgeConfig) -> None:
        bridge = FileBridge(bridge_config)
        assert bridge.config is bridge_config
        assert bridge._running is False


# ===========================================================================
# 5. FileBridge start / stop / context manager
# ===========================================================================

class TestFileBridgeStartStop:
    """Tests for start, stop, and async context manager."""

    @pytest.mark.asyncio
    async def test_start_creates_comm_dir_and_running(self, tmp_path: Path) -> None:
        subdir = tmp_path / "bridge_dir"
        config = BridgeConfig(comm_dir=subdir)
        bridge = FileBridge(config)

        await bridge.start()
        assert bridge._running is True
        assert subdir.exists()
        assert config.heartbeat_file.exists()

        await bridge.stop()
        assert bridge._running is False

    @pytest.mark.asyncio
    async def test_context_manager(self, bridge_config: BridgeConfig) -> None:
        async with FileBridge(bridge_config) as fb:
            assert fb._running is True
            assert isinstance(fb, FileBridge)
        assert fb._running is False

    @pytest.mark.asyncio
    async def test_start_writes_heartbeat(self, bridge_config: BridgeConfig) -> None:
        bridge = FileBridge(bridge_config)
        await bridge.start()
        assert bridge_config.heartbeat_file.exists()
        data = json.loads(bridge_config.heartbeat_file.read_text())
        assert data["python_alive"] is True
        assert "timestamp" in data
        await bridge.stop()

    @pytest.mark.asyncio
    async def test_start_sets_msg_counter(self, bridge_config: BridgeConfig) -> None:
        bridge = FileBridge(bridge_config)
        before = int(time.time() * 1000)
        await bridge.start()
        after = int(time.time() * 1000)
        assert before <= bridge._msg_counter <= after
        await bridge.stop()


# ===========================================================================
# 6. check_heartbeat
# ===========================================================================

class TestCheckHeartbeat:
    """Tests for FileBridge.check_heartbeat."""

    def test_recent_heartbeat_returns_true(self, bridge: FileBridge) -> None:
        _write_heartbeat(bridge.config, time.time())
        assert bridge.check_heartbeat() is True

    def test_expired_heartbeat_returns_false(self, bridge: FileBridge) -> None:
        # Write a heartbeat far in the past
        old_ts = time.time() - bridge.config.heartbeat_timeout_s - 10
        _write_heartbeat(bridge.config, old_ts)
        assert bridge.check_heartbeat() is False

    def test_missing_heartbeat_file_returns_false(self, bridge: FileBridge) -> None:
        # No heartbeat file written
        assert bridge.check_heartbeat() is False

    def test_invalid_json_in_heartbeat_returns_false(self, bridge: FileBridge) -> None:
        bridge.config.heartbeat_file.write_text("NOT VALID JSON{{{")
        assert bridge.check_heartbeat() is False


# ===========================================================================
# 7. poll_state
# ===========================================================================

class TestPollState:
    """Tests for FileBridge.poll_state."""

    @pytest.mark.asyncio
    async def test_no_state_file_returns_none(self, bridge: FileBridge) -> None:
        result = await bridge.poll_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_new_state_file_returns_data(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        _write_state_file(bridge_config, _make_state_data(turn_id=5))
        result = await bridge.poll_state()
        assert result is not None
        assert result.turn_id == 5
        assert bridge.stats.states_received == 1

    @pytest.mark.asyncio
    async def test_old_state_not_re_read(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        _write_state_file(bridge_config, _make_state_data())
        first = await bridge.poll_state()
        assert first is not None

        # Same file, mtime unchanged -> None
        second = await bridge.poll_state()
        assert second is None
        assert bridge.stats.states_received == 1

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.state_file.write_text("{{{{bad json", encoding="utf-8")
        result = await bridge.poll_state()
        assert result is None

    @pytest.mark.asyncio
    async def test_poll_state_updates_last_state(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        _write_state_file(bridge_config, _make_state_data(turn_id=7))
        await bridge.poll_state()
        last = bridge.get_last_state()
        assert last is not None
        assert last.turn_id == 7


# ===========================================================================
# 8. wait_for_state timeout
# ===========================================================================

class TestWaitForState:
    """Tests for FileBridge.wait_for_state."""

    @pytest.mark.asyncio
    async def test_timeout_raises_state_timeout_error(
        self, bridge: FileBridge
    ) -> None:
        with pytest.raises(StateTimeoutError):
            await bridge.wait_for_state(timeout=0.05)

    @pytest.mark.asyncio
    async def test_returns_state_when_available(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        # Write state before waiting
        _write_state_file(bridge_config, _make_state_data(turn_id=10))
        state = await bridge.wait_for_state(timeout=1.0)
        assert state.turn_id == 10


# ===========================================================================
# 9. send_command
# ===========================================================================

class TestSendCommand:
    """Tests for FileBridge.send_command."""

    @pytest.mark.asyncio
    async def test_send_command_success(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        bridge = FileBridge(bridge_config)
        # Disable throttling for fast tests
        bridge.config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )

        cmd = _make_command(msg_id=99)
        result = await bridge.send_command(cmd)
        assert result is True
        assert bridge.stats.commands_sent == 1
        assert bridge.config.commands_file.exists()

        # Verify written JSON
        written = json.loads(bridge.config.commands_file.read_text())
        assert written["msg_id"] == 99

    @pytest.mark.asyncio
    async def test_send_command_increments_stats(
        self, bridge_config: BridgeConfig
    ) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        await bridge.send_command(_make_command(msg_id=1))
        await bridge.send_command(_make_command(msg_id=2))
        assert bridge.stats.commands_sent == 2


# ===========================================================================
# 10. send_command_with_retry
# ===========================================================================

class TestSendCommandWithRetry:
    """Tests for FileBridge.send_command_with_retry."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_try(
        self, bridge_config: BridgeConfig
    ) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        cmd = _make_command(msg_id=10)
        result = await bridge.send_command_with_retry(cmd, max_retries=3)
        assert result is True

    @pytest.mark.asyncio
    async def test_retries_on_failure(self, bridge_config: BridgeConfig) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        call_count = 0
        original_send = bridge.send_command

        async def flaky_send(command: CommandMessage) -> bool:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return False
            return await original_send(command)

        bridge.send_command = flaky_send  # type: ignore[assignment]

        with patch("alpha_zero.bridge.file_bridge.asyncio.sleep", return_value=None):
            result = await bridge.send_command_with_retry(
                _make_command(), max_retries=5
            )
        assert result is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_fail(self, bridge_config: BridgeConfig) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        async def always_fail(command: CommandMessage) -> bool:
            return False

        bridge.send_command = always_fail  # type: ignore[assignment]

        with patch("alpha_zero.bridge.file_bridge.asyncio.sleep", return_value=None):
            result = await bridge.send_command_with_retry(
                _make_command(), max_retries=3
            )
        assert result is False


# ===========================================================================
# 11. _get_retry_delay exponential backoff
# ===========================================================================

class TestGetRetryDelay:
    """Tests for FileBridge._get_retry_delay."""

    def test_first_attempt_base_delay(self, bridge: FileBridge) -> None:
        # attempt=0 => base_delay * 2^0 = base_delay
        delay = bridge._get_retry_delay(0)
        assert delay == bridge.config.retry_delay_s

    def test_exponential_growth(self, bridge: FileBridge) -> None:
        base = bridge.config.retry_delay_s
        assert bridge._get_retry_delay(1) == base * 2
        assert bridge._get_retry_delay(2) == base * 4
        assert bridge._get_retry_delay(3) == base * 8

    def test_cap_at_max_retry_backoff(self, bridge: FileBridge) -> None:
        # Large attempt should be capped
        delay = bridge._get_retry_delay(100)
        assert delay == _MAX_RETRY_BACKOFF_SECONDS

    def test_cap_value_is_60(self) -> None:
        assert _MAX_RETRY_BACKOFF_SECONDS == 60.0


# ===========================================================================
# 12. clear_files
# ===========================================================================

class TestClearFiles:
    """Tests for FileBridge.clear_files."""

    def test_clear_existing_files(self, bridge: FileBridge) -> None:
        config = bridge.config
        config.ensure_comm_dir()

        # Create some files
        config.state_file.write_text("{}")
        config.commands_file.write_text("{}")
        config.ack_file.write_text("{}")
        config.errors_file.write_text("{}")

        bridge.clear_files()

        assert not config.state_file.exists()
        assert not config.commands_file.exists()
        assert not config.ack_file.exists()
        assert not config.errors_file.exists()

    def test_clear_resets_internal_state(self, bridge: FileBridge) -> None:
        bridge._last_state_mtime = 123.0
        bridge._last_state = object()  # type: ignore[assignment]
        bridge.clear_files()
        assert bridge._last_state_mtime == 0.0
        assert bridge._last_state is None

    def test_clear_no_files_no_error(self, bridge: FileBridge) -> None:
        # Should not raise even when files do not exist
        bridge.clear_files()


# ===========================================================================
# 13. get_stats
# ===========================================================================

class TestGetStats:
    """Tests for FileBridge.get_stats."""

    def test_initial_stats(self, bridge: FileBridge) -> None:
        stats = bridge.get_stats()
        assert stats["states_received"] == 0
        assert stats["commands_sent"] == 0
        assert stats["acks_received"] == 0
        assert stats["errors"] == 0
        assert stats["reconnects"] == 0
        assert stats["running"] is False

    def test_stats_reflect_operations(
        self, bridge_config: BridgeConfig
    ) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        bridge.stats.commands_sent = 3
        bridge.stats.errors = 1
        stats = bridge.get_stats()
        assert stats["commands_sent"] == 3
        assert stats["errors"] == 1

    def test_stats_includes_is_connected(self, bridge: FileBridge) -> None:
        stats = bridge.get_stats()
        assert "is_connected" in stats

    def test_stats_includes_running(self, bridge: FileBridge) -> None:
        stats = bridge.get_stats()
        assert "running" in stats
        assert stats["running"] is False


# ===========================================================================
# 14. is_connected property
# ===========================================================================

class TestIsConnected:
    """Tests for FileBridge.is_connected property."""

    @pytest.mark.asyncio
    async def test_not_running_not_connected(self, bridge: FileBridge) -> None:
        assert bridge.is_connected is False

    @pytest.mark.asyncio
    async def test_running_no_heartbeat_not_connected(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = FileBridge(bridge_config)
        bridge._running = True
        # No heartbeat file -> not connected
        assert bridge.is_connected is False

    @pytest.mark.asyncio
    async def test_running_with_heartbeat_connected(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge = FileBridge(bridge_config)
        await bridge.start()
        # start() writes a heartbeat
        assert bridge.is_connected is True
        await bridge.stop()


# ===========================================================================
# 15. _next_msg_id
# ===========================================================================

class TestNextMsgId:
    """Tests for FileBridge._next_msg_id."""

    def test_increments(self, bridge: FileBridge) -> None:
        first = bridge._next_msg_id()
        second = bridge._next_msg_id()
        assert second == first + 1

    def test_starts_from_counter(self, bridge: FileBridge) -> None:
        bridge._msg_counter = 100
        assert bridge._next_msg_id() == 101


# ===========================================================================
# 16. get_last_state
# ===========================================================================

class TestGetLastState:
    """Tests for FileBridge.get_last_state."""

    def test_initially_none(self, bridge: FileBridge) -> None:
        assert bridge.get_last_state() is None

    @pytest.mark.asyncio
    async def test_after_poll(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        _write_state_file(bridge_config, _make_state_data(turn_id=3))
        await bridge.poll_state()
        state = bridge.get_last_state()
        assert state is not None
        assert state.turn_id == 3


# ===========================================================================
# 17. MockFileBridge
# ===========================================================================

class TestMockFileBridge:
    """Tests for MockFileBridge."""

    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        mock = MockFileBridge()
        await mock.start()
        assert mock._running is True
        await mock.stop()
        assert mock._running is False

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        async with MockFileBridge() as mock:
            assert mock._running is True
        assert mock._running is False

    @pytest.mark.asyncio
    async def test_set_mock_state_and_poll(self) -> None:
        mock = MockFileBridge()
        mock.set_mock_state(_make_state_data(turn_id=42))
        state = await mock.poll_state()
        assert state is not None
        assert state.turn_id == 42
        assert mock.stats.states_received == 1

    @pytest.mark.asyncio
    async def test_poll_clears_state_after_read(self) -> None:
        mock = MockFileBridge()
        mock.set_mock_state(_make_state_data())
        first = await mock.poll_state()
        assert first is not None
        second = await mock.poll_state()
        assert second is None

    @pytest.mark.asyncio
    async def test_send_command_records(self) -> None:
        mock = MockFileBridge()
        cmd = _make_command(msg_id=77)
        result = await mock.send_command(cmd)
        assert result is True
        assert len(mock.get_sent_commands()) == 1
        assert mock.get_sent_commands()[0].msg_id == 77
        assert mock.stats.commands_sent == 1

    @pytest.mark.asyncio
    async def test_set_mock_ack_and_poll(self) -> None:
        mock = MockFileBridge()
        mock.set_mock_ack(_make_ack_data(msg_id=55, success=True))
        ack = await mock._poll_ack(55)
        assert ack is not None
        assert ack.msg_id == 55
        assert ack.success is True
        assert mock.stats.acks_received == 1

    @pytest.mark.asyncio
    async def test_mock_ack_wrong_id_returns_none(self) -> None:
        mock = MockFileBridge()
        mock.set_mock_ack(_make_ack_data(msg_id=10))
        ack = await mock._poll_ack(999)
        assert ack is None

    def test_check_heartbeat_always_true(self) -> None:
        mock = MockFileBridge()
        assert mock.check_heartbeat() is True

    @pytest.mark.asyncio
    async def test_start_updates_heartbeat_stats(self) -> None:
        mock = MockFileBridge()
        await mock.start()
        assert mock.stats.last_heartbeat_time > 0
        await mock.stop()

    @pytest.mark.asyncio
    async def test_multiple_commands_recorded(self) -> None:
        mock = MockFileBridge()
        await mock.send_command(_make_command(msg_id=1))
        await mock.send_command(_make_command(msg_id=2))
        await mock.send_command(_make_command(msg_id=3))
        cmds = mock.get_sent_commands()
        assert len(cmds) == 3
        assert [c.msg_id for c in cmds] == [1, 2, 3]


# ===========================================================================
# 18. _throttle_action
# ===========================================================================

class TestThrottleAction:
    """Tests for FileBridge._throttle_action rate limiting."""

    @pytest.mark.asyncio
    async def test_no_throttle_when_disabled(self, tmp_path: Path) -> None:
        config = BridgeConfig(comm_dir=tmp_path, max_actions_per_second=0.0)
        bridge = FileBridge(config)
        # Should return instantly with no sleep
        start = time.time()
        await bridge._throttle_action()
        elapsed = time.time() - start
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_throttle_sleeps_when_rate_exceeded(self, tmp_path: Path) -> None:
        config = BridgeConfig(comm_dir=tmp_path, max_actions_per_second=10.0)
        bridge = FileBridge(config)
        # First call sets the time, second should throttle
        bridge._last_action_time = time.time()

        with patch("alpha_zero.bridge.file_bridge.asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await bridge._throttle_action()
            # With rate=10, min_interval=0.1s. If called immediately after
            # _last_action_time=now, it should sleep.
            if mock_sleep.called:
                wait_arg = mock_sleep.call_args[0][0]
                assert 0 < wait_arg <= 0.1

    @pytest.mark.asyncio
    async def test_throttle_no_sleep_after_enough_time(
        self, tmp_path: Path
    ) -> None:
        config = BridgeConfig(comm_dir=tmp_path, max_actions_per_second=1.0)
        bridge = FileBridge(config)
        # Set last action far in the past
        bridge._last_action_time = time.time() - 10.0

        with patch("alpha_zero.bridge.file_bridge.asyncio.sleep") as mock_sleep:
            mock_sleep.return_value = None
            await bridge._throttle_action()
            mock_sleep.assert_not_called()


# ===========================================================================
# 19. _poll_ack
# ===========================================================================

class TestPollAck:
    """Tests for FileBridge._poll_ack."""

    @pytest.mark.asyncio
    async def test_no_ack_file_returns_none(self, bridge: FileBridge) -> None:
        result = await bridge._poll_ack(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_matching_ack(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        _write_ack_file(bridge_config, _make_ack_data(msg_id=42, success=True))
        result = await bridge._poll_ack(42)
        assert result is not None
        assert result.msg_id == 42
        assert result.success is True
        assert bridge.stats.acks_received == 1

    @pytest.mark.asyncio
    async def test_wrong_msg_id_returns_none(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        _write_ack_file(bridge_config, _make_ack_data(msg_id=42))
        result = await bridge._poll_ack(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_ack_returns_none(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        bridge_config.ack_file.write_text("not json", encoding="utf-8")
        result = await bridge._poll_ack(1)
        assert result is None


# ===========================================================================
# 20. wait_for_ack
# ===========================================================================

class TestWaitForAck:
    """Tests for FileBridge.wait_for_ack."""

    @pytest.mark.asyncio
    async def test_timeout_raises_ack_timeout_error(
        self, bridge: FileBridge
    ) -> None:
        with pytest.raises(AckTimeoutError):
            await bridge.wait_for_ack(msg_id=1, timeout=0.05)

    @pytest.mark.asyncio
    async def test_returns_ack_when_available(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        _write_ack_file(bridge_config, _make_ack_data(msg_id=7))
        ack = await bridge.wait_for_ack(msg_id=7, timeout=1.0)
        assert ack.msg_id == 7


# ===========================================================================
# 21. send_and_wait
# ===========================================================================

class TestSendAndWait:
    """Tests for FileBridge.send_and_wait."""

    @pytest.mark.asyncio
    async def test_send_and_wait_success(
        self, bridge_config: BridgeConfig
    ) -> None:
        config = BridgeConfig(
            comm_dir=bridge_config.comm_dir,
            max_actions_per_second=0.0,
        )
        config.ensure_comm_dir()
        bridge = FileBridge(config)

        cmd = _make_command(msg_id=33)
        # Pre-write ack
        _write_ack_file(config, _make_ack_data(msg_id=33))
        ack = await bridge.send_and_wait(cmd, timeout=1.0)
        assert ack.msg_id == 33


# ===========================================================================
# 22. read_errors
# ===========================================================================

class TestReadErrors:
    """Tests for FileBridge.read_errors."""

    def test_no_errors_file_returns_empty(self, bridge: FileBridge) -> None:
        errors = bridge.read_errors()
        assert errors == []

    def test_read_error_array(
        self, bridge: FileBridge, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        error_data = [
            {
                "type": "error",
                "error_code": "E001",
                "message": "Invalid action",
                "timestamp": time.time(),
            }
        ]
        bridge_config.errors_file.write_text(json.dumps(error_data), encoding="utf-8")
        errors = bridge.read_errors()
        assert len(errors) == 1
        assert errors[0].error_code == "E001"


# ===========================================================================
# 23. _write_heartbeat
# ===========================================================================

class TestWriteHeartbeat:
    """Tests for FileBridge._write_heartbeat."""

    @pytest.mark.asyncio
    async def test_heartbeat_updates_stats(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        bridge = FileBridge(bridge_config)
        bridge._write_heartbeat()
        assert bridge.stats.last_heartbeat_time > 0

    @pytest.mark.asyncio
    async def test_heartbeat_file_contents(
        self, bridge_config: BridgeConfig
    ) -> None:
        bridge_config.ensure_comm_dir()
        bridge = FileBridge(bridge_config)
        bridge._write_heartbeat()
        data = json.loads(bridge_config.heartbeat_file.read_text())
        assert "timestamp" in data
        assert data["python_alive"] is True
