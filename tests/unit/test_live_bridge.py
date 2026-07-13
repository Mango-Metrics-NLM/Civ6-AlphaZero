"""Tests for alpha_zero.bridge.live_bridge module.

Integration tests composing LogTailer + TunerBridge via LiveBridge.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.live_bridge import LiveBridge
from alpha_zero.bridge.log_tailer import AckTimeoutError, StateTimeoutError
from alpha_zero.bridge.protocol import create_end_turn_command


# --- Fixtures ---


@pytest.fixture
def tmp_log(tmp_path: Path) -> Path:
    """Create a temporary log file."""
    log_file = tmp_path / "Lua.log"
    log_file.write_text("")
    return log_file


@pytest.fixture
def bridge_config(tmp_path: Path) -> BridgeConfig:
    """Create a BridgeConfig pointing to tmp_path."""
    return BridgeConfig(comm_dir=tmp_path)


@pytest.fixture
def live_bridge(
    bridge_config: BridgeConfig, tmp_log: Path
) -> LiveBridge:
    """Create a LiveBridge instance (not started)."""
    return LiveBridge(bridge_config, tmp_log, poll_interval=0.05)


def _write_line(log_path: Path, line: str) -> None:
    """Append a line to the log file."""
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Lifecycle tests ---


@pytest.mark.asyncio
class TestLifecycle:
    async def test_context_manager(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            assert bridge._running
        assert not bridge._running

    async def test_start_stop(self, live_bridge: LiveBridge) -> None:
        await live_bridge.start()
        assert live_bridge._running
        await live_bridge.stop()
        assert not live_bridge._running


# --- State reading tests ---


@pytest.mark.asyncio
class TestStateReading:
    async def test_poll_state_none_initially(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            result = await bridge.poll_state()
            assert result is None

    async def test_wait_for_state_success(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            async def write_state() -> None:
                await asyncio.sleep(0.1)
                # width/height are required fields in the wire payload; explicit
                # zero is the pre-game representation. See MapData.from_dict.
                state = {"turn_id": 5, "map": {"width": 0, "height": 0, "tiles": []}}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_STATE:{json.dumps(state)}')

            asyncio.create_task(write_state())
            # wait_for_state returns a GameStateData object (not a dict).
            result = await bridge.wait_for_state(timeout=2.0)
            assert result.turn_id == 5

    async def test_wait_for_state_timeout(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            with pytest.raises(StateTimeoutError):
                await bridge.wait_for_state(timeout=0.2)

    async def test_wait_for_state_returns_existing(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            # Pre-inject state. wait_for_state now wraps via StateMessage.from_dict,
            # so the cached payload must be parseable. width/height are required
            # fields; explicit zero is the pre-game representation.
            bridge._tailer._latest_state = {
                "turn_id": 99,
                "map": {"width": 0, "height": 0, "tiles": []},
            }
            result = await bridge.wait_for_state(timeout=0.5, require_new=False)
            assert result.turn_id == 99


# --- Command sending tests ---


@pytest.mark.asyncio
class TestCommandSending:
    async def test_send_command_writes_file(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            msg_id = bridge.next_msg_id()
            cmd = create_end_turn_command(msg_id)
            result = await bridge.send_command(cmd)
            assert result is True

            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == msg_id

    async def test_msg_id_monotonic(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            ids = [bridge.next_msg_id() for _ in range(5)]
            assert ids == sorted(ids)
            assert len(set(ids)) == 5


# --- Ack tests ---


@pytest.mark.asyncio
class TestAckHandling:
    async def test_wait_for_ack_success(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            async def write_ack() -> None:
                await asyncio.sleep(0.1)
                ack = {"msg_id": 42, "success": True}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}')

            asyncio.create_task(write_ack())
            result = await bridge.wait_for_ack(msg_id=42, timeout=2.0)
            assert result["success"] is True

    async def test_wait_for_ack_timeout(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            with pytest.raises(AckTimeoutError):
                await bridge.wait_for_ack(msg_id=999, timeout=0.2)

    async def test_send_and_wait(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            msg_id = bridge.next_msg_id()
            cmd = create_end_turn_command(msg_id)

            async def write_ack() -> None:
                await asyncio.sleep(0.15)
                ack = {"msg_id": msg_id, "success": True}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}')

            asyncio.create_task(write_ack())
            result = await bridge.send_and_wait(cmd, timeout=2.0)
            assert result["success"] is True


# --- Heartbeat tests ---


@pytest.mark.asyncio
class TestHeartbeat:
    async def test_no_heartbeat_not_connected(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            assert not bridge.is_connected

    async def test_heartbeat_makes_connected(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            hb = {"lua_alive": True, "timestamp": int(time.time())}
            _write_line(tmp_log, f'AlphaZeroMain: AZ_HB:{json.dumps(hb)}')
            await asyncio.sleep(0.15)
            assert bridge.is_connected


# --- Stats tests ---


@pytest.mark.asyncio
class TestStats:
    async def test_combined_stats(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            # Send a command
            await bridge.send_command(create_end_turn_command(1))

            # Write a heartbeat
            _write_line(tmp_log, f'AZ_HB:{json.dumps({"lua_alive": True})}')
            await asyncio.sleep(0.15)

            stats = bridge.stats
            assert stats["commands_sent"] == 1
            assert stats["heartbeats_received"] >= 1
            assert stats["running"] is True

    async def test_get_stats_compat(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            stats = bridge.get_stats()
            assert isinstance(stats, dict)

    async def test_clear_files(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            # Write a command
            await bridge.send_command(create_end_turn_command(1))
            assert bridge_config.commands_file.exists()

            # Clear
            bridge.clear_files()
            content = bridge_config.commands_file.read_text()
            assert content == ""

    async def test_clear_files_logs_oserror(
        self,
        bridge_config: BridgeConfig,
        tmp_log: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """OSError while clearing commands.json is logged, not swallowed silently."""
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            await bridge.send_command(create_end_turn_command(1))
            assert bridge_config.commands_file.exists()

            def _raise_oserror(*args: Any, **kwargs: Any) -> None:
                raise OSError("file is locked")

            monkeypatch.setattr(Path, "write_text", _raise_oserror)

            with caplog.at_level(
                logging.DEBUG, logger="alpha_zero.bridge.live_bridge"
            ):
                bridge.clear_files()  # Should not raise.

            assert any(
                "Could not clear" in record.message for record in caplog.records
            )


# --- Poll interval configuration tests ---


@pytest.mark.asyncio
class TestPollIntervalConfig:
    async def test_default_poll_interval_derived_from_config(
        self, tmp_path: Path, tmp_log: Path
    ) -> None:
        """LiveBridge's default poll interval must honor BridgeConfig.poll_interval_ms.

        Regression test: previously LiveBridge hardcoded poll_interval=0.1 and
        never consulted config.poll_interval_ms (and thus CIV6_BRIDGE_POLL_MS),
        even though FileBridge and BridgeConfig document that env override.
        """
        config = BridgeConfig(comm_dir=tmp_path, poll_interval_ms=250)
        bridge = LiveBridge(config, tmp_log)
        assert bridge._tailer._poll_interval == pytest.approx(0.25)

    async def test_explicit_poll_interval_overrides_config(
        self, tmp_path: Path, tmp_log: Path
    ) -> None:
        """An explicitly passed poll_interval still wins over config."""
        config = BridgeConfig(comm_dir=tmp_path, poll_interval_ms=250)
        bridge = LiveBridge(config, tmp_log, poll_interval=0.05)
        assert bridge._tailer._poll_interval == pytest.approx(0.05)


# --- Additional coverage: default-argument and compatibility-shim paths ---


@pytest.mark.asyncio
class TestDefaultArgumentPaths:
    async def test_poll_state_returns_parsed_state_when_available(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        """poll_state() must return a parsed GameStateData, not just detect
        presence, once the tailer has cached a raw state payload."""
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            bridge._tailer._latest_state = {
                "turn_id": 7,
                "map": {"width": 0, "height": 0, "tiles": []},
            }
            result = await bridge.poll_state()
            assert result is not None
            assert result.turn_id == 7

    async def test_wait_for_state_uses_config_timeout_when_omitted(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        """Omitting `timeout` must derive it from config.command_timeout_ms
        rather than blocking forever or raising a TypeError."""
        bridge_config.command_timeout_ms = 200
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            with pytest.raises(StateTimeoutError):
                await bridge.wait_for_state()

    async def test_check_heartbeat_uses_config_max_age_when_omitted(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        """Omitting `max_age` must derive it from config.heartbeat_timeout_s."""
        bridge_config.heartbeat_timeout_s = 5.0
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            hb = {"lua_alive": True, "timestamp": int(time.time())}
            _write_line(tmp_log, f'AZ_HB:{json.dumps(hb)}')
            await asyncio.sleep(0.15)
            assert bridge.check_heartbeat() is True

    async def test_wait_for_ack_uses_config_timeout_when_omitted(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        bridge_config.command_timeout_ms = 200
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            with pytest.raises(AckTimeoutError):
                await bridge.wait_for_ack(msg_id=1)

    async def test_send_and_wait_with_retry_compat_shim(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        """FileBridge-API compatibility shim: agent_loop calls this method
        polymorphically regardless of which bridge implementation is active."""
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            msg_id = bridge.next_msg_id()
            cmd = create_end_turn_command(msg_id)

            async def write_ack() -> None:
                await asyncio.sleep(0.1)
                ack = {"msg_id": msg_id, "success": True}
                _write_line(tmp_log, f'AlphaZeroMain: AZ_ACK:{json.dumps(ack)}')

            asyncio.create_task(write_ack())
            result = await bridge.send_and_wait_with_retry(
                cmd, timeout=2.0, max_retries=3
            )
            assert result["success"] is True

    async def test_clear_files_when_commands_file_missing_is_a_noop(
        self, bridge_config: BridgeConfig, tmp_log: Path
    ) -> None:
        """clear_files() must not attempt to write when commands.json was
        never created (e.g. bridge started but no command sent yet)."""
        async with LiveBridge(bridge_config, tmp_log, poll_interval=0.05) as bridge:
            assert not bridge_config.commands_file.exists()
            bridge.clear_files()  # must not raise
            assert not bridge_config.commands_file.exists()
