"""Tests for alpha_zero.bridge.tuner_bridge module.

Covers: command writing, atomic file operations, convenience methods,
error handling, and message ID generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.protocol import (
    create_end_turn_command,
)
from alpha_zero.bridge.tuner_bridge import TunerBridge, TunerBridgeError

# --- Fixtures ---


@pytest.fixture
def bridge_config(tmp_path: Path) -> BridgeConfig:
    """Create a BridgeConfig with tmp_path comm_dir."""
    return BridgeConfig(comm_dir=tmp_path)


@pytest.fixture
def bridge(bridge_config: BridgeConfig) -> TunerBridge:
    """Create a TunerBridge instance (not started)."""
    return TunerBridge(bridge_config)


# --- Construction tests ---


class TestTunerBridgeConstruction:
    def test_init_defaults(self, bridge: TunerBridge) -> None:
        assert not bridge.is_running
        assert bridge.commands_sent == 0

    def test_msg_counter_seeded_int32_safe(self, bridge_config: BridgeConfig) -> None:
        # Seeded from wall-clock ms mapped into [1, MAX_MSG_ID) so the ack
        # msg_id round-trips as a positive signed int32 (not the raw ~1.7e12).
        from alpha_zero.bridge.utils import MAX_MSG_ID

        b = TunerBridge(bridge_config)
        assert 1 <= b._msg_counter < MAX_MSG_ID

    def test_stats_initial(self, bridge: TunerBridge) -> None:
        stats = bridge.stats
        assert stats["commands_sent"] == 0
        assert stats["running"] is False


# --- Lifecycle tests ---


@pytest.mark.asyncio
class TestLifecycle:
    async def test_start_stop(self, bridge: TunerBridge) -> None:
        await bridge.start()
        assert bridge.is_running
        await bridge.stop()
        assert not bridge.is_running

    async def test_context_manager(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            assert bridge.is_running
        assert not bridge.is_running

    async def test_start_creates_comm_dir(self, tmp_path: Path) -> None:
        subdir = tmp_path / "test_comm"
        config = BridgeConfig(comm_dir=subdir)
        bridge = TunerBridge(config)
        await bridge.start()
        assert subdir.exists()
        await bridge.stop()

    async def test_start_clears_stale_commands(self, bridge_config: BridgeConfig) -> None:
        # Write stale commands
        bridge_config.commands_file.write_text('{"old": "data"}')
        bridge = TunerBridge(bridge_config)
        await bridge.start()
        # Should be cleared
        content = bridge_config.commands_file.read_text()
        assert content == ""
        await bridge.stop()


# --- Command sending tests ---


@pytest.mark.asyncio
class TestCommandSending:
    async def test_send_command_writes_json(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            cmd = create_end_turn_command(msg_id=1)
            result = await bridge.send_command(cmd)
            assert result is True

            # Verify file contents
            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == 1
            assert data["commands"][0]["action_type"] == "end_turn"

    async def test_send_command_atomic_write(self, bridge_config: BridgeConfig) -> None:
        """Verify temp file doesn't linger after successful write."""
        async with TunerBridge(bridge_config) as bridge:
            cmd = create_end_turn_command(msg_id=1)
            await bridge.send_command(cmd)
            tmp_file = bridge_config.commands_file.with_suffix(".tmp")
            assert not tmp_file.exists()

    async def test_send_command_increments_counter(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            assert bridge.commands_sent == 0
            await bridge.send_command(create_end_turn_command(1))
            assert bridge.commands_sent == 1
            await bridge.send_command(create_end_turn_command(2))
            assert bridge.commands_sent == 2

    async def test_send_command_not_running_raises(self, bridge: TunerBridge) -> None:
        with pytest.raises(TunerBridgeError, match="not running"):
            await bridge.send_command(create_end_turn_command(1))

    async def test_send_unit_move(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            msg_id = await bridge.send_unit_move(unit_id=5, x=10, y=12)
            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == msg_id
            cmd = data["commands"][0]
            assert cmd["action_type"] == "unit_move"
            assert cmd["unit_id"] == 5
            assert cmd["target"]["x"] == 10
            assert cmd["target"]["y"] == 12

    async def test_send_end_turn(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            msg_id = await bridge.send_end_turn()
            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == msg_id
            assert data["commands"][0]["action_type"] == "end_turn"

    async def test_send_city_produce(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            msg_id = await bridge.send_city_produce(city_id=65536, item_id=42, item_type="building")
            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == msg_id
            cmd = data["commands"][0]
            assert cmd["action_type"] == "city_produce"
            assert cmd["city_id"] == 65536
            assert cmd["item_id"] == 42
            assert cmd["item_type"] == "building"

    async def test_overwrite_existing_commands(self, bridge_config: BridgeConfig) -> None:
        """Sending a new command overwrites the previous one."""
        async with TunerBridge(bridge_config) as bridge:
            await bridge.send_end_turn()
            msg_id2 = await bridge.send_end_turn()
            content = bridge_config.commands_file.read_text()
            data = json.loads(content)
            assert data["msg_id"] == msg_id2


# --- Message ID tests ---


class TestMessageId:
    def test_next_msg_id_monotonic(self, bridge: TunerBridge) -> None:
        ids = [bridge.next_msg_id() for _ in range(10)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 10  # All unique

    def test_next_msg_id_increments(self, bridge: TunerBridge) -> None:
        id1 = bridge.next_msg_id()
        id2 = bridge.next_msg_id()
        assert id2 == id1 + 1


# --- Stats tests ---


@pytest.mark.asyncio
class TestStats:
    async def test_stats_after_commands(self, bridge_config: BridgeConfig) -> None:
        async with TunerBridge(bridge_config) as bridge:
            await bridge.send_end_turn()
            await bridge.send_end_turn()
            stats = bridge.stats
            assert stats["commands_sent"] == 2
            assert stats["running"] is True
