"""Sanity and edge-case tests for the Civ6 Lua mod bridge integration.

Tests the Python-side bridge behaviour under conditions that mirror
what happens when the Civ6 Lua mod writes files with various issues:
corrupted JSON, missing fields, partial writes, stale heartbeats,
OneDrive-redirected paths, and IPC directory edge cases.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import (
    AckTimeoutError,
    ConnectionLostError,
    FileBridge,
    MockFileBridge,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import (
    AckMessage,
    CommandMessage,
    GameStateData,
    LuaCommand,
    ActionType,
    validate_state_json,
)


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def comm_dir(tmp_path: Path) -> Path:
    """Provide a clean temp comm directory."""
    d = tmp_path / "AlphaZeroAI"
    d.mkdir()
    return d


@pytest.fixture
def config(comm_dir: Path) -> BridgeConfig:
    return BridgeConfig(
        comm_dir=comm_dir,
        poll_interval_ms=5,
        command_timeout_ms=300,
        heartbeat_timeout_s=2.0,
    )


@pytest.fixture
def bridge(config: BridgeConfig) -> FileBridge:
    return FileBridge(config)


# ==============================================================================
# Heartbeat Edge Cases
# ==============================================================================


class TestHeartbeatEdgeCases:
    """Edge cases for heartbeat detection — mirrors Civ6 Lua sandbox issues."""

    def test_heartbeat_missing_lua_alive_returns_false(self, bridge: FileBridge) -> None:
        """Heartbeat without lua_alive key (Python's own heartbeat) should be ignored."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": time.time(), "python_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()

    def test_heartbeat_lua_alive_false_returns_false(self, bridge: FileBridge) -> None:
        """Heartbeat with lua_alive=false should be treated as not connected."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": time.time(), "lua_alive": False}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()

    def test_heartbeat_empty_file_returns_false(self, bridge: FileBridge) -> None:
        """Empty heartbeat file (Lua write interrupted) should not crash."""
        bridge.config.ensure_comm_dir()
        bridge.config.heartbeat_file.write_text("")
        assert not bridge.check_heartbeat()

    def test_heartbeat_truncated_json_returns_false(self, bridge: FileBridge) -> None:
        """Truncated JSON (partial Lua write) should be handled gracefully."""
        bridge.config.ensure_comm_dir()
        bridge.config.heartbeat_file.write_text('{"timestamp": 123456')
        assert not bridge.check_heartbeat()

    def test_heartbeat_negative_timestamp_returns_false(self, bridge: FileBridge) -> None:
        """Negative timestamp (Lua os.time() bug) should be stale."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": -1.0, "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()

    def test_heartbeat_zero_timestamp_returns_false(self, bridge: FileBridge) -> None:
        """Zero timestamp (epoch) should be stale."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": 0, "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()

    def test_heartbeat_future_timestamp_returns_true(self, bridge: FileBridge) -> None:
        """Timestamp slightly in the future (clock skew) should be fresh."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": time.time() + 1.0, "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert bridge.check_heartbeat()

    def test_heartbeat_non_numeric_timestamp_returns_false(self, bridge: FileBridge) -> None:
        """Non-numeric timestamp should be handled gracefully."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": "not_a_number", "lua_alive": True}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()


# ==============================================================================
# State Polling Edge Cases
# ==============================================================================


class TestStatePollEdgeCases:
    """Edge cases for state polling — mirrors Lua StateExporter output issues."""

    @pytest.mark.asyncio
    async def test_poll_empty_state_file(self, bridge: FileBridge) -> None:
        """Empty state file (Lua write started but nothing written yet)."""
        await bridge.start()
        bridge.config.state_file.write_text("")
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_with_only_whitespace(self, bridge: FileBridge) -> None:
        """Whitespace-only state file."""
        await bridge.start()
        bridge.config.state_file.write_text("   \n\n  ")
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_truncated_json(self, bridge: FileBridge) -> None:
        """Truncated JSON (Lua file write interrupted mid-stream)."""
        await bridge.start()
        bridge.config.state_file.write_text('{"turn_id": 5, "map": {"width": 44, "hei')
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_null_json(self, bridge: FileBridge) -> None:
        """JSON 'null' value (Lua encoder error)."""
        await bridge.start()
        bridge.config.state_file.write_text("null")
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_json_array_instead_of_object(self, bridge: FileBridge) -> None:
        """JSON array instead of object (Lua encoder bug)."""
        await bridge.start()
        bridge.config.state_file.write_text("[1, 2, 3]")
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_missing_map_key(self, bridge: FileBridge) -> None:
        """State JSON missing required 'map' key."""
        await bridge.start()
        state_data = {"turn_id": 5, "units": [], "cities": [], "legal_actions": []}
        bridge.config.state_file.write_text(json.dumps(state_data))
        state = await bridge.poll_state()
        # Should either return None or a state with defaults depending on implementation
        # The key thing is it doesn't crash

    @pytest.mark.asyncio
    async def test_poll_state_with_unicode(self, bridge: FileBridge) -> None:
        """State with Unicode city names (e.g. Chinese civilization)."""
        await bridge.start()
        state_data = {
            "turn_id": 1,
            "map": {"width": 44, "height": 26, "tiles": []},
            "units": [],
            "cities": [{"id": 1, "name": "北京", "owner": 0, "x": 10, "y": 10}],
            "legal_actions": [{"action": "end_turn"}],
        }
        bridge.config.state_file.write_text(
            json.dumps(state_data, ensure_ascii=False), encoding="utf-8"
        )
        state = await bridge.poll_state()
        assert state is not None
        assert state.cities[0].name == "北京"

    @pytest.mark.asyncio
    async def test_poll_state_very_large_file(self, bridge: FileBridge) -> None:
        """Large state file (many tiles, units, cities)."""
        await bridge.start()
        tiles = [{"x": i % 80, "y": i // 80, "terrain": "TERRAIN_PLAINS"} for i in range(2000)]
        state_data = {
            "turn_id": 100,
            "map": {"width": 80, "height": 60, "tiles": tiles},
            "units": [{"id": i, "type_name": "UNIT_WARRIOR", "owner": 0, "x": i, "y": 0} for i in range(50)],
            "cities": [],
            "legal_actions": [{"action": "end_turn"}],
        }
        bridge.config.state_file.write_text(json.dumps(state_data))
        state = await bridge.poll_state()
        assert state is not None
        assert state.turn_id == 100


# ==============================================================================
# Ack Edge Cases
# ==============================================================================


class TestAckEdgeCases:
    """Edge cases for ack handling."""

    @pytest.mark.asyncio
    async def test_ack_empty_file(self, bridge: FileBridge) -> None:
        """Empty ack file should not cause crash."""
        await bridge.start()
        bridge.config.ack_file.write_text("")
        result = await bridge._poll_ack(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_ack_wrong_type(self, bridge: FileBridge) -> None:
        """Ack file containing non-object JSON."""
        await bridge.start()
        bridge.config.ack_file.write_text('"just a string"')
        result = await bridge._poll_ack(1)
        assert result is None

    @pytest.mark.asyncio
    async def test_ack_missing_success_field(self, bridge: FileBridge) -> None:
        """Ack without success field should default gracefully."""
        await bridge.start()
        bridge.config.ack_file.write_text(json.dumps({"msg_id": 1}))
        result = await bridge._poll_ack(1)
        # Should either return None or an ack with success=False
        # Key: no crash

    @pytest.mark.asyncio
    async def test_ack_with_extra_fields(self, bridge: FileBridge) -> None:
        """Ack with extra unknown fields (Lua mod sends more than expected)."""
        await bridge.start()
        ack_data = {
            "msg_id": 42,
            "success": True,
            "extra_lua_field": "unexpected",
            "debug_info": {"lua_version": "5.1"},
        }
        bridge.config.ack_file.write_text(json.dumps(ack_data))
        result = await bridge._poll_ack(42)
        assert result is not None
        assert result.success is True


# ==============================================================================
# Command Sending Edge Cases
# ==============================================================================


class TestCommandSendEdgeCases:
    """Edge cases for command file writing."""

    @pytest.mark.asyncio
    async def test_send_command_to_readonly_dir(self, tmp_path: Path) -> None:
        """Sending a command when comm dir is read-only should fail gracefully."""
        config = BridgeConfig(
            comm_dir=tmp_path / "nonexistent_subdir" / "deep",
            poll_interval_ms=5,
        )
        bridge = FileBridge(config)
        cmd = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        # Should not crash even if dir doesn't exist
        result = await bridge.send_command(cmd)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_empty_command_list(self, bridge: FileBridge) -> None:
        """Sending command message with empty commands list."""
        await bridge.start()
        cmd = CommandMessage(msg_id=1, commands=[])
        result = await bridge.send_command(cmd)
        assert result is True
        content = json.loads(bridge.config.commands_file.read_text())
        assert content["commands"] == []


# ==============================================================================
# Config Path Edge Cases
# ==============================================================================


class TestConfigPathEdgeCases:
    """Edge cases for path resolution — mirrors OneDrive redirect issues."""

    def test_comm_dir_with_spaces(self, tmp_path: Path) -> None:
        """Path with spaces (common in Windows 'My Games')."""
        d = tmp_path / "My Games" / "Sid Meier's Civilization VI" / "AlphaZeroAI"
        d.mkdir(parents=True)
        config = BridgeConfig(comm_dir=d)
        assert config.state_file.parent.exists()

    def test_comm_dir_with_apostrophe(self, tmp_path: Path) -> None:
        """Path with apostrophe (Sid Meier's)."""
        d = tmp_path / "Sid Meier's Civilization VI" / "AlphaZeroAI"
        d.mkdir(parents=True)
        config = BridgeConfig(comm_dir=d)
        config.ensure_comm_dir()
        assert d.exists()

    def test_config_file_properties(self, tmp_path: Path) -> None:
        """All config file properties resolve to correct paths."""
        d = tmp_path / "ipc"
        config = BridgeConfig(comm_dir=d)
        assert config.state_file == d / "state.json"
        assert config.commands_file == d / "commands.json"
        assert config.ack_file == d / "ack.json"
        assert config.heartbeat_file == d / "heartbeat.txt"
        assert config.errors_file == d / "errors.json"


# ==============================================================================
# Mock Bridge Edge Cases
# ==============================================================================


class TestMockBridgeEdgeCases:
    """Edge cases for MockFileBridge used in testing."""

    @pytest.mark.asyncio
    async def test_mock_poll_state_without_set(self) -> None:
        """Polling without setting a mock state returns None."""
        bridge = MockFileBridge()
        await bridge.start()
        assert await bridge.poll_state() is None

    @pytest.mark.asyncio
    async def test_mock_multiple_states(self) -> None:
        """Setting multiple mock states only keeps the last one."""
        bridge = MockFileBridge()
        await bridge.start()
        bridge.set_mock_state({"turn_id": 1, "map": {"width": 1, "height": 1}, "units": [], "cities": [], "legal_actions": []})
        bridge.set_mock_state({"turn_id": 2, "map": {"width": 1, "height": 1}, "units": [], "cities": [], "legal_actions": []})
        state = await bridge.poll_state()
        assert state.turn_id == 2

    def test_mock_heartbeat_always_true(self) -> None:
        """MockFileBridge always reports alive."""
        bridge = MockFileBridge()
        assert bridge.check_heartbeat() is True


# ==============================================================================
# Protocol Validation Edge Cases
# ==============================================================================


class TestProtocolValidationEdgeCases:
    """Edge cases for protocol validation — mirrors Lua encoder quirks."""

    def test_validate_empty_string(self) -> None:
        is_valid, error = validate_state_json("")
        assert not is_valid

    def test_validate_null(self) -> None:
        is_valid, error = validate_state_json("null")
        assert not is_valid

    def test_validate_array(self) -> None:
        is_valid, error = validate_state_json("[1, 2]")
        assert not is_valid

    def test_validate_minimal_valid(self) -> None:
        """Minimal valid state with just turn_id and map."""
        data = json.dumps({"turn_id": 1, "map": {"width": 1, "height": 1}})
        is_valid, error = validate_state_json(data)
        assert is_valid

    def test_validate_turn_id_zero(self) -> None:
        """Turn 0 is valid (game start)."""
        data = json.dumps({"turn_id": 0, "map": {"width": 1, "height": 1}})
        is_valid, error = validate_state_json(data)
        assert is_valid

    def test_validate_negative_turn_id(self) -> None:
        """Negative turn_id (Lua bug) — still has required fields."""
        data = json.dumps({"turn_id": -1, "map": {"width": 1, "height": 1}})
        is_valid, error = validate_state_json(data)
        # Implementation-dependent: may pass field check even if value is odd
        # Key: no crash


# ==============================================================================
# Concurrent Access Edge Cases
# ==============================================================================


class TestConcurrentEdgeCases:
    """Edge cases for concurrent file access between Lua and Python."""

    @pytest.mark.asyncio
    async def test_state_file_deleted_during_poll(self, bridge: FileBridge) -> None:
        """State file deleted between exists check and read."""
        await bridge.start()
        state_data = {
            "turn_id": 1,
            "map": {"width": 44, "height": 26},
            "units": [],
            "cities": [],
            "legal_actions": [],
        }
        bridge.config.state_file.write_text(json.dumps(state_data))

        # First poll reads successfully
        state = await bridge.poll_state()
        assert state is not None

        # Delete the file
        bridge.config.state_file.unlink()

        # Second poll should handle missing file gracefully
        state2 = await bridge.poll_state()
        assert state2 is None

    @pytest.mark.asyncio
    async def test_rapid_state_changes(self, bridge: FileBridge) -> None:
        """Rapid state file rewrites (simulates fast Lua polling)."""
        await bridge.start()
        for turn in range(3):
            state_data = {
                "turn_id": turn,
                # Unique player_id + units ensure content hash changes every iteration
                "player_id": turn,
                "map": {"width": 44, "height": 26},
                "units": [
                    {"id": i, "type_name": f"UNIT_{turn}_{i}", "owner": 0, "x": i, "y": turn}
                    for i in range(turn + 1)
                ],
                "cities": [],
                "legal_actions": [{"action": "end_turn"}],
                "_nonce": turn,  # Extra field to guarantee unique JSON
            }
            bridge.config.state_file.write_text(json.dumps(state_data))
            await asyncio.sleep(0.01)  # Ensure file mtime changes
            state = await bridge.poll_state()
            assert state is not None, f"poll_state returned None for turn {turn}"
            assert state.turn_id == turn


# ==============================================================================
# IPC Directory Setup Edge Cases
# ==============================================================================


class TestIPCDirectoryEdgeCases:
    """Edge cases for IPC directory initialization."""

    def test_ensure_comm_dir_creates_nested(self, tmp_path: Path) -> None:
        """ensure_comm_dir creates deeply nested directories."""
        d = tmp_path / "a" / "b" / "c" / "AlphaZeroAI"
        config = BridgeConfig(comm_dir=d)
        config.ensure_comm_dir()
        assert d.exists()

    def test_ensure_comm_dir_idempotent(self, tmp_path: Path) -> None:
        """Calling ensure_comm_dir twice doesn't fail."""
        d = tmp_path / "AlphaZeroAI"
        config = BridgeConfig(comm_dir=d)
        config.ensure_comm_dir()
        config.ensure_comm_dir()  # Second call should be no-op
        assert d.exists()

    @pytest.mark.asyncio
    async def test_bridge_start_creates_comm_dir(self, tmp_path: Path) -> None:
        """Bridge.start() creates comm dir and heartbeat file."""
        d = tmp_path / "test_ipc"
        config = BridgeConfig(comm_dir=d, poll_interval_ms=5)
        bridge = FileBridge(config)
        await bridge.start()
        assert d.exists()
        await bridge.stop()
