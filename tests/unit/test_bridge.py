"""Unit tests for the Civ6 bridge module.

Tests cover:
- BridgeConfig configuration and validation
- Protocol message serialization/deserialization
- FileBridge file-based IPC operations
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from alpha_zero.bridge.config import (
    MAP_SIZES,
    BridgeConfig,
    MapSizeConfig,
    get_map_config,
)
from alpha_zero.bridge.file_bridge import (
    AckTimeoutError,
    FileBridge,
    MockFileBridge,
    StateTimeoutError,
)
from alpha_zero.bridge.protocol import (
    AckMessage,
    ActionType,
    CityData,
    CommandMessage,
    ErrorCode,
    GameStateData,
    LegalAction,
    LuaCommand,
    MapData,
    Position,
    StateMessage,
    UnitData,
    create_city_produce_command,
    create_end_turn_command,
    create_unit_move_command,
    validate_state_json,
)

# ==============================================================================
# Config Tests
# ==============================================================================


class TestBridgeConfig:
    """Tests for BridgeConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BridgeConfig()
        assert config.poll_interval_ms == 10
        assert config.command_timeout_ms == 5000
        assert config.heartbeat_timeout_s == 2.0
        assert config.max_retries == 5
        assert config.board_size == 16
        assert config.log_level == "INFO"
        assert not config.debug_mode

    def test_custom_config(self):
        """Test custom configuration values."""
        with tempfile.TemporaryDirectory() as tmp:
            config = BridgeConfig(
                comm_dir=Path(tmp),
                poll_interval_ms=20,
                board_size=32,
                debug_mode=True,
            )
            assert config.poll_interval_ms == 20
            assert config.board_size == 32
            assert config.debug_mode

    def test_validation_poll_interval(self):
        """Test poll_interval_ms validation."""
        with pytest.raises(ValueError, match="poll_interval_ms"):
            BridgeConfig(poll_interval_ms=0)

    def test_validation_command_timeout(self):
        """Test command_timeout_ms validation."""
        with pytest.raises(ValueError, match="command_timeout_ms"):
            BridgeConfig(command_timeout_ms=50)

    def test_validation_board_size(self):
        """Test board_size validation."""
        with pytest.raises(ValueError, match="board_size"):
            BridgeConfig(board_size=4)
        with pytest.raises(ValueError, match="board_size"):
            BridgeConfig(board_size=256)

    def test_validation_log_level(self):
        """Test log_level validation."""
        with pytest.raises(ValueError, match="log_level"):
            BridgeConfig(log_level="INVALID")

    def test_file_paths(self):
        """Test file path properties."""
        with tempfile.TemporaryDirectory() as tmp:
            config = BridgeConfig(comm_dir=Path(tmp))
            assert config.state_file == Path(tmp) / "state.json"
            assert config.commands_file == Path(tmp) / "commands.json"
            assert config.ack_file == Path(tmp) / "ack.json"
            assert config.errors_file == Path(tmp) / "errors.json"
            assert config.heartbeat_file == Path(tmp) / "heartbeat.txt"

    def test_ensure_comm_dir(self):
        """Test communication directory creation."""
        with tempfile.TemporaryDirectory() as tmp:
            comm_dir = Path(tmp) / "subdir" / "alphazero"
            config = BridgeConfig(comm_dir=comm_dir)
            assert not comm_dir.exists()
            config.ensure_comm_dir()
            assert comm_dir.exists()

    def test_from_env(self):
        """Test configuration from environment variables."""
        with patch.dict("os.environ", {
            "CIV6_BRIDGE_POLL_MS": "25",
            "CIV6_BOARD_SIZE": "24",
            "CIV6_LOG_LEVEL": "DEBUG",
            "CIV6_DEBUG_MODE": "true",
        }):
            config = BridgeConfig.from_env()
            assert config.poll_interval_ms == 25
            assert config.board_size == 24
            assert config.log_level == "DEBUG"
            assert config.debug_mode

    def test_to_dict(self):
        """Test configuration serialization."""
        config = BridgeConfig(board_size=24, debug_mode=True)
        d = config.to_dict()
        assert d["board_size"] == 24
        assert d["debug_mode"] is True
        assert "comm_dir" in d


class TestMapSizeConfig:
    """Tests for MapSizeConfig."""

    def test_standard_map_sizes(self):
        """Test standard Civ6 map sizes."""
        assert MAP_SIZES["duel"].width == 44
        assert MAP_SIZES["duel"].height == 26
        assert MAP_SIZES["standard"].width == 84
        assert MAP_SIZES["huge"].width == 106

    def test_total_tiles(self):
        """Test total tile calculation."""
        config = MapSizeConfig("Test", 10, 10)
        assert config.total_tiles == 100

    def test_scale_factor(self):
        """Test scale factor calculation."""
        config = MapSizeConfig("Test", 80, 50)
        x_scale, y_scale = config.get_scale_factor(16)
        assert x_scale == 0.2
        assert y_scale == 0.32

    def test_get_map_config_standard(self):
        """Test getting standard map config."""
        config = get_map_config(84, 54)
        assert config.name == "Standard"

    def test_get_map_config_custom(self):
        """Test getting custom map config."""
        config = get_map_config(100, 60)
        assert config.name.startswith("Custom")
        assert config.width == 100
        assert config.height == 60


# ==============================================================================
# Protocol Tests
# ==============================================================================


class TestPosition:
    """Tests for Position."""

    def test_create(self):
        """Test position creation."""
        pos = Position(x=10, y=20)
        assert pos.x == 10
        assert pos.y == 20

    def test_to_dict(self):
        """Test position serialization."""
        pos = Position(x=5, y=15)
        d = pos.to_dict()
        assert d == {"x": 5, "y": 15}

    def test_from_dict(self):
        """Test position deserialization."""
        pos = Position.from_dict({"x": 3, "y": 7})
        assert pos.x == 3
        assert pos.y == 7


class TestLuaCommand:
    """Tests for LuaCommand."""

    def test_end_turn(self):
        """Test end turn command."""
        cmd = LuaCommand(action_type=ActionType.END_TURN)
        d = cmd.to_dict()
        assert d["action_type"] == "end_turn"
        assert "unit_id" not in d

    def test_unit_move(self):
        """Test unit move command."""
        cmd = LuaCommand(
            action_type=ActionType.UNIT_MOVE,
            unit_id=42,
            target=Position(x=10, y=15),
        )
        d = cmd.to_dict()
        assert d["action_type"] == "unit_move"
        assert d["unit_id"] == 42
        assert d["target"] == {"x": 10, "y": 15}

    def test_city_produce(self):
        """Test city produce command."""
        cmd = LuaCommand(
            action_type=ActionType.CITY_PRODUCE,
            city_id=1,
            item_id=5,
            item_type="unit",
        )
        d = cmd.to_dict()
        assert d["action_type"] == "city_produce"
        assert d["city_id"] == 1
        assert d["item_id"] == 5
        assert d["item_type"] == "unit"

    def test_from_dict(self):
        """Test command deserialization."""
        cmd = LuaCommand.from_dict({
            "action_type": "unit_attack",
            "unit_id": 10,
            "target": {"x": 5, "y": 6},
        })
        assert cmd.action_type == ActionType.UNIT_ATTACK
        assert cmd.unit_id == 10
        assert cmd.target.x == 5


class TestUnitData:
    """Tests for UnitData."""

    def test_from_dict_lua_format(self):
        """Test parsing Lua mod format."""
        data = {
            "id": 123,
            "type_name": "UNIT_WARRIOR",
            "owner": 0,
            "x": 15,
            "y": 22,
            "hp": 80,
            "damage": 20,
            "movement": 1.5,
            "max_movement": 2.0,
            "combat_strength": 20,
            "is_civilian": False,
            "promotions": ["PROMOTION_TORTOISE"],
        }
        unit = UnitData.from_dict(data)
        assert unit.id == 123
        assert unit.unit_type == "UNIT_WARRIOR"
        assert unit.owner_id == 0
        assert unit.x == 15
        assert unit.y == 22
        assert unit.health == 80
        assert unit.moves_remaining == 1.5
        assert unit.combat_strength == 20
        assert unit.position == (15, 22)

    def test_from_dict_alternate_keys(self):
        """Test parsing with alternate key names."""
        data = {
            "id": 1,
            "type": "UNIT_ARCHER",
            "owner_id": 1,
            "x": 5,
            "y": 10,
            "health": 100,
            "moves_remaining": 2.0,
            "max_moves": 2.0,
        }
        unit = UnitData.from_dict(data)
        assert unit.unit_type == "UNIT_ARCHER"
        assert unit.owner_id == 1
        assert unit.health == 100


class TestCityData:
    """Tests for CityData."""

    def test_from_dict(self):
        """Test city parsing."""
        data = {
            "id": 1,
            "name": "Rome",
            "owner": 0,
            "x": 12,
            "y": 20,
            "population": 5,
            "buildings": ["BUILDING_MONUMENT", "BUILDING_GRANARY"],
            "current_production": "UNIT_WARRIOR",
        }
        city = CityData.from_dict(data)
        assert city.id == 1
        assert city.name == "Rome"
        assert city.population == 5
        assert len(city.buildings) == 2
        assert city.position == (12, 20)


class TestLegalAction:
    """Tests for LegalAction."""

    def test_end_turn(self):
        """Test end turn action."""
        action = LegalAction.from_dict({"action": "end_turn"})
        assert action.action_type == "end_turn"

    def test_unit_move_with_targets(self):
        """Test unit move with target positions."""
        action = LegalAction.from_dict({
            "action": "move",
            "unit_id": 5,
            "targets": [[10, 15], [11, 15], [10, 16]],
        })
        assert action.action_type == "move"
        assert action.unit_id == 5
        assert len(action.targets) == 3
        assert action.targets[0] == (10, 15)

    def test_city_produce_with_items(self):
        """Test city production with item list."""
        action = LegalAction.from_dict({
            "action": "produce",
            "city_id": 1,
            "item_type": "unit",
            "items": [1, 2, 5, 10],
        })
        assert action.city_id == 1
        assert action.items == [1, 2, 5, 10]


class TestGameStateData:
    """Tests for GameStateData."""

    def test_from_dict_lua_format(self):
        """Test parsing full Lua mod state format."""
        data = {
            "turn_id": 42,
            "player_id": 0,
            "player": {
                "id": 0,
                "gold": 500,
                "science_per_turn": 15,
                "culture_per_turn": 10,
                "faith": 0,
                "score": 100,
            },
            "map": {
                "width": 44,
                "height": 26,
                "tiles": [
                    {"x": 0, "y": 0, "is_water": False, "terrain": "TERRAIN_PLAINS"},
                ],
            },
            "units": [
                {"id": 1, "type_name": "UNIT_WARRIOR", "owner": 0, "x": 5, "y": 5},
            ],
            "cities": [
                {"id": 1, "name": "Rome", "owner": 0, "x": 10, "y": 10, "population": 3},
            ],
            "legal_actions": [
                {"action": "end_turn"},
                {"action": "move", "unit_id": 1, "targets": [[6, 5]]},
            ],
        }

        state = GameStateData.from_dict(data)
        assert state.turn_id == 42
        assert state.player_id == 0
        assert state.map_data.width == 44
        assert len(state.units) == 1
        assert len(state.cities) == 1
        assert len(state.legal_actions) == 2


class TestStateMessage:
    """Tests for StateMessage."""

    def test_from_dict(self):
        """Test state message parsing."""
        data = {
            "msg_id": 12345,
            "timestamp": 1706180000.0,
            "turn_id": 10,
            "player_id": 0,
            "map": {"width": 44, "height": 26, "tiles": []},
            "units": [],
            "cities": [],
            "legal_actions": [{"action": "end_turn"}],
        }

        msg = StateMessage.from_dict(data)
        assert msg.msg_id == 12345
        assert msg.state.turn_id == 10

    def test_to_json(self):
        """Test state message serialization."""
        state = GameStateData(
            turn_id=5,
            player_id=0,
            map_data=MapData(width=44, height=26),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )
        msg = StateMessage(msg_id=1, timestamp=time.time(), state=state)
        json_str = msg.to_json()
        data = json.loads(json_str)
        assert data["msg_id"] == 1
        assert data["type"] == "state"


class TestCommandMessage:
    """Tests for CommandMessage."""

    def test_single_command(self):
        """Test single command message."""
        cmd = CommandMessage(
            msg_id=100,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        json_str = cmd.to_json()
        data = json.loads(json_str)
        assert data["msg_id"] == 100
        assert data["type"] == "command"
        assert len(data["commands"]) == 1

    def test_multiple_commands(self):
        """Test multiple commands in one message."""
        cmd = CommandMessage(
            msg_id=101,
            commands=[
                LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1, target=Position(5, 6)),
                LuaCommand(action_type=ActionType.END_TURN),
            ],
        )
        json_str = cmd.to_json()
        parsed = CommandMessage.from_json(json_str)
        assert len(parsed.commands) == 2
        assert parsed.commands[0].unit_id == 1


class TestAckMessage:
    """Tests for AckMessage."""

    def test_success(self):
        """Test successful ack."""
        ack = AckMessage(msg_id=100, success=True)
        json_str = ack.to_json()
        parsed = AckMessage.from_json(json_str)
        assert parsed.msg_id == 100
        assert parsed.success is True

    def test_error(self):
        """Test error ack."""
        ack = AckMessage(
            msg_id=100,
            success=False,
            error_code="E001",
            error_message="Invalid action",
        )
        json_str = ack.to_json()
        parsed = AckMessage.from_json(json_str)
        assert parsed.success is False
        assert parsed.error_code == "E001"


class TestProtocolUtilities:
    """Tests for protocol utility functions."""

    def test_validate_state_json_valid(self):
        """Test validation of valid state JSON."""
        json_str = json.dumps({
            "turn_id": 1,
            "map": {"width": 44, "height": 26},
            "units": [],
            "cities": [],
        })
        is_valid, error = validate_state_json(json_str)
        assert is_valid
        assert error is None

    def test_validate_state_json_missing_turn(self):
        """Test validation with missing turn."""
        json_str = json.dumps({
            "map": {"width": 44, "height": 26},
        })
        is_valid, error = validate_state_json(json_str)
        assert not is_valid
        assert "turn" in error.lower()

    def test_validate_state_json_missing_map(self):
        """Test validation with missing map."""
        json_str = json.dumps({"turn_id": 1})
        is_valid, error = validate_state_json(json_str)
        assert not is_valid
        assert "map" in error.lower()

    def test_validate_state_json_invalid_json(self):
        """Test validation with invalid JSON."""
        is_valid, error = validate_state_json("not valid json")
        assert not is_valid
        assert "invalid json" in error.lower()

    def test_create_end_turn_command(self):
        """Test end turn command creation."""
        cmd = create_end_turn_command(msg_id=42)
        assert cmd.msg_id == 42
        assert len(cmd.commands) == 1
        assert cmd.commands[0].action_type == ActionType.END_TURN

    def test_create_unit_move_command(self):
        """Test unit move command creation."""
        cmd = create_unit_move_command(msg_id=43, unit_id=5, target_x=10, target_y=15)
        assert cmd.msg_id == 43
        assert cmd.commands[0].action_type == ActionType.UNIT_MOVE
        assert cmd.commands[0].unit_id == 5
        assert cmd.commands[0].target.x == 10

    def test_create_city_produce_command(self):
        """Test city produce command creation."""
        cmd = create_city_produce_command(msg_id=44, city_id=1, item_id=5, item_type="building")
        assert cmd.msg_id == 44
        assert cmd.commands[0].action_type == ActionType.CITY_PRODUCE
        assert cmd.commands[0].city_id == 1
        assert cmd.commands[0].item_type == "building"


# ==============================================================================
# FileBridge Tests
# ==============================================================================


class TestFileBridge:
    """Tests for FileBridge."""

    @pytest.fixture
    def temp_comm_dir(self):
        """Create temporary communication directory."""
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    @pytest.fixture
    def config(self, temp_comm_dir):
        """Create test configuration."""
        return BridgeConfig(
            comm_dir=temp_comm_dir,
            poll_interval_ms=5,
            command_timeout_ms=500,
            heartbeat_timeout_s=1.0,
        )

    @pytest.fixture
    def bridge(self, config):
        """Create FileBridge instance."""
        return FileBridge(config)

    @pytest.mark.asyncio
    async def test_start_stop(self, bridge):
        """Test bridge start and stop."""
        await bridge.start()
        assert bridge._running
        assert bridge.config.comm_dir.exists()

        await bridge.stop()
        assert not bridge._running

    @pytest.mark.asyncio
    async def test_context_manager(self, config):
        """Test async context manager."""
        async with FileBridge(config) as bridge:
            assert bridge._running
            assert bridge.config.comm_dir.exists()

    @pytest.mark.asyncio
    async def test_poll_state_no_file(self, bridge):
        """Test polling when no state file exists."""
        await bridge.start()
        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_poll_state_with_file(self, bridge):
        """Test polling with state file."""
        await bridge.start()

        # Write state file
        state_data = {
            "turn_id": 10,
            "player_id": 0,
            "map": {"width": 44, "height": 26, "tiles": []},
            "units": [],
            "cities": [],
            "legal_actions": [],
        }
        bridge.config.state_file.write_text(json.dumps(state_data))

        state = await bridge.poll_state()
        assert state is not None
        assert state.turn_id == 10
        assert bridge.stats.states_received == 1

    @pytest.mark.asyncio
    async def test_poll_state_invalid_json(self, bridge):
        """Test polling with invalid JSON."""
        await bridge.start()
        bridge.config.state_file.write_text("not valid json")

        state = await bridge.poll_state()
        assert state is None

    @pytest.mark.asyncio
    async def test_send_command(self, bridge):
        """Test sending command."""
        await bridge.start()

        cmd = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        success = await bridge.send_command(cmd)
        assert success
        assert bridge.config.commands_file.exists()
        assert bridge.stats.commands_sent == 1

        # Verify file contents
        content = bridge.config.commands_file.read_text()
        data = json.loads(content)
        assert data["msg_id"] == 1

    @pytest.mark.asyncio
    async def test_wait_for_ack_success(self, bridge):
        """Test waiting for acknowledgment."""
        await bridge.start()

        # Write ack file
        ack_data = {"msg_id": 42, "status": "success", "success": True}
        bridge.config.ack_file.write_text(json.dumps(ack_data))

        ack = await bridge.wait_for_ack(msg_id=42, timeout=1.0)
        assert ack.msg_id == 42
        assert ack.success is True
        assert bridge.stats.acks_received == 1

    @pytest.mark.asyncio
    async def test_wait_for_ack_timeout(self, bridge):
        """Test ack timeout."""
        await bridge.start()

        with pytest.raises(AckTimeoutError) as exc_info:
            await bridge.wait_for_ack(msg_id=99, timeout=0.1)

        assert exc_info.value.code == ErrorCode.TIMEOUT

    @pytest.mark.asyncio
    async def test_wait_for_state_timeout(self, bridge):
        """Test state timeout."""
        await bridge.start()

        with pytest.raises(StateTimeoutError) as exc_info:
            await bridge.wait_for_state(timeout=0.1)

        assert exc_info.value.code == ErrorCode.TIMEOUT

    @pytest.mark.asyncio
    async def test_send_and_wait(self, bridge):
        """Test send command and wait for ack."""
        await bridge.start()

        cmd = CommandMessage(
            msg_id=55,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        # Simulate ack being written after a short delay
        async def write_ack():
            await asyncio.sleep(0.05)
            ack_data = {"msg_id": 55, "success": True}
            bridge.config.ack_file.write_text(json.dumps(ack_data))

        asyncio.create_task(write_ack())
        ack = await bridge.send_and_wait(cmd, timeout=1.0)
        assert ack.msg_id == 55
        assert ack.success is True

    def test_check_heartbeat_no_file(self, bridge):
        """Test heartbeat check with no file."""
        assert not bridge.check_heartbeat()

    def test_check_heartbeat_fresh(self, bridge):
        """Test heartbeat check with fresh file."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": time.time()}
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert bridge.check_heartbeat()

    def test_check_heartbeat_stale(self, bridge):
        """Test heartbeat check with stale file."""
        bridge.config.ensure_comm_dir()
        heartbeat = {"timestamp": time.time() - 10}  # 10 seconds old
        bridge.config.heartbeat_file.write_text(json.dumps(heartbeat))
        assert not bridge.check_heartbeat()

    @pytest.mark.asyncio
    async def test_clear_files(self, bridge):
        """Test clearing communication files."""
        await bridge.start()

        # Create files
        bridge.config.state_file.write_text("{}")
        bridge.config.commands_file.write_text("{}")
        bridge.config.ack_file.write_text("{}")

        bridge.clear_files()

        assert not bridge.config.state_file.exists()
        assert not bridge.config.commands_file.exists()
        assert not bridge.config.ack_file.exists()

    def test_get_stats(self, bridge):
        """Test statistics retrieval."""
        stats = bridge.get_stats()
        assert "states_received" in stats
        assert "commands_sent" in stats
        assert "running" in stats

    @pytest.mark.asyncio
    async def test_get_last_state(self, bridge):
        """Test getting last received state."""
        await bridge.start()

        assert bridge.get_last_state() is None

        state_data = {
            "turn_id": 5,
            "map": {"width": 44, "height": 26},
            "units": [],
            "cities": [],
            "legal_actions": [],
        }
        bridge.config.state_file.write_text(json.dumps(state_data))
        await bridge.poll_state()

        last_state = bridge.get_last_state()
        assert last_state is not None
        assert last_state.turn_id == 5


class TestMockFileBridge:
    """Tests for MockFileBridge."""

    @pytest.mark.asyncio
    async def test_mock_state(self):
        """Test mock state setting and polling."""
        bridge = MockFileBridge()
        await bridge.start()

        state_data = {
            "turn_id": 100,
            "player_id": 0,
            "map": {"width": 44, "height": 26},
            "units": [],
            "cities": [],
            "legal_actions": [],
        }
        bridge.set_mock_state(state_data)

        state = await bridge.poll_state()
        assert state is not None
        assert state.turn_id == 100

        # State should be cleared after reading
        state2 = await bridge.poll_state()
        assert state2 is None

    @pytest.mark.asyncio
    async def test_mock_send_command(self):
        """Test mock command recording."""
        bridge = MockFileBridge()
        await bridge.start()

        cmd = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        await bridge.send_command(cmd)

        sent = bridge.get_sent_commands()
        assert len(sent) == 1
        assert sent[0].msg_id == 1

    @pytest.mark.asyncio
    async def test_mock_ack(self):
        """Test mock acknowledgment."""
        bridge = MockFileBridge()
        await bridge.start()

        bridge.set_mock_ack({"msg_id": 42, "success": True})
        ack = await bridge.wait_for_ack(msg_id=42, timeout=1.0)
        assert ack.success is True

    def test_mock_heartbeat(self):
        """Test mock always reports connected."""
        bridge = MockFileBridge()
        assert bridge.check_heartbeat()


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
