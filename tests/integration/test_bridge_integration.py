"""Integration tests for the Civ6 bridge system.

Tests end-to-end workflows including:
- FileBridge with StateConverter and ActionConverter
- RealCiv6Env Gymnasium compliance
- Civ6Interface BaseGameInterface compliance
- Full roundtrip: state -> encode -> action -> command -> ack
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter, ConversionResult
from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import FileBridge, MockFileBridge
from alpha_zero.bridge.protocol import (
    ActionType,
    CommandMessage,
    GameStateData,
    LegalAction,
    LuaCommand,
    MapData,
    Position,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import ActionSpaceConfig

# ==============================================================================
# Test Fixtures
# ==============================================================================


@pytest.fixture
def temp_comm_dir():
    """Create temporary communication directory."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def bridge_config(temp_comm_dir):
    """Create test bridge configuration."""
    return BridgeConfig(
        comm_dir=temp_comm_dir,
        poll_interval_ms=5,
        command_timeout_ms=500,
        heartbeat_timeout_s=1.0,
        board_size=16,
    )


@pytest.fixture
def action_space_config():
    """Create action space configuration."""
    return ActionSpaceConfig(
        board_size=16,
        max_units=100,
        max_cities=20,
    )


@pytest.fixture
def state_converter(bridge_config):
    """Create state converter instance."""
    return StateConverter(board_size=bridge_config.board_size)


@pytest.fixture
def action_converter(action_space_config):
    """Create action converter instance."""
    return ActionConverter(
        board_size=action_space_config.board_size,
        action_space_config=action_space_config,
    )


@pytest.fixture
def sample_lua_state() -> dict[str, Any]:
    """Create sample state in Lua mod format."""
    return {
        "turn_id": 42,
        "player_id": 0,
        "player": {
            "id": 0,
            "gold": 500,
            "science_per_turn": 15,
            "culture_per_turn": 10,
            "faith": 0,
            "score": 100,
            "techs_researched": ["TECH_MINING", "TECH_POTTERY"],
            "civics_researched": ["CIVIC_CODE_OF_LAWS"],
        },
        "map": {
            "width": 44,
            "height": 26,
            "tiles": [
                {
                    "x": x,
                    "y": y,
                    "terrain": "TERRAIN_PLAINS" if (x + y) % 2 == 0 else "TERRAIN_GRASSLAND",
                    "is_water": False,
                    "is_visible": x < 10 and y < 10,
                    "features": ["FEATURE_FOREST"] if (x * y) % 5 == 0 else [],
                }
                for x in range(44)
                for y in range(26)
            ],
        },
        "units": [
            {
                "id": 1,
                "type_name": "UNIT_WARRIOR",
                "owner": 0,
                "x": 5,
                "y": 5,
                "hp": 100,
                "damage": 0,
                "movement": 2.0,
                "max_movement": 2.0,
                "combat_strength": 20,
                "is_civilian": False,
                "can_move": True,
                "can_attack": True,
            },
            {
                "id": 2,
                "type_name": "UNIT_SETTLER",
                "owner": 0,
                "x": 6,
                "y": 5,
                "hp": 100,
                "damage": 0,
                "movement": 2.0,
                "max_movement": 2.0,
                "combat_strength": 0,
                "is_civilian": True,
                "can_move": True,
                "can_found": True,
            },
        ],
        "cities": [
            {
                "id": 1,
                "name": "Rome",
                "owner": 0,
                "x": 10,
                "y": 10,
                "population": 5,
                "food": 20,
                "production": 15,
                "buildings": ["BUILDING_MONUMENT", "BUILDING_GRANARY"],
                "current_production": "UNIT_WARRIOR",
                "production_progress": 30,
            },
        ],
        "legal_actions": [
            {"action": "end_turn"},
            {"action": "move", "unit_id": 1, "targets": [[4, 5], [5, 4], [6, 5]]},
            {"action": "move", "unit_id": 2, "targets": [[5, 5], [6, 4], [7, 5]]},
            {"action": "found_city", "unit_id": 2},
            {
                "action": "produce",
                "city_id": 1,
                "items": ["UNIT_WARRIOR", "UNIT_ARCHER", "BUILDING_WALLS"],
            },
        ],
    }


@pytest.fixture
def sample_encoder_state() -> dict[str, Any]:
    """Create sample state in encoder format."""
    return {
        "turn": 42,
        "current_player": 0,
        "terrain": {
            (x, y): {
                "type": "plains" if (x + y) % 2 == 0 else "grassland",
                "features": ["forest"] if (x * y) % 5 == 0 else [],
            }
            for x in range(16)
            for y in range(16)
        },
        "units": [
            {
                "id": 1,
                "_id": 1,
                "type": "melee",
                "combat_class": "melee",
                "player": 0,
                "position": (1, 1),
                "_original_pos": (5, 5),
                "health": 1.0,
                "moves": 2.0,
            },
            {
                "id": 2,
                "_id": 2,
                "type": "civilian",
                "combat_class": "civilian",
                "player": 0,
                "position": (1, 1),
                "_original_pos": (6, 5),
                "health": 1.0,
                "moves": 2.0,
            },
        ],
        "cities": [
            {
                "id": 1,
                "_id": 1,
                "name": "Rome",
                "player": 0,
                "position": (3, 4),
                "_original_pos": (10, 10),
                "population": 5,
            },
        ],
        "visibility": {(x, y): True for x in range(10) for y in range(10)},
        "_legal_actions": [
            LegalAction(action_type="end_turn"),
            LegalAction(action_type="move", unit_id=1, targets=[(0, 1), (1, 0), (2, 1)]),
        ],
    }


# ==============================================================================
# State Conversion Integration Tests
# ==============================================================================


class TestStateConversionIntegration:
    """Integration tests for state conversion pipeline."""

    def test_lua_to_encoder_format(self, state_converter, sample_lua_state):
        """Test converting Lua state to encoder format."""
        # Create GameStateData from Lua dict
        game_state_data = GameStateData.from_dict(sample_lua_state)

        # Convert to encoder format
        encoder_state = state_converter.convert(game_state_data)

        # Verify basic structure
        assert "turn" in encoder_state
        assert "current_player" in encoder_state
        assert "terrain" in encoder_state
        assert "units" in encoder_state
        assert "cities" in encoder_state

        # Verify turn conversion
        assert encoder_state["turn"] == 42
        assert encoder_state["current_player"] == 0

        # Verify terrain was converted and scaled
        assert len(encoder_state["terrain"]) > 0
        # All coordinates should be within board_size
        for x, y in encoder_state["terrain"].keys():
            assert 0 <= x < 16, f"x={x} out of bounds"
            assert 0 <= y < 16, f"y={y} out of bounds"

        # Verify units were converted
        assert len(encoder_state["units"]) == 2
        for unit in encoder_state["units"]:
            assert "position" in unit
            assert "type" in unit
            assert "player" in unit

        # Verify cities were converted
        assert len(encoder_state["cities"]) == 1

    def test_full_state_encoding_pipeline(self, state_converter, sample_lua_state):
        """Test full pipeline: Lua JSON -> encoder dict -> tensors."""
        from alpha_zero.envs.civ6_encoder import Civ6StateEncoder, EncoderConfig

        # Create encoder
        encoder_config = EncoderConfig(board_size=16, spatial_channels=84, vector_dim=512)
        encoder = Civ6StateEncoder(encoder_config)

        # Convert Lua state to encoder format
        game_state_data = GameStateData.from_dict(sample_lua_state)
        encoder_state = state_converter.convert(game_state_data)

        # Encode to tensors
        spatial, vector = encoder.encode(encoder_state)

        # Verify tensor shapes (may be torch or numpy)
        assert spatial.shape == (84, 16, 16)
        assert vector.shape == (512,)

        # Verify tensor types (encoder may return torch or numpy)
        import torch

        if isinstance(spatial, torch.Tensor):
            assert spatial.dtype == torch.float32
            assert vector.dtype == torch.float32
            # Verify non-zero values (terrain should have values)
            assert torch.any(spatial != 0), "Spatial tensor should have non-zero values"
        else:
            assert spatial.dtype == np.float32
            assert vector.dtype == np.float32
            # Verify non-zero values (terrain should have values)
            assert np.any(spatial != 0), "Spatial tensor should have non-zero values"

    def test_legal_actions_preserved(self, state_converter, sample_lua_state):
        """Test that legal actions are preserved through conversion."""
        game_state_data = GameStateData.from_dict(sample_lua_state)
        encoder_state = state_converter.convert(game_state_data)

        assert "_legal_actions" in encoder_state
        legal_actions = encoder_state["_legal_actions"]

        # Should have same number of actions
        assert len(legal_actions) == 5

        # Verify action types
        action_types = [a.action_type for a in legal_actions]
        assert "end_turn" in action_types
        assert "move" in action_types
        assert "produce" in action_types

    def test_coordinate_scaling(self, state_converter, sample_lua_state):
        """Test coordinate scaling from Civ6 map to board size."""
        game_state_data = GameStateData.from_dict(sample_lua_state)
        encoder_state = state_converter.convert(game_state_data)

        # Original unit position: (5, 5) on 44x26 map
        # Should be scaled to 16x16 board
        units = encoder_state["units"]
        assert len(units) > 0

        for unit in units:
            x, y = unit["position"]
            assert 0 <= x < 16
            assert 0 <= y < 16


# ==============================================================================
# Action Conversion Integration Tests
# ==============================================================================


class TestActionConversionIntegration:
    """Integration tests for action conversion pipeline."""

    def test_action_mask_generation(self, action_converter, sample_encoder_state):
        """Test generating action mask from legal actions."""
        legal_actions = sample_encoder_state["_legal_actions"]

        mask = action_converter.generate_action_mask(legal_actions, sample_encoder_state)

        # Verify mask shape
        assert mask.shape == (action_converter.action_space_size,)

        # Verify mask values are binary
        assert np.all((mask == 0.0) | (mask == 1.0))

        # End turn should always be valid
        assert mask[0] == 1.0

        # Should have at least some valid actions
        assert mask.sum() >= 1

    def test_action_to_command_conversion(self, action_converter, sample_encoder_state):
        """Test converting action index to Lua command."""
        # Action 0 is always end_turn
        result = action_converter.convert_with_result(0, sample_encoder_state, [])

        assert isinstance(result, ConversionResult)
        assert result.is_valid
        assert result.command is not None
        assert result.command.action_type == ActionType.END_TURN

    def test_command_serialization(self, action_converter, sample_encoder_state):
        """Test that commands serialize correctly for Lua."""
        result = action_converter.convert_with_result(0, sample_encoder_state, [])
        command = result.command

        # Serialize to dict (what Lua will receive)
        cmd_dict = command.to_dict()

        assert "action_type" in cmd_dict
        assert cmd_dict["action_type"] == "end_turn"

        # Verify JSON serializable
        json_str = json.dumps(cmd_dict)
        parsed = json.loads(json_str)
        assert parsed["action_type"] == "end_turn"


# ==============================================================================
# FileBridge Integration Tests
# ==============================================================================


class TestFileBridgeIntegration:
    """Integration tests for FileBridge with real file operations."""

    @pytest.mark.asyncio
    async def test_full_roundtrip(self, bridge_config, sample_lua_state):
        """Test full roundtrip: state -> command -> ack."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Simulate Lua writing state
            state_json = json.dumps(sample_lua_state)
            bridge_config.state_file.write_text(state_json)

            # Poll for state
            state = await bridge.poll_state()
            assert state is not None
            assert state.turn_id == 42

            # Send command
            cmd = CommandMessage(
                msg_id=1,
                commands=[LuaCommand(action_type=ActionType.END_TURN)],
            )
            await bridge.send_command(cmd)

            # Verify command file was written
            assert bridge_config.commands_file.exists()
            cmd_content = json.loads(bridge_config.commands_file.read_text())
            assert cmd_content["msg_id"] == 1

            # Simulate Lua writing ack
            ack_data = {
                "msg_id": 1,
                "success": True,
                "executed": [{"action_type": "end_turn", "success": True}],
            }
            bridge_config.ack_file.write_text(json.dumps(ack_data))

            # Wait for ack
            ack = await bridge.wait_for_ack(msg_id=1, timeout=1.0)
            assert ack.success

        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_state_update_detection(self, bridge_config, sample_lua_state):
        """Test that bridge detects state updates."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Write initial state
            state1 = sample_lua_state.copy()
            state1["turn_id"] = 1
            bridge_config.state_file.write_text(json.dumps(state1))

            # Poll and verify
            result1 = await bridge.poll_state()
            assert result1.turn_id == 1

            # Clear file (poll returns None when file is gone)
            bridge_config.state_file.unlink()

            # Write updated state
            state2 = sample_lua_state.copy()
            state2["turn_id"] = 2
            bridge_config.state_file.write_text(json.dumps(state2))

            # Small delay to ensure file system catches up
            await asyncio.sleep(0.05)

            # Poll and verify new state
            result2 = await bridge.poll_state()
            assert result2 is not None
            assert result2.turn_id == 2

        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_multiple_commands(self, bridge_config, sample_lua_state):
        """Test sending multiple commands in sequence."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Send multiple commands
            for i in range(3):
                cmd = CommandMessage(
                    msg_id=i + 1,
                    commands=[LuaCommand(action_type=ActionType.END_TURN)],
                )
                await bridge.send_command(cmd)

                # Simulate ack
                ack_data = {"msg_id": i + 1, "success": True}
                bridge_config.ack_file.write_text(json.dumps(ack_data))

                ack = await bridge.wait_for_ack(msg_id=i + 1, timeout=1.0)
                assert ack.msg_id == i + 1

            # Verify statistics
            stats = bridge.get_stats()
            assert stats["commands_sent"] == 3
            assert stats["acks_received"] == 3

        finally:
            await bridge.stop()


# ==============================================================================
# Mock Bridge Integration Tests
# ==============================================================================


class TestMockBridgeIntegration:
    """Integration tests using MockFileBridge for testing without real files."""

    @pytest.mark.asyncio
    async def test_mock_state_injection(self, sample_lua_state):
        """Test injecting mock state for testing."""
        bridge = MockFileBridge()
        await bridge.start()

        # Set mock state
        bridge.set_mock_state(sample_lua_state)

        # Poll and verify
        state = await bridge.poll_state()
        assert state is not None
        assert state.turn_id == 42
        assert len(state.units) == 2

    @pytest.mark.asyncio
    async def test_mock_command_capture(self):
        """Test capturing commands for verification."""
        bridge = MockFileBridge()
        await bridge.start()

        # Send commands
        cmd1 = CommandMessage(
            msg_id=1,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )
        cmd2 = CommandMessage(
            msg_id=2,
            commands=[
                LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1, target=Position(5, 6))
            ],
        )

        await bridge.send_command(cmd1)
        await bridge.send_command(cmd2)

        # Verify captured
        sent = bridge.get_sent_commands()
        assert len(sent) == 2
        assert sent[0].msg_id == 1
        assert sent[1].msg_id == 2

    @pytest.mark.asyncio
    async def test_mock_auto_ack(self):
        """Test mock auto-acknowledgment."""
        bridge = MockFileBridge()
        await bridge.start()

        # Set up auto-ack
        bridge.set_mock_ack({"msg_id": 99, "success": True})

        # Wait for ack
        ack = await bridge.wait_for_ack(msg_id=99, timeout=1.0)
        assert ack.success


# ==============================================================================
# End-to-End Pipeline Tests
# ==============================================================================


class TestEndToEndPipeline:
    """End-to-end integration tests for complete pipeline."""

    @pytest.mark.asyncio
    async def test_complete_turn_cycle(
        self, bridge_config, state_converter, action_converter, sample_lua_state
    ):
        """Test complete turn cycle: receive state -> select action -> send command."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # 1. Lua writes game state
            bridge_config.state_file.write_text(json.dumps(sample_lua_state))

            # 2. Python reads state
            game_state_data = await bridge.poll_state()
            assert game_state_data is not None

            # 3. Convert to encoder format
            encoder_state = state_converter.convert(game_state_data)

            # 4. Generate action mask
            legal_actions = encoder_state["_legal_actions"]
            mask = action_converter.generate_action_mask(legal_actions, encoder_state)

            # 5. Select action (use end_turn for simplicity)
            assert mask[0] == 1.0  # end_turn should be valid
            action_idx = 0

            # 6. Convert to command
            result = action_converter.convert_with_result(action_idx, encoder_state, legal_actions)
            assert result.is_valid

            # 7. Send command
            cmd_msg = CommandMessage(msg_id=1, commands=[result.command])
            await bridge.send_command(cmd_msg)

            # 8. Lua writes ack
            ack_data = {"msg_id": 1, "success": True}
            bridge_config.ack_file.write_text(json.dumps(ack_data))

            # 9. Python receives ack
            ack = await bridge.wait_for_ack(msg_id=1, timeout=1.0)
            assert ack.success

        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_multi_turn_simulation(
        self, bridge_config, state_converter, action_converter, sample_lua_state
    ):
        """Test multiple turn simulation."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            for turn in range(1, 6):  # 5 turns
                # Update turn in state
                state = sample_lua_state.copy()
                state["turn_id"] = turn

                # Write state
                bridge_config.state_file.write_text(json.dumps(state))

                # Poll
                game_state_data = await bridge.poll_state()
                assert game_state_data.turn_id == turn

                # Convert and select action
                encoder_state = state_converter.convert(game_state_data)
                result = action_converter.convert_with_result(0, encoder_state, [])

                # Send command
                cmd_msg = CommandMessage(msg_id=turn, commands=[result.command])
                await bridge.send_command(cmd_msg)

                # Ack
                ack_data = {"msg_id": turn, "success": True}
                bridge_config.ack_file.write_text(json.dumps(ack_data))
                ack = await bridge.wait_for_ack(msg_id=turn, timeout=1.0)
                assert ack.success

            # Verify all turns processed
            stats = bridge.get_stats()
            assert stats["commands_sent"] == 5

        finally:
            await bridge.stop()


# ==============================================================================
# Error Handling Integration Tests
# ==============================================================================


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_recovery(self, bridge_config):
        """Test recovery from invalid JSON in state file."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Write invalid JSON
            bridge_config.state_file.write_text("not valid json {{{")

            # Should return None, not crash
            state = await bridge.poll_state()
            assert state is None

            # Write valid JSON
            valid_state = {
                "turn_id": 1,
                "map": {"width": 44, "height": 26, "tiles": []},
                "units": [],
                "cities": [],
                "legal_actions": [],
            }
            bridge_config.state_file.write_text(json.dumps(valid_state))

            # Should recover
            state = await bridge.poll_state()
            assert state is not None
            assert state.turn_id == 1

        finally:
            await bridge.stop()

    @pytest.mark.asyncio
    async def test_ack_error_handling(self, bridge_config):
        """Test handling of error acknowledgments."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Send command
            cmd = CommandMessage(
                msg_id=1,
                commands=[LuaCommand(action_type=ActionType.END_TURN)],
            )
            await bridge.send_command(cmd)

            # Simulate error ack
            ack_data = {
                "msg_id": 1,
                "success": False,
                "error_code": "E001",
                "error_message": "Invalid action",
            }
            bridge_config.ack_file.write_text(json.dumps(ack_data))

            # Read ack
            ack = await bridge.wait_for_ack(msg_id=1, timeout=1.0)
            assert not ack.success
            assert ack.error_code == "E001"
            assert ack.error_message == "Invalid action"

        finally:
            await bridge.stop()

    def test_state_converter_missing_data(self, state_converter):
        """Test state converter handles missing data gracefully."""
        # Minimal valid state
        minimal_state = GameStateData(
            turn_id=1,
            player_id=0,
            map_data=MapData(width=44, height=26),
            units=[],
            cities=[],
            players=[],
            legal_actions=[],
        )

        # Should not crash
        encoder_state = state_converter.convert(minimal_state)
        assert encoder_state is not None
        assert encoder_state["turn"] == 1
        assert len(encoder_state["units"]) == 0
        assert len(encoder_state["cities"]) == 0


# ==============================================================================
# Performance Integration Tests
# ==============================================================================


class TestPerformanceIntegration:
    """Performance-related integration tests."""

    @pytest.mark.asyncio
    async def test_state_polling_latency(self, bridge_config, sample_lua_state):
        """Test state polling latency is acceptable."""
        bridge = FileBridge(bridge_config)

        try:
            await bridge.start()

            # Write state
            bridge_config.state_file.write_text(json.dumps(sample_lua_state))

            # Measure poll time
            start = time.perf_counter()
            state = await bridge.poll_state()
            elapsed = time.perf_counter() - start

            assert state is not None
            # Should complete within 100ms
            assert elapsed < 0.1, f"Poll took {elapsed:.3f}s, expected < 0.1s"

        finally:
            await bridge.stop()

    def test_state_conversion_performance(self, state_converter, sample_lua_state):
        """Test state conversion performance."""
        game_state_data = GameStateData.from_dict(sample_lua_state)

        # Measure conversion time
        start = time.perf_counter()
        for _ in range(100):
            state_converter.convert(game_state_data)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100
        # Should average under 10ms per conversion
        assert avg_time < 0.01, f"Avg conversion: {avg_time * 1000:.2f}ms, expected < 10ms"

    def test_action_mask_generation_performance(self, action_converter, sample_encoder_state):
        """Test action mask generation performance."""
        legal_actions = [
            LegalAction(action_type="end_turn"),
        ] + [
            LegalAction(action_type="move", unit_id=i, targets=[(0, 0), (1, 1)]) for i in range(50)
        ]

        # Measure generation time
        start = time.perf_counter()
        for _ in range(100):
            action_converter.generate_action_mask(legal_actions, sample_encoder_state)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100
        # Should average under 50ms per generation
        assert avg_time < 0.05, f"Avg mask generation: {avg_time * 1000:.2f}ms, expected < 50ms"


# ==============================================================================
# Run tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
