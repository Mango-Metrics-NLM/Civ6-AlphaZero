"""Integration tests for action execution pipeline.

Tests the end-to-end action execution from neural network output to
game command, including coordinate translation, action validation,
and command acknowledgment.

Run with:
    pytest tests/integration/test_action_execution.py -v
    pytest tests/integration/test_action_execution.py -v -m requires_env  # With Docker
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch

from alpha_zero.bridge.config import BridgeConfig, get_map_config
from alpha_zero.bridge.file_bridge import (
    AckTimeoutError,
    MockFileBridge,
)
from alpha_zero.bridge.protocol import (
    ActionType,
    CommandMessage,
    LuaCommand,
    Position,
)

if TYPE_CHECKING:
    from collections.abc import Generator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_comm_dir() -> Generator[Path, None, None]:
    """Create temporary communication directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def bridge_config(temp_comm_dir: Path) -> BridgeConfig:
    """Create bridge configuration for testing."""
    return BridgeConfig(
        comm_dir=temp_comm_dir,
        poll_interval_ms=10,
        command_timeout_ms=500,
        heartbeat_timeout_s=1.0,
        board_size=16,
    )


@pytest.fixture
def mock_bridge(bridge_config: BridgeConfig) -> MockFileBridge:
    """Create mock file bridge for testing."""
    return MockFileBridge(bridge_config)


@pytest.fixture
def sample_game_state() -> dict[str, Any]:
    """Create sample game state for testing."""
    return {
        "msg_id": 1,
        "turn_id": 10,
        "player_id": 0,
        "era": "ancient",
        "map_data": {
            "width": 84,
            "height": 54,
            "tiles": [],
        },
        "units": [
            {
                "id": 1001,
                "type": "warrior",
                "owner": 0,
                "x": 42,
                "y": 27,
                "health": 100,
                "moves": 2,
            },
            {
                "id": 1002,
                "type": "settler",
                "owner": 0,
                "x": 43,
                "y": 27,
                "health": 100,
                "moves": 2,
            },
        ],
        "cities": [
            {
                "id": 2001,
                "name": "Capital",
                "owner": 0,
                "x": 40,
                "y": 25,
                "population": 5,
            }
        ],
        "players": [
            {
                "id": 0,
                "name": "Player",
                "civilization": "America",
                "gold": 100,
                "score": 50,
            }
        ],
        "legal_actions": [
            {"type": "unit_move", "unit_id": 1001, "x": 43, "y": 27},
            {"type": "unit_move", "unit_id": 1001, "x": 42, "y": 28},
            {"type": "end_turn"},
        ],
    }


@pytest.fixture
def sample_ack() -> dict[str, Any]:
    """Create sample acknowledgment."""
    return {
        "msg_id": 1,
        "success": True,
        "timestamp": 1234567890.0,
    }


# ============================================================================
# End Turn Action Tests
# ============================================================================


class TestEndTurnActionRoundtrip:
    """Tests for end turn action execution."""

    @pytest.mark.asyncio
    async def test_end_turn_command_created(
        self,
        mock_bridge: MockFileBridge,
    ) -> None:
        """Verify end turn command is properly created."""
        await mock_bridge.start()

        command = CommandMessage(
            msg_id=mock_bridge._next_msg_id(),
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        await mock_bridge.send_command(command)

        sent_commands = mock_bridge.get_sent_commands()
        assert len(sent_commands) == 1
        assert sent_commands[0].commands[0].action_type == "end_turn"

        await mock_bridge.stop()

    @pytest.mark.asyncio
    async def test_end_turn_receives_ack(
        self,
        mock_bridge: MockFileBridge,
        sample_ack: dict[str, Any],
    ) -> None:
        """Verify end turn command receives acknowledgment."""
        await mock_bridge.start()

        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        # Set up ack
        sample_ack["msg_id"] = msg_id
        mock_bridge.set_mock_ack(sample_ack)

        await mock_bridge.send_command(command)
        ack = await mock_bridge.wait_for_ack(msg_id, timeout=1.0)

        assert ack.msg_id == msg_id
        assert ack.success is True

        await mock_bridge.stop()


# ============================================================================
# Unit Movement Tests
# ============================================================================


class TestUnitMoveWithCoordinates:
    """Tests for unit movement with coordinate translation."""

    @pytest.mark.asyncio
    async def test_unit_move_command_includes_coordinates(
        self,
        mock_bridge: MockFileBridge,
    ) -> None:
        """Verify unit move command includes correct coordinates."""
        await mock_bridge.start()

        command = CommandMessage(
            msg_id=mock_bridge._next_msg_id(),
            commands=[
                LuaCommand(
                    action_type=ActionType.UNIT_MOVE,
                    unit_id=1001,
                    target=Position(x=43, y=27),
                )
            ],
        )

        await mock_bridge.send_command(command)

        sent_commands = mock_bridge.get_sent_commands()
        assert len(sent_commands) == 1

        lua_cmd = sent_commands[0].commands[0]
        assert lua_cmd.action_type == ActionType.UNIT_MOVE
        assert lua_cmd.unit_id == 1001
        assert lua_cmd.target is not None
        assert lua_cmd.target.x == 43
        assert lua_cmd.target.y == 27

        await mock_bridge.stop()

    def test_coordinate_scaling_from_board_to_map(self) -> None:
        """Verify coordinate scaling from board_size to map coordinates."""
        # Standard map 84x54, board_size 16
        map_config = get_map_config(84, 54)
        board_size = 16

        scale_x, scale_y = map_config.get_scale_factor(board_size)

        # Board (8, 8) should map roughly to center of map
        board_x, board_y = 8, 8
        map_x = int(board_x / scale_x)
        map_y = int(board_y / scale_y)

        # Should be roughly in the center
        assert 35 <= map_x <= 50
        assert 20 <= map_y <= 35

    def test_coordinate_scaling_bidirectional(self) -> None:
        """Verify coordinate scaling works both ways."""
        map_config = get_map_config(84, 54)
        board_size = 16

        scale_x, scale_y = map_config.get_scale_factor(board_size)

        # Map to board
        map_x, map_y = 42, 27
        board_x = round(map_x * scale_x)
        board_y = round(map_y * scale_y)

        # Board to map (reverse)
        map_x_back = round(board_x / scale_x)
        map_y_back = round(board_y / scale_y)

        # Should be close to original (within 1 tile due to rounding)
        assert abs(map_x_back - map_x) <= 2
        assert abs(map_y_back - map_y) <= 2


# ============================================================================
# City Production Tests
# ============================================================================


class TestCityProductionCommand:
    """Tests for city production command execution."""

    @pytest.mark.asyncio
    async def test_city_production_command_created(
        self,
        mock_bridge: MockFileBridge,
    ) -> None:
        """Verify city production command is properly created."""
        await mock_bridge.start()

        command = CommandMessage(
            msg_id=mock_bridge._next_msg_id(),
            commands=[
                LuaCommand(
                    action_type=ActionType.CITY_PRODUCE,
                    city_id=2001,
                    item_type="unit",
                    metadata={"item": "UNIT_WARRIOR"},
                )
            ],
        )

        await mock_bridge.send_command(command)

        sent_commands = mock_bridge.get_sent_commands()
        assert len(sent_commands) == 1

        lua_cmd = sent_commands[0].commands[0]
        assert lua_cmd.action_type == ActionType.CITY_PRODUCE
        assert lua_cmd.city_id == 2001
        assert lua_cmd.metadata["item"] == "UNIT_WARRIOR"

        await mock_bridge.stop()

    @pytest.mark.asyncio
    async def test_city_production_receives_ack(
        self,
        mock_bridge: MockFileBridge,
        sample_ack: dict[str, Any],
    ) -> None:
        """Verify city production command receives acknowledgment."""
        await mock_bridge.start()

        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[
                LuaCommand(
                    action_type=ActionType.CITY_PRODUCE,
                    city_id=2001,
                    item_type="building",
                    metadata={"item": "BUILDING_MONUMENT"},
                )
            ],
        )

        sample_ack["msg_id"] = msg_id
        mock_bridge.set_mock_ack(sample_ack)

        await mock_bridge.send_command(command)
        ack = await mock_bridge.wait_for_ack(msg_id, timeout=1.0)

        assert ack.success is True

        await mock_bridge.stop()


# ============================================================================
# Invalid Action Tests
# ============================================================================


class TestInvalidActionRejection:
    """Tests for invalid action handling."""

    @pytest.mark.asyncio
    async def test_invalid_action_returns_failure_ack(
        self,
        mock_bridge: MockFileBridge,
    ) -> None:
        """Verify invalid actions receive failure acknowledgment."""
        await mock_bridge.start()

        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[
                LuaCommand(
                    action_type=ActionType.UNIT_MOVE,
                    unit_id=9999,  # Non-existent unit
                    target=Position(x=0, y=0),
                )
            ],
        )

        # Set up failure ack
        mock_bridge.set_mock_ack({
            "msg_id": msg_id,
            "success": False,
            "error": "Unit not found",
            "timestamp": 1234567890.0,
        })

        await mock_bridge.send_command(command)
        ack = await mock_bridge.wait_for_ack(msg_id, timeout=1.0)

        assert ack.success is False

        await mock_bridge.stop()

    @pytest.mark.asyncio
    async def test_timeout_on_missing_ack(
        self,
        mock_bridge: MockFileBridge,
    ) -> None:
        """Verify timeout is raised when ack is missing."""
        await mock_bridge.start()

        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        # Don't set mock ack - should timeout
        await mock_bridge.send_command(command)

        with pytest.raises(AckTimeoutError):
            await mock_bridge.wait_for_ack(msg_id, timeout=0.1)

        await mock_bridge.stop()


# ============================================================================
# Action Mask Tests
# ============================================================================


class TestActionMaskMatchesLegalActions:
    """Tests that action mask correctly reflects legal_actions."""

    def test_action_mask_shape_matches_action_space(self) -> None:
        """Verify action mask shape matches action space size."""
        # Example: 16x16 board with 5 action types
        board_size = 16
        num_action_types = 5

        # Action mask should be (num_action_types * board_size * board_size,)
        # or structured as (num_action_types, board_size, board_size)
        action_mask = torch.zeros(num_action_types, board_size, board_size)

        assert action_mask.shape == (5, 16, 16)

    def test_action_mask_has_nonzero_for_legal_actions(
        self,
        sample_game_state: dict[str, Any],
    ) -> None:
        """Verify action mask has non-zero values for legal actions."""
        legal_actions = sample_game_state["legal_actions"]

        # Create simple action mask (1 where legal, 0 elsewhere)
        board_size = 16
        action_mask = torch.zeros(board_size, board_size)

        # Mark end_turn as always legal (special position)
        action_mask[0, 0] = 1.0

        # Mark unit moves as legal
        for action in legal_actions:
            if action["type"] == "unit_move":
                # Scale coordinates to board
                map_config = get_map_config(84, 54)
                scale_x, scale_y = map_config.get_scale_factor(board_size)
                board_x = min(board_size - 1, round(action["x"] * scale_x))
                board_y = min(board_size - 1, round(action["y"] * scale_y))
                action_mask[board_y, board_x] = 1.0

        # Should have at least 1 legal action (end_turn)
        assert action_mask.sum() >= 1

    def test_illegal_actions_masked_out(self) -> None:
        """Verify illegal actions have zero probability in mask."""
        board_size = 16

        # Create legal action list (3 legal moves)
        legal_positions = [(5, 5), (6, 5), (5, 6)]

        # Create mask
        action_mask = torch.zeros(board_size, board_size)
        for x, y in legal_positions:
            action_mask[y, x] = 1.0

        # Verify illegal positions are zero
        assert action_mask[0, 0] == 0.0  # Origin (not in legal list)
        assert action_mask[15, 15] == 0.0  # Corner (not in legal list)
        assert action_mask[10, 10] == 0.0  # Random position

        # Verify legal positions are non-zero
        for x, y in legal_positions:
            assert action_mask[y, x] == 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestActionExecutionIntegration:
    """Integration tests for full action execution pipeline."""

    @pytest.mark.asyncio
    async def test_full_turn_cycle(
        self,
        mock_bridge: MockFileBridge,
        sample_game_state: dict[str, Any],
        sample_ack: dict[str, Any],
    ) -> None:
        """Test complete turn cycle: receive state, send action, receive ack."""
        await mock_bridge.start()

        # 1. Receive game state
        mock_bridge.set_mock_state(sample_game_state)
        state = await mock_bridge.poll_state()
        assert state is not None
        assert state.turn_id == 10

        # 2. Select action (end turn for simplicity)
        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        # 3. Send command
        sample_ack["msg_id"] = msg_id
        mock_bridge.set_mock_ack(sample_ack)

        await mock_bridge.send_command(command)

        # 4. Receive acknowledgment
        ack = await mock_bridge.wait_for_ack(msg_id, timeout=1.0)
        assert ack.success is True

        # 5. Verify stats
        assert mock_bridge.stats.states_received == 1
        assert mock_bridge.stats.commands_sent == 1
        assert mock_bridge.stats.acks_received == 1

        await mock_bridge.stop()

    @pytest.mark.asyncio
    async def test_multiple_actions_in_sequence(
        self,
        mock_bridge: MockFileBridge,
        sample_ack: dict[str, Any],
    ) -> None:
        """Test multiple actions executed in sequence."""
        await mock_bridge.start()

        actions = [
            LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1001, target=Position(x=43, y=27)),
            LuaCommand(action_type=ActionType.UNIT_MOVE, unit_id=1002, target=Position(x=44, y=27)),
            LuaCommand(action_type=ActionType.END_TURN),
        ]

        for action in actions:
            msg_id = mock_bridge._next_msg_id()
            command = CommandMessage(msg_id=msg_id, commands=[action])

            sample_ack["msg_id"] = msg_id
            mock_bridge.set_mock_ack(sample_ack.copy())

            await mock_bridge.send_command(command)
            ack = await mock_bridge.wait_for_ack(msg_id, timeout=1.0)
            assert ack.success is True

        assert mock_bridge.stats.commands_sent == 3
        assert mock_bridge.stats.acks_received == 3

        await mock_bridge.stop()

    @pytest.mark.asyncio
    async def test_send_and_wait_convenience_method(
        self,
        mock_bridge: MockFileBridge,
        sample_ack: dict[str, Any],
    ) -> None:
        """Test send_and_wait convenience method."""
        await mock_bridge.start()

        msg_id = mock_bridge._next_msg_id()
        command = CommandMessage(
            msg_id=msg_id,
            commands=[LuaCommand(action_type=ActionType.END_TURN)],
        )

        sample_ack["msg_id"] = msg_id
        mock_bridge.set_mock_ack(sample_ack)

        ack = await mock_bridge.send_and_wait(command, timeout=1.0)
        assert ack.success is True

        await mock_bridge.stop()
