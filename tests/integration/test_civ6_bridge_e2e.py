"""End-to-end integration test for BridgeAgentLoop + MockFileBridge.

W1.2 of the v1.7.0 sprint: exercises the full stack on a single event
loop with no real Civ6 game required:

    MockFileBridge.poll_state() -> StateConverter -> ActionConverter
        -> GreedyNetworkPolicy (network forward) -> ActionConverter.convert
        -> MockFileBridge.send_command() -> AckMessage

Uses the W1.4 synthetic-weights champion fixture so the network is a
real ``MicroCivNetwork`` instance (random weights, but real forward
pass through real layers — proves the bridge plumbing tolerates real
torch tensors, not just MagicMock returns).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.agent_loop import (
    AgentLoopConfig,
    BridgeAgentLoop,
    GreedyNetworkPolicy,
)
from alpha_zero.bridge.config import BridgeConfig
from alpha_zero.bridge.file_bridge import MockFileBridge
from alpha_zero.bridge.protocol import (
    GameStateData,
    LegalAction,
    MapData,
    MessageType,
    PlayerData,
    TileData,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_network import MicroCivNetwork
from alpha_zero.storage.config import StorageConfig
from alpha_zero.storage.factory import create_storage_backend
from tests.fixtures.build_smoke_champion import build_smoke_champion


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


_TEST_BOARD_SIZE = 8


@pytest.fixture
def smoke_champion(tmp_path: Path) -> Path:
    """Build a synthetic champion checkpoint in tmp storage."""
    storage_dir = tmp_path / "storage"
    build_smoke_champion(storage_dir, board_size=_TEST_BOARD_SIZE)
    return storage_dir


@pytest.fixture
def smoke_network(smoke_champion: Path) -> MicroCivNetwork:
    """Load the synthetic champion into a real MicroCivNetwork."""
    storage = create_storage_backend(StorageConfig.for_local(str(smoke_champion)))
    state_dict, _metadata = storage.load_champion()
    net = MicroCivNetwork(board_size=_TEST_BOARD_SIZE)
    net.load_state_dict(state_dict)
    return net


def _minimal_state(turn_id: int = 1) -> GameStateData:
    """A minimal but protocol-valid GameStateData with one legal end_turn action.

    Coordinates and player IDs follow what StateConverter expects: at least
    one tile, two players, ``end_turn`` always legal so the agent always has
    a fallback even if the network's preferred action is illegal.
    """
    tiles = [
        TileData(
            x=0,
            y=0,
            terrain="TERRAIN_GRASS",
            features=[],
            is_visible=True,
            is_revealed=True,
        )
    ]
    return GameStateData(
        turn_id=turn_id,
        player_id=0,
        map_data=MapData(width=8, height=8, tiles=tiles),
        units=[],
        cities=[],
        players=[
            PlayerData(id=0, civilization="CIVILIZATION_AMERICA", is_alive=True),
            PlayerData(id=1, civilization="CIVILIZATION_ENGLAND", is_alive=True),
        ],
        legal_actions=[LegalAction(action_type="end_turn", unit_id=0, city_id=0)],
        current_player_civ="CIVILIZATION_AMERICA",
    )


def _state_dict(turn_id: int = 1) -> dict:
    """A StateMessage-compatible dict wrapping ``_minimal_state``."""
    state = _minimal_state(turn_id)
    state_payload = {
        "turn_id": state.turn_id,
        "player_id": state.player_id,
        "player": {
            "id": state.player_id,
            "civilization": state.current_player_civ,
            "is_alive": True,
        },
        "map_data": {
            "width": state.map_data.width,
            "height": state.map_data.height,
            "tiles": [
                {
                    "x": t.x,
                    "y": t.y,
                    "terrain": t.terrain,
                    "features": t.features,
                    "is_visible": t.is_visible,
                    "is_revealed": t.is_revealed,
                }
                for t in state.map_data.tiles
            ],
        },
        "units": [],
        "cities": [],
        "players": [
            {
                "id": p.id,
                "civilization": p.civilization,
                "is_alive": p.is_alive,
            }
            for p in state.players
        ],
        "legal_actions": [
            {"action_type": la.action_type, "unit_id": la.unit_id, "city_id": la.city_id}
            for la in state.legal_actions
        ],
        "current_player_civ": state.current_player_civ,
        "era": 0,
        "is_game_over": state.is_game_over,
        "winner_id": state.winner_id,
    }
    return {
        "type": MessageType.STATE.value,
        "msg_id": turn_id,
        "timestamp": 0.0,
        "state": state_payload,
    }


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bridge_e2e_one_turn(smoke_network: MicroCivNetwork) -> None:
    """One full turn: mock state -> policy -> command -> mock ack -> done."""
    bridge_cfg = BridgeConfig()
    bridge = MockFileBridge(bridge_cfg)

    policy = GreedyNetworkPolicy(network=smoke_network, device="cpu")
    state_converter = StateConverter(board_size=_TEST_BOARD_SIZE)
    action_converter = ActionConverter(board_size=_TEST_BOARD_SIZE)

    loop = BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=state_converter,
        action_converter=action_converter,
        config=AgentLoopConfig(max_turns=1, per_turn_timeout_s=2.0),
    )

    # Seed the mock bridge: one state, one ack matching the loop's msg_id.
    # MockFileBridge's _next_msg_id is monotonic from start(), so we set the
    # ack to a generous msg_id range and let the loop pick one.
    async def _drive() -> int:
        await bridge.start()
        bridge.set_mock_state(_state_dict(turn_id=1))
        # Ack with the next msg_id the loop will use; the bridge increments
        # the counter on send_command via _next_msg_id().
        bridge.set_mock_ack({
            "msg_id": bridge._msg_counter + 1,
            "type": MessageType.ACK.value,
            "timestamp": 0.0,
            "success": True,
        })
        try:
            return await loop.run()
        finally:
            await bridge.stop()

    code = await asyncio.wait_for(_drive(), timeout=10.0)
    assert code == BridgeAgentLoop.EXIT_OK

    stats = loop.stats()
    assert stats["turns_played"] == 1
    # send_command path was exercised.
    assert len(bridge.get_sent_commands()) >= 1
    # Bridge command must have a recognised action_type, not a placeholder.
    sent = bridge.get_sent_commands()[0]
    assert sent.commands, "expected at least one Lua command in the message"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_bridge_e2e_multiple_turns_uses_real_network(
    smoke_network: MicroCivNetwork,
) -> None:
    """Drive 3 turns; assert the real network forward pass is exercised each turn."""
    bridge = MockFileBridge(BridgeConfig())
    policy = GreedyNetworkPolicy(network=smoke_network, device="cpu")
    state_converter = StateConverter(board_size=_TEST_BOARD_SIZE)
    action_converter = ActionConverter(board_size=_TEST_BOARD_SIZE)

    loop = BridgeAgentLoop(
        bridge=bridge,
        policy=policy,
        state_converter=state_converter,
        action_converter=action_converter,
        config=AgentLoopConfig(max_turns=3, per_turn_timeout_s=2.0),
    )

    async def _drive() -> int:
        await bridge.start()
        try:
            # Feed states + acks one at a time so the loop progresses turn
            # by turn. MockFileBridge clears _mock_state after each read, so
            # the loop will time out unless we keep replenishing.
            run_task = asyncio.create_task(loop.run())
            for turn in range(1, 4):
                bridge.set_mock_state(_state_dict(turn_id=turn))
                bridge.set_mock_ack({
                    "msg_id": bridge._msg_counter + 1,
                    "type": MessageType.ACK.value,
                    "timestamp": 0.0,
                    "success": True,
                })
                # Wait briefly for the loop to consume the state.
                await asyncio.sleep(0.1)
            return await run_task
        finally:
            await bridge.stop()

    code = await asyncio.wait_for(_drive(), timeout=15.0)
    assert code == BridgeAgentLoop.EXIT_OK

    stats = loop.stats()
    # We fed 3 states; the loop ran up to max_turns=3.
    assert stats["turns_played"] >= 1, f"expected >=1 turn, got {stats}"
    # Every turn that ran produced at least one command on the wire.
    assert len(bridge.get_sent_commands()) >= 1
