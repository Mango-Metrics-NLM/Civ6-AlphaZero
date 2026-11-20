"""GM-B2: the Python side of exporter parity (protocol fields and the agent loop).

Covers ``schema_version``, the end-turn blocker fields, ``ResearchStatus``,
``LegalAction.item_names`` and the ``END_TURN_BLOCKED`` handling in
``BridgeAgentLoop``. Only ``TestRealExport`` needs the Lua runtime (lupa).
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from alpha_zero.bridge import protocol
from alpha_zero.bridge.agent_loop import AgentLoopConfig, BridgeAgentLoop
from alpha_zero.bridge.protocol import (
    AckMessage,
    GameStateData,
    LegalAction,
    LuaCommand,
    PlayerData,
)
from tests.fixtures import lua_harness as h

_STATUS = {
    "type": "TECH_MINING",
    "index": 1,
    "cost": 50,
    "progress": 20,
    "turns_left": 8,
    "boost_triggered": True,
}


def _payload(**extra: Any) -> dict[str, Any]:
    """The smallest legacy-shaped state payload the reader accepts."""
    base: dict[str, Any] = {
        "turn_id": 3,
        "player_id": 0,
        "map": {"width": 1, "height": 1, "tiles": []},
        "units": [],
        "cities": [],
        "legal_actions": [{"action": "end_turn"}],
        "player": {"id": 0, "civilization": "CIVILIZATION_STUB_A", "current_research": 1},
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# protocol
# ---------------------------------------------------------------------------
class TestProtocolConstants:
    def test_end_turn_blocked_constant(self) -> None:
        assert protocol.END_TURN_BLOCKED == "END_TURN_BLOCKED"


class TestGameStateFields:
    def test_legacy_payload_defaults(self) -> None:
        parsed = GameStateData.from_dict(_payload())
        assert parsed.schema_version == 1
        assert parsed.end_turn_blocking is None
        assert parsed.end_turn_blocking_message is None
        # The legacy raw-integer research survives untouched, with no status object.
        assert parsed.players[0].current_research == 1
        assert parsed.players[0].research_status is None
        assert parsed.players[0].civic_status is None
        assert parsed.legal_actions[0].action_type == "end_turn"
        assert parsed.legal_actions[0].item_names == []

    def test_new_fields_are_read(self) -> None:
        parsed = GameStateData.from_dict(
            _payload(
                schema_version=2,
                end_turn_blocking="ENDTURN_BLOCKING_RESEARCH",
                end_turn_blocking_message="Choose a technology",
            )
        )
        assert parsed.schema_version == 2
        assert parsed.end_turn_blocking == "ENDTURN_BLOCKING_RESEARCH"
        assert parsed.end_turn_blocking_message == "Choose a technology"


class TestResearchStatus:
    def test_from_dict_reads_every_field(self) -> None:
        status = protocol.ResearchStatus.from_dict(_STATUS)
        assert status.type == "TECH_MINING"
        assert status.index == 1
        assert status.cost == 50
        assert status.progress == 20
        assert status.turns_left == 8
        assert status.boost_triggered is True

    def test_missing_optional_fields_are_none(self) -> None:
        status = protocol.ResearchStatus.from_dict({"type": "TECH_MINING"})
        assert status.type == "TECH_MINING"
        assert status.index is None
        assert status.cost is None
        assert status.progress is None
        assert status.turns_left is None
        assert status.boost_triggered is None

    def test_to_dict_round_trips(self) -> None:
        status = protocol.ResearchStatus.from_dict(_STATUS)
        again = protocol.ResearchStatus.from_dict(status.to_dict())
        assert again == status
        assert status.to_dict()["type"] == "TECH_MINING"
        assert status.to_dict()["turns_left"] == 8

    @pytest.mark.parametrize("field", ["research_status", "civic_status"])
    def test_player_parses_the_status_object(self, field: str) -> None:
        player = PlayerData.from_dict({"id": 0, "civilization": "X", field: _STATUS})
        status = getattr(player, field)
        assert isinstance(status, protocol.ResearchStatus)
        assert (status.type, status.turns_left) == ("TECH_MINING", 8)

    @pytest.mark.parametrize("field", ["research_status", "civic_status"])
    @pytest.mark.parametrize("junk", ["TECH_MINING", 7, [1, 2], True])
    def test_non_dict_status_parses_as_none(self, field: str, junk: Any) -> None:
        player = PlayerData.from_dict({"id": 0, "civilization": "X", field: junk})
        assert getattr(player, field) is None
        # A bad status must not cost the rest of the player block.
        assert player.civilization == "X"


class TestLegalActionItemNames:
    def test_default_is_empty(self) -> None:
        assert LegalAction(action_type="end_turn").item_names == []

    def test_from_dict_reads_item_names(self) -> None:
        action = LegalAction.from_dict(
            {"action": "research_tech", "items": [2], "item_names": ["TECH_WRITING"]}
        )
        assert action.items == [2]
        assert action.item_names == ["TECH_WRITING"]

    def test_to_dict_writes_item_names_only_when_non_empty(self) -> None:
        named = LegalAction(action_type="research_tech", items=[2], item_names=["TECH_WRITING"])
        assert named.to_dict()["item_names"] == ["TECH_WRITING"]
        assert "item_names" not in LegalAction(action_type="research_tech", items=[2]).to_dict()


class TestRealExport:
    """The mod's own export, through the harness, parses and carries the new fields."""

    def test_new_fields_come_through(self, caplog: pytest.LogCaptureFixture) -> None:
        runtime = h.load_mod()
        runtime.execute(
            "Civ6Stub.notifications[0] = { { id = 5, blocking = EndTurnBlockingTypes."
            "ENDTURN_BLOCKING_RESEARCH, message = 'Choose a technology', dismissed = false } }"
        )
        state = h.export_state(runtime)
        with caplog.at_level(logging.WARNING):
            parsed = GameStateData.from_dict(state)
        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
        assert parsed.schema_version == 2
        assert parsed.end_turn_blocking == "ENDTURN_BLOCKING_RESEARCH"
        assert parsed.end_turn_blocking_message == "Choose a technology"
        player = parsed.players[0]
        assert player.current_research == "TECH_MINING"
        assert player.techs_researched == ["TECH_POTTERY"]
        assert player.civics_researched == ["CIVIC_CODE_OF_LAWS"]
        assert player.research_status is not None
        assert (player.research_status.type, player.research_status.index) == ("TECH_MINING", 1)
        assert player.research_status.cost == 50
        assert player.research_status.progress == 20
        assert player.research_status.turns_left == 8
        assert player.research_status.boost_triggered is True
        assert player.civic_status is not None
        assert player.civic_status.type == "CIVIC_CRAFTSMANSHIP"
        assert player.civic_status.turns_left == 15
        assert player.civic_status.boost_triggered is False
        by_type = {a.action_type: a for a in parsed.legal_actions}
        assert by_type["research_tech"].items == [2]
        assert by_type["research_tech"].item_names == ["TECH_WRITING"]
        assert by_type["research_civic"].item_names == ["CIVIC_FOREIGN_TRADE"]


# ---------------------------------------------------------------------------
# agent loop
# ---------------------------------------------------------------------------
@pytest.fixture
def bridge() -> MagicMock:
    from alpha_zero.bridge.config import BridgeConfig

    mock = MagicMock()
    mock.config = BridgeConfig()
    mock.next_msg_id = MagicMock(return_value=1)
    mock.check_heartbeat = MagicMock(return_value=True)
    state = MagicMock(name="GameStateData")
    state.turn_id = 1
    state.player_id = 0
    state.is_game_over = False
    state.winner_id = None
    state.legal_actions = []
    mock.wait_for_state = AsyncMock(return_value=state)
    mock.send_command = AsyncMock(return_value=True)
    mock.send_and_wait_with_retry = AsyncMock(
        return_value=AckMessage(msg_id=1, success=True, timestamp=0.0)
    )
    return mock


@pytest.fixture
def make_loop(bridge: MagicMock) -> Any:
    state_converter = MagicMock(name="StateConverter")
    state_converter.convert.return_value = {
        "current_player": 0,
        "units": [],
        "cities": [],
        "_legal_actions": [],
        "_map_size": (8, 8),
        "_obs": {
            "spatial": np.zeros((13, 8, 8), dtype=np.float32),
            "vector": np.zeros(64, dtype=np.float32),
        },
    }
    action_converter = MagicMock(name="ActionConverter")
    action_converter.generate_action_mask.return_value = np.ones(8, dtype=np.float32)
    action_converter.convert.side_effect = lambda *a, **k: LuaCommand.from_dict(
        {"action_type": "end_turn"}
    )
    policy = AsyncMock(name="PolicyEngine")
    policy.select_action.return_value = (0, {"value_estimate": 0.0})

    def _build(turns: int = 1) -> BridgeAgentLoop:
        return BridgeAgentLoop(
            bridge=bridge,
            policy=policy,
            state_converter=state_converter,
            action_converter=action_converter,
            config=AgentLoopConfig(max_turns=turns, per_turn_timeout_s=0.5),
        )

    return _build


def _failed_ack(executed: list[dict[str, Any]], code: str = "EXECUTION_FAILED") -> AckMessage:
    return AckMessage(
        msg_id=1,
        success=False,
        timestamp=0.0,
        error_code=code,
        error_message="boom",
        executed=executed,
    )


class TestAgentLoopBlockedEndTurn:
    @pytest.mark.asyncio
    async def test_blocked_end_turn_is_counted_apart_from_invalid_actions(
        self, bridge: MagicMock, make_loop: Any
    ) -> None:
        message = "End turn blocked: ENDTURN_BLOCKING_RESEARCH (Choose a technology)"
        bridge.send_and_wait_with_retry.return_value = _failed_ack(
            [
                {
                    "action_type": "end_turn",
                    "success": False,
                    "error_code": protocol.END_TURN_BLOCKED,
                    "message": message,
                }
            ],
            code=protocol.END_TURN_BLOCKED,
        )
        loop = make_loop(turns=1)

        code = await loop.run()

        assert code == BridgeAgentLoop.EXIT_OK
        stats = loop.stats()
        assert stats["blocked_end_turns"] == 1
        assert stats["last_end_turn_blocker"] == message
        assert stats["invalid_actions"] == 0
        assert stats["turns_played"] == 1

    @pytest.mark.asyncio
    async def test_blocker_found_among_several_executed_entries(
        self, bridge: MagicMock, make_loop: Any
    ) -> None:
        bridge.send_and_wait_with_retry.return_value = _failed_ack(
            [
                {"action_type": "skip", "success": True},
                {
                    "action_type": "end_turn",
                    "success": False,
                    "error_code": "END_TURN_BLOCKED",
                    "message": "End turn blocked: ENDTURN_BLOCKING_CIVIC",
                },
            ]
        )
        loop = make_loop(turns=1)
        await loop.run()
        stats = loop.stats()
        assert stats["blocked_end_turns"] == 1
        assert stats["last_end_turn_blocker"] == "End turn blocked: ENDTURN_BLOCKING_CIVIC"
        assert stats["invalid_actions"] == 0

    @pytest.mark.asyncio
    async def test_other_failures_still_count_as_invalid(
        self, bridge: MagicMock, make_loop: Any
    ) -> None:
        bridge.send_and_wait_with_retry.return_value = _failed_ack(
            [{"action_type": "move", "success": False, "error_code": "INVALID_TARGET"}]
        )
        loop = make_loop(turns=1)
        await loop.run()
        stats = loop.stats()
        assert stats["invalid_actions"] == 1
        assert stats["blocked_end_turns"] == 0
        assert stats["last_end_turn_blocker"] is None

    @pytest.mark.asyncio
    async def test_blocker_counter_accumulates_per_turn(
        self, bridge: MagicMock, make_loop: Any
    ) -> None:
        bridge.send_and_wait_with_retry.return_value = _failed_ack(
            [
                {
                    "action_type": "end_turn",
                    "success": False,
                    "error_code": "END_TURN_BLOCKED",
                    "message": "End turn blocked: ENDTURN_BLOCKING_UNITS",
                }
            ]
        )
        loop = make_loop(turns=2)
        await loop.run()
        assert loop.stats()["blocked_end_turns"] == 2
        assert loop.stats()["invalid_actions"] == 0

    @pytest.mark.asyncio
    async def test_a_successful_ack_leaves_the_counters_alone(
        self, bridge: MagicMock, make_loop: Any
    ) -> None:
        loop = make_loop(turns=1)
        await loop.run()
        stats = loop.stats()
        assert stats["turns_played"] == 1
        assert stats["blocked_end_turns"] == 0
        assert stats["last_end_turn_blocker"] is None
