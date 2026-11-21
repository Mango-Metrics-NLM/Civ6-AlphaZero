"""Three findings of the automated review on the loop PR (GM-B7c-2).

* the command key left out the operands a diplomacy or citizen command
  carries in its metadata, so one refused diplomatic action suppressed every
  other for the turn;
* a production blocker reopened every city's choice, also for cities already
  building, so a deterministic policy could keep changing the busy city's
  queue while the idle one kept the blocker up;
* a fallback ``end_turn`` whose send raised stayed pending, and the next
  state's blocker was counted as its answer.
"""

from __future__ import annotations

import asyncio
from typing import Any

from alpha_zero.bridge.agent_loop import BridgeAgentLoop
from alpha_zero.bridge.protocol import GameStateData
from alpha_zero.bridge.turn_ledger import TurnLedger, command_key
from tests.fixtures.ledger_states import exchange, ok, refused
from tests.unit.test_bridge_loop_review import BLOCKER, Bridge, game_state


# ---------------------------------------------------------------------------
# The key carries the operands the executor reads from metadata
# ---------------------------------------------------------------------------
def diplomacy(sub_action: int, target: int, **extra: Any) -> dict[str, Any]:
    return {
        "action_type": "diplomacy",
        "metadata": {"sub_action": sub_action, "target_player_id": target, **extra},
    }


def citizen(lock: bool) -> dict[str, Any]:
    return {"action_type": "city_citizen", "city_id": 301, "item_id": 7, "metadata": {"lock": lock}}


class TestKeyOperands:
    def test_another_diplomatic_action_is_another_command(self) -> None:
        assert command_key(diplomacy(1, 2)) != command_key(diplomacy(3, 2))

    def test_another_target_player_is_another_command(self) -> None:
        assert command_key(diplomacy(1, 2)) != command_key(diplomacy(1, 4))

    def test_lock_and_unlock_are_different_commands(self) -> None:
        assert command_key(citizen(True)) != command_key(citizen(False))

    def test_a_descriptive_field_does_not_make_a_new_command(self) -> None:
        assert command_key(diplomacy(1, 2)) == command_key(diplomacy(1, 2, why="because"))

    def test_metadata_without_operands_is_the_same_as_none(self) -> None:
        move = {"action_type": "unit_move", "unit_id": 5, "target": {"x": 1, "y": 2}}
        assert command_key(move) == command_key({**move, "metadata": {"note": "x"}})
        assert command_key(move) == command_key({**move, "metadata": None})

    def test_one_refused_diplomatic_action_does_not_suppress_the_others(self) -> None:
        ledger = TurnLedger()
        on = two_cities()
        ledger.observe(on)
        exchange(ledger, diplomacy(1, 2), on, refused())
        assert ledger.is_suppressed(diplomacy(1, 2))
        assert not ledger.is_suppressed(diplomacy(3, 2))
        assert not ledger.is_suppressed(diplomacy(1, 4))


# ---------------------------------------------------------------------------
# A production blocker reopens only the idle city
# ---------------------------------------------------------------------------
def two_cities(first: str | None = None, second: str | None = None, blocking: str | None = None):
    data: dict[str, Any] = {
        "turn_id": 12,
        "player_id": 0,
        "map": {"width": 4, "height": 4},
        "units": [],
        "cities": [
            {"id": 301, "owner": 0, "x": 0, "y": 0, "current_production": first},
            {"id": 302, "owner": 0, "x": 3, "y": 3, "current_production": second},
        ],
        "players": [{"id": 0, "current_research": "TECH_A", "current_civic": "CIVIC_A"}],
    }
    if blocking is not None:
        data["end_turn_blocking"] = blocking
    return GameStateData.from_dict(data)


def produce(city_id: int, item: str = "UNIT_WARRIOR") -> dict[str, Any]:
    return {"action_type": "city_produce", "city_id": city_id, "item_id": item, "item_type": "unit"}


PRODUCTION_BLOCKER = "ENDTURN_BLOCKING_PRODUCTION"


class TestProductionBlockerAndTwoCities:
    def _first_city_chosen(self) -> TurnLedger:
        ledger = TurnLedger()
        before = two_cities(blocking=PRODUCTION_BLOCKER)
        ledger.observe(before)
        exchange(ledger, produce(301), before, ok())
        # The first city now builds; the second is still idle and blocks.
        ledger.observe(two_cities(first="UNIT_WARRIOR", blocking=PRODUCTION_BLOCKER))
        return ledger

    def test_the_busy_city_keeps_its_choice(self) -> None:
        ledger = self._first_city_chosen()
        assert ledger.is_suppressed(produce(301, "UNIT_SETTLER"))

    def test_the_idle_city_can_be_chosen(self) -> None:
        ledger = self._first_city_chosen()
        assert not ledger.is_suppressed(produce(302))

    def test_a_blocker_with_no_idle_city_named_reopens_all(self) -> None:
        """The state shows both building yet the game asks: it cannot be told which."""
        ledger = self._first_city_chosen()
        ledger.observe(two_cities("UNIT_WARRIOR", "UNIT_WARRIOR", blocking=PRODUCTION_BLOCKER))
        assert not ledger.is_suppressed(produce(301, "UNIT_SETTLER"))

    def test_without_a_blocker_a_building_city_stays_held(self) -> None:
        ledger = self._first_city_chosen()
        ledger.observe(two_cities("UNIT_WARRIOR", "UNIT_WARRIOR"))
        assert ledger.is_suppressed(produce(301, "UNIT_SETTLER"))


# ---------------------------------------------------------------------------
# A fallback end_turn that never went out is not answered by the next state
# ---------------------------------------------------------------------------
class FailingSendBridge(Bridge):
    async def send_command(self, command: Any) -> bool:
        self._record(command)
        raise ConnectionError("the transport is down")


class RaisingConv:
    """A state converter that fails, so every step takes the fallback path."""

    def convert(self, state: Any) -> dict[str, Any]:
        raise RuntimeError("cannot convert")


class TestFallbackSendThatRaises:
    def test_three_failed_fallback_sends_are_not_a_stall(self) -> None:
        from alpha_zero.bridge.agent_loop import AgentLoopConfig

        states = [game_state(blocking=BLOCKER) for _ in range(4)] + [game_state(over=True)]
        bridge = FailingSendBridge(states)
        loop = BridgeAgentLoop(
            bridge=bridge,  # type: ignore[arg-type]
            policy=object(),  # type: ignore[arg-type]
            state_converter=RaisingConv(),  # type: ignore[arg-type]
            action_converter=object(),  # type: ignore[arg-type]
            config=AgentLoopConfig(per_turn_timeout_s=0.5, max_turns=0),
        )
        code = asyncio.run(loop.run())
        stats = loop.stats()
        assert stats["fallback_end_turns"] == 4, "premise: every step took the fallback"
        assert len(bridge.sent) == 4, "premise: each send was attempted"
        assert code == BridgeAgentLoop.EXIT_OK
        assert stats["stalled_turns"] == 0
        assert stats["blocked_end_turns"] == 0

    def test_control_fallbacks_that_go_out_and_are_blocked_do_stall(self) -> None:
        from alpha_zero.bridge.agent_loop import AgentLoopConfig

        states = [game_state(blocking=BLOCKER) for _ in range(5)] + [game_state(over=True)]
        bridge = Bridge(states)
        loop = BridgeAgentLoop(
            bridge=bridge,  # type: ignore[arg-type]
            policy=object(),  # type: ignore[arg-type]
            state_converter=RaisingConv(),  # type: ignore[arg-type]
            action_converter=object(),  # type: ignore[arg-type]
            config=AgentLoopConfig(per_turn_timeout_s=0.5, max_turns=0),
        )
        assert asyncio.run(loop.run()) == BridgeAgentLoop.EXIT_STALLED
