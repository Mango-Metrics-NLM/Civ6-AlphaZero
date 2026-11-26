"""States, commands and acknowledgements for the turn-ledger tests (GM-B7c-2).

Real ``GameStateData`` built with ``from_dict``; real ``AckMessage`` objects.
"""

from __future__ import annotations

from typing import Any

from alpha_zero.bridge.protocol import AckMessage, GameStateData
from alpha_zero.bridge.turn_ledger import TurnLedger

BLOCKER = "ENDTURN_BLOCKING_RESEARCH"
END_TURN = {"action_type": "end_turn"}
MOVE_101 = {"action_type": "unit_move", "unit_id": 101, "target": {"x": 0, "y": 1}}
MOVE_102 = {"action_type": "unit_move", "unit_id": 102, "target": {"x": 0, "y": 1}}
PRODUCE = {"action_type": "city_produce", "city_id": 301, "item_id": "UNIT_WARRIOR"}
UPGRADE = {"action_type": "unit_upgrade", "unit_id": 101}  # a type state_diff does not judge


def state(
    turn: int = 12,
    *,
    units: dict[int, tuple[int, int, float]] | None = None,
    blocking: str | None = None,
    research: str = "TECH_A",
    producing: str = "UNIT_WARRIOR",
) -> GameStateData:
    """An own-player state; ``units`` maps id -> (x, y, moves), in the given order."""
    units = {101: (0, 0, 2.0), 102: (1, 0, 2.0)} if units is None else units
    data: dict[str, Any] = {
        "turn_id": turn,
        "player_id": 0,
        "map": {"width": 4, "height": 4},
        "units": [
            {"id": i, "owner": 0, "x": x, "y": y, "moves_remaining": m}
            for i, (x, y, m) in units.items()
        ],
        "cities": [{"id": 301, "owner": 0, "x": 0, "y": 0, "current_production": producing}],
        "players": [{"id": 0, "current_research": research, "current_civic": "CIVIC_A"}],
    }
    if blocking is not None:
        data["end_turn_blocking"] = blocking
    return GameStateData.from_dict(data)


def ok(msg_id: int = 1) -> AckMessage:
    return AckMessage(msg_id=msg_id, success=True)


def refused(msg_id: int = 1) -> AckMessage:
    executed = [{"success": False, "error_code": "OPERATION_REFUSED", "message": "no"}]
    return AckMessage(msg_id=msg_id, success=False, error_code="COMMAND_FAILED", executed=executed)


def blocked(msg_id: int = 1) -> AckMessage:
    executed = [{"success": False, "error_code": "END_TURN_BLOCKED", "message": BLOCKER}]
    return AckMessage(msg_id=msg_id, success=False, error_code="COMMAND_FAILED", executed=executed)


def exchange(ledger: TurnLedger, command: dict[str, Any], on: GameStateData, ack: Any) -> Any:
    """The loop's three calls for one command; returns the outcome."""
    ledger.sent(command, on)
    return ledger.answered(command, ack)


def no_effect_once(ledger: TurnLedger, command: dict[str, Any], turn: int = 12) -> None:
    """Send ``command`` on a state, get success, and see an identical state afterwards."""
    before = state(turn)
    ledger.observe(before)
    exchange(ledger, command, before, ok())
    ledger.observe(state(turn))


def current_research(value: GameStateData) -> str:
    """The research name of an own-player state, so a blocked state can be rebuilt equal."""
    (player,) = [p for p in value.players if p.id == value.player_id]
    return str(player.current_research)
