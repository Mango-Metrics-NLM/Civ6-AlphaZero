"""A command policy's answer is checked against what the game offered (review of PR #163).

``choose`` checked the answer against the turn's ledger only, so a policy that
answered ``unit_skip`` for a unit nobody offered had it sent unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from alpha_zero.bridge.offers import is_offered
from alpha_zero.bridge.protocol import LegalAction, LuaCommand
from tests.fixtures.bridge_loop_fakes import END, MOVE_1, RESEARCH, SKIP_1, game_state, play

OFFERS = [
    LegalAction("end_turn"),
    LegalAction("skip", unit_id=1),
    LegalAction("move", unit_id=1, targets=[(0, 1), (1, 1)]),
    LegalAction("attack", unit_id=1, targets=[(2, 2)]),
    LegalAction("ranged_attack", unit_id=4, targets=[(5, 5)]),
    LegalAction("found_city", unit_id=3),
    LegalAction("produce", city_id=7, items=["UNIT_WARRIOR", "BUILDING_GRANARY"]),
    LegalAction("research_tech", items=[2], item_names=["TECH_WRITING"]),
]


def _cmd(action: str, **fields: Any) -> dict[str, Any]:
    return {"action_type": action, **fields}


class TestIsOffered:
    @pytest.mark.parametrize(
        "command",
        [
            _cmd("end_turn"),
            _cmd("unit_skip", unit_id=1),
            _cmd("unit_move", unit_id=1, target={"x": 1, "y": 1}),
            _cmd("unit_attack", unit_id=1, target={"x": 2, "y": 2}),
            _cmd("unit_ranged_attack", unit_id=4, target={"x": 5, "y": 5}),
            _cmd("unit_found_city", unit_id=3),
            _cmd("city_produce", city_id=7, item_id="UNIT_WARRIOR", item_type="unit"),
            _cmd("research_tech", item_id="TECH_WRITING"),
            _cmd("research_tech", item_id=2),
        ],
    )
    def test_an_offered_command_is_offered(self, command: dict[str, Any]) -> None:
        assert is_offered(command, OFFERS)
        assert is_offered(LuaCommand.from_dict(command), OFFERS)

    @pytest.mark.parametrize(
        "command",
        [
            _cmd("unit_skip", unit_id=999),
            _cmd("unit_move", unit_id=1, target={"x": 9, "y": 9}),
            _cmd("unit_move", unit_id=4, target={"x": 0, "y": 1}),
            _cmd("unit_attack", unit_id=1, target={"x": 0, "y": 1}),  # a move tile, not an attack
            _cmd("unit_move", unit_id=1, target={"x": 2, "y": 2}),  # an attack tile, not a move
            _cmd("unit_found_city", unit_id=1),
            _cmd("city_produce", city_id=8, item_id="UNIT_WARRIOR"),
            _cmd("city_produce", city_id=7, item_id="UNIT_SETTLER"),
            _cmd("research_tech", item_id="TECH_MINING"),
            _cmd("research_civic", item_id="TECH_WRITING"),
            _cmd("unit_fortify", unit_id=1),
            _cmd(""),
            {},
        ],
    )
    def test_anything_else_is_not(self, command: dict[str, Any]) -> None:
        assert not is_offered(command, OFFERS)

    def test_nothing_is_offered_by_no_offers_or_an_unreadable_list(self) -> None:
        assert not is_offered(_cmd("end_turn"), [])
        assert not is_offered(_cmd("end_turn"), None)
        assert not is_offered(_cmd("end_turn"), [object()])
        assert not is_offered(None, OFFERS)


class _Answers:
    """A command policy that answers with one fixed command."""

    def __init__(self, command: dict[str, Any]) -> None:
        self.command = command

    async def select_command(
        self, converted_state: Any, *, allowed: Any = None, decision_seed: Any = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.command, {"fallback": None}


class TestTheLoopDoesNotSendAnUnofferedAnswer:
    def test_an_unoffered_skip_is_not_sent(self) -> None:
        states = [game_state(5), game_state(over=True)]
        loop, bridge, _, _ = play(states, _Answers(_cmd("unit_skip", unit_id=999)), offered=(0,))
        assert bridge.sent == [END]
        stats = loop.stats()
        assert stats["search_fallbacks"] == 1
        assert stats["search_failures"] == 0

    def test_a_move_to_a_tile_nobody_offered_is_not_sent(self) -> None:
        states = [game_state(5), game_state(over=True)]
        answer = _cmd("unit_move", unit_id=1, target={"x": 7, "y": 7})
        _, bridge, _, _ = play(states, _Answers(answer))
        assert bridge.sent == [END]

    @pytest.mark.parametrize("command", [SKIP_1, MOVE_1, RESEARCH])
    def test_control_an_offered_answer_is_sent(self, command: dict[str, Any]) -> None:
        states = [game_state(5), game_state(over=True)]
        _, bridge, _, _ = play(states, _Answers(command))
        assert bridge.sent == [command]
