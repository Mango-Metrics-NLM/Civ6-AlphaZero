"""GM-B7a (P1): a production command may name its item by the game's type name."""

from __future__ import annotations

import pytest

from alpha_zero.bridge.protocol import ActionType, LuaCommand, ProtocolValidationError


def produce(item_id: object, **extra: object) -> dict[str, object]:
    return {"action_type": "city_produce", "city_id": 301, "item_id": item_id, **extra}


class TestCityProduceItemNames:
    def test_a_type_name_is_accepted_and_kept(self) -> None:
        command = LuaCommand.from_dict(produce("UNIT_WARRIOR"))
        assert command.action_type is ActionType.CITY_PRODUCE
        assert command.item_id == "UNIT_WARRIOR"

    def test_a_type_name_round_trips_with_its_item_type(self) -> None:
        raw = produce("BUILDING_GRANARY", item_type="building")
        command = LuaCommand.from_dict(raw)
        data = command.to_dict()
        assert data["item_id"] == "BUILDING_GRANARY"
        assert data["item_type"] == "building"
        assert LuaCommand.from_dict(data) == command

    def test_an_int_still_works_and_stays_an_int(self) -> None:
        command = LuaCommand.from_dict(produce(3))
        assert command.item_id == 3 and type(command.item_id) is int
        assert command.to_dict()["item_id"] == 3

    @pytest.mark.parametrize("bad", ["not a type name", "1BAD", "", "UNIT-WARRIOR", "A;B"])
    def test_a_string_that_is_not_a_type_name_is_still_rejected(self, bad: str) -> None:
        with pytest.raises(ProtocolValidationError):
            LuaCommand.from_dict(produce(bad))

    def test_control_research_names_are_still_accepted(self) -> None:
        raw = {"action_type": "research_tech", "item_id": "TECH_MINING"}
        assert LuaCommand.from_dict(raw).item_id == "TECH_MINING"

    @pytest.mark.parametrize("action", ["unit_move", "unit_build"])
    def test_control_other_actions_still_refuse_a_name(self, action: str) -> None:
        with pytest.raises(ProtocolValidationError):
            LuaCommand.from_dict({"action_type": action, "item_id": "UNIT_WARRIOR"})
