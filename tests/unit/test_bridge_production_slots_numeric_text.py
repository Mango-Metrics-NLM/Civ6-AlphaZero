"""A production row offered as text ("5") is the row 5, however the offer was built.

``LegalAction.from_dict`` turns numeric strings into integers; a ``LegalAction``
constructed directly kept ``"5"``, and the command then carried the string,
which the mod reads as a type name and refuses.
"""

from __future__ import annotations

import pytest

from alpha_zero.bridge import production_slots as slots
from alpha_zero.bridge.protocol import LegalAction


def test_numeric_text_and_number_are_the_same_item() -> None:
    assert slots.offered(["5", 3, "UNIT_WARRIOR"]) == slots.offered([5, "3", "UNIT_WARRIOR"])
    assert slots.offered(["5", 5]) == {"unit": (5,)}


def test_item_at_gives_the_number_not_the_text() -> None:
    item = slots.item_at("unit", 0, ["5"])
    assert item == 5 and isinstance(item, int)


def test_slot_of_finds_either_spelling() -> None:
    assert slots.slot_of("5", [3, 5]) == ("unit", 1)
    assert slots.slot_of(5, ["3", "5"]) == ("unit", 1)


def test_the_kind_of_a_text_row_is_kept() -> None:
    kinds = {"5": "building", "7": "unit"}
    assert slots.offered(["5", "7"], kinds) == {"building": (5,), "unit": (7,)}
    assert slots.slot_of(5, ["5", "7"], kinds) == ("building", 0)


def test_names_are_untouched() -> None:
    assert slots.offered(["UNIT_WARRIOR", " 5x", "5.0", "5 "]) == {
        "unit": (" 5x", "5 ", "5.0", "UNIT_WARRIOR")
    }


def test_the_rule_is_the_one_from_dict_uses() -> None:
    texts = [
        "5",
        "-5",
        "05",
        "5.0",
        " 5",
        "UNIT_WARRIOR",
        chr(0x665),
    ]  # the last is an Arabic-Indic five
    parsed = LegalAction.from_dict({"action_type": "produce", "city_id": 1, "items": texts})
    assert slots.offered(texts) == slots.offered(parsed.items)


@pytest.mark.parametrize("direct", [True, False])
def test_both_ways_of_building_an_offer_give_the_same_slots(direct: bool) -> None:
    if direct:
        action = LegalAction(action_type="produce", city_id=301, items=["5", "2"])
    else:
        action = LegalAction.from_dict(
            {"action_type": "produce", "city_id": 301, "items": ["5", "2"]}
        )
    assert slots.offered(action.items, action.item_types) == {"unit": (2, 5)}
    assert slots.item_at("unit", 1, action.items, action.item_types) == 5
