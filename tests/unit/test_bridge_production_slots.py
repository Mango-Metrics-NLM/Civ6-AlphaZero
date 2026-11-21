"""GM-B7b (P1-P3, E10): production choices by position, commands by name.

The old mapping hashed a type name into 200 buckets with Python's per-process salted ``hash``:
two processes disagreed, and with a smaller production space most items fell off the end.
``alpha_zero.bridge.production_slots`` does not exist yet, so it is imported inside each test and
each test fails on its own. Tests marked CONTROL pass on the tree before the slice.
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter, InvalidActionError
from alpha_zero.bridge.protocol import ActionType, GameStateData, LegalAction, LuaCommand
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.core.civ6_actions import (
    ActionCategory,
    ActionSpaceConfig,
    CityActionType,
    HierarchicalAction,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONVERTER_LINES_BEFORE = 1369

ITEMS = ["UNIT_WARRIOR", "BUILDING_GRANARY", "UNIT_SETTLER"]
KINDS = {"UNIT_WARRIOR": "unit", "BUILDING_GRANARY": "building", "UNIT_SETTLER": "unit"}
EXPECTED_SLOT = {
    "UNIT_SETTLER": ("unit", 0),
    "UNIT_WARRIOR": ("unit", 1),
    "BUILDING_GRANARY": ("building", 0),
}


def slots() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.production_slots")


# ---------------------------------------------------------------------------
# P1: the module
# ---------------------------------------------------------------------------
class TestOffered:
    def test_groups_by_kind_sorted_and_deduplicated(self) -> None:
        """Fails on the current tree: the module does not exist."""
        items: list[Any] = ["UNIT_WARRIOR", "BUILDING_GRANARY", "UNIT_SETTLER", "UNIT_WARRIOR", 7]
        types = {**KINDS, "7": "building"}
        got = slots().offered([*items, 3], types)
        assert {k: v for k, v in got.items() if v} == {
            "unit": (3, "UNIT_SETTLER", "UNIT_WARRIOR"),
            "building": (7, "BUILDING_GRANARY"),
        }
        assert all(isinstance(v, tuple) for v in got.values())

    def test_the_default_kind_is_for_items_the_map_does_not_name(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        got = mod.offered(["B", "A"], {})
        assert {k: v for k, v in got.items() if v} == {"unit": ("A", "B")}
        got = mod.offered(["B", "A"], {}, default_type="building")
        assert {k: v for k, v in got.items() if v} == {"building": ("A", "B")}

    def test_all_four_kinds_are_kept_apart(self) -> None:
        """Fails on the current tree: the module does not exist."""
        types = {"U": "unit", "B": "building", "W": "wonder", "D": "district"}
        got = slots().offered(["D", "W", "B", "U"], types)
        assert {k: v for k, v in got.items() if v} == {
            "unit": ("U",),
            "building": ("B",),
            "wonder": ("W",),
            "district": ("D",),
        }

    def test_nothing_offered_gives_nothing(self) -> None:
        """Fails on the current tree: the module does not exist."""
        assert not any(slots().offered([], {}).values())


class TestSlotOfAndItemAt:
    def test_slots_are_positions_in_the_sorted_group(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        for name, slot in EXPECTED_SLOT.items():
            assert mod.slot_of(name, ITEMS, KINDS) == slot

    def test_item_at_is_the_inverse_for_every_offered_item(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        mixed: list[Any] = [*ITEMS, 4, 2]
        types = {**KINDS, "4": "building"}
        for item in mixed:
            kind, position = mod.slot_of(item, mixed, types)
            assert mod.item_at(kind, position, mixed, types) == item

    def test_item_at_beyond_the_group_or_of_a_kind_not_offered_is_none(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        assert mod.item_at("unit", 2, ITEMS, KINDS) is None
        assert mod.item_at("building", 1, ITEMS, KINDS) is None
        assert mod.item_at("wonder", 0, ITEMS, KINDS) is None
        assert mod.item_at("district", 0, ITEMS, KINDS) is None

    def test_slot_of_an_item_that_was_not_offered_is_none(self) -> None:
        """Fails on the current tree: the module does not exist."""
        assert slots().slot_of("UNIT_ARCHER", ITEMS, KINDS) is None

    def test_integers_sort_before_strings_in_a_group(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        items: list[Any] = ["UNIT_B", 10, "UNIT_A", 2]
        assert mod.offered(items, {})["unit"] == (2, 10, "UNIT_A", "UNIT_B")
        assert mod.slot_of(10, items, {}) == ("unit", 1)

    def test_the_result_does_not_depend_on_the_order_of_items(self) -> None:
        """Fails on the current tree: the module does not exist."""
        mod = slots()
        items: list[Any] = [*ITEMS, "BUILDING_MONUMENT", 5]
        types = {**KINDS, "BUILDING_MONUMENT": "building"}
        reference = {k: v for k, v in mod.offered(items, types).items() if v}
        reference_slots = {str(i): mod.slot_of(i, items, types) for i in items}
        for order in itertools.islice(itertools.permutations(items), 0, None, 37):
            shuffled = list(order)
            assert {k: v for k, v in mod.offered(shuffled, types).items() if v} == reference
            assert {str(i): mod.slot_of(i, shuffled, types) for i in items} == reference_slots

    def test_the_slot_of_a_name_is_the_same_in_two_processes(self) -> None:
        """E10. Fails on the current tree: the module does not exist (the old mapping hashed the
        name with Python's salted ``hash``, so two processes disagreed)."""
        code = (
            "import json, sys\n"
            "from alpha_zero.bridge import production_slots as p\n"
            "names = ['UNIT_WARRIOR','UNIT_SETTLER','BUILDING_GRANARY','BUILDING_MONUMENT',"
            "'UNIT_ARCHER','WONDER_PYRAMIDS']\n"
            "kinds = {n: ('building' if n.startswith('BUILDING') else 'wonder' "
            "if n.startswith('WONDER') else 'unit') for n in names}\n"
            "print(json.dumps({n: p.slot_of(n, names, kinds) for n in names}))\n"
            "print(json.dumps(sorted(m for m in ('torch', 'numpy') if m in sys.modules)))\n"
        )
        outputs = []
        for seed in ("1", "2", "31337"):
            env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(REPO_ROOT)}
            done = subprocess.run(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            assert done.returncode == 0, done.stderr
            outputs.append(done.stdout.strip().splitlines())
        assert outputs[0] == outputs[1] == outputs[2]
        by_name = json.loads(outputs[0][0])
        assert by_name["UNIT_ARCHER"] == ["unit", 0]
        assert by_name["UNIT_WARRIOR"] == ["unit", 2]
        assert by_name["BUILDING_GRANARY"] == ["building", 0]
        assert json.loads(outputs[0][1]) == [], "the module pulled in torch or numpy"


# ---------------------------------------------------------------------------
# P2: through ActionConverter
# ---------------------------------------------------------------------------
def make_converter(production_items: int = 20) -> ActionConverter:
    config = ActionSpaceConfig(
        board_size=8, max_units=4, max_cities=2, num_production_items=production_items
    )
    return ActionConverter(board_size=8, action_space_config=config, research_registry=False)


STATE: dict[str, Any] = {"current_player": 0, "units": [], "cities": [{"_id": 301, "player": 0}]}


def produce_action(items: list[Any], types: dict[str, str] | None = None) -> LegalAction:
    return LegalAction(action_type="produce", city_id=301, items=items, item_types=types or {})


def legal(*items: LegalAction) -> list[LegalAction]:
    return [LegalAction(action_type="end_turn"), *items]


def production_indices(
    conv: ActionConverter, mask: np.ndarray
) -> list[tuple[int, HierarchicalAction]]:
    """The marked indices of the city category, decoded."""
    offset = conv.action_space.category_offsets[ActionCategory.CITY_ACTION]
    size = conv.action_space.category_sizes[ActionCategory.CITY_ACTION]
    return [
        (int(i) + offset, conv.action_space.decode(int(i) + offset))
        for i in np.nonzero(mask[offset : offset + size])[0]
    ]


ACTION_OF_KIND = {
    "unit": CityActionType.PRODUCE_UNIT.value,
    "building": CityActionType.PRODUCE_BUILDING.value,
}
KIND_OF_TYPE = {
    CityActionType.PRODUCE_UNIT.value: "unit",
    CityActionType.PRODUCE_BUILDING.value: "building",
    CityActionType.PRODUCE_WONDER.value: "wonder",
    CityActionType.PRODUCE_DISTRICT.value: "district",
}


def names_by_index(
    conv: ActionConverter, indices: list[tuple[int, HierarchicalAction]], legal_actions: Any
) -> dict[int, LuaCommand]:
    return {i: conv.convert(i, STATE, legal_actions) for i, _ in indices}


class TestConverterByName:
    def test_each_offered_item_gets_one_index_at_its_slot(self) -> None:
        """E10. Fails on the current tree: the slot is a salted hash, not the position."""
        conv = make_converter()
        actions = legal(produce_action(ITEMS, KINDS))
        marked = production_indices(conv, conv.generate_action_mask(actions, STATE))
        assert len(marked) == 3
        found = {(KIND_OF_TYPE[d.action_type], d.target) for _, d in marked}
        assert found == {("unit", 0), ("unit", 1), ("building", 0)}
        assert all(d.selector == 0 for _, d in marked)

    def test_the_command_carries_the_name_and_its_kind(self) -> None:
        """E10. Fails on the current tree: the command carries an integer."""
        conv = make_converter()
        actions = legal(produce_action(ITEMS, KINDS))
        mask = conv.generate_action_mask(actions, STATE)
        commands = names_by_index(conv, production_indices(conv, mask), actions)
        assert len(commands) == 3
        for command in commands.values():
            assert command.action_type is ActionType.CITY_PRODUCE
            assert command.city_id == 301
            assert isinstance(command.item_id, str)
            assert command.item_type == KINDS[command.item_id]

    def test_the_round_trip_names_every_offered_item_exactly_once(self) -> None:
        """E10. Fails on the current tree: integers come back, and most were dropped."""
        conv = make_converter()
        actions = legal(produce_action(ITEMS, KINDS))
        mask = conv.generate_action_mask(actions, STATE)
        commands = names_by_index(conv, production_indices(conv, mask), actions)
        assert sorted(str(c.item_id) for c in commands.values()) == sorted(ITEMS)

    def test_every_item_within_the_bound_is_choosable_and_the_rest_are_not(self) -> None:
        """E10. 25 units into 20 slots: the first 20 by sorted name are marked. Fails on the
        current tree: a salted hash into 200 buckets lands most of them at or above 20."""
        names = [f"UNIT_{i:02d}" for i in range(25)]
        kinds = dict.fromkeys(names, "unit")
        conv = make_converter(20)
        actions = legal(produce_action(list(reversed(names)), kinds))
        marked = production_indices(conv, conv.generate_action_mask(actions, STATE))
        assert sorted(d.target for _, d in marked) == list(range(20))
        commands = names_by_index(conv, marked, actions)
        assert sorted(str(c.item_id) for c in commands.values()) == names[:20]

    def test_a_larger_space_marks_all_25(self) -> None:
        """E10. Fails on the current tree: with 200 buckets the hashes collide and drop items
        (25 names in 200 buckets collide in about 78% of draws)."""
        names = [f"UNIT_{i:02d}" for i in range(25)]
        kinds = dict.fromkeys(names, "unit")
        conv = make_converter(200)
        actions = legal(produce_action(names, kinds))
        marked = production_indices(conv, conv.generate_action_mask(actions, STATE))
        assert sorted(d.target for _, d in marked) == list(range(25))

    def test_an_item_offered_in_a_different_order_gets_the_same_index(self) -> None:
        """E10. Fails on the current tree: the indices are salted hashes, not the positions."""
        conv = make_converter()
        first = conv.generate_action_mask(legal(produce_action(ITEMS, KINDS)), STATE)
        second = conv.generate_action_mask(legal(produce_action(ITEMS[::-1], KINDS)), STATE)
        assert np.array_equal(first, second)
        space = conv.action_space
        end = space.category_offsets[ActionCategory.END_TURN]
        expected = {end}
        for kind, position in EXPECTED_SLOT.values():
            expected.add(
                space.encode(
                    HierarchicalAction(
                        category=ActionCategory.CITY_ACTION,
                        selector=0,
                        action_type=ACTION_OF_KIND[kind],
                        target=position,
                    )
                )
            )
        assert {int(i) for i in np.nonzero(first)[0]} == expected

    @pytest.mark.parametrize(
        ("kind", "slot"),
        [("unit", 5), ("unit", 2), ("building", 1), ("wonder", 0), ("district", 0)],
    )
    def test_a_slot_nothing_was_offered_at_is_refused(self, kind: str, slot: int) -> None:
        """CONTROL (passes on the current tree, by the legality check): an index for a
        production slot with nothing offered raises ``InvalidActionError``."""
        conv = make_converter()
        action_type = {
            "unit": CityActionType.PRODUCE_UNIT,
            "building": CityActionType.PRODUCE_BUILDING,
            "wonder": CityActionType.PRODUCE_WONDER,
            "district": CityActionType.PRODUCE_DISTRICT,
        }[kind]
        index = conv.action_space.encode(
            HierarchicalAction(
                category=ActionCategory.CITY_ACTION,
                selector=0,
                action_type=action_type.value,
                target=slot,
            )
        )
        actions = legal(produce_action(ITEMS, KINDS))
        mask = conv.generate_action_mask(actions, STATE)
        assert mask[index] == 0.0
        with pytest.raises(InvalidActionError):
            conv.convert(index, STATE, actions)

    def test_the_marked_slot_is_legal_and_its_neighbour_is_not(self) -> None:
        """E10. Fails on the current tree: slot 1 of the units is not what the hash says."""
        conv = make_converter()
        actions = legal(produce_action(ITEMS, KINDS))
        marked = production_indices(conv, conv.generate_action_mask(actions, STATE))
        units = sorted(d.target for _, d in marked if KIND_OF_TYPE[d.action_type] == "unit")
        assert units == [0, 1]
        beyond = conv.action_space.encode(
            HierarchicalAction(
                category=ActionCategory.CITY_ACTION,
                selector=0,
                action_type=CityActionType.PRODUCE_UNIT.value,
                target=2,
            )
        )
        with pytest.raises(InvalidActionError):
            conv.convert(beyond, STATE, actions)

    def test_integer_items_still_work_and_the_command_carries_the_integer(self) -> None:
        """CONTROL: passes on the current tree (identity mapping); must still hold."""
        conv = make_converter()
        actions = legal(produce_action([5, 6, 7], {"5": "unit", "6": "unit", "7": "unit"}))
        mask = conv.generate_action_mask(actions, STATE)
        marked = production_indices(conv, mask)
        assert len(marked) == 3
        commands = names_by_index(conv, marked, actions)
        assert sorted(c.item_id for c in commands.values()) == [5, 6, 7]
        assert all(type(c.item_id) is int for c in commands.values())
        assert all(c.item_type == "unit" for c in commands.values())

    def test_integer_items_of_different_kinds_keep_their_kind(self) -> None:
        """CONTROL: passes on the current tree."""
        conv = make_converter()
        actions = legal(produce_action([5, 6], {"5": "unit", "6": "building"}))
        commands = names_by_index(
            conv, production_indices(conv, conv.generate_action_mask(actions, STATE)), actions
        )
        assert sorted((c.item_type, c.item_id) for c in commands.values()) == [
            ("building", 6),
            ("unit", 5),
        ]

    def test_converter_module_did_not_grow(self) -> None:
        """CONTROL: passes on the current tree (1,369 lines); the slice may not lengthen it."""
        path = REPO_ROOT / "alpha_zero" / "bridge" / "action_converter.py"
        assert len(path.read_text(encoding="utf-8").splitlines()) <= CONVERTER_LINES_BEFORE


# ---------------------------------------------------------------------------
# P3: end to end against the mod
# ---------------------------------------------------------------------------
class TestEndToEnd:
    @pytest.mark.parametrize("name", ["UNIT_WARRIOR", "BUILDING_GRANARY"])
    def test_a_choice_from_the_exported_state_starts_that_production(self, name: str) -> None:
        """E4 and E10 together. Fails on the current tree: the exporter lists nothing, so the
        mask has no production index."""
        from tests.fixtures import lua_harness as h

        runtime = h.load_mod()
        parsed = GameStateData.from_dict(h.export_state(runtime))
        converted = StateConverter(research_registry=False).convert(parsed)
        conv = ActionConverter(research_registry=False)
        mask = conv.generate_action_mask(parsed.legal_actions, converted)
        marked = production_indices(conv, mask)
        assert len(marked) == 2, "the warrior and the granary should both be choosable"
        by_name = {}
        for index, _ in marked:
            command = conv.convert(index, converted, parsed.legal_actions)
            by_name[command.item_id] = command
        assert set(by_name) == {"UNIT_WARRIOR", "BUILDING_GRANARY"}
        command = by_name[name]
        assert runtime.eval("Civ6Stub.city(301).building") is None, "already building; vacuous"
        assert h.execute(runtime, command.to_dict())[0] is True
        runtime.execute("Civ6Stub.applyRequests()")
        assert runtime.eval("Civ6Stub.city(301).building") == name
        assert int(runtime.eval("#Civ6Stub.dropped")) == 0


class TestReviewFindings:
    """Review of GM-B7b: a slot nothing was offered at, and items beyond the slots."""

    def _unit_slot_index(self, conv: ActionConverter, slot: int) -> int:
        return conv.action_space.encode(
            HierarchicalAction(
                category=ActionCategory.CITY_ACTION,
                selector=0,
                action_type=CityActionType.PRODUCE_UNIT.value,
                target=slot,
            )
        )

    def test_a_slot_with_no_offered_item_is_not_sent_as_a_number(self) -> None:
        """Without the offered items a slot names nothing. Sending the bare slot would be read
        by the mod as a row index: another item."""
        from alpha_zero.bridge.action_converter import InvalidActionError

        conv = make_converter(20)
        index = self._unit_slot_index(conv, 0)
        with pytest.raises(InvalidActionError):
            conv.convert(index, STATE, None)
        with pytest.raises(InvalidActionError):
            conv.convert(index, STATE, [], validate=False)

    def test_items_beyond_the_slots_are_counted(self) -> None:
        names = [f"UNIT_{i:02d}" for i in range(25)]
        conv = make_converter(20)
        conv.generate_action_mask(legal(produce_action(names, dict.fromkeys(names, "unit"))), STATE)
        assert conv.production_stats["items_beyond_slots"] == 5

    def test_nothing_is_counted_when_everything_fits(self) -> None:
        conv = make_converter(20)
        conv.generate_action_mask(legal(produce_action(ITEMS, KINDS)), STATE)
        assert conv.production_stats["items_beyond_slots"] == 0
