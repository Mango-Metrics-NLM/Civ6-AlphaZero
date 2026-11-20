"""GM-B3: research wiring, end to end against the mod, through the Lua harness.

Only these tests need lupa. Shared helpers live in ``tests/fixtures/research_wiring.py``.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from alpha_zero.bridge.action_converter import ActionConverter
from alpha_zero.bridge.protocol import (
    ActionType,
    GameStateData,
    LuaCommand,
)
from alpha_zero.bridge.state_converter import StateConverter
from alpha_zero.rules import parse_ruleset
from tests.fixtures import lua_harness as h
from tests.fixtures.research_wiring import (
    CIVIC_RANGE,
    CIVIC_TYPES,
    END_TURN,
    RULESET,
    TECH_IDS,
    TECH_RANGE,
    encode,
    flat,
    registry_from,
    research_bits,
)


@pytest.fixture
def tr() -> ModuleType:
    return importlib.import_module("alpha_zero.bridge.tech_registry")


@pytest.fixture
def reg(tr: ModuleType) -> Any:
    techs = "\n".join(f"  - id: {i}\n    era: 1\n    cost: 10" for i in TECH_IDS)
    return tr.from_ruleset(parse_ruleset(RULESET.format(techs=techs)), CIVIC_TYPES)


@pytest.fixture
def reg_gameinfo(tr: ModuleType) -> Any:
    """Ids as the GameInfo importer writes them: ``tech_`` prefixed."""
    return registry_from(tr, [f"tech_{i}" for i in TECH_IDS])


@pytest.fixture
def reg_no_civics(tr: ModuleType) -> Any:
    return registry_from(tr, TECH_IDS, ())


# ---------------------------------------------------------------------------
# End to end against the mod, through the Lua harness
# ---------------------------------------------------------------------------
BLOCKERS = {
    "tech": (
        "{ id = 5, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_RESEARCH,"
        " message = 'Choose a technology', typeName = 'NOTIFICATION_CHOOSE_TECH',"
        " dismissed = false }"
    ),
    "civic": (
        "{ id = 6, blocking = EndTurnBlockingTypes.ENDTURN_BLOCKING_CIVIC,"
        " message = 'Choose a civic', typeName = 'NOTIFICATION_CHOOSE_CIVIC',"
        " dismissed = false }"
    ),
}
BLOCKER_NAME = {"tech": "ENDTURN_BLOCKING_RESEARCH", "civic": "ENDTURN_BLOCKING_CIVIC"}
CATEGORY = {"tech": TECH_RANGE, "civic": CIVIC_RANGE}
ACTION_TYPE = {"tech": ActionType.RESEARCH_TECH, "civic": ActionType.RESEARCH_CIVIC}
CHOICE = {"tech": "TECH_WRITING", "civic": "CIVIC_FOREIGN_TRADE"}
MESSAGE = {
    "tech": "Research requested: TECH_WRITING",
    "civic": "Civic requested: CIVIC_FOREIGN_TRADE",
}
# What the choice is, with a registry (slot) and without one (the game's row index).
SELECTOR = {True: {"tech": 0, "civic": 0}, False: {"tech": 2, "civic": 2}}
# Everything offered once the current research is cleared, by the same two readings.
OFFERED_WHEN_CLEARED = {
    True: {"tech": {0, 3}, "civic": {0, 3}},
    False: {"tech": {1, 2}, "civic": {1, 2}},
}
STATE_KEY = {"tech": "tech", "civic": "civic"}


def current_of(parsed: GameStateData, kind: str) -> Any:
    player = parsed.players[0]
    if kind == "tech":
        return player.current_research
    return player.civic_status.type if player.civic_status else None


class World:
    """One mod runtime, its latest export and the converters, with or without a registry."""

    def __init__(self, runtime: Any, registry: Any) -> None:
        self.runtime = runtime
        self.registry = registry
        self.with_registry = registry is not None
        self.state = StateConverter(research_registry=registry)
        self.actions = ActionConverter(research_registry=registry)
        self.refresh()

    def refresh(self) -> None:
        self.raw = h.export_state(self.runtime)
        self.parsed = GameStateData.from_dict(self.raw)
        self.converted = self.state.convert(self.parsed)
        self.mask = self.actions.generate_action_mask(self.parsed.legal_actions, self.converted)

    def bits(self, kind: str) -> set[int]:
        return research_bits(self.actions, self.mask, CATEGORY[kind])

    def command(self, kind: str, selector: int) -> LuaCommand:
        index = flat(self.actions, CATEGORY[kind], selector)
        assert self.mask[index] == 1.0, f"selector {selector} is not offered for {kind}"
        return self.actions.convert(index, self.converted, self.parsed.legal_actions)

    def apply(self) -> Any:
        return self.runtime.eval("Civ6Stub.applyRequests()")


@pytest.fixture(params=[True, False], ids=["registry", "no-registry"])
def world(request: pytest.FixtureRequest) -> World:
    registry = request.getfixturevalue("reg") if request.param else None
    return World(h.load_mod(), registry)


class TestStubPremises:
    """What the end-to-end tests assume about the mod and the stub, without the feature."""

    def test_default_offers_are_writing_and_foreign_trade(self) -> None:
        state = h.export_state(h.load_mod())
        by_name = {a["action"]: a for a in state["legal_actions"]}
        assert by_name["research_tech"]["items"] == [2]
        assert by_name["research_tech"]["item_names"] == ["TECH_WRITING"]
        assert by_name["research_civic"]["items"] == [2]
        assert by_name["research_civic"]["item_names"] == ["CIVIC_FOREIGN_TRADE"]

    def test_clearing_the_current_choice_offers_two_of_each(self) -> None:
        runtime = h.load_mod()
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        runtime.execute("Civ6Stub.players[0].civic.current = -1")
        by_name = {a["action"]: a for a in h.export_state(runtime)["legal_actions"]}
        assert by_name["research_tech"]["item_names"] == ["TECH_MINING", "TECH_WRITING"]
        assert by_name["research_civic"]["item_names"] == [
            "CIVIC_CRAFTSMANSHIP",
            "CIVIC_FOREIGN_TRADE",
        ]

    @pytest.mark.parametrize("item", ["TECH_WRITING", 2])
    def test_the_executor_takes_a_type_name_or_a_row_index(self, item: Any) -> None:
        runtime = h.load_mod()
        command = {"action_type": "research_tech", "item_id": item}
        assert h.execute(runtime, command) == (True, None, "Research requested: TECH_WRITING")
        assert runtime.eval("Civ6Stub.applyRequests()") == 1
        parsed = GameStateData.from_dict(h.export_state(runtime))
        assert parsed.players[0].current_research == "TECH_WRITING"

    def test_the_default_blocker_is_the_one_the_tests_set(self) -> None:
        runtime = h.load_mod()
        runtime.execute(f"Civ6Stub.notifications[0] = {{ {BLOCKERS['tech']} }}")
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        success, code, _ = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        command = {"action_type": "research_tech", "item_id": "TECH_WRITING"}
        assert h.execute(runtime, command)[0] is True
        assert runtime.eval("Civ6Stub.applyRequests()") == 1
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")


class TestEndToEnd:
    def test_exported_research_reaches_the_encoder_at_the_registry_slots(self, reg: Any) -> None:
        runtime = h.load_mod()
        state = GameStateData.from_dict(h.export_state(runtime))
        conv = StateConverter(research_registry=reg)
        vector = encode(conv, state)
        # The stub's player has TECH_POTTERY (slot 2, row 0) and CIVIC_CODE_OF_LAWS (slot 1).
        assert vector[70 + 2] == 1.0
        assert vector[180 + 1] == 1.0
        assert float(vector[70:170].sum()) == 1.0
        assert float(vector[180:280].sum()) == 1.0
        assert conv.research_stats == {
            "conversions_without_registry": 0,
            "unmapped_techs": 0,
            "unmapped_civics": 0,
            "slots_out_of_range": 0,
        }

    def test_without_a_registry_the_exported_research_is_not_encoded(self) -> None:
        runtime = h.load_mod()
        state = GameStateData.from_dict(h.export_state(runtime))
        conv = StateConverter()
        vector = encode(conv, state)
        assert float(vector[70:170].sum()) == 0.0
        assert float(vector[180:280].sum()) == 0.0
        assert conv.research_stats["conversions_without_registry"] == 1

    def test_the_mask_offers_exactly_the_default_choices(self, world: World) -> None:
        with_registry = world.with_registry
        assert world.bits("tech") == {SELECTOR[with_registry]["tech"]}
        assert world.bits("civic") == {SELECTOR[with_registry]["civic"]}

    def test_a_choice_goes_through_to_the_game(self, world: World) -> None:
        with_registry = world.with_registry
        for kind in ("tech", "civic"):
            assert current_of(world.parsed, kind) != CHOICE[kind], "already chosen; vacuous"
            cmd = world.command(kind, SELECTOR[with_registry][kind])
            assert cmd.action_type == ACTION_TYPE[kind]
            if with_registry:
                assert cmd.item_id == CHOICE[kind]
            else:
                assert cmd.item_id == SELECTOR[False][kind]
                assert type(cmd.item_id) is int
            result = h.execute(world.runtime, cmd.to_dict())
            assert result == (True, None, MESSAGE[kind])
            assert world.apply() == 1
            world.refresh()
            assert current_of(world.parsed, kind) == CHOICE[kind]

    @pytest.mark.parametrize("kind", ["tech", "civic"])
    def test_a_blocked_turn_is_unblocked_by_the_choice(self, world: World, kind: str) -> None:
        with_registry = world.with_registry
        runtime = world.runtime
        runtime.execute(f"Civ6Stub.notifications[0] = {{ {BLOCKERS[kind]} }}")
        runtime.execute(f"Civ6Stub.players[0].{STATE_KEY[kind]}.current = -1")

        success, code, message = h.execute(runtime, END_TURN)
        assert (success, code) == (False, "END_TURN_BLOCKED")
        assert BLOCKER_NAME[kind] in message

        world.refresh()
        assert world.parsed.end_turn_blocking == BLOCKER_NAME[kind]
        assert world.bits(kind) == OFFERED_WHEN_CLEARED[with_registry][kind]

        cmd = world.command(kind, SELECTOR[with_registry][kind])
        if with_registry:
            assert cmd.item_id == CHOICE[kind]
        else:
            assert cmd.item_id == SELECTOR[False][kind]
        assert h.execute(runtime, cmd.to_dict()) == (True, None, MESSAGE[kind])
        assert world.apply() == 1

        world.refresh()
        assert world.parsed.end_turn_blocking is None
        assert current_of(world.parsed, kind) == CHOICE[kind]
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")


# ---------------------------------------------------------------------------
# D12: end to end, review coverage
# ---------------------------------------------------------------------------
class TestEndToEndReviewCoverage:
    def test_a_gameinfo_shaped_rule_set(self, reg_gameinfo: Any) -> None:
        assert reg_gameinfo.tech_ids[0] == "tech_writing"
        world = World(h.load_mod(), reg_gameinfo)
        vector = np.asarray(world.state.convert_and_encode(world.parsed)[1].detach().numpy())
        assert vector[70 + 2] == 1.0  # TECH_POTTERY is slot 2
        assert world.bits("tech") == {0}
        cmd = world.command("tech", 0)
        assert cmd.item_id == "TECH_WRITING"
        assert h.execute(world.runtime, cmd.to_dict()) == (
            True,
            None,
            "Research requested: TECH_WRITING",
        )
        assert world.apply() == 1
        world.refresh()
        assert current_of(world.parsed, "tech") == "TECH_WRITING"

    def test_without_civic_slots_no_civic_is_offered_and_it_is_counted(
        self, reg_no_civics: Any
    ) -> None:
        world = World(h.load_mod(), reg_no_civics)
        assert world.bits("tech") == {0}
        assert world.bits("civic") == set()
        assert world.actions.research_stats["unmapped_research_items"] == 1
        assert world.state.research_stats["unmapped_civics"] == 1

    def test_without_civic_slots_a_civic_blocked_turn_stays_blocked(
        self, reg_no_civics: Any
    ) -> None:
        # The documented limit: no civic list, no civic choice; the turn cannot end.
        world = World(h.load_mod(), reg_no_civics)
        runtime = world.runtime
        runtime.execute(f"Civ6Stub.notifications[0] = {{ {BLOCKERS['civic']} }}")
        runtime.execute("Civ6Stub.players[0].civic.current = -1")
        world.refresh()
        assert world.parsed.end_turn_blocking == "ENDTURN_BLOCKING_CIVIC"
        assert world.bits("civic") == set()
        assert world.actions.research_stats["unmapped_research_items"] == 3
        assert h.execute(runtime, END_TURN)[:2] == (False, "END_TURN_BLOCKED")
        assert world.apply() == 0
        assert h.execute(runtime, END_TURN)[:2] == (False, "END_TURN_BLOCKED")

    def test_without_civic_slots_a_tech_choice_still_works(self, reg_no_civics: Any) -> None:
        world = World(h.load_mod(), reg_no_civics)
        runtime = world.runtime
        runtime.execute(f"Civ6Stub.notifications[0] = {{ {BLOCKERS['tech']} }}")
        runtime.execute("Civ6Stub.players[0].tech.current = -1")
        world.refresh()
        assert world.bits("tech") == {0, 3}
        cmd = world.command("tech", 0)
        assert cmd.item_id == "TECH_WRITING"
        assert h.execute(runtime, cmd.to_dict())[0] is True
        assert world.apply() == 1
        assert h.execute(runtime, END_TURN) == (True, None, "End turn requested")
